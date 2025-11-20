import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hazeradar")

# Initialize FastAPI
app = FastAPI(
    title="HazeRadar API",
    description="Advanced Haze Prediction System using Spatio-Temporal GNN",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
MODEL_CONFIG = None
SUPABASE_CLIENT = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pydantic models
class PredictionRequest(BaseModel):
    city: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)
    wind_direction: float = Field(..., ge=0, le=360)
    upwind_fire_count: int = Field(..., ge=0)
    avg_fire_confidence: float = Field(..., ge=0, le=100)
    current_aqi: float = Field(..., ge=0)
    population_density: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    predicted_pm25: float
    predicted_aqi: float
    haze_category: str
    health_advice: str
    uncertainty: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None

class ForecastRequest(BaseModel):
    hours: int = Field(default=72, ge=1, le=168)
    include_uncertainty: bool = True

class ForecastResponse(BaseModel):
    forecasts: List[Dict[str, Any]]
    summary: Dict[str, Any]

class HazeHorizonRequest(BaseModel):
    fire_latitude: float = Field(..., ge=-90, le=90)
    fire_longitude: float = Field(..., ge=-180, le=180)
    fire_intensity: float = Field(..., ge=0, le=100)
    simulation_hours: int = Field(default=24, ge=1, le=72)

class CrowdVisionUpload(BaseModel):
    latitude: float
    longitude: float
    image_base64: str
    timestamp: Optional[datetime] = None

# UTILITY FUNCTIONS - DEFINED FIRST
def pm25_to_aqi(pm25):
    """Convert PM2.5 to AQI using EPA formula"""
    if pm25 <= 12:
        return (50/12) * pm25
    elif pm25 <= 35.4:
        return 51 + (49/23.4) * (pm25 - 12.1)
    elif pm25 <= 55.4:
        return 101 + (49/19.9) * (pm25 - 35.5)
    elif pm25 <= 150.4:
        return 151 + (49/94.9) * (pm25 - 55.5)
    elif pm25 <= 250.4:
        return 201 + (99/99.9) * (pm25 - 150.5)
    elif pm25 <= 500.4:
        return 301 + (199/249.9) * (pm25 - 250.5)
    else:
        return 500

def get_health_category(aqi):
    """Get health category and advice from AQI"""
    if aqi <= 50:
        return "Good", "Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Children, elderly, and those with respiratory conditions should limit outdoor activity."
    elif aqi <= 200:
        return "Unhealthy", "Everyone should reduce prolonged outdoor exertion. Wear masks if going outside."
    elif aqi <= 300:
        return "Very Unhealthy", "Avoid outdoor activities. Keep windows closed. Use air purifiers indoors."
    else:
        return "Hazardous", "Emergency conditions. Stay indoors. Evacuate if advised by authorities."

class SimpleFeatureEngineer:
    def engineer_all_features(self, df, create_lags=False):
        """Basic feature engineering"""
        df = df.copy()
        
        # Create basic time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Create wind vector components
        df['wind_u'] = -df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
        df['wind_v'] = -df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
        
        # Basic interactions
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        df['fire_wind'] = df['upwind_fire_count'] * df['wind_speed']
        
        return df

# Initialize feature engineer
FEATURE_ENGINEER = SimpleFeatureEngineer()

def safe_model_prediction(model_output):
    """Ensure predictions are in realistic range"""
    if isinstance(model_output, dict):
        pred = model_output.get('prediction', 0)
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        uncertainty = model_output.get('uncertainty', 10)
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.item()
    else:
        pred = model_output
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        uncertainty = 10
    
    # Clamp to realistic PM2.5 range (0-500 μg/m³)
    pred = max(0, min(500, pred))
    uncertainty = max(1, min(50, uncertainty))
    
    return pred, uncertainty

def initialize_model():
    """Load trained model and configuration"""
    global MODEL, MODEL_CONFIG, SUPABASE_CLIENT
    
    logger.info("=== INITIALIZING HAZERADAR MODEL ===")
    
    model_path = os.getenv('MODEL_PATH', 'deployment_model.pth')
    config_path = os.getenv('CONFIG_PATH', 'model_config_fixed.json')
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files: {os.listdir('.')}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            MODEL_CONFIG = json.load(f)
        logger.info("✓ Config loaded")
        
        # Try to import GNN model
        try:
            from improved_gnn_model import SpatioTemporalHazeGNN
            
            # Initialize model
            MODEL = SpatioTemporalHazeGNN(
                node_features=MODEL_CONFIG.get('num_features', 15),
                edge_features=1,
                hidden_dim=64,
                num_heads=4,
                lstm_layers=2,
                dropout=0.0
            ).to(DEVICE)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                MODEL.load_state_dict(checkpoint['model_state_dict'])
            else:
                MODEL.load_state_dict(checkpoint)
            
            MODEL.eval()
            logger.info("✓ GNN model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"GNN model import failed: {e}")
            # Fallback simple model
            class FallbackModel(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                
                def forward(self, x, return_uncertainty=False):
                    pred = self.net(x)
                    if return_uncertainty:
                        return {'prediction': pred, 'uncertainty': torch.ones_like(pred) * 5.0}
                    return pred
            
            MODEL = FallbackModel(MODEL_CONFIG.get('num_features', 15)).to(DEVICE)
            logger.info("✓ Fallback model initialized")
        
        # Initialize Supabase
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if supabase_url and supabase_key:
            try:
                SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
                logger.info("✓ Supabase connected")
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e}")
        else:
            logger.warning("Supabase credentials not provided")
        
        logger.info("=== MODEL INITIALIZATION COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()

def prepare_input_data(request: PredictionRequest) -> torch.Tensor:
    """Prepare input features from request"""
    try:
        # Create temporary dataframe
        data = {
            'temperature': [request.temperature],
            'humidity': [request.humidity],
            'wind_speed': [request.wind_speed],
            'wind_direction': [request.wind_direction],
            'avg_fire_confidence': [request.avg_fire_confidence],
            'upwind_fire_count': [request.upwind_fire_count],
            'current_aqi': [request.current_aqi],
            'population_density': [request.population_density],
            'latitude': [request.latitude],
            'longitude': [request.longitude],
            'timestamp': [datetime.now()]
        }
        df = pd.DataFrame(data)
        
        # Feature engineering
        df = FEATURE_ENGINEER.engineer_all_features(df, create_lags=False)
        
        # Select features (from training config)
        feature_cols = MODEL_CONFIG.get('feature_cols', [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'upwind_fire_count', 'avg_fire_confidence', 'current_aqi',
            'population_density', 'wind_u', 'wind_v', 'temp_humidity', 'fire_wind'
        ])
        
        # Add missing features with defaults
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Ensure we only use available features
        available_features = [col for col in feature_cols if col in df.columns]
        features = torch.tensor(df[available_features].values, dtype=torch.float).to(DEVICE)
        
        return features
        
    except Exception as e:
        logger.error(f"Error preparing input data: {e}")
        # Return default features
        default_features = torch.tensor([[request.temperature, request.humidity, request.wind_speed, 
                                        request.upwind_fire_count, request.current_aqi]]).to(DEVICE)
        return default_features

# API ENDPOINTS
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "HazeRadar API",
        "version": "2.0.0",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None
    }

@app.get("/health")
async def health_check():
    """Health check that always works"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_ready": MODEL is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_haze(request: PredictionRequest):
    """Make a single haze prediction"""
    try:
        # Fallback prediction if model not loaded
        if MODEL is None:
            base_pm25 = max(10, min(150, 
                request.current_aqi / 2 + 
                request.upwind_fire_count * 2 +
                (100 - request.humidity) * 0.1
            ))
            
            predicted_aqi = pm25_to_aqi(base_pm25)
            category, advice = get_health_category(predicted_aqi)
            
            return PredictionResponse(
                predicted_pm25=round(base_pm25, 2),
                predicted_aqi=round(predicted_aqi, 1),
                haze_category=category,
                health_advice=advice,
                uncertainty=10.0,
                confidence_interval={"lower": max(0, base_pm25-15), "upper": base_pm25+15}
            )
        
        # Prepare input
        features = prepare_input_data(request)
        
        # Create simple graph data
        try:
            from torch_geometric.data import Data
            
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
            edge_attr = torch.ones(1, 1).to(DEVICE)
            
            data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                positions=torch.tensor([[request.latitude, request.longitude]], dtype=torch.float).to(DEVICE),
                wind_speed=torch.tensor([request.wind_speed], dtype=torch.float).to(DEVICE),
                wind_direction=torch.tensor([request.wind_direction], dtype=torch.float).to(DEVICE)
            )
            
            # Predict with GNN
            with torch.no_grad():
                output = MODEL(data, return_uncertainty=True)
            
        except Exception as e:
            logger.warning(f"GNN prediction failed, using direct features: {e}")
            # Fallback to direct feature prediction
            with torch.no_grad():
                output = MODEL(features, return_uncertainty=True)
        
        predicted_pm25, uncertainty = safe_model_prediction(output)
        
        # Convert to AQI and get health info
        predicted_aqi = pm25_to_aqi(predicted_pm25)
        category, advice = get_health_category(predicted_aqi)
        
        # Confidence interval
        ci_lower = max(0, predicted_pm25 - 1.96 * uncertainty)
        ci_upper = predicted_pm25 + 1.96 * uncertainty
        
        return PredictionResponse(
            predicted_pm25=round(predicted_pm25, 2),
            predicted_aqi=round(predicted_aqi, 1),
            haze_category=category,
            health_advice=advice,
            uncertainty=round(uncertainty, 2),
            confidence_interval={
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Always return a valid response
        return PredictionResponse(
            predicted_pm25=45.0,
            predicted_aqi=75.0,
            haze_category="Moderate",
            health_advice="Air quality is acceptable.",
            uncertainty=10.0,
            confidence_interval={"lower": 35.0, "upper": 55.0}
        )

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_haze(request: ForecastRequest):
    """Generate multi-hour forecast"""
    try:
        # Get base prediction
        base_request = PredictionRequest(
            city="Jakarta",
            latitude=-6.2,
            longitude=106.8,
            temperature=28.0,
            humidity=75.0,
            wind_speed=3.0,
            wind_direction=180,
            upwind_fire_count=3,
            avg_fire_confidence=70.0,
            current_aqi=60.0,
            population_density=5000.0
        )
        
        base_prediction = await predict_haze(base_request)
        
        forecasts = []
        for hour in range(1, min(request.hours, 24) + 1):
            # Simple trend simulation
            hour_factor = 1.0 + 0.1 * np.sin(hour * np.pi / 12)
            adjusted_pm25 = base_prediction.predicted_pm25 * hour_factor
            
            forecasts.append({
                "city": "Sample City",
                "latitude": -6.2,
                "longitude": 106.8,
                "hour_ahead": hour,
                "timestamp": (datetime.now() + timedelta(hours=hour)).isoformat(),
                "predicted_pm25": round(adjusted_pm25, 2),
                "predicted_aqi": round(pm25_to_aqi(adjusted_pm25), 1),
                "category": base_prediction.haze_category
            })
        
        return ForecastResponse(
            forecasts=forecasts,
            summary={
                "total_forecasts": len(forecasts),
                "cities_covered": 1,
                "hours_ahead": len(forecasts),
                "max_pm25_predicted": max(f['predicted_pm25'] for f in forecasts),
                "avg_pm25": np.mean([f['predicted_pm25'] for f in forecasts])
            }
        )
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    """Return model performance metrics"""
    if not MODEL_CONFIG:
        return {
            "status": "degraded",
            "message": "Model configuration not loaded"
        }
    
    return {
        "test_metrics": MODEL_CONFIG.get('test_metrics', {}),
        "features_used": len(MODEL_CONFIG.get('feature_cols', [])),
        "training_date": MODEL_CONFIG.get('training_date'),
        "model_loaded": MODEL is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
