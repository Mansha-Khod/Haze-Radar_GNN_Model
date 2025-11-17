from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from supabase import create_client, Client

# Import our models
from improved_gnn_model import SpatioTemporalHazeGNN, HazeHorizonSimulator, DynamicGraphBuilder
from training_pipeline import FeatureEngineering

# Initialize FastAPI
app = FastAPI(
    title="HazeRadar API",
    description="Advanced Haze Prediction System using Spatio-Temporal GNN",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
FEATURE_ENGINEER = None
MODEL_CONFIG = None
SUPABASE_CLIENT = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request for real-time prediction"""
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


class ForecastRequest(BaseModel):
    """Request for multi-hour forecast"""
    hours: int = Field(default=72, ge=1, le=168)  # Up to 7 days
    include_uncertainty: bool = True


class HazeHorizonRequest(BaseModel):
    """Request for what-if scenario simulation"""
    fire_latitude: float = Field(..., ge=-90, le=90)
    fire_longitude: float = Field(..., ge=-180, le=180)
    fire_intensity: float = Field(..., ge=0, le=100)
    simulation_hours: int = Field(default=24, ge=1, le=72)


class CrowdVisionUpload(BaseModel):
    """Citizen haze image upload"""
    latitude: float
    longitude: float
    image_base64: str
    timestamp: Optional[datetime] = None


class PredictionResponse(BaseModel):
    """Response for predictions"""
    predicted_pm25: float
    predicted_aqi: float
    haze_category: str
    health_advice: str
    uncertainty: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None


class ForecastResponse(BaseModel):
    """Response for forecast"""
    forecasts: List[Dict[str, Any]]
    summary: Dict[str, Any]


# Utility functions
# Add this to the beginning of your backend_api.py, replacing the initialize_model function

def initialize_model():
    """Load trained model and configuration"""
    global MODEL, FEATURE_ENGINEER, MODEL_CONFIG, SUPABASE_CLIENT
    
    model_path = os.getenv('MODEL_PATH', 'deployment_model.pth')
    config_path = os.getenv('CONFIG_PATH', 'model_config_fixed.json')
    
    print(f"Looking for model at: {model_path}")
    print(f"Looking for config at: {config_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Available files:", os.listdir('.'))
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        print("Available files:", os.listdir('.'))
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            MODEL_CONFIG = json.load(f)
        print(f"✓ Config loaded successfully")
    except Exception as e:
        print(f"ERROR loading config: {e}")
        raise
    
    # Initialize model
    try:
        MODEL = SpatioTemporalHazeGNN(
            node_features=MODEL_CONFIG['num_features'],
            edge_features=1,
            hidden_dim=64,
            num_heads=4,
            lstm_layers=2,
            dropout=0.0  # No dropout for inference
        ).to(DEVICE)
        print(f"✓ Model architecture initialized")
    except Exception as e:
        print(f"ERROR initializing model architecture: {e}")
        raise
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
        print(f"✓ Model weights loaded")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        raise
    
    # Initialize feature engineer
    try:
        FEATURE_ENGINEER = FeatureEngineering()
        print(f"✓ Feature engineer initialized")
    except Exception as e:
        print(f"ERROR initializing feature engineer: {e}")
        raise
    
    # Initialize Supabase
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if supabase_url and supabase_key:
        try:
            SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
            print(f"✓ Supabase connected")
        except Exception as e:
            print(f"WARNING: Supabase connection failed: {e}")
    else:
        print("WARNING: Supabase credentials not provided")
    
    print(f"\n{'='*60}")
    print(f"MODEL LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Test MAE: {MODEL_CONFIG.get('test_metrics', {}).get('mae', 'N/A'):.2f}")
    print(f"Test R²: {MODEL_CONFIG.get('test_metrics', {}).get('r2', 'N/A'):.3f}")
    print(f"Features: {MODEL_CONFIG.get('num_features', 'N/A')}")
    print(f"{'='*60}\n")

def pm25_to_aqi(pm25):
    """Convert PM2.5 to AQI using EPA formula"""
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
    
    return 500  # Hazardous


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


def prepare_input_data(request: PredictionRequest) -> torch.Tensor:
    """Prepare input features from request"""
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
    feature_cols = MODEL_CONFIG['feature_cols']
    # Remove lagged features if not available
    feature_cols = [col for col in feature_cols if col in df.columns or not 'lag' in col]
    
    # Add missing features with defaults
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Transform
    features = torch.tensor(df[feature_cols].values, dtype=torch.float).to(DEVICE)
    
    return features


# API Endpoints

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
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "feature_engineer_ready": FEATURE_ENGINEER is not None,
        "supabase_connected": SUPABASE_CLIENT is not None,
        "test_metrics": MODEL_CONFIG.get('test_metrics') if MODEL_CONFIG else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_haze(request: PredictionRequest):
    """
    Make a single haze prediction for given location and conditions
    """
    try:
        # Prepare input
        features = prepare_input_data(request)
        
        # Create minimal PyG data structure
        from torch_geometric.data import Data
        
        positions = torch.tensor([[request.latitude, request.longitude]], dtype=torch.float).to(DEVICE)
        wind_speed = torch.tensor([request.wind_speed], dtype=torch.float).to(DEVICE)
        wind_direction = torch.tensor([request.wind_direction], dtype=torch.float).to(DEVICE)
        
        # Self-loop edge
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
        edge_attr = torch.ones(1, 1).to(DEVICE)
        
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            positions=positions,
            wind_speed=wind_speed,
            wind_direction=wind_direction
        )
        
        # Predict
        with torch.no_grad():
            output = MODEL(data, return_uncertainty=True)
        
        predicted_pm25 = output['prediction'].item()
        uncertainty = output['uncertainty'].item()
        
        # Convert to AQI
        predicted_aqi = pm25_to_aqi(predicted_pm25)
        category, advice = get_health_category(predicted_aqi)
        
        # Confidence interval (95%)
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_haze(request: ForecastRequest):
    """
    Generate multi-hour haze forecast for all monitored cities
    """
    try:
        if not SUPABASE_CLIENT:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Fetch latest data from Supabase
        response = SUPABASE_CLIENT.table("gnn_training_data").select("*").order('timestamp', desc=True).limit(50).execute()
        current_data = pd.DataFrame(response.data)
        
        if current_data.empty:
            raise HTTPException(status_code=404, detail="No current data available")
        
        # Get unique cities
        cities = current_data['city'].unique()
        
        forecasts = []
        
        for hour_ahead in range(1, request.hours + 1):
            hour_forecasts = []
            
            for city in cities:
                city_data = current_data[current_data['city'] == city].iloc[0]
                
                # Simulate time progression (simplified)
                # In production, integrate weather forecast API
                forecast_time = datetime.now() + timedelta(hours=hour_ahead)
                
                pred_request = PredictionRequest(
                    city=city,
                    latitude=city_data['latitude'],
                    longitude=city_data['longitude'],
                    temperature=city_data['temperature'],
                    humidity=city_data['humidity'],
                    wind_speed=city_data['wind_speed'],
                    wind_direction=city_data['wind_direction'],
                    upwind_fire_count=city_data['upwind_fire_count'],
                    avg_fire_confidence=city_data['avg_fire_confidence'],
                    current_aqi=city_data['current_aqi'],
                    population_density=city_data['population_density']
                )
                
                prediction = await predict_haze(pred_request)
                
                hour_forecasts.append({
                    "city": city,
                    "latitude": city_data['latitude'],
                    "longitude": city_data['longitude'],
                    "hour_ahead": hour_ahead,
                    "timestamp": forecast_time.isoformat(),
                    "predicted_pm25": prediction.predicted_pm25,
                    "predicted_aqi": prediction.predicted_aqi,
                    "category": prediction.haze_category,
                    "uncertainty": prediction.uncertainty if request.include_uncertainty else None
                })
            
            forecasts.extend(hour_forecasts)
        
        # Generate summary
        df_forecast = pd.DataFrame(forecasts)
        summary = {
            "total_forecasts": len(forecasts),
            "cities_covered": len(cities),
            "hours_ahead": request.hours,
            "max_pm25_predicted": float(df_forecast['predicted_pm25'].max()),
            "max_pm25_city": df_forecast.loc[df_forecast['predicted_pm25'].idxmax(), 'city'],
            "avg_pm25": float(df_forecast['predicted_pm25'].mean()),
            "unhealthy_hours": int((df_forecast['predicted_aqi'] > 150).sum())
        }
        
        return ForecastResponse(
            forecasts=forecasts,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/haze-horizon")
async def simulate_haze_horizon(request: HazeHorizonRequest):
    """
    Simulate 'what-if' scenario with analyst-drawn fire location
    """
    try:
        if not SUPABASE_CLIENT:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Get current system state
        response = SUPABASE_CLIENT.table("gnn_training_data").select("*").order('timestamp', desc=True).limit(100).execute()
        current_df = pd.DataFrame(response.data)
        
        # Prepare data similar to training
        from training_pipeline import HazeTrainer
        temp_trainer = HazeTrainer(MODEL)
        temp_trainer.feature_engineer = FEATURE_ENGINEER
        
        city_graph_response = SUPABASE_CLIENT.table("city_graph_structure").select("*").execute()
        city_graph = pd.DataFrame(city_graph_response.data)
        
        # Create PyG data (simplified)
        from torch_geometric.data import Data
        
        # Feature engineering
        current_df = FEATURE_ENGINEER.engineer_all_features(current_df, create_lags=False)
        feature_cols = [col for col in MODEL_CONFIG['feature_cols'] if col in current_df.columns and 'lag' not in col]
        
        x = torch.tensor(current_df[feature_cols].values, dtype=torch.float).to(DEVICE)
        positions = torch.tensor(current_df[['latitude', 'longitude']].values, dtype=torch.float).to(DEVICE)
        wind_speed = torch.tensor(current_df['wind_speed'].values, dtype=torch.float).to(DEVICE)
        wind_direction = torch.tensor(current_df['wind_direction'].values, dtype=torch.float).to(DEVICE)
        
        # Simple graph
        n_nodes = len(current_df)
        edge_index = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long).t().to(DEVICE)
        edge_attr = torch.ones(n_nodes, 1).to(DEVICE)
        
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            positions=positions, wind_speed=wind_speed, wind_direction=wind_direction
        )
        
        # Initialize simulator
        simulator = HazeHorizonSimulator(MODEL).to(DEVICE)
        
        # Run simulation
        new_fire_location = torch.tensor([request.fire_latitude, request.fire_longitude], dtype=torch.float).to(DEVICE)
        
        predictions = simulator.simulate_new_fire(
            data, 
            new_fire_location, 
            request.fire_intensity,
            request.simulation_hours
        )
        
        # Format results
        simulation_results = []
        for hour, pred in enumerate(predictions):
            pm25_values = pred['prediction'].cpu().numpy().flatten()
            
            for i, city in enumerate(current_df['city'].unique()[:len(pm25_values)]):
                simulation_results.append({
                    "hour": hour + 1,
                    "city": city,
                    "latitude": current_df.iloc[i]['latitude'],
                    "longitude": current_df.iloc[i]['longitude'],
                    "predicted_pm25": float(pm25_values[i]),
                    "predicted_aqi": float(pm25_to_aqi(pm25_values[i]))
                })
        
        return {
            "simulation": {
                "fire_location": {"lat": request.fire_latitude, "lon": request.fire_longitude},
                "fire_intensity": request.fire_intensity,
                "hours_simulated": request.simulation_hours
            },
            "results": simulation_results,
            "summary": {
                "max_impact_pm25": max([r['predicted_pm25'] for r in simulation_results]),
                "cities_affected": len(set([r['city'] for r in simulation_results if r['predicted_aqi'] > 100])),
                "peak_hour": max(simulation_results, key=lambda x: x['predicted_pm25'])['hour']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/crowdvision/upload")
async def upload_crowdvision(request: CrowdVisionUpload, background_tasks: BackgroundTasks):
    """
    Accept citizen haze image uploads for validation
    """
    try:
        # Store in database
        if SUPABASE_CLIENT:
            upload_data = {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "image_data": request.image_base64[:100],  # Store thumbnail reference
                "timestamp": request.timestamp.isoformat() if request.timestamp else datetime.now().isoformat(),
                "validated": False
            }
            
            SUPABASE_CLIENT.table("crowdvision_uploads").insert(upload_data).execute()
        
        # TODO: Implement CNN-based haze density estimation from image
        # For now, return mock validation
        
        return {
            "status": "received",
            "message": "Thank you for your contribution to HazeRadar!",
            "estimated_haze": "moderate",  # Placeholder
            "points_earned": 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/metrics")
async def get_model_metrics():
    """Return model performance metrics"""
    if not MODEL_CONFIG:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "test_metrics": MODEL_CONFIG.get('test_metrics'),
        "features_used": len(MODEL_CONFIG.get('feature_cols', [])),
        "training_date": MODEL_CONFIG.get('training_date'),
        "model_architecture": {
            "type": "Spatio-Temporal Graph Neural Network",
            "layers": "GAT + LSTM + Attention",
            "dynamic_graphs": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
