import os
import sys
import json
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hazeradar")

# Initialize FastAPI
app = FastAPI(title="HazeRadar API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
FEATURE_ENGINEER = None
MODEL_CONFIG = None
STARTUP_ERROR = None
DEVICE = None


class PredictionRequest(BaseModel):
    """Request with validation"""
    city: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    temperature: float = Field(..., ge=-50, le=60)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0, le=50)
    wind_direction: float = Field(..., ge=0, le=360)
    upwind_fire_count: int = Field(..., ge=0, le=1000)
    avg_fire_confidence: float = Field(..., ge=0, le=100)
    current_aqi: float = Field(..., ge=0, le=500)
    population_density: float = Field(..., ge=0, le=50000)


def safe_import(module_name, description):
    """Safely import and log"""
    try:
        if module_name == "torch":
            import torch
            logger.info(f"✓ {description}: {torch.__version__}")
            return torch
        elif module_name == "torch_geometric":
            import torch_geometric
            logger.info(f"✓ {description}: {torch_geometric.__version__}")
            return torch_geometric
        elif module_name == "improved_gnn_model":
            from improved_gnn_model import SpatioTemporalHazeGNN
            logger.info(f"✓ {description}")
            return SpatioTemporalHazeGNN
        elif module_name == "training_pipeline":
            from training_pipeline import FeatureEngineering
            logger.info(f"✓ {description}")
            return FeatureEngineering
    except Exception as e:
        logger.error(f"✗ {description}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def initialize_model():
    """Load model with detailed error handling"""
    global MODEL, FEATURE_ENGINEER, MODEL_CONFIG, STARTUP_ERROR, DEVICE
    
    logger.info("="*60)
    logger.info("STARTING MODEL INITIALIZATION")
    logger.info("="*60)
    
    try:
        # Step 1: Import torch
        logger.info("\n[1/8] Importing PyTorch...")
        torch = safe_import("torch", "PyTorch")
        if torch is None:
            raise ImportError("Failed to import torch")
        
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {DEVICE}")
        
        # Step 2: Import torch_geometric
        logger.info("\n[2/8] Importing PyTorch Geometric...")
        pyg = safe_import("torch_geometric", "PyTorch Geometric")
        if pyg is None:
            raise ImportError("Failed to import torch_geometric")
        
        # Step 3: Import model architecture
        logger.info("\n[3/8] Importing model architecture...")
        SpatioTemporalHazeGNN = safe_import("improved_gnn_model", "Model architecture")
        if SpatioTemporalHazeGNN is None:
            raise ImportError("Failed to import model architecture")
        
        # Step 4: Import feature engineering
        logger.info("\n[4/8] Importing feature engineering...")
        FeatureEngineering = safe_import("training_pipeline", "Feature engineering")
        if FeatureEngineering is None:
            raise ImportError("Failed to import feature engineering")
        
        # Step 5: Check files
        logger.info("\n[5/8] Checking model files...")
        model_path = os.getenv('MODEL_PATH', 'deployment_model.pth')
        config_path = os.getenv('CONFIG_PATH', 'model_config_fixed.json')
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Config path: {config_path}")
        logger.info(f"Current dir: {os.getcwd()}")
        logger.info(f"Files: {os.listdir('.')}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        logger.info("✓ Files exist")
        
        # Step 6: Load config
        logger.info("\n[6/8] Loading configuration...")
        with open(config_path, 'r') as f:
            MODEL_CONFIG = json.load(f)
        
        logger.info(f"✓ Config loaded")
        logger.info(f"  Features: {MODEL_CONFIG.get('num_features')}")
        logger.info(f"  Test MAE: {MODEL_CONFIG.get('test_metrics', {}).get('mae', 'N/A')}")
        
        # Step 7: Initialize model
        logger.info("\n[7/8] Initializing model architecture...")
        MODEL = SpatioTemporalHazeGNN(
            node_features=MODEL_CONFIG['num_features'],
            edge_features=1,
            hidden_dim=64,
            num_heads=4,
            lstm_layers=2,
            dropout=0.0
        ).to(DEVICE)
        
        logger.info("✓ Model architecture created")
        
        # Step 8: Load weights
        logger.info("\n[8/8] Loading model weights...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                MODEL.load_state_dict(checkpoint['model_state_dict'])
            else:
                MODEL.load_state_dict(checkpoint)
        else:
            MODEL.load_state_dict(checkpoint)
        
        MODEL.eval()
        logger.info("✓ Weights loaded")
        
        # Initialize feature engineer
        FEATURE_ENGINEER = FeatureEngineering()
        logger.info("✓ Feature engineer initialized")
        
        logger.info("\n" + "="*60)
        logger.info("✅ MODEL LOADED SUCCESSFULLY")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        STARTUP_ERROR = str(e)
        logger.error("\n" + "="*60)
        logger.error("❌ MODEL LOADING FAILED")
        logger.error("="*60)
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting HazeRadar API...")
    success = initialize_model()
    if not success:
        logger.warning("Model failed to load - API will run in degraded mode")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "HazeRadar API",
        "version": "2.0.0",
        "model_loaded": MODEL is not None,
        "startup_error": STARTUP_ERROR,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Always return 200 for health check, but include status
    status = {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "feature_engineer_ready": FEATURE_ENGINEER is not None,
        "config_loaded": MODEL_CONFIG is not None,
        "startup_error": STARTUP_ERROR,
        "device": str(DEVICE) if DEVICE else None,
        "timestamp": datetime.now().isoformat()
    }
    
    if MODEL_CONFIG:
        status["test_metrics"] = MODEL_CONFIG.get('test_metrics')
        status["num_features"] = MODEL_CONFIG.get('num_features')
    
    return status


@app.get("/debug")
async def debug_info():
    """Detailed debug information"""
    
    import_status = {}
    
    # Test imports
    try:
        import torch
        import_status["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
    except Exception as e:
        import_status["torch"] = {"error": str(e)}
    
    try:
        import torch_geometric
        import_status["torch_geometric"] = {"version": torch_geometric.__version__}
    except Exception as e:
        import_status["torch_geometric"] = {"error": str(e)}
    
    try:
        from improved_gnn_model import SpatioTemporalHazeGNN
        import_status["improved_gnn_model"] = {"status": "ok"}
    except Exception as e:
        import_status["improved_gnn_model"] = {"error": str(e)}
    
    try:
        from training_pipeline import FeatureEngineering
        import_status["training_pipeline"] = {"status": "ok"}
    except Exception as e:
        import_status["training_pipeline"] = {"error": str(e)}
    
    # File check
    model_path = os.getenv('MODEL_PATH', 'deployment_model.pth')
    config_path = os.getenv('CONFIG_PATH', 'model_config_fixed.json')
    
    files_status = {
        "model_file": {
            "path": model_path,
            "exists": os.path.exists(model_path),
            "size_mb": round(os.path.getsize(model_path) / (1024**2), 2) if os.path.exists(model_path) else None
        },
        "config_file": {
            "path": config_path,
            "exists": os.path.exists(config_path),
            "size_kb": round(os.path.getsize(config_path) / 1024, 2) if os.path.exists(config_path) else None
        }
    }
    
    return {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
        "environment_variables": {
            "MODEL_PATH": os.getenv('MODEL_PATH', 'NOT_SET'),
            "CONFIG_PATH": os.getenv('CONFIG_PATH', 'NOT_SET'),
            "PORT": os.getenv('PORT', 'NOT_SET')
        },
        "imports": import_status,
        "files": files_status,
        "model_loaded": MODEL is not None,
        "startup_error": STARTUP_ERROR
    }


@app.post("/predict")
async def predict_haze(request: PredictionRequest):
    """Make prediction"""
    
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {STARTUP_ERROR or 'Unknown'}"
        )
    
    # Simple mock response for now
    # TODO: Implement actual prediction with feature engineering
    return {
        "predicted_pm25": 50.0,
        "predicted_aqi": 120.0,
        "haze_category": "Moderate",
        "health_advice": "Model loaded but prediction not fully implemented",
        "warning": "This is a placeholder response"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
