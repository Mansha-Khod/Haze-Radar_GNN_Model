import os
import sys
import json
import logging
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Configure logging to see ALL errors
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hazeradar")

# Debug startup immediately
logger.info("=== STARTING DEBUG VERSION ===")
logger.info(f"Python: {sys.version}")
logger.info(f"Working dir: {os.getcwd()}")
logger.info(f"Files: {os.listdir('.')}")

app = FastAPI(title="HazeRadar Debug API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test imports one by one
def test_imports():
    logger.info("=== TESTING IMPORTS ===")
    
    imports_to_test = [
        "torch",
        "torch.nn", 
        "numpy",
        "pandas",
        "supabase",
        "torch_geometric",
        "improved_gnn_model"
    ]
    
    for import_name in imports_to_test:
        try:
            if import_name == "torch":
                import torch
                logger.info(f"✓ torch {torch.__version__}")
                logger.info(f"  CUDA available: {torch.cuda.is_available()}")
            elif import_name == "torch.nn":
                import torch.nn as nn
                logger.info("✓ torch.nn")
            elif import_name == "numpy":
                import numpy as np
                logger.info(f"✓ numpy {np.__version__}")
            elif import_name == "pandas":
                import pandas as pd
                logger.info(f"✓ pandas {pd.__version__}")
            elif import_name == "supabase":
                from supabase import create_client
                logger.info("✓ supabase")
            elif import_name == "torch_geometric":
                import torch_geometric
                logger.info(f"✓ torch_geometric {torch_geometric.__version__}")
            elif import_name == "improved_gnn_model":
                from improved_gnn_model import SpatioTemporalHazeGNN
                logger.info("✓ improved_gnn_model")
        except Exception as e:
            logger.error(f"✗ {import_name}: {e}")
            traceback.print_exc()

# Test model loading
def test_model_loading():
    logger.info("=== TESTING MODEL LOADING ===")
    
    model_path = os.getenv('MODEL_PATH', 'deployment_model.pth')
    config_path = os.getenv('CONFIG_PATH', 'model_config_fixed.json')
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Exists - Model: {os.path.exists(model_path)}")
    logger.info(f"Exists - Config: {os.path.exists(config_path)}")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✓ Config loaded: {config.keys()}")
        except Exception as e:
            logger.error(f"✗ Config load failed: {e}")
    
    if os.path.exists(model_path):
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            logger.info(f"✓ Model loaded: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                logger.info(f"  Keys: {checkpoint.keys()}")
        except Exception as e:
            logger.error(f"✗ Model load failed: {e}")
            traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    logger.info("=== STARTUP EVENT ===")
    test_imports()
    test_model_loading()
    logger.info("=== STARTUP COMPLETE ===")

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": "HazeRadar Debug API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug")
async def debug_info():
    """Detailed debug information"""
    import_info = {}
    
    # Test basic functionality
    try:
        import torch
        import_info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        import_info["torch"] = {"error": str(e)}
    
    try:
        import torch_geometric
        import_info["torch_geometric"] = {
            "version": torch_geometric.__version__
        }
    except Exception as e:
        import_info["torch_geometric"] = {"error": str(e)}
    
    return {
        "status": "debug",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files": os.listdir('.'),
        "environment_vars": {
            "MODEL_PATH": os.getenv('MODEL_PATH'),
            "CONFIG_PATH": os.getenv('CONFIG_PATH'),
            "SUPABASE_URL": "SET" if os.getenv('SUPABASE_URL') else "NOT_SET",
            "SUPABASE_KEY": "SET" if os.getenv('SUPABASE_KEY') else "NOT_SET"
        },
        "imports": import_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
