#!/bin/bash

echo "=========================================="
echo "HazeRadar Startup Script"
echo "=========================================="

# Get PORT from environment, default to 8000
PORT=${PORT:-8000}

echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  Working Directory: $(pwd)"
echo "  PORT: $PORT"

# Show files
echo ""
echo "Files in directory:"
ls -lh

# Check model files
echo ""
echo "Checking model files:"
if [ -f "deployment_model.pth" ]; then
    echo "  ✓ deployment_model.pth exists ($(du -h deployment_model.pth | cut -f1))"
else
    echo "  ✗ deployment_model.pth NOT FOUND"
fi

if [ -f "model_config_fixed.json" ]; then
    echo "  ✓ model_config_fixed.json exists"
else
    echo "  ✗ model_config_fixed.json NOT FOUND"
fi

# Test imports
echo ""
echo "Testing imports:"
python -c "import torch; print('  ✓ torch')" 2>&1 || echo "  ✗ torch"
python -c "import torch_geometric; print('  ✓ torch_geometric')" 2>&1 || echo "  ✗ torch_geometric"
python -c "from improved_gnn_model import SpatioTemporalHazeGNN; print('  ✓ improved_gnn_model')" 2>&1 || echo "  ✗ improved_gnn_model"

echo ""
echo "=========================================="
echo "Starting uvicorn on port $PORT..."
echo "=========================================="
echo ""

# Start the application - PORT is now a proper number
exec uvicorn backend_api:app --host 0.0.0.0 --port $PORT --log-level info
