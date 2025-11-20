#!/bin/bash

echo "=========================================="
echo "HazeRadar Startup Script"
echo "=========================================="

# Show environment
echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  Working Directory: $(pwd)"
echo "  PORT: ${PORT:-8000}"

# Show files
echo ""
echo "Files in directory:"
ls -lh

# Show Python packages
echo ""
echo "Installed packages:"
pip list | grep -E "(torch|fastapi|pydantic|supabase)"

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
    echo "  Content preview:"
    head -10 model_config_fixed.json | sed 's/^/    /'
else
    echo "  ✗ model_config_fixed.json NOT FOUND"
fi

# Check Python files
echo ""
echo "Checking Python files:"
for file in backend_api.py improved_gnn_model.py training_pipeline.py; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file NOT FOUND"
    fi
done

# Test imports
echo ""
echo "Testing imports:"
python -c "import torch; print(f'  ✓ torch {torch.__version__}')" 2>&1
python -c "import torch_geometric; print(f'  ✓ torch_geometric {torch_geometric.__version__}')" 2>&1
python -c "from improved_gnn_model import SpatioTemporalHazeGNN; print('  ✓ improved_gnn_model')" 2>&1
python -c "from training_pipeline import FeatureEngineering; print('  ✓ training_pipeline')" 2>&1

echo ""
echo "=========================================="
echo "Starting uvicorn..."
echo "=========================================="
echo ""

# Start the application
exec uvicorn backend_api:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
