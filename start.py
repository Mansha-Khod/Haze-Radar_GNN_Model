"""
HazeRadar Startup Script
Handles PORT environment variable correctly for Railway/Render
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment, default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print("=" * 50)
    print("HazeRadar API Starting...")
    print(f"Port: {port}")
    print("=" * 50)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
```

## Step 3: Check your file structure

Make sure your files are named correctly:
- Your FastAPI app should be in **`main.py`** (not `backend_api.py`)
- You should have **`start.py`** (the new file)

## Step 4: Alternative - Use Railway's Procfile

Create a file named **`Procfile`** (no extension) in your project root:
```
web: python start.py
