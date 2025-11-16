FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in correct order
RUN pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install PyG dependencies
RUN pip install torch-scatter==2.1.2+pt21cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-sparse==0.6.18+pt21cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-spline-conv==1.2.2+pt21cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-geometric==2.4.0

# Install other requirements
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
