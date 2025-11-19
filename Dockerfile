# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install PyG more reliably
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install torch-geometric

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
