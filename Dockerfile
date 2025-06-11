# Use CUDA-enabled PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Download model weights
RUN chmod +x download_weights.sh && \
    ./download_weights.sh

# Set Python path
ENV PYTHONPATH=/app

# Expose port for the API
EXPOSE 8000

# Set the entrypoint
CMD ["python3", "scripts/api.py"]
