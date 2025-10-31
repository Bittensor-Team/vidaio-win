#!/bin/bash

# Vidaio VSR Server Startup Script
# Optimized for Vidaio subnet integration

echo "🚀 Starting Vidaio VSR Server..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vidaio

# Check if model exists
MODEL_PATH="/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ VSR model not found at $MODEL_PATH"
    echo "Please ensure the model is downloaded and placed correctly"
    exit 1
fi

# Create necessary directories
mkdir -p /tmp/vidaio_vsr
mkdir -p /tmp/vidaio_vsr_output

# Check GPU availability
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ GPU detected and available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  GPU not detected, falling back to CPU"
fi

# Start the server
echo "🎬 Starting VSR server on port 29115..."
python /workspace/vidaio-win/vidaio_vsr_server.py





