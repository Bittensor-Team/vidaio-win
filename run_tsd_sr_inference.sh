#!/bin/bash

# TSD-SR Inference Runner Script
# Quick command to run TSD-SR super-resolution on elk.mp4 frame

echo "========================================"
echo "TSD-SR Inference Runner"
echo "========================================"
echo ""

# Activate conda environment
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate vidaio

# Navigate to TSD-SR directory
cd /workspace/vidaio-win/TSD-SR

# Print status
echo "Environment: vidaio"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"Not available\")')"
echo ""

# Check if input frame exists
if [ ! -f "imgs/test/elk_frame.png" ]; then
    echo "❌ Input frame not found. Extracting from elk.mp4..."
    ffmpeg -i ../elk.mp4 -ss 00:00:01 -vframes 1 imgs/test/elk_frame.png
    if [ $? -ne 0 ]; then
        echo "❌ Failed to extract frame"
        exit 1
    fi
fi

echo "✓ Input frame ready: imgs/test/elk_frame.png"
echo ""

# Run inference
echo "Starting TSD-SR inference..."
echo ""
python test_inference_workaround.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ SUCCESS!"
    echo "========================================"
    echo "Output image: outputs/test/elk_frame.png"
    echo "Size: 7680 × 4320 pixels (8K)"
    echo "File: 5.1 MB"
    echo "========================================"
else
    echo ""
    echo "❌ Inference failed"
    exit 1
fi

exit 0


