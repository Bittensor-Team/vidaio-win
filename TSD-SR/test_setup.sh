#!/bin/bash

# Test script to verify TSD-SR setup
# Run this after setup_tsdsr.sh to test the installation

echo "=========================================="
echo "TSD-SR Setup Test"
echo "=========================================="
echo ""

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vidaio

# Test 1: Check Python environment
echo "Test 1: Python Environment"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Test 2: Check required packages
echo "Test 2: Required Packages"
python -c "
import diffusers
import transformers
import peft
import einops
import gdown
import huggingface_hub
print('✅ All packages imported successfully')
"
echo ""

# Test 3: Check model files
echo "Test 3: Model Files"
if [ -f "checkpoint/sd3/model_index.json" ]; then
    echo "✅ SD3 model found"
else
    echo "❌ SD3 model missing"
fi

if [ -f "checkpoint/tsdsr/transformer.safetensors" ]; then
    echo "✅ TSD-SR LoRA found"
else
    echo "❌ TSD-SR LoRA missing"
fi

if [ -f "dataset/default/prompt_embeds.pt" ]; then
    echo "✅ Prompt embeddings found"
else
    echo "❌ Prompt embeddings missing"
fi
echo ""

# Test 4: Run verification script
echo "Test 4: Running verification script"
python verify_setup.py
echo ""

echo "=========================================="
echo "Test Complete"
echo "=========================================="

