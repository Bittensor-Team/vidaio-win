#!/bin/bash

# TSD-SR Complete Setup Script
# This script sets up TSD-SR from a fresh GitHub repository clone
# Includes environment creation, package installation, and model downloads

set -e  # Exit on any error

echo "=========================================="
echo "TSD-SR Complete Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the TSD-SR directory
if [ ! -f "test/test_tsdsr.py" ]; then
    print_error "Please run this script from the TSD-SR root directory"
    exit 1
fi

# Step 1: Create conda environment
print_status "Creating conda environment 'vidaio'..."
if conda env list | grep -q "vidaio"; then
    print_warning "Environment 'vidaio' already exists. Skipping creation."
else
    conda create -n vidaio python=3.12 -y
    print_success "Conda environment 'vidaio' created"
fi

# Step 2: Activate environment and install packages
print_status "Activating environment and installing packages..."

# Source conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vidaio

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
print_status "Installing core dependencies..."
pip install diffusers==0.29.1
pip install transformers==4.49.0
pip install accelerate
pip install peft==0.15.0
pip install pillow==10.3.0
pip install numpy==1.24.4
pip install wandb
pip install loralib==0.1.2
pip install pyiqa==0.1.13

# Install additional packages we discovered were needed
print_status "Installing additional required packages..."
pip install einops
pip install gdown
pip install huggingface_hub[cli]
pip install opencv-python
pip install tqdm
pip install safetensors

print_success "All packages installed"

# Step 3: Create necessary directories
print_status "Creating directory structure..."
mkdir -p checkpoint/sd3
mkdir -p checkpoint/tsdsr
mkdir -p checkpoint/teacher
mkdir -p dataset/default
mkdir -p dataset/null
mkdir -p imgs/test
mkdir -p outputs/test

print_success "Directory structure created"

# Step 4: Download TSD-SR models from Google Drive
print_status "Downloading TSD-SR models from Google Drive..."
print_warning "This may take several minutes as models are large (~1.5GB total)"

# Download the entire TSD-SR models folder
gdown --folder https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI

print_success "TSD-SR models downloaded"

# Step 5: Organize downloaded models
print_status "Organizing downloaded models..."

# Copy TSD-SR LoRA weights
if [ -d "TSD-SR-Models/checkpoint/tsdsr" ]; then
    cp -r TSD-SR-Models/checkpoint/tsdsr/* checkpoint/tsdsr/
    print_success "TSD-SR LoRA weights copied"
else
    print_error "TSD-SR LoRA weights not found in downloaded folder"
    exit 1
fi

# Copy teacher model weights
if [ -d "TSD-SR-Models/checkpoint/teacher" ]; then
    cp -r TSD-SR-Models/checkpoint/teacher/* checkpoint/teacher/
    print_success "Teacher model weights copied"
fi

# Copy prompt embeddings
if [ -d "TSD-SR-Models/dataset/default" ]; then
    cp -r TSD-SR-Models/dataset/default/* dataset/default/
    print_success "Default prompt embeddings copied"
fi

if [ -d "TSD-SR-Models/dataset/null" ]; then
    cp -r TSD-SR-Models/dataset/null/* dataset/null/
    print_success "Null prompt embeddings copied"
fi

# Step 6: Download SD3 model from HuggingFace
print_status "Downloading Stable Diffusion 3 model from HuggingFace..."
print_warning "This requires HuggingFace authentication and may take 10-15 minutes (~5GB download)"

# Check if user has HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    print_warning "HUGGINGFACE_TOKEN environment variable not set"
    print_warning "You may need to authenticate with HuggingFace to download SD3 model"
    print_warning "Run: huggingface-cli login"
    print_warning "Or set HUGGINGFACE_TOKEN environment variable"
fi

# Download SD3 model
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir checkpoint/sd3 --local-dir-use-symlinks False

print_success "SD3 model downloaded"

# Step 7: Create sample input image if none exists
if [ ! -f "imgs/test/elk_frame.png" ]; then
    print_status "Creating sample input image..."
    
    # Check if elk.mp4 exists in parent directory
    if [ -f "../elk.mp4" ]; then
        print_status "Extracting frame from elk.mp4..."
        ffmpeg -i ../elk.mp4 -ss 00:00:01 -vframes 1 imgs/test/elk_frame.png
        print_success "Sample frame extracted from elk.mp4"
    else
        print_warning "elk.mp4 not found. Creating a placeholder image..."
        # Create a simple test image
        python -c "
from PIL import Image
import numpy as np
img = Image.new('RGB', (1920, 1080), color='lightblue')
img.save('imgs/test/elk_frame.png')
print('Placeholder image created')
"
        print_success "Placeholder test image created"
    fi
fi

# Step 8: Create inference runner script
print_status "Creating inference runner script..."

cat > run_inference.sh << 'EOF'
#!/bin/bash

# TSD-SR Inference Runner
# Quick command to run TSD-SR super-resolution

echo "========================================"
echo "TSD-SR Inference Runner"
echo "========================================"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vidaio

# Navigate to TSD-SR directory
cd "$(dirname "$0")"

# Print status
echo "Environment: vidaio"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "Not available")' 2>/dev/null || echo 'Not available')"
echo ""

# Check if input image exists
if [ ! -f "imgs/test/elk_frame.png" ]; then
    echo "❌ Input image not found: imgs/test/elk_frame.png"
    echo "Please place your input image in imgs/test/ directory"
    exit 1
fi

echo "✓ Input image ready: imgs/test/elk_frame.png"
echo ""

# Run inference with memory optimization
echo "Starting TSD-SR inference..."
echo "Using optimized settings for GPU memory..."
echo ""

python test/test_tsdsr.py \
    --pretrained_model_name_or_path checkpoint/sd3 \
    -i imgs/test/elk_frame.png \
    -o outputs/test \
    --lora_dir checkpoint/tsdsr \
    --embedding_dir dataset/default \
    --upscale 2 \
    --process_size 128 \
    --device cuda

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ SUCCESS!"
    echo "========================================"
    echo "Output image: outputs/test/elk_frame.png"
    if [ -f "outputs/test/elk_frame.png" ]; then
        echo "Size: $(identify -format '%wx%h' outputs/test/elk_frame.png 2>/dev/null || echo 'Unknown')"
        echo "File: $(ls -lh outputs/test/elk_frame.png | awk '{print $5}')"
    fi
    echo "========================================"
else
    echo ""
    echo "❌ Inference failed"
    exit 1
fi

exit 0
EOF

chmod +x run_inference.sh
print_success "Inference runner script created"

# Step 9: Create setup verification script
print_status "Creating setup verification script..."

cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""
TSD-SR Setup Verification Script
Checks if all required components are properly installed and configured
"""

import os
import sys
import torch
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ {description}: {filepath} ({size:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def main():
    print("=" * 60)
    print("TSD-SR Setup Verification")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check Python and PyTorch
    print("Python Environment:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Check required directories
    print("Directory Structure:")
    dirs = [
        "checkpoint/sd3",
        "checkpoint/tsdsr", 
        "dataset/default",
        "imgs/test",
        "outputs/test"
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path} (MISSING)")
            all_good = False
    print()
    
    # Check SD3 model files
    print("SD3 Model Files:")
    sd3_files = [
        "checkpoint/sd3/model_index.json",
        "checkpoint/sd3/transformer/config.json",
        "checkpoint/sd3/vae/config.json"
    ]
    
    for file_path in sd3_files:
        if not check_file_exists(file_path, "SD3"):
            all_good = False
    print()
    
    # Check TSD-SR LoRA weights
    print("TSD-SR LoRA Weights:")
    tsdsr_files = [
        "checkpoint/tsdsr/transformer.safetensors",
        "checkpoint/tsdsr/vae.safetensors"
    ]
    
    for file_path in tsdsr_files:
        if not check_file_exists(file_path, "TSD-SR LoRA"):
            all_good = False
    print()
    
    # Check prompt embeddings
    print("Prompt Embeddings:")
    embed_files = [
        "dataset/default/prompt_embeds.pt",
        "dataset/default/pool_embeds.pt"
    ]
    
    for file_path in embed_files:
        if not check_file_exists(file_path, "Embeddings"):
            all_good = False
    print()
    
    # Check input image
    print("Input Image:")
    if check_file_exists("imgs/test/elk_frame.png", "Input"):
        pass
    else:
        print("⚠️  No input image found. You can add your own image to imgs/test/")
    print()
    
    # Summary
    print("=" * 60)
    if all_good:
        print("✅ SETUP COMPLETE!")
        print("You can now run: ./run_inference.sh")
    else:
        print("❌ SETUP INCOMPLETE!")
        print("Please check the missing components above.")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x verify_setup.py
print_success "Setup verification script created"

# Step 10: Create requirements.txt for reference
print_status "Creating requirements.txt for reference..."

cat > requirements.txt << 'EOF'
# TSD-SR Requirements
# Core dependencies
diffusers==0.29.1
torch==2.2.2
torchvision==0.17.2
pyiqa==0.1.13
transformers==4.49.0
peft==0.15.0
pillow==10.3.0
accelerate
numpy==1.24.4
wandb
loralib==0.1.2

# Additional packages discovered during setup
einops
gdown
huggingface_hub[cli]
opencv-python
tqdm
safetensors
EOF

print_success "Requirements file created"

# Step 11: Clean up temporary files
print_status "Cleaning up temporary files..."
if [ -d "TSD-SR-Models" ]; then
    rm -rf TSD-SR-Models
    print_success "Temporary files cleaned up"
fi

# Step 12: Run verification
print_status "Running setup verification..."
python verify_setup.py

# Final summary
echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Environment: vidaio (activated)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "Not available")')"
echo ""
echo "Next steps:"
echo "1. Place your input image in: imgs/test/"
echo "2. Run inference: ./run_inference.sh"
echo "3. Check output in: outputs/test/"
echo ""
echo "Available scripts:"
echo "  ./run_inference.sh    - Run TSD-SR inference"
echo "  ./verify_setup.py     - Verify setup status"
echo ""
echo "For custom inference:"
echo "  python test/test_tsdsr.py --help"
echo ""
echo "Note: This script preserves your HUGGINGFACE_TOKEN"
echo "environment variable for model downloads."
echo "=========================================="

print_success "TSD-SR setup completed successfully!"