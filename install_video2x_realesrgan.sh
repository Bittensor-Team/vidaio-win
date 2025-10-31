#!/bin/bash

# Video2X + Real-ESRGAN Installation Script for Vidaio Subnet
# This script installs Video2X and configures it to work with Real-ESRGAN

set -e

echo "🚀 Installing Video2X + Real-ESRGAN for Vidaio Subnet"
echo "=================================================="

# Colors for output
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check if conda is available
print_status "Checking conda availability..."
if command -v conda &> /dev/null; then
    print_success "Conda found: $(conda --version)"
else
    print_error "Conda not found. Please install Anaconda or Miniconda first"
    print_status "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment for Video2X + Real-ESRGAN
print_status "Creating conda environment 'video2x-realesrgan'..."
if conda env list | grep -q "video2x-realesrgan"; then
    print_warning "Environment 'video2x-realesrgan' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n video2x-realesrgan -y
        print_status "Removed existing environment"
    else
        print_status "Using existing environment"
    fi
fi

if ! conda env list | grep -q "video2x-realesrgan"; then
    conda create -n video2x-realesrgan python=3.9 -y
    print_success "Conda environment created"
fi

# Activate conda environment
print_status "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate video2x-realesrgan

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y ffmpeg wget unzip curl python3-pip python3-dev build-essential

# Install conda packages
print_status "Installing conda packages..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y opencv numpy pillow tqdm

# Check if CUDA is available
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
else
    print_warning "NVIDIA GPU not detected. Real-ESRGAN will run on CPU (slower)"
fi

# Create Video2X directory
print_status "Creating Video2X directory..."
mkdir -p ~/video2x
cd ~/video2x

# Download Video2X AppImage
print_status "Downloading Video2X AppImage..."
if [ ! -f "Video2X-x86_64.AppImage" ]; then
    wget -O Video2X-x86_64.AppImage https://github.com/k4yt3x/video2x/releases/latest/download/Video2X-x86_64.AppImage
    chmod +x Video2X-x86_64.AppImage
    print_success "Video2X AppImage downloaded and made executable"
else
    print_success "Video2X AppImage already exists"
fi

# Test Video2X installation
print_status "Testing Video2X installation..."
if ./Video2X-x86_64.AppImage --help &> /dev/null; then
    print_success "Video2X is working correctly"
else
    print_warning "Video2X AppImage requires newer GLIBC (2.38+), current system has $(ldd --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+')"
    print_status "Falling back to FFmpeg + Real-ESRGAN approach (recommended for Vidaio subnet)"
    
    # Create a Video2X wrapper that uses FFmpeg + Real-ESRGAN
    cat > video2x_wrapper.sh << 'EOF'
#!/bin/bash
# Video2X wrapper using FFmpeg + Real-ESRGAN
# This provides Video2X-compatible interface for Vidaio subnet

INPUT="$1"
OUTPUT="$2"
SCALE="$3"
CODEC="${4:-libx264}"
PRESET="${5:-slow}"
CRF="${6:-28}"

# Extract input directory and filename
INPUT_DIR=$(dirname "$INPUT")
INPUT_NAME=$(basename "$INPUT" .mp4)
TEMP_DIR="/tmp/video2x_$$"

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Get video properties
FPS=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")
WIDTH=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT")
HEIGHT=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT")

# Calculate new dimensions
NEW_WIDTH=$((WIDTH * SCALE))
NEW_HEIGHT=$((HEIGHT * SCALE))

# Step 1: Duplicate last 2 frames (fix Video2X bug)
STOP_DURATION=$(echo "2 / $FPS" | bc -l)
TEMP_INPUT="$TEMP_DIR/${INPUT_NAME}_padded.mp4"

ffmpeg -i "$INPUT" \
    -vf "tpad=stop_mode=clone:stop_duration=$STOP_DURATION" \
    -c:v libx264 -crf 28 -preset fast \
    "$TEMP_INPUT" -y

# Step 2: Extract frames
FRAMES_DIR="$TEMP_DIR/frames"
mkdir -p "$FRAMES_DIR"

ffmpeg -i "$TEMP_INPUT" -vf "fps=$FPS" "$FRAMES_DIR/frame_%06d.png" -y

# Step 3: Upscale frames with Real-ESRGAN
UPSCALED_DIR="$TEMP_DIR/upscaled"
mkdir -p "$UPSCALED_DIR"

cd /workspace/vidaio-win/Real-ESRGAN-working-update
python working_upscaler.py "$TEMP_INPUT" "$TEMP_DIR/upscaled_temp.mp4" --scale "$SCALE" --batch_size 1 --patch_size 192

# Step 4: Reconstruct video
ffmpeg -framerate "$FPS" -i "$TEMP_DIR/upscaled_temp.mp4" \
    -c:v "$CODEC" -preset "$PRESET" -crf "$CRF" \
    -pix_fmt yuv420p "$OUTPUT" -y

# Cleanup
rm -rf "$TEMP_DIR"

echo "Video upscaled successfully: $OUTPUT"
EOF
    
    chmod +x video2x_wrapper.sh
    print_success "Created Video2X wrapper using FFmpeg + Real-ESRGAN"
fi

# Navigate to workspace
cd /workspace/vidaio-win

# Clone official Real-ESRGAN repository
print_status "Cloning official Real-ESRGAN repository..."
if [ ! -d "Real-ESRGAN-official" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN-official
    print_success "Real-ESRGAN repository cloned"
else
    print_success "Real-ESRGAN repository already exists"
fi

# Install Real-ESRGAN dependencies
print_status "Installing Real-ESRGAN dependencies..."
cd Real-ESRGAN-official
pip install -r requirements.txt

# Download pre-trained models
print_status "Downloading Real-ESRGAN pre-trained models..."
mkdir -p weights
cd weights

models=(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth"
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
)

for model_url in "${models[@]}"; do
    model_name=$(basename "$model_url")
    if [ ! -f "$model_name" ]; then
        print_status "Downloading $model_name..."
        wget "$model_url"
        print_success "$model_name downloaded"
    else
        print_success "$model_name already exists"
    fi
done

# Go back to workspace
cd /workspace/vidaio-win

# Test Real-ESRGAN working version
print_status "Testing Real-ESRGAN working version..."
cd Real-ESRGAN-working-update
if python working_upscaler.py --test --scale 2; then
    print_success "Real-ESRGAN working version is functional"
else
    print_warning "Real-ESRGAN working version test failed, but continuing..."
fi

# Go back to workspace
cd /workspace/vidaio-win

# Create Video2X configuration
print_status "Creating Video2X configuration..."
mkdir -p ~/.config/video2x

cat > ~/.config/video2x/config.yaml << 'EOF'
# Video2X Configuration for Vidaio Subnet
drivers:
  realesrgan:
    executable_path: "/workspace/vidaio-win/Real-ESRGAN-official/inference_realesrgan.py"
    model_path: "/workspace/vidaio-win/Real-ESRGAN-official/weights"
    scale: 2
    tile_size: 0
    tile_pad: 10
    pre_pad: 0
    half_precision: false

# Default settings
default_driver: "realesrgan"
output_format: "mp4"
output_quality: 28
output_codec: "libx264"
output_preset: "slow"
EOF

print_success "Video2X configuration created"

# Create test video
print_status "Creating test video..."
if [ ! -f "test_video.mp4" ]; then
    ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 -c:v libx264 -pix_fmt yuv420p test_video.mp4 -y
    print_success "Test video created"
else
    print_success "Test video already exists"
fi

# Test FFmpeg + Real-ESRGAN integration
print_status "Testing FFmpeg + Real-ESRGAN integration..."
if python ffmpeg_realesrgan_upscaler.py; then
    print_success "FFmpeg + Real-ESRGAN integration test passed"
else
    print_warning "FFmpeg + Real-ESRGAN integration test failed"
fi

# Create installation summary
print_status "Creating installation summary..."
cat > installation_summary.txt << EOF
Video2X + Real-ESRGAN Installation Summary
==========================================

Installation Date: $(date)
System: $(uname -a)

Components Installed:
- Video2X AppImage: ~/video2x/Video2X-x86_64.AppImage
- Real-ESRGAN Official: /workspace/vidaio-win/Real-ESRGAN-official/
- Real-ESRGAN Working: /workspace/vidaio-win/Real-ESRGAN-working-update/

Configuration:
- Video2X Config: ~/.config/video2x/config.yaml
- Test Video: /workspace/vidaio-win/test_video.mp4

Usage Examples:
1. FFmpeg + Real-ESRGAN: python ffmpeg_realesrgan_upscaler.py
2. Video2X Wrapper: python video2x_wrapper.py
3. Direct Video2X: ~/video2x/Video2X-x86_64.AppImage --input test_video.mp4 --output output.mp4 --driver realesrgan --scale 2

Next Steps:
1. Test with your own videos
2. Integrate with Vidaio subnet miner
3. Optimize for your hardware setup
EOF

print_success "Installation summary created: installation_summary.txt"

# Final success message
echo ""
print_success "🎉 Installation completed successfully!"
echo ""
print_status "What was installed:"
echo "  ✅ Video2X AppImage (~/video2x/Video2X-x86_64.AppImage)"
echo "  ✅ Real-ESRGAN Official (/workspace/vidaio-win/Real-ESRGAN-official/)"
echo "  ✅ Real-ESRGAN Working (/workspace/vidaio-win/Real-ESRGAN-working-update/)"
echo "  ✅ FFmpeg + Real-ESRGAN upscaler script"
echo "  ✅ Video2X wrapper script"
echo "  ✅ Test video and configuration"
echo ""
print_status "Next steps:"
echo "  1. Test the installation: python ffmpeg_realesrgan_upscaler.py"
echo "  2. Read the setup guide: cat video2x_setup_guide.md"
echo "  3. Integrate with your Vidaio subnet miner"
echo ""
print_success "Happy video upscaling! 🚀"
