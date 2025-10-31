# 🚀 Quick Start Guide: Video2X + Real-ESRGAN

## Prerequisites
- Ubuntu 20.04+ (or similar Linux distribution)
- NVIDIA GPU with CUDA support
- Anaconda or Miniconda installed

## 🎯 One-Command Installation

```bash
# Run the automated installation script
./install_video2x_realesrgan.sh
```

This script will:
- ✅ Create a conda environment `video2x-realesrgan`
- ✅ Install Video2X AppImage
- ✅ Install Real-ESRGAN (official + working versions)
- ✅ Download pre-trained models
- ✅ Create test video
- ✅ Test the installation

## 🔧 Manual Setup (Alternative)

### 1. Create Conda Environment
```bash
# Create environment from file
conda env create -f environment.yml

# Or create manually
conda create -n video2x-realesrgan python=3.9 -y
conda activate video2x-realesrgan
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install Video2X
```bash
mkdir -p ~/video2x
cd ~/video2x
wget https://github.com/k4yt3x/video2x/releases/latest/download/Video2X-x86_64.AppImage
chmod +x Video2X-x86_64.AppImage
```

### 3. Install Real-ESRGAN
```bash
cd /workspace/vidaio-win
git clone https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN-official
cd Real-ESRGAN-official
pip install -r requirements.txt
```

## 🧪 Testing

### Test Real-ESRGAN
```bash
conda activate video2x-realesrgan
cd /workspace/vidaio-win/Real-ESRGAN-working-update
python working_upscaler.py --test --scale 2
```

### Test Video2X
```bash
conda activate video2x-realesrgan
~/video2x/Video2X-x86_64.AppImage --help
```

### Test FFmpeg + Real-ESRGAN
```bash
conda activate video2x-realesrgan
cd /workspace/vidaio-win
python ffmpeg_realesrgan_upscaler.py
```

## 🎬 Usage Examples

### 1. FFmpeg + Real-ESRGAN (Recommended)
```python
from ffmpeg_realesrgan_upscaler import FFmpegRealESRGANUpscaler

upscaler = FFmpegRealESRGANUpscaler()
success = upscaler.upscale_video(
    input_path="input.mp4",
    output_path="output_2x.mp4", 
    scale=2,
    task_type="SD2HD"
)
```

### 2. Video2X Wrapper
```python
from video2x_wrapper import Video2XUpscaler

upscaler = Video2XUpscaler()
success = upscaler.upscale_video(
    input_path="input.mp4",
    output_path="output_2x.mp4",
    scale=2,
    task_type="SD2HD"
)
```

### 3. Direct Video2X CLI
```bash
conda activate video2x-realesrgan
~/video2x/Video2X-x86_64.AppImage \
  --input input.mp4 \
  --output output_2x.mp4 \
  --driver realesrgan \
  --scale 2 \
  --format mp4 \
  --quality 28 \
  --codec libx264 \
  --preset slow
```

## 🔧 Integration with Vidaio Subnet

### Update your upscaler worker server:
```python
# Add to upscaler_worker_server.py
from ffmpeg_realesrgan_upscaler import FFmpegRealESRGANUpscaler

@app.post("/upscale_video")
async def upscale_video(file: UploadFile = File(...), task_type: str = "SD2HD"):
    """Upscale video using FFmpeg + Real-ESRGAN"""
    try:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Upscale video
        upscaler = FFmpegRealESRGANUpscaler()
        output_path = tmp_path.replace('.mp4', '_upscaled.mp4')
        
        success = upscaler.upscale_video(tmp_path, output_path, task_type=task_type)
        
        if not success:
            raise HTTPException(status_code=500, detail="Video upscaling failed")
        
        # Return upscaled video
        response = FileResponse(output_path, media_type="video/mp4")
        response.headers["X-Temp-File"] = output_path
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 📊 Task Type Mapping

| Task Type | Scale Factor | Description |
|-----------|--------------|-------------|
| `SD2HD`   | 2x           | 480p → 1080p |
| `HD24K`   | 2x           | 1080p → 4K |
| `SD24K`   | 4x           | 480p → 4K |
| `4K28K`   | 2x           | 4K → 8K |

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA not found**: Make sure NVIDIA drivers and CUDA are installed
2. **Video2X fails**: Check if the AppImage is executable and has proper permissions
3. **Real-ESRGAN fails**: Ensure the conda environment is activated and dependencies are installed
4. **Memory issues**: Reduce batch_size and patch_size in the upscaler

### Debug Commands:
```bash
# Check CUDA
nvidia-smi

# Check conda environment
conda info --envs
conda list

# Check Video2X
~/video2x/Video2X-x86_64.AppImage --help

# Test Real-ESRGAN
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 Notes

- **Performance**: FFmpeg + Real-ESRGAN is more reliable but slower than Video2X
- **Memory**: Large videos require significant RAM (32GB+ recommended)
- **GPU**: Real-ESRGAN benefits greatly from GPU acceleration
- **Quality**: Both approaches maintain high quality output

## 🎉 Success!

Once everything is working, you'll have:
- ✅ Isolated conda environment
- ✅ Video2X + Real-ESRGAN integration
- ✅ FFmpeg + Real-ESRGAN alternative
- ✅ Test videos and scripts
- ✅ Ready for Vidaio subnet integration

Happy video upscaling! 🚀