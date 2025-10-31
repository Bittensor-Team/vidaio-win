# TSD-SR Complete Setup Guide

This repository contains a complete setup script that automates the entire TSD-SR (Text-to-Image Diffusion Super-Resolution) installation and configuration process.

## Quick Start

For a fresh GitHub repository clone, simply run:

```bash
chmod +x setup_tsdsr.sh
./setup_tsdsr.sh
```

This single command will:
- Create a conda environment named `vidaio`
- Install all required Python packages
- Download TSD-SR models from Google Drive
- Download Stable Diffusion 3 from HuggingFace
- Set up directory structure
- Create helper scripts
- Verify the installation

## Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python 3.12**
- **CUDA-compatible GPU** (recommended)
- **HuggingFace Account** (for SD3 model download)
- **~10GB free disk space**

## What the Setup Script Does

### 1. Environment Creation
- Creates conda environment `vidaio` with Python 3.12
- Installs PyTorch with CUDA support
- Installs all required dependencies

### 2. Package Installation
The script installs these packages in the correct order:

**Core Dependencies:**
- `torch==2.2.2` (with CUDA 12.1)
- `torchvision==0.17.2`
- `diffusers==0.29.1`
- `transformers==4.49.0`
- `accelerate`
- `peft==0.15.0`
- `pillow==10.3.0`
- `numpy==1.24.4`
- `wandb`
- `loralib==0.1.2`
- `pyiqa==0.1.13`

**Additional Packages:**
- `einops` (tensor operations)
- `gdown` (Google Drive downloads)
- `huggingface_hub[cli]` (HuggingFace integration)
- `opencv-python` (image processing)
- `tqdm` (progress bars)
- `safetensors` (model format)

### 3. Model Downloads
- **TSD-SR Models** (~1.5GB) from Google Drive
  - LoRA weights (transformer.safetensors, vae.safetensors)
  - Prompt embeddings (prompt_embeds.pt, pool_embeds.pt)
  - Teacher model weights
- **Stable Diffusion 3** (~5GB) from HuggingFace
  - Requires HuggingFace authentication
  - Downloads to `checkpoint/sd3/`

### 4. Directory Structure
Creates the following structure:
```
TSD-SR/
├── checkpoint/
│   ├── sd3/              # Stable Diffusion 3 model
│   ├── tsdsr/            # TSD-SR LoRA weights
│   └── teacher/          # Teacher model weights
├── dataset/
│   ├── default/          # Default prompt embeddings
│   └── null/             # Null prompt embeddings
├── imgs/test/            # Input images
├── outputs/test/         # Output images
├── setup_tsdsr.sh       # Main setup script
├── run_inference.sh      # Inference runner
├── verify_setup.py       # Setup verification
└── requirements.txt      # Package list
```

## Usage After Setup

### Run Inference
```bash
./run_inference.sh
```

### Custom Inference
```bash
# Activate environment
conda activate vidaio

# Run with custom parameters
python test/test_tsdsr.py \
    --pretrained_model_name_or_path checkpoint/sd3 \
    -i imgs/test/your_image.png \
    -o outputs/test \
    --lora_dir checkpoint/tsdsr \
    --embedding_dir dataset/default \
    --upscale 4 \
    --process_size 256 \
    --device cuda
```

### Verify Setup
```bash
python verify_setup.py
```

## Environment Variables

The script preserves your existing environment variables, including:
- `HUGGINGFACE_TOKEN` - For HuggingFace model downloads
- `CUDA_VISIBLE_DEVICES` - For GPU selection

## Troubleshooting

### GPU Memory Issues
If you encounter CUDA out of memory errors:
- Reduce `--process_size` (e.g., 128, 256)
- Reduce `--upscale` factor (e.g., 2 instead of 4)
- Use CPU: `--device cpu`

### HuggingFace Authentication
If SD3 download fails:
```bash
huggingface-cli login
# Enter your HuggingFace token
```

### Missing Dependencies
If any packages are missing:
```bash
conda activate vidaio
pip install -r requirements.txt
```

## File Sizes

- **TSD-SR LoRA**: ~380MB (transformer + VAE)
- **Prompt Embeddings**: ~5.5MB
- **SD3 Model**: ~5GB
- **Total**: ~5.4GB

## Performance Notes

- **Processing Time**: ~16 seconds for 2x upscale
- **Memory Usage**: ~32GB GPU memory for full resolution
- **Optimized Settings**: Uses tiled processing for large images

## Script Features

- **Error Handling**: Exits on any error with clear messages
- **Progress Tracking**: Shows download and installation progress
- **Verification**: Automatically verifies setup completion
- **Cleanup**: Removes temporary files after setup
- **Colored Output**: Easy-to-read status messages

## Support

If you encounter issues:
1. Check `python verify_setup.py` output
2. Ensure all environment variables are set
3. Verify GPU memory availability
4. Check HuggingFace authentication status

The setup script is designed to be idempotent - you can run it multiple times safely.

