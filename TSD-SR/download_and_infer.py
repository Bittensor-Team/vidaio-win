#!/usr/bin/env python3
"""
Script to download models and run inference with TSD-SR.
Uses the vidaio conda environment with proper CUDA support.
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

# Set environment variables for CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_cuda():
    """Check CUDA availability"""
    print("=" * 50)
    print("CUDA Configuration")
    print("=" * 50)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

def check_directories():
    """Check if required directories exist"""
    dirs = [
        "checkpoint/sd3",
        "checkpoint/tsdsr",
        "dataset/default",
        "outputs/test",
        "imgs/test"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def download_sd3_model():
    """Download SD3 model from HuggingFace"""
    sd3_path = "checkpoint/sd3"
    
    # Check if already downloaded
    if os.path.exists(f"{sd3_path}/transformer/config.json"):
        print("✓ SD3 model already exists")
        return True
    
    print("Attempting to download SD3 model...")
    print("Note: This requires HuggingFace authentication for the gated model")
    
    try:
        from huggingface_hub import snapshot_download
        
        print("Downloading SD3 model (this may take 10-30 minutes)...")
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
            local_dir=sd3_path,
            local_dir_use_symlinks=False
        )
        print("✓ SD3 model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download SD3 model: {e}")
        print("\nTry manual download:")
        print("1. Visit: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers")
        print("2. Accept terms and authenticate")
        print("3. Download and extract to checkpoint/sd3/")
        return False

def create_placeholder_embeddings():
    """Create placeholder embeddings"""
    embedding_dir = "dataset/default"
    
    prompt_embeds_path = f"{embedding_dir}/prompt_embeds.pt"
    pool_embeds_path = f"{embedding_dir}/pool_embeds.pt"
    
    if os.path.exists(prompt_embeds_path) and os.path.exists(pool_embeds_path):
        print("✓ Prompt embeddings already exist")
        return True
    
    print("Creating placeholder embeddings...")
    try:
        prompt_embeds = torch.randn(333, 4096)
        pool_embeds = torch.randn(2048)
        
        torch.save(prompt_embeds, prompt_embeds_path)
        torch.save(pool_embeds, pool_embeds_path)
        
        print("✓ Placeholder embeddings created")
        print("⚠️  For best results, download real embeddings from Google Drive")
        return True
    except Exception as e:
        print(f"✗ Failed to create embeddings: {e}")
        return False

def create_dummy_lora_weights():
    """Create dummy LoRA weights for testing"""
    lora_path = "checkpoint/tsdsr"
    
    if os.path.exists(f"{lora_path}/transformer.safetensors"):
        print("✓ LoRA weights already exist")
        return True
    
    print("Creating placeholder LoRA weights...")
    try:
        # Create minimal safetensors files
        from safetensors.torch import save_file
        
        # Transformer LoRA (minimal structure)
        transformer_lora = {
            "lora_A.default.weight": torch.randn(64, 1),
            "lora_B.default.weight": torch.randn(1, 64)
        }
        
        vae_lora = {
            "lora_A.default.weight": torch.randn(64, 1),
            "lora_B.default.weight": torch.randn(1, 64)
        }
        
        save_file(transformer_lora, f"{lora_path}/transformer.safetensors")
        save_file(vae_lora, f"{lora_path}/vae.safetensors")
        
        print("⚠️  Created placeholder LoRA weights")
        print("⚠️  Download real weights from: https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI")
        return True
    except Exception as e:
        print(f"✗ Failed to create placeholder LoRA weights: {e}")
        return False

def check_extracted_frame():
    """Check if frame is extracted"""
    frame_path = "imgs/test/elk_frame.png"
    
    if os.path.exists(frame_path):
        size = os.path.getsize(frame_path)
        print(f"✓ Extracted frame found: {frame_path} ({size / 1024:.1f} KB)")
        return True
    
    print(f"✗ Frame not found: {frame_path}")
    return False

def run_inference():
    """Run TSD-SR inference"""
    print("\n" + "=" * 50)
    print("Running TSD-SR Inference")
    print("=" * 50)
    
    check_cuda()
    
    cmd = [
        sys.executable, "test/test_tsdsr.py",
        "--pretrained_model_name_or_path", "checkpoint/sd3",
        "-i", "imgs/test/elk_frame.png",
        "-o", "outputs/test",
        "--lora_dir", "checkpoint/tsdsr",
        "--embedding_dir", "dataset/default",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--upscale", "4",
        "--process_size", "512",
        "--mixed_precision", "fp16" if torch.cuda.is_available() else "fp32"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✓ Inference completed successfully!")
            
            # Check output
            output_path = "outputs/test/elk_frame.png"
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"✓ Output saved: {output_path} ({size / 1024 / 1024:.2f} MB)")
                return True
        else:
            print(f"\n✗ Inference failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error running inference: {e}")
        return False

def main():
    print("TSD-SR Download and Inference")
    print("=" * 50)
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Check we're in the right directory
    if not os.path.exists("test/test_tsdsr.py"):
        print("✗ Please run from TSD-SR root directory")
        return 1
    
    # Setup
    check_directories()
    check_extracted_frame()
    
    print("\n" + "=" * 50)
    print("Downloading Models")
    print("=" * 50)
    
    # Try to download SD3 model
    sd3_success = download_sd3_model()
    
    # Create placeholder embeddings
    create_placeholder_embeddings()
    
    # Create placeholder LoRA weights (for testing without real weights)
    create_dummy_lora_weights()
    
    if not sd3_success:
        print("\n✗ Cannot continue without SD3 model")
        print("Please download manually and place in checkpoint/sd3/")
        return 1
    
    # Run inference
    if run_inference():
        print("\n" + "=" * 50)
        print("SUCCESS! Upscaled image saved to outputs/test/")
        print("=" * 50)
        return 0
    else:
        print("\n✗ Inference failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())


