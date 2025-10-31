#!/usr/bin/env python3
"""
Script to run TSD-SR inference on a single image frame.
This script will help you set up and run inference on the elk frame.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import diffusers
        import transformers
        from PIL import Image
        print("✓ Required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        "imgs/test",
        "checkpoint/tsdsr", 
        "dataset/default",
        "outputs/test"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
    
    if missing_dirs:
        print(f"Created {len(missing_dirs)} missing directories")
    return True

def check_models():
    """Check if required models are available"""
    sd3_path = "checkpoint/sd3"
    lora_path = "checkpoint/tsdsr"
    embedding_path = "dataset/default"
    
    missing_models = []
    
    if not os.path.exists(sd3_path):
        missing_models.append(f"SD3 model at {sd3_path}")
    
    if not os.path.exists(f"{lora_path}/transformer.safetensors"):
        missing_models.append(f"TSD-SR LoRA weights at {lora_path}")
    
    if not os.path.exists(f"{embedding_path}/prompt_embeds.pt"):
        missing_models.append(f"Prompt embeddings at {embedding_path}")
    
    if missing_models:
        print("✗ Missing required models:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease download the required models:")
        print("1. SD3 model from: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers")
        print("2. TSD-SR LoRA weights from: https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI")
        print("3. Prompt embeddings from the same Google Drive link")
        return False
    
    print("✓ All required models are available")
    return True

def create_default_embeddings():
    """Create default prompt embeddings if they don't exist"""
    embedding_dir = "dataset/default"
    prompt_embeds_path = f"{embedding_dir}/prompt_embeds.pt"
    pool_embeds_path = f"{embedding_dir}/pool_embeds.pt"
    
    if os.path.exists(prompt_embeds_path) and os.path.exists(pool_embeds_path):
        print("✓ Prompt embeddings already exist")
        return True
    
    print("Creating default prompt embeddings...")
    try:
        import torch
        from diffusers import StableDiffusion3Pipeline
        
        # This is a placeholder - you'll need the actual SD3 model to generate embeddings
        print("Note: You need to download the actual prompt embeddings from the Google Drive link")
        print("Place them in the dataset/default/ directory as:")
        print("  - prompt_embeds.pt")
        print("  - pool_embeds.pt")
        return False
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False

def run_inference(input_image, output_dir, sd3_path, lora_dir, embedding_dir):
    """Run the inference"""
    cmd = [
        "python", "test/test_tsdsr.py",
        "--pretrained_model_name_or_path", sd3_path,
        "-i", input_image,
        "-o", output_dir,
        "--lora_dir", lora_dir,
        "--embedding_dir", embedding_dir,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--upscale", "4",
        "--process_size", "512"
    ]
    
    print(f"Running inference command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Inference completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Inference failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run TSD-SR inference on a single image")
    parser.add_argument("--input", "-i", default="imgs/test/elk_frame.png", help="Input image path")
    parser.add_argument("--output", "-o", default="outputs/test", help="Output directory")
    parser.add_argument("--sd3_path", default="checkpoint/sd3", help="Path to SD3 model")
    parser.add_argument("--lora_dir", default="checkpoint/tsdsr", help="Path to TSD-SR LoRA weights")
    parser.add_argument("--embedding_dir", default="dataset/default", help="Path to prompt embeddings")
    parser.add_argument("--skip_checks", action="store_true", help="Skip model availability checks")
    
    args = parser.parse_args()
    
    print("TSD-SR Inference Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check/create directories
    check_directories()
    
    # Check models (unless skipped)
    if not args.skip_checks and not check_models():
        print("\nTo skip model checks, use --skip_checks flag")
        return 1
    
    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"✗ Input image not found: {args.input}")
        return 1
    
    print(f"✓ Input image found: {args.input}")
    
    # Run inference
    print("\nStarting inference...")
    success = run_inference(
        args.input, 
        args.output, 
        args.sd3_path, 
        args.lora_dir, 
        args.embedding_dir
    )
    
    if success:
        print(f"\n✓ Inference completed! Check results in: {args.output}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())


