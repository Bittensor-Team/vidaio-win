#!/usr/bin/env python3
"""
Script to help set up the required models for TSD-SR inference.
This script will guide you through downloading the necessary models.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_huggingface_cli():
    """Check if huggingface-cli is available"""
    # Try different possible locations
    possible_paths = [
        "huggingface-cli",
        "/home/administrator/miniconda3/bin/huggingface-cli",
        "/usr/local/bin/huggingface-cli"
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ huggingface-cli is available at {path}")
                return path
        except FileNotFoundError:
            continue
    
    print("✗ huggingface-cli not found")
    print("Please install it with: pip install huggingface_hub[cli]")
    return None

def download_sd3_model(hf_cli_path):
    """Download SD3 model from HuggingFace"""
    sd3_path = "checkpoint/sd3"
    
    if os.path.exists(sd3_path):
        print(f"✓ SD3 model already exists at {sd3_path}")
        return True
    
    print("Downloading SD3 model...")
    print("This may take a while as the model is large (~5GB)")
    
    try:
        cmd = [
            hf_cli_path, "download",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "--local-dir", sd3_path,
            "--local-dir-use-symlinks", "False"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print("✓ SD3 model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download SD3 model: {e}")
        return False

def create_placeholder_embeddings():
    """Create placeholder embeddings for testing"""
    embedding_dir = "dataset/default"
    os.makedirs(embedding_dir, exist_ok=True)
    
    prompt_embeds_path = f"{embedding_dir}/prompt_embeds.pt"
    pool_embeds_path = f"{embedding_dir}/pool_embeds.pt"
    
    if os.path.exists(prompt_embeds_path) and os.path.exists(pool_embeds_path):
        print("✓ Prompt embeddings already exist")
        return True
    
    print("Creating placeholder embeddings...")
    print("Note: These are placeholder embeddings for testing only!")
    print("For best results, download the actual embeddings from the Google Drive link.")
    
    try:
        import torch
        
        # Create placeholder tensors with the expected shapes
        # Based on the test script, these should be:
        # prompt_embeds: (333, 4096) - from SD3
        # pool_embeds: (2048,) - from SD3
        
        prompt_embeds = torch.randn(333, 4096)
        pool_embeds = torch.randn(2048)
        
        torch.save(prompt_embeds, prompt_embeds_path)
        torch.save(pool_embeds, pool_embeds_path)
        
        print(f"✓ Created placeholder embeddings at {embedding_dir}")
        print("⚠️  These are random embeddings - download real ones for best results!")
        return True
    except Exception as e:
        print(f"✗ Failed to create placeholder embeddings: {e}")
        return False

def main():
    print("TSD-SR Model Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("test/test_tsdsr.py"):
        print("✗ Please run this script from the TSD-SR root directory")
        return 1
    
    # Create necessary directories
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("dataset/default", exist_ok=True)
    os.makedirs("outputs/test", exist_ok=True)
    
    print("✓ Created necessary directories")
    
    # Check huggingface CLI
    hf_cli_path = check_huggingface_cli()
    if not hf_cli_path:
        print("\nPlease install huggingface-cli first:")
        print("pip install huggingface_hub[cli]")
        return 1
    
    # Download SD3 model
    if not download_sd3_model(hf_cli_path):
        print("\nFailed to download SD3 model. Please try manually:")
        print("huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir checkpoint/sd3")
        return 1
    
    # Create placeholder embeddings
    if not create_placeholder_embeddings():
        print("\nFailed to create placeholder embeddings")
        return 1
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Download TSD-SR LoRA weights from:")
    print("   https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI")
    print("   Place them in: checkpoint/tsdsr/")
    print("\n2. Download real prompt embeddings from the same Google Drive link")
    print("   Place them in: dataset/default/")
    print("\n3. Run inference with:")
    print("   python run_inference.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
