#!/usr/bin/env python3
"""
Simple test script to verify the setup is working.
This script will check if all required files are in place.
"""

import os
import sys
import torch
from pathlib import Path

def check_setup():
    """Check if the setup is complete"""
    print("TSD-SR Setup Check")
    print("=" * 30)
    
    # Check directories
    required_dirs = [
        "checkpoint/sd3",
        "checkpoint/tsdsr",
        "dataset/default", 
        "outputs/test",
        "imgs/test"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path}")
            missing_dirs.append(dir_path)
    
    # Check files
    required_files = [
        "imgs/test/elk_frame.png",
        "dataset/default/prompt_embeds.pt",
        "dataset/default/pool_embeds.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    # Check models
    model_files = [
        "checkpoint/sd3/transformer/config.json",
        "checkpoint/tsdsr/transformer.safetensors",
        "checkpoint/tsdsr/vae.safetensors"
    ]
    
    missing_models = []
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_models.append(file_path)
    
    print("\n" + "=" * 30)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
    
    if missing_models:
        print(f"Missing models: {missing_models}")
        print("\nTo complete setup:")
        print("1. Download SD3 model from HuggingFace (requires authentication)")
        print("2. Download TSD-SR LoRA weights from Google Drive")
        print("3. Download real prompt embeddings from Google Drive")
    
    if not missing_dirs and not missing_files and not missing_models:
        print("✓ Setup is complete! You can run inference now.")
        return True
    else:
        print("✗ Setup is incomplete. Please download the missing components.")
        return False

if __name__ == "__main__":
    check_setup()
