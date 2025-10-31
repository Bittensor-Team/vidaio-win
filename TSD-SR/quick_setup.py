#!/usr/bin/env python3
"""
Quick setup script for TSD-SR inference.
This creates the necessary structure and placeholder files for testing.
"""

import os
import sys
import torch
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "checkpoint/sd3",
        "checkpoint/tsdsr", 
        "dataset/default",
        "outputs/test",
        "imgs/test"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_placeholder_embeddings():
    """Create placeholder embeddings for testing"""
    embedding_dir = "dataset/default"
    
    prompt_embeds_path = f"{embedding_dir}/prompt_embeds.pt"
    pool_embeds_path = f"{embedding_dir}/pool_embeds.pt"
    
    if os.path.exists(prompt_embeds_path) and os.path.exists(pool_embeds_path):
        print("✓ Prompt embeddings already exist")
        return True
    
    print("Creating placeholder embeddings...")
    print("⚠️  These are random embeddings - download real ones for best results!")
    
    try:
        # Create placeholder tensors with the expected shapes
        # Based on the test script, these should be:
        # prompt_embeds: (333, 4096) - from SD3
        # pool_embeds: (2048,) - from SD3
        
        prompt_embeds = torch.randn(333, 4096)
        pool_embeds = torch.randn(2048)
        
        torch.save(prompt_embeds, prompt_embeds_path)
        torch.save(pool_embeds, pool_embeds_path)
        
        print(f"✓ Created placeholder embeddings at {embedding_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to create placeholder embeddings: {e}")
        return False

def create_test_script():
    """Create a simple test script that can run without the full model"""
    script_content = '''#!/usr/bin/env python3
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
    
    print("\\n" + "=" * 30)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
    
    if missing_models:
        print(f"Missing models: {missing_models}")
        print("\\nTo complete setup:")
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
'''
    
    with open("check_setup.py", "w") as f:
        f.write(script_content)
    
    print("✓ Created check_setup.py script")

def main():
    print("TSD-SR Quick Setup")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists("test/test_tsdsr.py"):
        print("✗ Please run this script from the TSD-SR root directory")
        return 1
    
    # Create directories
    create_directories()
    
    # Create placeholder embeddings
    if not create_placeholder_embeddings():
        return 1
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 30)
    print("Quick setup completed!")
    print("\nNext steps:")
    print("1. Run: python check_setup.py")
    print("2. Download the required models (see check_setup.py output)")
    print("3. Run inference: python run_inference.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


