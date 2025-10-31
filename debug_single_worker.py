#!/usr/bin/env python3
"""
Debug single worker performance issue
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_single_batch():
    """Test processing a single batch to identify the bottleneck"""
    print("🔍 Testing Single Batch Processing")
    print("=" * 40)
    
    # Start one worker
    print("🚀 Starting single worker...")
    cmd = "WORKER_ID=0 /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port 8090 --worker-id 0"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for worker to be ready
    print("⏳ Waiting for worker to be ready...")
    for attempt in range(30):
        try:
            response = requests.get("http://127.0.0.1:8090/health", timeout=2)
            if response.status_code == 200:
                print("✅ Worker ready")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ Worker not ready")
        return
    
    # Create test setup with just 8 frames (1 batch)
    print("📸 Creating test setup with 8 frames...")
    test_dir = "/tmp/single_batch_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 8 test frames
    from PIL import Image
    for i in range(1, 9):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Create manifest for single batch
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("# Worker 0 Test Manifest\n")
        f.write("# Input Dir: /tmp/single_batch_test/input_frames\n")
        f.write("# Output Dir: /tmp/single_batch_test/output_frames\n")
        f.write("# Scale: 2x\n")
        f.write("# Batch Size: 8\n")
        f.write("# Patch Size: 256\n")
        f.write("# Total Frames: 8\n")
        f.write("# Worker Batches: [0]\n")
        f.write("# Total Batches: 1\n")
        f.write("\n")
        f.write("batch_0,frame_000001.png,frame_000002.png,frame_000003.png,frame_000004.png,frame_000005.png,frame_000006.png,frame_000007.png,frame_000008.png\n")
    
    print("📋 Created single batch manifest")
    
    # Test processing with detailed timing
    print("⚡ Testing single batch processing...")
    start_time = time.time()
    
    # Send manifest
    response = requests.post("http://127.0.0.1:8090/process_manifest",
                           json={"manifest_path": manifest_path}, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Failed to send manifest: {response.text}")
        return
    
    print("✅ Manifest sent")
    
    # Monitor progress with detailed timing
    print("⏳ Monitoring progress...")
    last_progress = 0
    last_time = time.time()
    
    while True:
        try:
            response = requests.get("http://127.0.0.1:8090/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                current_time = time.time()
                elapsed = current_time - start_time
                
                if status['progress_percent'] != last_progress:
                    progress_time = current_time - last_time
                    print(f"  📊 Progress: {status['progress_percent']:.1f}% (elapsed: {elapsed:.1f}s, delta: {progress_time:.1f}s)")
                    last_progress = status['progress_percent']
                    last_time = current_time
                
                if status['status'] in ['completed', 'error']:
                    total_time = current_time - start_time
                    print(f"✅ Single batch completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(1)
    
    # Check results
    output_frames = list(Path(output_dir).glob("*.png"))
    print(f"📊 Output frames: {len(output_frames)}/8")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def test_direct_model_loading():
    """Test model loading directly to measure timing"""
    print("\n🔍 Testing Direct Model Loading")
    print("=" * 40)
    
    sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')
    
    try:
        from RealESRGAN import RealESRGAN
        import torch
        from PIL import Image
        
        print("🖥️  Testing GPU availability...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        print("🤖 Testing model loading time...")
        start_time = time.time()
        model = RealESRGAN(device, scale=2)
        model.load_weights(f'weights/RealESRGAN_x2.pth', download=True)
        load_time = time.time() - start_time
        print(f"   Model loading time: {load_time:.2f}s")
        
        print("⚡ Testing inference time...")
        # Create test image
        test_img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        
        # Warm up
        for i in range(3):
            _ = model.predict(test_img, batch_size=1, patches_size=192)
        
        # Time single inference
        start_time = time.time()
        for i in range(10):
            _ = model.predict(test_img, batch_size=1, patches_size=192)
        single_time = (time.time() - start_time) / 10
        
        print(f"   Single inference time: {single_time:.3f}s")
        
        # Time batch inference
        start_time = time.time()
        for i in range(5):
            _ = model.predict(test_img, batch_size=8, patches_size=256)
        batch_time = (time.time() - start_time) / 5
        
        print(f"   Batch inference time (8 frames): {batch_time:.3f}s")
        print(f"   Per frame time: {batch_time/8:.3f}s")
        
        # Test with multiple images in batch
        print("📦 Testing batch processing with 8 images...")
        images = [test_img] * 8
        
        start_time = time.time()
        for i in range(3):
            for img in images:
                _ = model.predict(img, batch_size=1, patches_size=256)
        sequential_time = (time.time() - start_time) / 3
        
        print(f"   Sequential processing (8 images): {sequential_time:.3f}s")
        print(f"   Per image time: {sequential_time/8:.3f}s")
        
    except Exception as e:
        print(f"❌ Direct model test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run single worker debug tests"""
    print("🔍 Single Worker Debug Suite")
    print("=" * 40)
    
    test_direct_model_loading()
    test_single_batch()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()





