#!/usr/bin/env python3
"""
GPU and model loading diagnostic script
"""

import sys
import os
import time
import torch
import psutil
import subprocess

sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')

def check_gpu_memory():
    """Check GPU memory usage"""
    try:
        if torch.cuda.is_available():
            print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
            print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.3f} GB")
            print(f"   Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.1f} GB")
        else:
            print("❌ CUDA not available")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")

def check_system_resources():
    """Check system resources"""
    print(f"💾 RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available")
    print(f"🖥️  CPU: {psutil.cpu_count()} cores")
    
    # Check if multiple Python processes are running
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and 'upscaler_worker' in ' '.join(proc.info['cmdline']):
                python_procs.append(proc.info['pid'])
        except:
            pass
    
    print(f"🐍 Python processes: {len(python_procs)}")

def test_model_loading():
    """Test model loading time and memory usage"""
    print("\n🤖 Testing model loading...")
    
    try:
        from RealESRGAN import RealESRGAN
        
        # Check memory before
        print("📊 Memory before model loading:")
        check_gpu_memory()
        
        # Load model
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=2)
        model.load_weights(f'weights/RealESRGAN_x2.pth', download=True)
        load_time = time.time() - start_time
        
        print(f"⏱️  Model loading time: {load_time:.2f}s")
        
        # Check memory after
        print("📊 Memory after model loading:")
        check_gpu_memory()
        
        # Test inference time
        print("\n⚡ Testing inference speed...")
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        
        # Warm up
        for i in range(3):
            _ = model.predict(test_img, batch_size=1, patches_size=192)
        
        # Time inference
        start_time = time.time()
        for i in range(10):
            _ = model.predict(test_img, batch_size=1, patches_size=192)
        inference_time = (time.time() - start_time) / 10
        
        print(f"⏱️  Average inference time (100x100): {inference_time:.3f}s")
        
        # Test with batch processing
        print("\n📦 Testing batch processing...")
        start_time = time.time()
        for i in range(5):
            _ = model.predict(test_img, batch_size=8, patches_size=256)
        batch_time = (time.time() - start_time) / 5
        
        print(f"⏱️  Average batch time (batch_size=8): {batch_time:.3f}s")
        
        # Check memory after inference
        print("📊 Memory after inference:")
        check_gpu_memory()
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()

def test_concurrent_loading():
    """Test what happens when multiple models try to load"""
    print("\n🔄 Testing concurrent model loading...")
    
    # Check if multiple workers are running
    result = subprocess.run("ps aux | grep upscaler_worker_batch | grep -v grep | wc -l", 
                          shell=True, capture_output=True, text=True)
    worker_count = int(result.stdout.strip())
    
    print(f"🔍 Currently running workers: {worker_count}")
    
    if worker_count > 0:
        print("📊 Memory with workers running:")
        check_gpu_memory()
        
        # Check worker logs for model loading
        print("\n📋 Checking worker logs...")
        for i in range(6):
            log_file = f"/tmp/batch_worker_{8090 + i}.log"
            if os.path.exists(log_file):
                print(f"\nWorker {i+1} log (last 10 lines):")
                result = subprocess.run(f"tail -10 {log_file}", shell=True, capture_output=True, text=True)
                print(result.stdout)

def main():
    """Run all diagnostics"""
    print("🔍 GPU and Model Loading Diagnostics")
    print("=" * 40)
    
    check_system_resources()
    check_gpu_memory()
    test_model_loading()
    test_concurrent_loading()
    
    print("\n✅ Diagnostics complete!")

if __name__ == "__main__":
    main()





