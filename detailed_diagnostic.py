#!/usr/bin/env python3
"""
Detailed diagnostic to understand the performance bottleneck
"""

import os
import sys
import time
import subprocess
import requests
import json
import threading
from pathlib import Path

def test_single_worker_detailed():
    """Test single worker with detailed timing"""
    print("🔍 Detailed Single Worker Test")
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
    
    # Create test setup
    test_dir = "/tmp/detailed_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create test frames
    print("📸 Creating test frames...")
    from PIL import Image
    for i in range(1, 9):  # 8 frames
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img.save(frame_path)
    
    # Create manifest
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("# Worker 0 Test Manifest\n")
        f.write("# Input Dir: /tmp/detailed_test/input_frames\n")
        f.write("# Output Dir: /tmp/detailed_test/output_frames\n")
        f.write("# Scale: 2x\n")
        f.write("# Batch Size: 8\n")
        f.write("# Patch Size: 256\n")
        f.write("# Total Frames: 8\n")
        f.write("# Worker Batches: [0]\n")
        f.write("# Total Batches: 1\n")
        f.write("\n")
        f.write("batch_0,frame_000001.png,frame_000002.png,frame_000003.png,frame_000004.png,frame_000005.png,frame_000006.png,frame_000007.png,frame_000008.png\n")
    
    # Test processing with detailed timing
    print("⚡ Testing processing with detailed timing...")
    
    # Send manifest
    start_time = time.time()
    response = requests.post("http://127.0.0.1:8090/process_manifest",
                           json={"manifest_path": manifest_path}, timeout=10)
    send_time = time.time() - start_time
    
    print(f"📤 Manifest sent in {send_time:.3f}s")
    
    if response.status_code != 200:
        print(f"❌ Failed to send manifest: {response.text}")
        return
    
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
                    print(f"✅ Processing completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(0.5)
    
    # Check results
    output_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Output frames: {output_frames}/8")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def test_worker_startup_time():
    """Test how long it takes for workers to start up"""
    print("\n🔍 Worker Startup Time Test")
    print("=" * 40)
    
    # Test single worker startup
    print("🚀 Testing single worker startup...")
    start_time = time.time()
    
    cmd = "WORKER_ID=0 /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port 8090 --worker-id 0"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for worker to be ready
    for attempt in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get("http://127.0.0.1:8090/health", timeout=2)
            if response.status_code == 200:
                startup_time = time.time() - start_time
                print(f"✅ Single worker startup: {startup_time:.1f}s")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ Single worker startup failed")
        process.terminate()
        return
    
    # Test multiple workers startup
    print("🚀 Testing 6 workers startup...")
    start_time = time.time()
    
    processes = []
    for i in range(6):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        time.sleep(1)  # Stagger startup
    
    # Wait for all workers to be ready
    ready_count = 0
    for attempt in range(60):
        ready_count = 0
        for i in range(6):
            port = 8090 + i
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    ready_count += 1
            except:
                pass
        
        if ready_count == 6:
            startup_time = time.time() - start_time
            print(f"✅ 6 workers startup: {startup_time:.1f}s")
            break
        
        time.sleep(1)
    else:
        print(f"❌ Only {ready_count}/6 workers ready after 60s")
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    
    # Kill any remaining processes
    subprocess.run("pkill -f upscaler_worker_batch.py", shell=True)

def main():
    """Run detailed diagnostics"""
    print("🔍 Detailed Performance Diagnostics")
    print("=" * 50)
    
    test_worker_startup_time()
    test_single_worker_detailed()
    
    print("\n✅ Detailed diagnostics complete!")

if __name__ == "__main__":
    main()





