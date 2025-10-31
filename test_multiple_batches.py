#!/usr/bin/env python3
"""
Test multiple batches to identify the performance issue
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_multiple_batches():
    """Test processing multiple batches to find the bottleneck"""
    print("🔍 Testing Multiple Batches")
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
    
    # Create test setup with 24 frames (3 batches of 8)
    print("📸 Creating test setup with 24 frames (3 batches)...")
    test_dir = "/tmp/multiple_batches_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 24 test frames
    from PIL import Image
    for i in range(1, 25):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifest for single worker with 3 batches
    print("📋 Generating manifest...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 24 1 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Manifest generation failed: {result.stderr}")
        return
    
    # Check manifest content
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            content = f.read()
            batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
            print(f"📊 Manifest has {len(batch_lines)} batches")
    
    # Test processing with detailed timing
    print("⚡ Testing multiple batch processing...")
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
    batch_times = []
    
    while True:
        try:
            response = requests.get("http://127.0.0.1:8090/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                current_time = time.time()
                elapsed = current_time - start_time
                
                if status['progress_percent'] != last_progress:
                    progress_time = current_time - last_time
                    batch_time = progress_time / (status['progress_percent'] - last_progress) * 100 if last_progress > 0 else 0
                    batch_times.append(batch_time)
                    
                    print(f"  📊 Progress: {status['progress_percent']:.1f}% (elapsed: {elapsed:.1f}s, delta: {progress_time:.1f}s, batch_time: {batch_time:.1f}s)")
                    last_progress = status['progress_percent']
                    last_time = current_time
                
                if status['status'] in ['completed', 'error']:
                    total_time = current_time - start_time
                    print(f"✅ Multiple batches completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(1)
    
    # Check results
    output_frames = list(Path(output_dir).glob("*.png"))
    print(f"📊 Output frames: {len(output_frames)}/24")
    
    # Analyze batch times
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"📊 Average batch time: {avg_batch_time:.1f}s")
        print(f"📊 Batch times: {[f'{t:.1f}s' for t in batch_times]}")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run multiple batch test"""
    print("🔍 Multiple Batch Debug Test")
    print("=" * 40)
    
    test_multiple_batches()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()





