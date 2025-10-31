#!/usr/bin/env python3
"""
Test large workload with single worker to reproduce the slowdown
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_large_workload_single():
    """Test large workload with single worker"""
    print("🔍 Testing Large Workload with Single Worker")
    print("=" * 50)
    
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
    
    # Create test setup with 80 frames (10 batches of 8) - similar to elk.mp4
    print("📸 Creating test setup with 80 frames (10 batches)...")
    test_dir = "/tmp/large_workload_single"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 80 test frames
    from PIL import Image
    print("📸 Creating 80 test frames...")
    for i in range(1, 81):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifest for single worker with 10 batches
    print("📋 Generating manifest...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 80 1 8"
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
    print("⚡ Testing large workload processing...")
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
                    if last_progress > 0:
                        batch_time = progress_time / (status['progress_percent'] - last_progress) * 100
                        batch_times.append(batch_time)
                    
                    print(f"  📊 Progress: {status['progress_percent']:.1f}% (elapsed: {elapsed:.1f}s, delta: {progress_time:.1f}s)")
                    last_progress = status['progress_percent']
                    last_time = current_time
                
                if status['status'] in ['completed', 'error']:
                    total_time = current_time - start_time
                    print(f"✅ Large workload completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(2)
    
    # Check results
    output_frames = list(Path(output_dir).glob("*.png"))
    print(f"📊 Output frames: {len(output_frames)}/80")
    
    # Analyze performance
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"📊 Average batch time: {avg_batch_time:.1f}s")
        print(f"📊 Expected time for 10 batches: {avg_batch_time * 10:.1f}s")
        print(f"📊 Actual time: {total_time:.1f}s")
        
        if total_time > avg_batch_time * 10 * 1.5:  # 50% tolerance
            print("⚠️  Performance degradation detected!")
        else:
            print("✅ Performance is within expected range")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run large workload test"""
    print("🔍 Large Workload Single Worker Test")
    print("=" * 50)
    
    test_large_workload_single()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()





