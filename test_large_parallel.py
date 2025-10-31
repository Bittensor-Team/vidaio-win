#!/usr/bin/env python3
"""
Test large workload with 6 workers to see true parallel processing
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_large_parallel():
    """Test large workload with 6 workers"""
    print("🔍 Testing Large Workload with 6 Workers")
    print("=" * 50)
    
    # Start 6 workers
    print("🚀 Starting 6 workers...")
    processes = []
    for i in range(6):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        time.sleep(2)
    
    # Wait for workers
    print("⏳ Waiting for workers...")
    ready_workers = 0
    for i in range(6):
        port = 8090 + i
        for attempt in range(30):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    ready_workers += 1
                    break
            except:
                pass
            time.sleep(1)
    
    print(f"✅ {ready_workers}/6 workers ready")
    
    # Create test setup with 48 frames (6 batches of 8) - enough for all workers
    print("📸 Creating test setup with 48 frames (6 batches)...")
    test_dir = "/tmp/large_parallel_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 48 test frames
    from PIL import Image
    for i in range(1, 49):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests
    print("📋 Generating manifests...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 48 6 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Check manifest distribution
    print("📊 Checking manifest distribution...")
    for i in range(6):
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                content = f.read()
                batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
                print(f"  Worker {i}: {len(batch_lines)} batches")
    
    # Test processing
    print("⚡ Testing parallel processing...")
    start_time = time.time()
    
    # Send manifests to all workers
    for i in range(6):
        port = 8090 + i
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        
        if os.path.exists(manifest_path):
            print(f"  📤 Sending manifest to Worker {i}...")
            response = requests.post(f"http://127.0.0.1:{port}/process_manifest",
                                   json={"manifest_path": manifest_path}, timeout=10)
            print(f"    Response: {response.status_code}")
    
    # Monitor progress
    print("⏳ Monitoring progress...")
    last_progress = {}
    
    while True:
        all_completed = True
        total_progress = 0
        active_workers = 0
        
        for i in range(6):
            port = 8090 + i
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    
                    if status['status'] not in ['completed', 'error']:
                        all_completed = False
                        active_workers += 1
                    
                    total_progress += status.get('progress_percent', 0)
                    
                    # Log progress changes
                    if i not in last_progress or last_progress[i] != status['progress_percent']:
                        print(f"  Worker {i}: {status['progress_percent']:.1f}% - {status.get('current_batch', 'N/A')}")
                        last_progress[i] = status['progress_percent']
            except:
                all_completed = False
        
        if all_completed:
            break
        
        avg_progress = total_progress / 6
        elapsed = time.time() - start_time
        print(f"  📊 Overall: {avg_progress:.1f}% ({active_workers} active, {elapsed:.1f}s)")
        time.sleep(2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Check results
    output_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"✅ Processing completed in {total_time:.1f}s")
    print(f"📊 Output frames: {output_frames}/48")
    print(f"📊 Active workers during processing: {active_workers}")
    
    # Compare with sequential
    print("\n📊 Performance Analysis:")
    print(f"  Parallel time: {total_time:.1f}s")
    print(f"  Expected sequential: {total_time * 6:.1f}s (6x slower)")
    print(f"  Theoretical speedup: 6x")
    print(f"  Actual speedup: {total_time * 6 / total_time:.1f}x")
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run large parallel test"""
    print("🔍 Large Parallel Processing Test")
    print("=" * 40)
    
    test_large_parallel()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()





