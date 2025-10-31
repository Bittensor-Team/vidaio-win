#!/usr/bin/env python3
"""
Debug worker processing to see what's actually happening
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def debug_worker_processing():
    """Debug what workers are actually doing"""
    print("🔍 Debugging Worker Processing")
    print("=" * 40)
    
    # Start 2 workers
    print("🚀 Starting 2 workers...")
    processes = []
    for i in range(2):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        time.sleep(3)
    
    # Wait for workers
    print("⏳ Waiting for workers...")
    for i in range(2):
        port = 8090 + i
        for attempt in range(30):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  ✅ Worker {i} ready")
                    break
            except:
                pass
            time.sleep(1)
    
    # Create test setup
    test_dir = "/tmp/debug_worker"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 16 test frames
    from PIL import Image
    for i in range(1, 17):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests
    print("📋 Generating manifests...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 16 2 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Check manifest contents
    print("📊 Checking manifest contents...")
    for i in range(2):
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                content = f.read()
                batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
                print(f"  Worker {i}: {len(batch_lines)} batches")
                for line in batch_lines[:2]:  # Show first 2 batches
                    print(f"    {line}")
    
    # Send manifests and monitor
    print("🚀 Sending manifests...")
    start_time = time.time()
    
    for i in range(2):
        port = 8090 + i
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        
        print(f"  📤 Sending manifest to Worker {i}...")
        response = requests.post(f"http://127.0.0.1:{port}/process_manifest",
                               json={"manifest_path": manifest_path}, timeout=10)
        print(f"    Response: {response.status_code} - {response.text}")
    
    # Monitor progress with detailed logging
    print("⏳ Monitoring progress...")
    last_status = {}
    
    while True:
        all_completed = True
        current_time = time.time()
        elapsed = current_time - start_time
        
        for i in range(2):
            port = 8090 + i
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    
                    # Check if status changed
                    if i not in last_status or last_status[i] != status:
                        print(f"  Worker {i}: {status['status']} - {status.get('progress_percent', 0):.1f}% - {status.get('current_batch', 'N/A')}")
                        last_status[i] = status
                    
                    if status['status'] not in ['completed', 'error']:
                        all_completed = False
                else:
                    all_completed = False
            except Exception as e:
                print(f"  Worker {i}: Error - {e}")
                all_completed = False
        
        if all_completed:
            break
        
        time.sleep(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Processing completed in {total_time:.1f}s")
    
    # Check results
    output_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Output frames: {output_frames}/16")
    
    # Check worker logs
    print("📋 Checking worker logs...")
    for i in range(2):
        log_file = f"/tmp/batch_worker_{8090 + i}.log"
        if os.path.exists(log_file):
            print(f"  Worker {i} log (last 10 lines):")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"    {line.strip()}")
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run worker processing debug"""
    print("🔍 Worker Processing Debug")
    print("=" * 30)
    
    debug_worker_processing()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()





