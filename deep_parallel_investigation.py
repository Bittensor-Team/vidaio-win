#!/usr/bin/env python3
"""
Deep investigation into parallel vs sequential processing
"""

import os
import sys
import time
import subprocess
import requests
import json
import threading
from pathlib import Path

def test_worker_threading():
    """Test if workers are actually running in parallel"""
    print("🔍 Testing Worker Threading")
    print("=" * 40)
    
    # Start 6 workers
    print("🚀 Starting 6 workers...")
    processes = []
    for i in range(6):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        time.sleep(2)
    
    # Wait for workers to be ready
    print("⏳ Waiting for workers to be ready...")
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
    
    # Test concurrent processing
    print("⚡ Testing concurrent processing...")
    
    # Create test setup
    test_dir = "/tmp/parallel_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 24 test frames (3 batches of 8)
    from PIL import Image
    for i in range(1, 25):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests for each worker
    print("📋 Generating manifests...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 24 6 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Manifest generation failed: {result.stderr}")
        return
    
    # Send manifests to all workers simultaneously
    print("🚀 Sending manifests to all workers simultaneously...")
    start_time = time.time()
    
    threads = []
    results = {}
    
    def send_manifest(worker_id, port):
        """Send manifest to a specific worker"""
        manifest_path = f"{manifest_dir}/worker_{worker_id}_manifest.txt"
        
        try:
            response = requests.post(f"http://127.0.0.1:{port}/process_manifest",
                                   json={"manifest_path": manifest_path}, timeout=10)
            results[worker_id] = {
                'status': response.status_code,
                'response': response.text,
                'start_time': time.time()
            }
        except Exception as e:
            results[worker_id] = {
                'status': 'error',
                'response': str(e),
                'start_time': time.time()
            }
    
    # Start all workers simultaneously
    for i in range(6):
        port = 8090 + i
        thread = threading.Thread(target=send_manifest, args=(i, port))
        thread.start()
        threads.append(thread)
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    send_time = time.time() - start_time
    print(f"📤 All manifests sent in {send_time:.3f}s")
    
    # Check which workers actually got work
    print("📊 Checking worker assignments...")
    for i in range(6):
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                content = f.read()
                batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
                print(f"  Worker {i}: {len(batch_lines)} batches")
    
    # Monitor progress
    print("⏳ Monitoring progress...")
    all_completed = False
    start_time = time.time()
    
    while not all_completed and (time.time() - start_time) < 60:  # 1 minute timeout
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
            except:
                all_completed = False
        
        if not all_completed:
            avg_progress = total_progress / 6
            elapsed = time.time() - start_time
            print(f"  📊 Progress: {avg_progress:.1f}% ({active_workers} active workers, {elapsed:.1f}s)")
            time.sleep(2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Check results
    output_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"✅ Processing completed in {total_time:.1f}s")
    print(f"📊 Output frames: {output_frames}/24")
    print(f"📊 Active workers during processing: {active_workers}")
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def test_sequential_vs_parallel():
    """Test sequential vs parallel processing directly"""
    print("\n🔍 Testing Sequential vs Parallel Processing")
    print("=" * 50)
    
    # Test 1: Sequential processing (1 worker)
    print("📊 Test 1: Sequential Processing (1 worker)")
    start_time = time.time()
    
    # Start 1 worker
    cmd = "WORKER_ID=0 /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port 8090 --worker-id 0"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for worker
    for attempt in range(30):
        try:
            response = requests.get("http://127.0.0.1:8090/health", timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    
    # Create test with 16 frames (2 batches)
    test_dir = "/tmp/seq_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    from PIL import Image
    for i in range(1, 17):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifest
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 16 1 8"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Process
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    response = requests.post("http://127.0.0.1:8090/process_manifest",
                           json={"manifest_path": manifest_path}, timeout=10)
    
    # Wait for completion
    while True:
        try:
            response = requests.get("http://127.0.0.1:8090/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                if status['status'] in ['completed', 'error']:
                    break
        except:
            break
        time.sleep(1)
    
    sequential_time = time.time() - start_time
    print(f"  ✅ Sequential processing: {sequential_time:.1f}s")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)
    
    # Test 2: Parallel processing (2 workers)
    print("\n📊 Test 2: Parallel Processing (2 workers)")
    start_time = time.time()
    
    # Start 2 workers
    processes = []
    for i in range(2):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        time.sleep(2)
    
    # Wait for workers
    for i in range(2):
        port = 8090 + i
        for attempt in range(30):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
    
    # Create test with 16 frames (2 batches)
    test_dir = "/tmp/par_test"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    for i in range(1, 17):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 16 2 8"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Process with both workers
    for i in range(2):
        port = 8090 + i
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        response = requests.post(f"http://127.0.0.1:{port}/process_manifest",
                               json={"manifest_path": manifest_path}, timeout=10)
    
    # Wait for completion
    while True:
        all_completed = True
        for i in range(2):
            port = 8090 + i
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    if status['status'] not in ['completed', 'error']:
                        all_completed = False
            except:
                all_completed = False
        
        if all_completed:
            break
        time.sleep(1)
    
    parallel_time = time.time() - start_time
    print(f"  ✅ Parallel processing: {parallel_time:.1f}s")
    
    # Analysis
    speedup = sequential_time / parallel_time
    print(f"\n📊 Analysis:")
    print(f"  Sequential time: {sequential_time:.1f}s")
    print(f"  Parallel time: {parallel_time:.1f}s")
    print(f"  Speedup: {speedup:.1f}x")
    
    if speedup > 1.5:
        print("  ✅ True parallel processing detected")
    else:
        print("  ❌ Sequential processing detected (no speedup)")
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run deep parallel investigation"""
    print("🔍 Deep Parallel Processing Investigation")
    print("=" * 50)
    
    test_worker_threading()
    test_sequential_vs_parallel()
    
    print("\n✅ Investigation complete!")

if __name__ == "__main__":
    main()





