#!/usr/bin/env python3
"""
Performance test script to compare different worker counts
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_worker_count(num_workers, test_name):
    """Test performance with specific number of workers"""
    print(f"\n🧪 Testing {test_name} ({num_workers} workers)")
    print("=" * 50)
    
    # Start workers
    print(f"🚀 Starting {num_workers} workers...")
    for i in range(num_workers):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    
    # Wait for workers to be ready
    print("⏳ Waiting for workers to be ready...")
    ready_workers = 0
    for i in range(num_workers):
        port = 8090 + i
        for attempt in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    ready_workers += 1
                    break
            except:
                pass
            time.sleep(1)
    
    if ready_workers != num_workers:
        print(f"❌ Only {ready_workers}/{num_workers} workers ready")
        return None
    
    print(f"✅ {ready_workers} workers ready")
    
    # Create test manifest
    test_dir = f"/tmp/performance_test_{num_workers}"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create test frames (smaller test - 24 frames)
    print("📸 Creating test frames...")
    from PIL import Image
    for i in range(1, 25):  # 24 frames
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests
    print("📋 Generating manifests...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 24 {num_workers} 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Manifest generation failed: {result.stderr}")
        return None
    
    # Start processing
    print("⚡ Starting batch processing...")
    start_time = time.time()
    
    # Send manifests to workers
    for i in range(num_workers):
        port = 8090 + i
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        
        if os.path.exists(manifest_path):
            try:
                response = requests.post(f"http://127.0.0.1:{port}/process_manifest",
                                       json={"manifest_path": manifest_path}, timeout=10)
                if response.status_code != 200:
                    print(f"❌ Failed to send manifest to worker {i}")
            except Exception as e:
                print(f"❌ Error sending manifest to worker {i}: {e}")
    
    # Monitor progress
    print("⏳ Monitoring progress...")
    all_completed = False
    max_wait_time = 300  # 5 minutes max
    check_interval = 5
    
    while not all_completed and (time.time() - start_time) < max_wait_time:
        all_completed = True
        total_progress = 0
        
        for i in range(num_workers):
            port = 8090 + i
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    if status['status'] not in ['completed', 'error']:
                        all_completed = False
                    total_progress += status.get('progress_percent', 0)
            except:
                all_completed = False
        
        if not all_completed:
            avg_progress = total_progress / num_workers
            elapsed = time.time() - start_time
            print(f"  📊 Progress: {avg_progress:.1f}% ({elapsed:.1f}s elapsed)")
            time.sleep(check_interval)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Check results
    output_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    print(f"✅ Processing completed in {total_time:.1f}s")
    print(f"📊 Output frames: {output_frames}/24")
    
    # Cleanup
    subprocess.run(f"pkill -f upscaler_worker_batch.py", shell=True)
    subprocess.run(f"rm -rf {test_dir}", shell=True)
    
    return {
        'workers': num_workers,
        'time': total_time,
        'frames': output_frames,
        'frames_per_second': output_frames / total_time if total_time > 0 else 0
    }

def main():
    """Run performance tests with different worker counts"""
    print("🚀 Performance Test Suite")
    print("=========================")
    
    # Test configurations
    tests = [
        (1, "Single Worker"),
        (2, "Two Workers"),
        (3, "Three Workers"),
        (6, "Six Workers")
    ]
    
    results = []
    
    for num_workers, test_name in tests:
        result = test_worker_count(num_workers, test_name)
        if result:
            results.append(result)
        time.sleep(5)  # Wait between tests
    
    # Print results
    print("\n📊 Performance Results")
    print("=" * 50)
    print(f"{'Workers':<8} {'Time (s)':<10} {'FPS':<8} {'Speedup':<8}")
    print("-" * 50)
    
    baseline_time = None
    for result in results:
        if baseline_time is None:
            baseline_time = result['time']
            speedup = 1.0
        else:
            speedup = baseline_time / result['time']
        
        print(f"{result['workers']:<8} {result['time']:<10.1f} {result['frames_per_second']:<8.2f} {speedup:<8.1f}x")
    
    # Analysis
    print("\n🔍 Analysis:")
    if len(results) >= 2:
        single_worker = next((r for r in results if r['workers'] == 1), None)
        six_workers = next((r for r in results if r['workers'] == 6), None)
        
        if single_worker and six_workers:
            expected_speedup = 6.0
            actual_speedup = single_worker['time'] / six_workers['time']
            efficiency = (actual_speedup / expected_speedup) * 100
            
            print(f"  Expected 6-worker speedup: {expected_speedup:.1f}x")
            print(f"  Actual 6-worker speedup: {actual_speedup:.1f}x")
            print(f"  Efficiency: {efficiency:.1f}%")
            
            if efficiency < 50:
                print("  ⚠️  Low efficiency - possible bottlenecks:")
                print("     - GPU memory contention")
                print("     - Model loading overhead")
                print("     - I/O bottlenecks")
                print("     - Threading issues")

if __name__ == "__main__":
    main()





