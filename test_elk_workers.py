#!/usr/bin/env python3
"""
Test elk.mp4 with different worker counts
"""

import os
import sys
import time
import subprocess

def test_elk_with_workers(num_workers):
    """Test elk.mp4 with specific number of workers"""
    print(f"\n🧪 Testing elk.mp4 with {num_workers} workers")
    print("=" * 50)
    
    # Start workers
    print(f"🚀 Starting {num_workers} workers...")
    processes = []
    for i in range(num_workers):
        port = 8090 + i
        cmd = f"WORKER_ID={i} /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port {port} --worker-id {i}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)
        time.sleep(2)
    
    # Wait for workers to be ready
    print("⏳ Waiting for workers to be ready...")
    import requests
    ready_workers = 0
    for i in range(num_workers):
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
    
    if ready_workers != num_workers:
        print(f"❌ Only {ready_workers}/{num_workers} workers ready")
        for proc in processes:
            proc.terminate()
        return None
    
    print(f"✅ {ready_workers} workers ready")
    
    # Test elk.mp4
    output_file = f"output_elk_{num_workers}w.mp4"
    print(f"🎬 Processing elk.mp4 -> {output_file}")
    
    start_time = time.time()
    
    # Run video2x wrapper
    cmd = f"./video2x_wrapper_batch.sh elk.mp4 {output_file} 2"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if result.returncode == 0:
        # Get file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✅ Processing completed in {total_time:.1f}s")
            print(f"📁 Output: {output_file} ({file_size:.1f} MB)")
        else:
            print(f"❌ Output file not created")
            return None
    else:
        print(f"❌ Processing failed: {result.stderr}")
        return None
    
    # Cleanup
    for proc in processes:
        proc.terminate()
    
    return {
        'workers': num_workers,
        'time': total_time,
        'file_size': file_size if os.path.exists(output_file) else 0
    }

def main():
    """Test elk.mp4 with different worker counts"""
    print("🚀 elk.mp4 Performance Test")
    print("=" * 30)
    
    # Test configurations
    worker_counts = [1, 2, 3, 6]
    results = []
    
    for num_workers in worker_counts:
        result = test_elk_with_workers(num_workers)
        if result:
            results.append(result)
        time.sleep(5)  # Wait between tests
    
    # Print results
    print("\n📊 elk.mp4 Results")
    print("=" * 50)
    print(f"{'Workers':<8} {'Time (s)':<10} {'Size (MB)':<10} {'Speedup':<8}")
    print("-" * 50)
    
    baseline_time = None
    for result in results:
        if baseline_time is None:
            baseline_time = result['time']
            speedup = 1.0
        else:
            speedup = baseline_time / result['time']
        
        print(f"{result['workers']:<8} {result['time']:<10.1f} {result['file_size']:<10.1f} {speedup:<8.1f}x")
    
    # Analysis
    print("\n🔍 Analysis:")
    if len(results) >= 2:
        single_worker = next((r for r in results if r['workers'] == 1), None)
        two_workers = next((r for r in results if r['workers'] == 2), None)
        six_workers = next((r for r in results if r['workers'] == 6), None)
        
        if single_worker and two_workers:
            speedup_2w = single_worker['time'] / two_workers['time']
            print(f"  2 workers speedup: {speedup_2w:.1f}x")
        
        if single_worker and six_workers:
            speedup_6w = single_worker['time'] / six_workers['time']
            efficiency = (speedup_6w / 6.0) * 100
            print(f"  6 workers speedup: {speedup_6w:.1f}x")
            print(f"  6 workers efficiency: {efficiency:.1f}%")
            
            if speedup_6w < 2.0:
                print("  💡 Recommendation: Use 2 workers for optimal performance")

if __name__ == "__main__":
    main()





