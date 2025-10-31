#!/usr/bin/env python3
"""
VSR Performance Experiment - Test different worker counts
"""

import requests
import time
import math
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import subprocess

def test_worker_performance(worker_url, frame_paths):
    """Test single worker performance"""
    start_time = time.time()
    processed = 0
    
    for frame_path in frame_paths:
        try:
            with open(frame_path, 'rb') as f:
                files = {'file': (frame_path.split('/')[-1], f, 'image/png')}
                resp = requests.post(f'{worker_url}/upscale', files=files, timeout=60)
                if resp.status_code == 200:
                    processed += 1
                else:
                    print(f"❌ Worker error: Status {resp.status_code}")
        except Exception as e:
            print(f"❌ Worker error: {e}")
    
    elapsed = time.time() - start_time
    return processed, elapsed

def get_healthy_workers(base_port, num_workers):
    """Get list of healthy workers"""
    healthy_workers = []
    
    for i in range(num_workers):
        port = base_port + i
        url = f"http://127.0.0.1:{port}"
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded", False):
                    healthy_workers.append(url)
        except:
            pass
    
    return healthy_workers

def run_experiment(worker_count, test_frames=20):
    """Run performance experiment with specific worker count"""
    print(f"\n🧪 EXPERIMENT: {worker_count} workers, {test_frames} frames")
    print("="*50)
    
    # Get healthy workers
    healthy_workers = get_healthy_workers(8100, worker_count)
    
    if len(healthy_workers) < worker_count:
        print(f"❌ Only {len(healthy_workers)}/{worker_count} workers available")
        return None
    
    # Create test frames
    frames_dir = f'/tmp/exp_frames_{worker_count}w'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Use existing test frame
    test_frame = '/workspace/vidaio-win/optimization_tests/test_frame.png'
    frames = []
    for i in range(test_frames):
        frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
        cv2.imwrite(frame_path, cv2.imread(test_frame))
        frames.append(frame_path)
    
    # Distribute frames among workers
    frames_per_worker = math.ceil(len(frames) / len(healthy_workers))
    worker_tasks = []
    
    for i, worker_url in enumerate(healthy_workers):
        start_idx = i * frames_per_worker
        end_idx = min((i + 1) * frames_per_worker, len(frames))
        worker_frames = frames[start_idx:end_idx]
        worker_tasks.append((worker_url, worker_frames))
        print(f"   Worker {i+1}: {len(worker_frames)} frames")
    
    # Process frames in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
        results = list(executor.map(lambda args: test_worker_performance(*args), worker_tasks))
    
    total_elapsed = time.time() - start_time
    
    # Calculate results
    total_processed = sum(processed for processed, _ in results)
    max_worker_time = max(elapsed for _, elapsed in results)
    min_worker_time = min(elapsed for _, elapsed in results)
    avg_worker_time = sum(elapsed for _, elapsed in results) / len(results)
    
    frames_per_second = len(frames) / total_elapsed
    seconds_per_frame = total_elapsed / len(frames)
    
    print(f"📊 RESULTS:")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Frames processed: {total_processed}/{len(frames)}")
    print(f"   Frames per second: {frames_per_second:.2f}")
    print(f"   Seconds per frame: {seconds_per_frame:.3f}")
    print(f"   Worker times: min={min_worker_time:.1f}s, avg={avg_worker_time:.1f}s, max={max_worker_time:.1f}s")
    print(f"   Efficiency: {min_worker_time/max_worker_time:.2f}")
    
    return {
        'workers': worker_count,
        'frames': test_frames,
        'total_time': total_elapsed,
        'frames_per_second': frames_per_second,
        'seconds_per_frame': seconds_per_frame,
        'min_worker_time': min_worker_time,
        'max_worker_time': max_worker_time,
        'efficiency': min_worker_time/max_worker_time
    }

def main():
    print("🚀 VSR Performance Experiment")
    print("Testing different worker counts with 20 frames each")
    print("="*60)
    
    # Test different worker counts
    worker_counts = [1, 2, 4, 6, 8, 10, 12, 15, 20]
    results = []
    
    for worker_count in worker_counts:
        result = run_experiment(worker_count, test_frames=20)
        if result:
            results.append(result)
    
    # Print summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Workers':<8} {'FPS':<6} {'s/frame':<8} {'Efficiency':<10} {'Speedup':<8}")
    print("-" * 60)
    
    baseline_fps = None
    for result in results:
        if baseline_fps is None:
            baseline_fps = result['frames_per_second']
            speedup = 1.0
        else:
            speedup = result['frames_per_second'] / baseline_fps
        
        print(f"{result['workers']:<8} {result['frames_per_second']:<6.2f} {result['seconds_per_frame']:<8.3f} {result['efficiency']:<10.2f} {speedup:<8.2f}")
    
    # Find optimal worker count
    best_result = max(results, key=lambda x: x['frames_per_second'])
    print(f"\n🏆 OPTIMAL: {best_result['workers']} workers")
    print(f"   Best FPS: {best_result['frames_per_second']:.2f}")
    print(f"   Best s/frame: {best_result['seconds_per_frame']:.3f}")

if __name__ == "__main__":
    main()





