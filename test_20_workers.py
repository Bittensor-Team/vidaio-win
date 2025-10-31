#!/usr/bin/env python3
"""
Test VSR performance with 20 workers
"""

import subprocess
import time
import requests
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import math

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames, 'duration': duration}

def extract_frames(video_path, frames_dir, max_frames=50):
    """Extract frames for testing"""
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // max_frames)
    
    frame_count = 0
    extracted = 0
    
    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(frames_dir, f"frame_{extracted:06d}.png")
            cv2.imwrite(frame_path, frame)
            extracted += 1
            
        frame_count += 1
    
    cap.release()
    return extracted

def test_worker_performance(worker_url, frame_paths):
    """Test single worker performance"""
    start_time = time.time()
    processed = 0
    
    for frame_path in frame_paths:
        try:
            with open(frame_path, 'rb') as f:
                files = {'file': (os.path.basename(frame_path), f, 'image/png')}
                resp = requests.post(f"{worker_url}/upscale", files=files, timeout=60)
                
                if resp.status_code == 200:
                    processed += 1
                else:
                    print(f"❌ Worker error: Status {resp.status_code}")
        except Exception as e:
            print(f"❌ Worker error: {e}")
    
    elapsed = time.time() - start_time
    return processed, elapsed

def test_20_workers():
    """Test with 20 VSR workers"""
    print("🚀 Testing VSR Performance with 20 Workers")
    print("="*60)
    
    # Check GPU memory
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        total, used, free = map(int, result.stdout.strip().split(', '))
        print(f"🖥️  GPU Memory: {total}MB total, {used}MB used, {free}MB free")
        print(f"📊 Available for VSR: {free}MB ÷ 1200MB = {free//1200} workers max")
    
    # Test video
    input_video = "/workspace/vidaio-win/elk.mp4"
    info = get_video_info(input_video)
    print(f"📹 Video: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    print(f"⏱️  Duration: {info['duration']:.1f}s, Frames: {info['total_frames']}")
    
    # Extract test frames
    frames_dir = "/tmp/test_frames_20w"
    test_frames = 100  # Test with 100 frames
    extracted = extract_frames(input_video, frames_dir, test_frames)
    print(f"🎬 Extracted {extracted} test frames")
    
    # Setup 20 workers
    num_workers = 20
    base_port = 8100
    worker_urls = [f"http://127.0.0.1:{base_port + i}" for i in range(num_workers)]
    
    # Check worker health
    healthy_workers = []
    for i, url in enumerate(worker_urls):
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded", False):
                    healthy_workers.append(url)
                    print(f"✅ Worker {i+1}: {data.get('model_name', 'unknown')}")
        except:
            print(f"❌ Worker {i+1}: Not responding")
    
    if len(healthy_workers) < num_workers:
        print(f"⚠️  Only {len(healthy_workers)}/{num_workers} workers healthy")
        return
    
    print(f"✅ {len(healthy_workers)} workers ready")
    
    # Distribute frames among workers
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames_per_worker = math.ceil(len(frames) / len(healthy_workers))
    
    worker_tasks = []
    for i, worker_url in enumerate(healthy_workers):
        start_idx = i * frames_per_worker
        end_idx = min((i + 1) * frames_per_worker, len(frames))
        worker_frames = [os.path.join(frames_dir, f) for f in frames[start_idx:end_idx]]
        worker_tasks.append((worker_url, worker_frames))
        print(f"   Worker {i+1}: {len(worker_frames)} frames")
    
    # Process frames in parallel
    print(f"\n🎬 Processing {len(frames)} frames with {len(healthy_workers)} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
        results = list(executor.map(lambda args: test_worker_performance(*args), worker_tasks))
    
    total_elapsed = time.time() - start_time
    
    # Calculate results
    total_processed = sum(processed for processed, _ in results)
    max_worker_time = max(elapsed for _, elapsed in results)
    
    print(f"\n📊 RESULTS:")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Slowest worker: {max_worker_time:.1f}s")
    print(f"   Processed: {total_processed}/{len(frames)}")
    print(f"   Frames per second: {len(frames)/total_elapsed:.2f}")
    print(f"   Seconds per frame: {total_elapsed/len(frames):.3f}")
    
    # Project full video performance
    full_video_frames = info['total_frames']
    projected_time = (total_elapsed / len(frames)) * full_video_frames
    print(f"\n🔮 PROJECTION for full video ({full_video_frames} frames):")
    print(f"   Estimated time: {projected_time:.1f}s ({projected_time/60:.1f} minutes)")
    print(f"   Speedup vs 6 workers: ~{255.9/projected_time:.1f}x faster")

if __name__ == "__main__":
    test_20_workers()





