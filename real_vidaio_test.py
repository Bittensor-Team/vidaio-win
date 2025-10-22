#!/usr/bin/env python3
"""
REAL Vidaio Subnet Test - Using Actual Real-ESRGAN Workers
"""

import os
import cv2
import subprocess
import time
import requests
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fix division by zero
    if fps > 0:
        duration = total_frames / fps
    else:
        duration = 0
        fps = 30  # Default fallback
    
    cap.release()
    
    return {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames, 'duration': duration}

def crop_video(input_path, output_path, duration_sec=5):
    """Crop video to specific duration"""
    log(f"✂️  Cropping to {duration_sec}s...")
    
    cmd = ['ffmpeg', '-y', '-i', input_path, '-t', str(duration_sec), '-c', 'copy', output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        log(f"✅ Cropped: {size_mb:.2f} MB")
        return True
    else:
        log(f"❌ Cropping failed")
        return False

def extract_frames(video_path, frames_dir, max_frames=30):
    """Extract frames for Real-ESRGAN processing"""
    log(f"🎬 Extracting {max_frames} frames...")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames evenly
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
    log(f"✅ Extracted {extracted} frames")
    return extracted

def upscale_frame_real_esrgan(frame_path, output_path, worker_url):
    """Upscale single frame using Real-ESRGAN worker"""
    try:
        with open(frame_path, 'rb') as f:
            files = {'file': (os.path.basename(frame_path), f, 'image/png')}
            resp = requests.post(f"{worker_url}/upscale", files=files, timeout=120)
            
            if resp.status_code == 200:
                with open(output_path, 'wb') as out:
                    out.write(resp.content)
                return True
            else:
                log(f"❌ Worker error: Status {resp.status_code}")
                return False
    except Exception as e:
        log(f"❌ Worker error: {e}")
        return False

def upscale_video_real_esrgan_chunked(input_video, output_video, target_w, target_h, task_name, num_workers=5):
    """Upscale video using Real-ESRGAN workers with chunked approach"""
    log(f"🤖 REAL-ESRGAN CHUNKED {task_name}: {target_w}x{target_h}")
    
    info = get_video_info(input_video)
    log(f"   Input: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    log(f"   Duration: {info['duration']:.1f}s, Frames: {info['total_frames']}")
    
    # Extract frames
    frames_dir = f"/tmp/frames_{task_name}_{int(time.time())}"
    max_frames = min(50, info['total_frames'])  # More frames for better testing
    extracted = extract_frames(input_video, frames_dir, max_frames)
    
    if extracted == 0:
        log("❌ No frames extracted")
        return False
    
    # Setup workers
    worker_urls = [f"http://127.0.0.1:{8090 + i}" for i in range(num_workers)]
    
    # Check worker health
    healthy_workers = []
    for i, url in enumerate(worker_urls):
        try:
            resp = requests.get(f"{url}/health", timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded", False):
                    healthy_workers.append((i, url))
                    log(f"   ✅ Worker {i+1} ({url}) is healthy")
                else:
                    log(f"   ❌ Worker {i+1} ({url}) model not loaded")
            else:
                log(f"   ❌ Worker {i+1} ({url}) status {resp.status_code}")
        except Exception as e:
            log(f"   ❌ Worker {i+1} ({url}) error: {e}")
    
    if not healthy_workers:
        log("❌ No healthy workers found")
        return False
    
    log(f"✅ {len(healthy_workers)} workers ready")
    
    # Upscale frames using chunked approach
    upscaled_dir = f"/tmp/upscaled_{task_name}_{int(time.time())}"
    os.makedirs(upscaled_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frames)
    
    # Split frames into chunks for each worker
    chunk_size = math.ceil(total_frames / len(healthy_workers))
    chunks = []
    
    for i, (worker_id, worker_url) in enumerate(healthy_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        chunk_frames = frames[start_idx:end_idx]
        chunks.append((worker_id, worker_url, chunk_frames, frames_dir, upscaled_dir))
        log(f"   Worker {worker_id+1}: frames {start_idx}-{end_idx-1} ({len(chunk_frames)} frames)")
    
    log(f"🎬 Processing {total_frames} frames in {len(chunks)} chunks...")
    start_time = time.time()
    
    # Process chunks in parallel using ThreadPoolExecutor
    def process_chunk(args):
        worker_id, worker_url, frame_files, frames_dir, upscaled_dir = args
        
        log(f"🔧 Worker {worker_id+1} processing {len(frame_files)} frames...")
        chunk_start = time.time()
        
        processed = 0
        failed = 0
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame_idx = int(frame_file.split('_')[1].split('.')[0])
            output_path = os.path.join(upscaled_dir, f"upscaled_{frame_idx:06d}.png")
            
            try:
                with open(frame_path, 'rb') as f:
                    files = {'file': (frame_file, f, 'image/png')}
                    resp = requests.post(f"{worker_url}/upscale", files=files, timeout=120)
                    
                    if resp.status_code == 200:
                        with open(output_path, 'wb') as out:
                            out.write(resp.content)
                        processed += 1
                    else:
                        failed += 1
                        
            except Exception as e:
                failed += 1
                log(f"   ⚠️  Worker {worker_id+1} frame {frame_idx} failed: {str(e)[:50]}")
        
        chunk_elapsed = time.time() - chunk_start
        log(f"✅ Worker {worker_id+1} completed: {processed}/{len(frame_files)} frames in {chunk_elapsed:.1f}s")
        
        return (worker_id, processed, failed, chunk_elapsed)
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Collect results
    total_processed = 0
    total_failed = 0
    max_worker_time = 0
    
    for worker_id, processed, failed, elapsed in results:
        total_processed += processed
        total_failed += failed
        max_worker_time = max(max_worker_time, elapsed)
    
    total_elapsed = time.time() - start_time
    
    log(f"✅ Chunked Real-ESRGAN complete!")
    log(f"   Total time: {total_elapsed:.1f}s")
    log(f"   Slowest worker: {max_worker_time:.1f}s")
    log(f"   Processed: {total_processed}/{total_frames}")
    log(f"   Speedup: ~{max_worker_time/total_elapsed:.1f}x vs single worker")
    
    if total_processed == 0:
        log("❌ No frames upscaled successfully")
        return False
    
    # Scale upscaled frames to target resolution
    log(f"🔧 Scaling to target resolution: {target_w}x{target_h}")
    
    scaled_dir = f"/tmp/scaled_{task_name}_{int(time.time())}"
    os.makedirs(scaled_dir, exist_ok=True)
    
    upscaled_frames = sorted([f for f in os.listdir(upscaled_dir) if f.endswith('.png')])
    
    for frame_file in upscaled_frames:
        input_path = os.path.join(upscaled_dir, frame_file)
        output_path = os.path.join(scaled_dir, frame_file)
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f'scale={target_w}:{target_h}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            output_path
        ]
        subprocess.run(cmd, capture_output=True)
    
    # Create final video
    log("🎬 Creating final video...")
    
    # Check if scaled frames exist
    scaled_frames = sorted([f for f in os.listdir(scaled_dir) if f.endswith('.png')])
    log(f"   Found {len(scaled_frames)} scaled frames")
    
    if not scaled_frames:
        log("❌ No scaled frames found")
        return False
    
    frame_pattern = os.path.join(scaled_dir, "upscaled_%06d.png")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(info['fps']),
        '-i', frame_pattern,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_video
    ]
    
    log(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(output_video):
        size_mb = os.path.getsize(output_video) / 1024 / 1024
        log(f"✅ {task_name}: {size_mb:.2f} MB")
        return True
    else:
        log(f"❌ Video creation failed:")
        log(f"   Return code: {result.returncode}")
        log(f"   Stderr: {result.stderr}")
        log(f"   Stdout: {result.stdout}")
        return False

def test_real_vidaio_tasks():
    """Test real Vidaio tasks with Real-ESRGAN"""
    log("🚀 REAL VIDAIO SUBNET TEST (Real-ESRGAN)")
    log("="*60)
    
    input_video = "/workspace/vidaio-subnet/elk.mp4"
    output_dir = "/workspace/vidaio-subnet/real_vidaio_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original video info
    info = get_video_info(input_video)
    log(f"📹 Original: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    log(f"⏱️  Duration: {info['duration']:.1f}s, Frames: {info['total_frames']}")
    
    # Test 1: 5s cropped video
    log(f"\n{'='*60}")
    log("TEST 1: 5-SECOND CROPPED VIDEO (Real-ESRGAN)")
    log(f"{'='*60}")
    
    cropped_5s = os.path.join(output_dir, "elk_5s.mp4")
    if crop_video(input_video, cropped_5s, 5):
        info_5s = get_video_info(cropped_5s)
        log(f"✅ 5s video: {info_5s['total_frames']} frames @ {info_5s['fps']:.1f} fps")
        
        # Test upscaling tasks on 5s video
        log(f"\n📊 REAL UPSCALING TASKS (5s video):")
        upscale_video_real_esrgan_chunked(cropped_5s, os.path.join(output_dir, "real_sd2hd_5s.mp4"), 1920, 1080, "SD2HD_5s")
        upscale_video_real_esrgan_chunked(cropped_5s, os.path.join(output_dir, "real_hd24k_5s.mp4"), 3840, 2160, "HD24K_5s")
    
    # Test 2: Original 10s video (limited frames for speed)
    log(f"\n{'='*60}")
    log("TEST 2: 10-SECOND VIDEO (Real-ESRGAN, Limited Frames)")
    log(f"{'='*60}")
    
    log(f"\n📊 REAL UPSCALING TASKS (10s video, 50 frames):")
    upscale_video_real_esrgan_chunked(input_video, os.path.join(output_dir, "real_hd24k_10s.mp4"), 3840, 2160, "HD24K_10s")
    
    log(f"\n✅ Real Vidaio tests completed!")
    log(f"📁 Results in: {output_dir}")

if __name__ == "__main__":
    test_real_vidaio_tasks()
