#!/usr/bin/env python3
"""
Quick Vidaio Subnet Test - Focus on Requirements
"""

import os
import cv2
import subprocess
import time
from datetime import datetime

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
    duration = total_frames / fps
    cap.release()
    
    return {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames, 'duration': duration}

def crop_video(input_path, output_path, duration_sec=5):
    """Crop video to 5 seconds"""
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

def test_upscaling(input_path, output_path, target_w, target_h, task_name):
    """Test upscaling task"""
    log(f"🖼️  {task_name}: {target_w}x{target_h}")
    
    info = get_video_info(input_path)
    log(f"   Input: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    log(f"   Duration: {info['duration']:.1f}s, Frames: {info['total_frames']}")
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale={target_w}:{target_h}:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        output_path
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        log(f"✅ {task_name}: {size_mb:.2f} MB in {elapsed:.1f}s")
        return True
    else:
        log(f"❌ {task_name} failed")
        return False

def test_compression(input_path, output_path, crf, task_name):
    """Test compression task"""
    log(f"🗜️  {task_name}: CRF {crf}")
    
    info = get_video_info(input_path)
    ref_size = os.path.getsize(input_path) / 1024 / 1024
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '128k',
        output_path
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        comp_size = os.path.getsize(output_path) / 1024 / 1024
        ratio = ref_size / comp_size
        log(f"✅ {task_name}: {comp_size:.2f} MB, ratio: {ratio:.1f}x in {elapsed:.1f}s")
        return True
    else:
        log(f"❌ {task_name} failed")
        return False

def main():
    log("🚀 QUICK VIDAIO SUBNET TEST")
    log("="*60)
    
    input_video = "/workspace/vidaio-subnet/elk.mp4"
    output_dir = "/workspace/vidaio-subnet/vidaio_quick_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original video info
    info = get_video_info(input_video)
    log(f"📹 Original: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    log(f"⏱️  Duration: {info['duration']:.1f}s, Frames: {info['total_frames']}")
    
    # Test 1: 5s cropped video
    log(f"\n{'='*60}")
    log("TEST 1: 5-SECOND CROPPED VIDEO")
    log(f"{'='*60}")
    
    cropped_5s = os.path.join(output_dir, "elk_5s.mp4")
    if crop_video(input_video, cropped_5s, 5):
        info_5s = get_video_info(cropped_5s)
        log(f"✅ 5s video: {info_5s['total_frames']} frames @ {info_5s['fps']:.1f} fps")
        
        # Test upscaling tasks on 5s video
        log(f"\n📊 UPSCALING TASKS (5s video):")
        test_upscaling(cropped_5s, os.path.join(output_dir, "sd2hd_5s.mp4"), 1920, 1080, "SD2HD")
        test_upscaling(cropped_5s, os.path.join(output_dir, "hd24k_5s.mp4"), 3840, 2160, "HD24K")
        test_upscaling(cropped_5s, os.path.join(output_dir, "sd24k_5s.mp4"), 3840, 2160, "SD24K")
        
        # Test compression tasks on 5s video
        log(f"\n📊 COMPRESSION TASKS (5s video):")
        test_compression(cropped_5s, os.path.join(output_dir, "comp_vmaf80_5s.mp4"), 45, "VMAF80")
        test_compression(cropped_5s, os.path.join(output_dir, "comp_vmaf90_5s.mp4"), 32, "VMAF90")
    
    # Test 2: Original 10s video
    log(f"\n{'='*60}")
    log("TEST 2: ORIGINAL 10-SECOND VIDEO")
    log(f"{'='*60}")
    
    # Test upscaling tasks on 10s video
    log(f"\n📊 UPSCALING TASKS (10s video):")
    test_upscaling(input_video, os.path.join(output_dir, "sd2hd_10s.mp4"), 1920, 1080, "SD2HD")
    test_upscaling(input_video, os.path.join(output_dir, "hd24k_10s.mp4"), 3840, 2160, "HD24K")
    test_upscaling(input_video, os.path.join(output_dir, "sd24k_10s.mp4"), 3840, 2160, "SD24K")
    test_upscaling(input_video, os.path.join(output_dir, "4k28k_10s.mp4"), 7680, 4320, "4K28K")
    
    # Test compression tasks on 10s video
    log(f"\n📊 COMPRESSION TASKS (10s video):")
    test_compression(input_video, os.path.join(output_dir, "comp_vmaf80_10s.mp4"), 45, "VMAF80")
    test_compression(input_video, os.path.join(output_dir, "comp_vmaf90_10s.mp4"), 32, "VMAF90")
    
    log(f"\n✅ All tests completed!")
    log(f"📁 Results in: {output_dir}")

if __name__ == "__main__":
    main()
