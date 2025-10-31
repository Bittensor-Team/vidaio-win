#!/usr/bin/env python3
"""
Test SD2HD Resolution Fix
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}

def load_vsr_model(model_path, worker_id):
    """Load VSR model for a specific worker"""
    print(f"🤖 Worker {worker_id}: Loading VSR model...")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        model = ort.InferenceSession(model_path, providers=providers)
        input_info = model.get_inputs()[0]
        output_info = model.get_outputs()[0]
        
        print(f"✅ Worker {worker_id}: Model loaded - {input_info.shape} -> {output_info.shape}")
        return model
    except Exception as e:
        print(f"❌ Worker {worker_id}: Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for VSR model"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    return img_tensor, (h, w)

def postprocess_output(output_tensor, original_size):
    """Postprocess model output back to image"""
    output_img = np.transpose(output_tensor[0], (1, 2, 0))
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    new_h, new_w = output_img.shape[:2]
    upscale_factor = new_h / original_size[0]
    return output_img, upscale_factor

def process_frames_worker(worker_id, model, frame_paths, output_dir):
    """Process frames for a single worker and save upscaled frames"""
    print(f"🔧 Worker {worker_id}: Processing {len(frame_paths)} frames...")
    
    start_time = time.time()
    processed = 0
    total_inference_time = 0
    
    for frame_path in frame_paths:
        try:
            image = cv2.imread(frame_path)
            if image is None:
                continue
                
            input_tensor, original_size = preprocess_image(image)
            
            inference_start = time.time()
            input_info = model.get_inputs()[0]
            output_info = model.get_outputs()[0]
            
            outputs = model.run([output_info.name], {input_info.name: input_tensor})
            
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            output_img, upscale_factor = postprocess_output(outputs[0], original_size)
            
            # Save upscaled frame
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(output_dir, f"upscaled_{frame_name}")
            cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            
            processed += 1
            
        except Exception as e:
            print(f"❌ Worker {worker_id} frame error: {e}")
    
    elapsed = time.time() - start_time
    avg_inference_time = total_inference_time / processed if processed > 0 else 0
    
    print(f"✅ Worker {worker_id}: {processed}/{len(frame_paths)} frames in {elapsed:.1f}s")
    print(f"   Avg inference time: {avg_inference_time:.3f}s/frame")
    
    return {
        'worker_id': worker_id,
        'processed': processed,
        'total_time': elapsed,
        'avg_inference_time': avg_inference_time,
        'frames_per_second': processed / elapsed if elapsed > 0 else 0
    }

def extract_video_frames(video_path, frames_dir, max_frames=50):
    """Extract frames from video"""
    print(f"🎬 Extracting {max_frames} frames from {video_path}...")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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
    print(f"✅ Extracted {extracted} frames")
    return extracted

def create_video_from_frames(frames_dir, output_video, fps=30):
    """Create video from upscaled frames"""
    print(f"🎬 Creating video from frames...")
    
    frame_pattern = os.path.join(frames_dir, "upscaled_frame_%06d.png")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_video
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            size_mb = os.path.getsize(output_video) / 1024 / 1024
            print(f"✅ Video created: {size_mb:.2f} MB")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False

def main():
    print("🧪 TESTING SD2HD RESOLUTION FIX")
    print("="*50)
    
    # Create 480p input video
    input_video = "/tmp/test_input_480p.mp4"
    if not os.path.exists(input_video):
        print("🎬 Creating 480p input video...")
        cmd = [
            'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
            '-vf', 'scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-t', '5',  # 5 seconds
            input_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created {input_video}")
        else:
            print(f"❌ Failed to create input video: {result.stderr}")
            return
    
    # Extract frames
    frames_dir = '/tmp/test_frames_sd2hd'
    extracted = extract_video_frames(input_video, frames_dir, max_frames=20)
    if extracted == 0:
        print("❌ No frames extracted")
        return
    
    # Create output directory for upscaled frames
    upscaled_dir = '/tmp/test_upscaled_sd2hd'
    os.makedirs(upscaled_dir, exist_ok=True)
    
    # Load VSR model
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    model = load_vsr_model(model_path, 1)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Get frame list
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames = [os.path.join(frames_dir, f) for f in frames]
    
    print(f"🎬 Processing {len(frames)} frames...")
    start_time = time.time()
    
    # Process frames
    for frame_path in frames:
        try:
            image = cv2.imread(frame_path)
            if image is None:
                continue
                
            input_tensor, original_size = preprocess_image(image)
            
            input_info = model.get_inputs()[0]
            output_info = model.get_outputs()[0]
            
            outputs = model.run([output_info.name], {input_info.name: input_tensor})
            
            output_img, upscale_factor = postprocess_output(outputs[0], original_size)
            
            # Save upscaled frame
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(upscaled_dir, f"upscaled_{frame_name}")
            cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            
            print(f"   Processed {frame_name}: {original_size[1]}x{original_size[0]} → {output_img.shape[1]}x{output_img.shape[0]} ({upscale_factor:.2f}x)")
            
        except Exception as e:
            print(f"❌ Frame error: {e}")
    
    total_elapsed = time.time() - start_time
    print(f"✅ Processed {len(frames)} frames in {total_elapsed:.1f}s")
    
    # Create video from upscaled frames
    output_video = "/tmp/test_sd2hd_upscaled.mp4"
    if create_video_from_frames(upscaled_dir, output_video):
        print(f"✅ Output video created: {output_video}")
        
        # Check resolution
        info = get_video_info(output_video)
        if 'streams' in info and info['streams']:
            stream = info['streams'][0]
            resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
            print(f"📹 VSR Output Resolution: {resolution}")
            
            # Fix resolution
            print(f"🔧 Fixing resolution: {resolution} → 1920x1080")
            fixed_video = "/tmp/test_sd2hd_fixed.mp4"
            resize_cmd = [
                'ffmpeg', '-y', '-i', output_video,
                '-vf', 'scale=1920:1080:flags=lanczos',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                fixed_video
            ]
            
            if subprocess.run(resize_cmd, capture_output=True, text=True).returncode == 0:
                print(f"✅ Fixed video created: {fixed_video}")
                
                # Check fixed resolution
                fixed_info = get_video_info(fixed_video)
                if 'streams' in fixed_info and fixed_info['streams']:
                    stream = fixed_info['streams'][0]
                    fixed_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
                    print(f"📹 Fixed Resolution: {fixed_resolution}")
                    
                    if fixed_resolution == "1920x1080":
                        print("🎉 SUCCESS! SD2HD resolution fix works!")
                    else:
                        print(f"❌ Still wrong resolution: {fixed_resolution}")
                else:
                    print("❌ Could not get fixed video info")
            else:
                print("❌ Failed to fix resolution")
        else:
            print("❌ Could not get video info")

if __name__ == "__main__":
    main()





