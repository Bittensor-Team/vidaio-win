#!/usr/bin/env python3
"""
Vidaio Pipeline Test - HD24K & 4K28K Only
Test HD24K (1080p→4K) and 4K28K (4K→8K) tasks with DXM-FP32 model
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
import json

def resize_video(input_path, output_path, target_resolution):
    """Resize video to target resolution using ffmpeg"""
    print(f"🎬 Resizing video to {target_resolution}...")
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale={target_resolution}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ Video resized to {target_resolution}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ Error resizing video: {e}")
        return False

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

def run_task_experiment(task_name, input_video, target_resolution, scale_factor, num_workers, test_frames=50, duration="5s"):
    """Run VSR experiment for a specific task"""
    print(f"\n🧪 TASK: {task_name} ({duration})")
    print(f"   Input: {input_video}")
    print(f"   Target: {target_resolution}")
    print(f"   Scale: {scale_factor}x")
    print(f"   Workers: {num_workers}")
    print(f"   Frames: {test_frames}")
    print("="*70)
    
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Extract frames
    frames_dir = f'/tmp/frames_{task_name}_{num_workers}w_{duration}'
    extracted = extract_video_frames(input_video, frames_dir, test_frames)
    if extracted == 0:
        print("❌ No frames extracted")
        return None
    
    # Create output directory for upscaled frames
    upscaled_dir = f'/tmp/upscaled_{task_name}_{num_workers}w_{duration}'
    os.makedirs(upscaled_dir, exist_ok=True)
    
    # Load models for each worker
    print(f"🤖 Loading {num_workers} VSR models...")
    models = []
    for i in range(num_workers):
        model = load_vsr_model(model_path, i+1)
        if model is None:
            print(f"❌ Failed to load model for worker {i+1}")
            return None
        models.append(model)
    
    # Get frame list
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames = [os.path.join(frames_dir, f) for f in frames]
    
    if len(frames) == 0:
        print("❌ No frames found in directory")
        return None
    
    # Distribute frames among workers
    frames_per_worker = math.ceil(len(frames) / num_workers)
    worker_tasks = []
    
    for i in range(num_workers):
        start_idx = i * frames_per_worker
        end_idx = min((i + 1) * frames_per_worker, len(frames))
        worker_frames = frames[start_idx:end_idx]
        worker_tasks.append((i+1, models[i], worker_frames, upscaled_dir))
        print(f"   Worker {i+1}: {len(worker_frames)} frames")
    
    # Process frames in parallel
    print(f"🎬 Processing {len(frames)} frames with {num_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda args: process_frames_worker(*args), worker_tasks))
    
    total_elapsed = time.time() - start_time
    
    # Calculate results
    total_processed = sum(result['processed'] for result in results)
    max_worker_time = max(result['total_time'] for result in results)
    min_worker_time = min(result['total_time'] for result in results)
    avg_inference_time = sum(result['avg_inference_time'] for result in results) / len(results)
    
    frames_per_second = len(frames) / total_elapsed if total_elapsed > 0 else 0
    seconds_per_frame = total_elapsed / len(frames) if len(frames) > 0 else 0
    
    print(f"\n📊 RESULTS:")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Frames processed: {total_processed}/{len(frames)}")
    print(f"   Frames per second: {frames_per_second:.2f}")
    print(f"   Seconds per frame: {seconds_per_frame:.3f}")
    print(f"   Worker times: min={min_worker_time:.1f}s, max={max_worker_time:.1f}s")
    print(f"   Avg inference time: {avg_inference_time:.3f}s/frame")
    print(f"   Efficiency: {min_worker_time/max_worker_time:.2f}")
    
    # Create output video
    output_video = f"/workspace/vidaio-win/real_vidaio_tests/{task_name}_{duration}_{num_workers}w.mp4"
    if create_video_from_frames(upscaled_dir, output_video):
        print(f"✅ Output video saved: {output_video}")
    
    return {
        'task': task_name,
        'duration': duration,
        'workers': num_workers,
        'frames': len(frames),
        'total_time': total_elapsed,
        'frames_per_second': frames_per_second,
        'seconds_per_frame': seconds_per_frame,
        'avg_inference_time': avg_inference_time,
        'efficiency': min_worker_time/max_worker_time,
        'output_video': output_video
    }

def find_optimal_workers(task_name, input_video, target_resolution, scale_factor, duration="5s", test_frames=150):
    """Find optimal worker count for a task"""
    print(f"\n🔍 FINDING OPTIMAL WORKERS FOR {task_name} ({duration})")
    print("="*70)
    
    # Based on previous experiments, test around known optimal values
    if "HD24K" in task_name:
        worker_counts = [6, 8, 10, 12, 15]  # HD24K optimal around 15
    elif "4K28K" in task_name:
        worker_counts = [4, 6, 8, 10, 12]  # 4K28K optimal around 8
    else:
        worker_counts = [6, 8, 10, 12, 15]  # Default range
    
    results = []
    
    for worker_count in worker_counts:
        result = run_task_experiment(
            f"{task_name}_{worker_count}w", 
            input_video, 
            target_resolution, 
            scale_factor, 
            worker_count, 
            test_frames,
            duration
        )
        if result:
            results.append(result)
    
    # Find optimal worker count
    if results:
        best_result = max(results, key=lambda x: x['frames_per_second'])
        print(f"\n🏆 OPTIMAL FOR {task_name} ({duration}): {best_result['workers']} workers")
        print(f"   Best FPS: {best_result['frames_per_second']:.2f}")
        print(f"   Best s/frame: {best_result['seconds_per_frame']:.3f}")
        print(f"   Inference time: {best_result['avg_inference_time']:.3f}s/frame")
        
        return best_result
    return None

def main():
    print("🚀 VIDAIO PIPELINE TEST - HD24K & 4K28K Only")
    print("Testing HD24K and 4K28K tasks with DXM-FP32 model")
    print("="*80)
    
    # Create output directory
    output_dir = "/workspace/vidaio-win/real_vidaio_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing input videos
    print("🎬 Checking input videos...")
    
    # 1080p for HD tasks
    hd_video = "/tmp/elk_1080p.mp4"
    if not os.path.exists(hd_video):
        print(f"❌ HD video not found: {hd_video}")
        return
    print(f"✅ HD video found: {hd_video}")
    
    # 4K for 4K tasks
    k4_video = "/tmp/elk_4k.mp4"
    if not os.path.exists(k4_video):
        print(f"❌ 4K video not found: {k4_video}")
        return
    print(f"✅ 4K video found: {k4_video}")
    
    print("✅ Input videos ready")
    
    # Test HD24K and 4K28K tasks with different durations
    task_results = {}
    durations = [
        ("5s", 150),   # 5 seconds, 150 frames
        ("10s", 300)   # 10 seconds, 300 frames
    ]
    
    for duration, test_frames in durations:
        print(f"\n{'='*80}")
        print(f"TESTING {duration.upper()} DURATION ({test_frames} frames)")
        print(f"{'='*80}")
        
        # HD24K: 1080p → 4K (2x scale)
        print(f"\n{'='*80}")
        print(f"TESTING HD24K: 1080p → 4K (2x scale) - {duration}")
        print(f"{'='*80}")
        hd24k_result = find_optimal_workers("HD24K", hd_video, "3840x2160", 2, duration, test_frames)
        if hd24k_result:
            task_results[f"HD24K_{duration}"] = hd24k_result
        
        # 4K28K: 4K → 8K (2x scale)
        print(f"\n{'='*80}")
        print(f"TESTING 4K28K: 4K → 8K (2x scale) - {duration}")
        print(f"{'='*80}")
        k4k8_result = find_optimal_workers("4K28K", k4_video, "7680x4320", 2, duration, test_frames)
        if k4k8_result:
            task_results[f"4K28K_{duration}"] = k4k8_result
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for task_name, result in task_results.items():
        print(f"{task_name:>12}: {result['workers']:>2} workers, {result['frames_per_second']:>6.2f} FPS, {result['seconds_per_frame']:>6.3f} s/frame")
    
    # Save results to JSON
    results_file = "/workspace/vidaio-win/vidaio_pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump(task_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"📁 Output videos saved to: {output_dir}")

if __name__ == "__main__":
    main()
