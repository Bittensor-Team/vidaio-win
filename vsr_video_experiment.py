#!/usr/bin/env python3
"""
VSR Video Processing Experiment
Test with 150 frames from elk.mp4 for both SD and HD upscaling
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
from concurrent.futures import ThreadPoolExecutor
import subprocess

def load_vsr_model(model_path, worker_id):
    """Load VSR model for a specific worker"""
    print(f"🤖 Worker {worker_id}: Loading VSR model...")
    
    # Set up ONNX Runtime providers
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
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = img_rgb.shape[:2]
    
    # Convert to float32 and normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension: [1, C, H, W]
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    
    return img_tensor, (h, w)

def postprocess_output(output_tensor, original_size):
    """Postprocess model output back to image"""
    # Remove batch dimension and transpose back to HWC
    output_img = np.transpose(output_tensor[0], (1, 2, 0))
    
    # Denormalize from [0, 1] to [0, 255]
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    
    # Get new dimensions
    new_h, new_w = output_img.shape[:2]
    
    # Calculate upscaling factor
    upscale_factor = new_h / original_size[0]
    
    return output_img, upscale_factor

def process_frames_worker(worker_id, model, frame_paths):
    """Process frames for a single worker"""
    print(f"🔧 Worker {worker_id}: Processing {len(frame_paths)} frames...")
    
    start_time = time.time()
    processed = 0
    total_inference_time = 0
    
    for frame_path in frame_paths:
        try:
            # Load and preprocess image
            image = cv2.imread(frame_path)
            if image is None:
                continue
                
            input_tensor, original_size = preprocess_image(image)
            
            # Run inference
            inference_start = time.time()
            input_info = model.get_inputs()[0]
            output_info = model.get_outputs()[0]
            
            outputs = model.run([output_info.name], {input_info.name: input_tensor})
            
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Postprocess
            output_img, upscale_factor = postprocess_output(outputs[0], original_size)
            
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

def extract_video_frames(video_path, frames_dir, max_frames=150):
    """Extract frames from video"""
    print(f"🎬 Extracting {max_frames} frames from {video_path}...")
    
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
    print(f"✅ Extracted {extracted} frames")
    return extracted

def run_video_experiment(num_workers, test_frames=150, resolution="SD"):
    """Run VSR experiment with video frames"""
    print(f"\n🧪 VIDEO EXPERIMENT: {num_workers} workers, {test_frames} frames, {resolution}")
    print("="*70)
    
    # Model path
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Extract frames from elk.mp4
    video_path = "/workspace/vidaio-win/elk.mp4"
    frames_dir = f'/tmp/video_frames_{resolution}_{num_workers}w'
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    extracted = extract_video_frames(video_path, frames_dir, test_frames)
    if extracted == 0:
        print("❌ No frames extracted")
        return None
    
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
    
    # Distribute frames among workers
    frames_per_worker = math.ceil(len(frames) / num_workers)
    worker_tasks = []
    
    for i in range(num_workers):
        start_idx = i * frames_per_worker
        end_idx = min((i + 1) * frames_per_worker, len(frames))
        worker_frames = frames[start_idx:end_idx]
        worker_tasks.append((i+1, models[i], worker_frames))
        print(f"   Worker {i+1}: {len(worker_frames)} frames")
    
    # Process frames in parallel using ThreadPoolExecutor
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
    
    frames_per_second = len(frames) / total_elapsed
    seconds_per_frame = total_elapsed / len(frames)
    
    print(f"\n📊 RESULTS:")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Frames processed: {total_processed}/{len(frames)}")
    print(f"   Frames per second: {frames_per_second:.2f}")
    print(f"   Seconds per frame: {seconds_per_frame:.3f}")
    print(f"   Worker times: min={min_worker_time:.1f}s, max={max_worker_time:.1f}s")
    print(f"   Avg inference time: {avg_inference_time:.3f}s/frame")
    print(f"   Efficiency: {min_worker_time/max_worker_time:.2f}")
    
    return {
        'workers': num_workers,
        'frames': len(frames),
        'resolution': resolution,
        'total_time': total_elapsed,
        'frames_per_second': frames_per_second,
        'seconds_per_frame': seconds_per_frame,
        'avg_inference_time': avg_inference_time,
        'efficiency': min_worker_time/max_worker_time
    }

def main():
    print("🚀 VSR Video Processing Experiment")
    print("Testing with 150 frames from elk.mp4 - SD and HD upscaling")
    print("="*80)
    
    # Test different worker counts for both SD and HD
    worker_counts = [1, 2, 4, 6, 8, 10, 12, 15, 20]
    resolutions = ["SD", "HD"]
    all_results = []
    
    for resolution in resolutions:
        print(f"\n{'='*80}")
        print(f"TESTING {resolution} UPSCALING")
        print(f"{'='*80}")
        
        results = []
        
        for worker_count in worker_counts:
            result = run_video_experiment(worker_count, test_frames=150, resolution=resolution)
            if result:
                results.append(result)
                all_results.append(result)
        
        # Print summary for this resolution
        print(f"\n📈 {resolution} PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Workers':<8} {'FPS':<6} {'s/frame':<8} {'Inference':<10} {'Efficiency':<10} {'Speedup':<8}")
        print("-" * 70)
        
        baseline_fps = None
        for result in results:
            if baseline_fps is None:
                baseline_fps = result['frames_per_second']
                speedup = 1.0
            else:
                speedup = result['frames_per_second'] / baseline_fps
            
            print(f"{result['workers']:<8} {result['frames_per_second']:<6.2f} {result['seconds_per_frame']:<8.3f} {result['avg_inference_time']:<10.3f} {result['efficiency']:<10.2f} {speedup:<8.2f}")
        
        # Find optimal worker count for this resolution
        best_result = max(results, key=lambda x: x['frames_per_second'])
        print(f"\n🏆 {resolution} OPTIMAL: {best_result['workers']} workers")
        print(f"   Best FPS: {best_result['frames_per_second']:.2f}")
        print(f"   Best s/frame: {best_result['seconds_per_frame']:.3f}")
        print(f"   Inference time: {best_result['avg_inference_time']:.3f}s/frame")
    
    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")
    
    sd_results = [r for r in all_results if r['resolution'] == 'SD']
    hd_results = [r for r in all_results if r['resolution'] == 'HD']
    
    print(f"{'Resolution':<10} {'Workers':<8} {'FPS':<6} {'s/frame':<8} {'Inference':<10}")
    print("-" * 60)
    
    for sd_result, hd_result in zip(sd_results, hd_results):
        print(f"{'SD':<10} {sd_result['workers']:<8} {sd_result['frames_per_second']:<6.2f} {sd_result['seconds_per_frame']:<8.3f} {sd_result['avg_inference_time']:<10.3f}")
        print(f"{'HD':<10} {hd_result['workers']:<8} {hd_result['frames_per_second']:<6.2f} {hd_result['seconds_per_frame']:<8.3f} {hd_result['avg_inference_time']:<10.3f}")
        print()

if __name__ == "__main__":
    main()





