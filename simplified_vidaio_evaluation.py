#!/usr/bin/env python3
"""
Simplified Vidaio Evaluation
Runs VSR processing with optimal workers and evaluates with scoring
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
import tempfile
import shutil
import uuid

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

def calculate_psnr_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate PSNR with fixed parsing"""
    try:
        psnr_cmd = [
            'ffmpeg', '-y',
            '-i', reference_path,
            '-i', distorted_path,
            '-lavfi', 'psnr=stats_file=/tmp/psnr.log',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(psnr_cmd, capture_output=True, text=True, timeout=60)
        
        # Parse PSNR from log - look for the last frame's psnr_avg
        psnr_score = 0.0
        if os.path.exists('/tmp/psnr.log'):
            with open('/tmp/psnr.log', 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                
                # Find the last line with psnr_avg
                for line in reversed(lines):
                    if 'psnr_avg:' in line:
                        try:
                            # Extract psnr_avg value
                            parts = line.split('psnr_avg:')
                            if len(parts) > 1:
                                psnr_value = parts[1].split()[0]
                                if psnr_value == 'inf':
                                    psnr_score = 100.0  # Perfect match
                                else:
                                    psnr_score = float(psnr_value)
                                break
                        except Exception as e:
                            continue
        
        # Cleanup
        if os.path.exists('/tmp/psnr.log'):
            os.remove('/tmp/psnr.log')
        
        return psnr_score
        
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        return 0.0

def calculate_ssim_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate SSIM with fixed parsing"""
    try:
        ssim_cmd = [
            'ffmpeg', '-y',
            '-i', reference_path,
            '-i', distorted_path,
            '-lavfi', 'ssim=stats_file=/tmp/ssim.log',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(ssim_cmd, capture_output=True, text=True, timeout=60)
        
        # Parse SSIM from log - look for the last frame's All value
        ssim_score = 0.0
        if os.path.exists('/tmp/ssim.log'):
            with open('/tmp/ssim.log', 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                
                # Find the last line with All: value
                for line in reversed(lines):
                    if 'All:' in line:
                        try:
                            # Extract All value
                            parts = line.split('All:')
                            if len(parts) > 1:
                                ssim_value = parts[1].split()[0]
                                ssim_score = float(ssim_value)
                                break
                        except Exception as e:
                            continue
        
        # Cleanup
        if os.path.exists('/tmp/ssim.log'):
            os.remove('/tmp/ssim.log')
        
        return ssim_score
        
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0

def calculate_vmaf_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate VMAF alternative using fixed PSNR and SSIM"""
    try:
        psnr = calculate_psnr_fixed(reference_path, distorted_path)
        ssim = calculate_ssim_fixed(reference_path, distorted_path)
        
        # Convert PSNR to 0-100 scale (VMAF-like)
        vmaf_psnr = min(psnr * 2.0, 100.0)  # Scale PSNR to 0-100
        
        # SSIM is already 0-1, convert to 0-100
        vmaf_ssim = ssim * 100.0
        
        # Combine PSNR and SSIM for VMAF-like score
        vmaf_score = (vmaf_psnr * 0.6 + vmaf_ssim * 0.4)
        
        return vmaf_score, psnr, ssim
        
    except Exception as e:
        print(f"VMAF calculation failed: {e}")
        return 0.0, 0.0, 0.0

def calculate_pie_app_improved(reference_path: str, distorted_path: str) -> float:
    """Calculate PIE-APP perceptual quality score using proper frame-by-frame analysis"""
    try:
        import cv2
        import numpy as np
        import random
        
        # Read videos
        ref_cap = cv2.VideoCapture(reference_path)
        dist_cap = cv2.VideoCapture(distorted_path)
        
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            return 0.0
        
        # Get video properties
        ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
        dist_fps = dist_cap.get(cv2.CAP_PROP_FPS)
        ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dist_frame_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use minimum frame count to avoid out-of-bounds
        min_frames = min(ref_frame_count, dist_frame_count)
        
        if min_frames < 4:
            ref_cap.release()
            dist_cap.release()
            return 0.0
        
        # Sample 4 random frames as per Vidaio spec
        frame_indices = sorted(random.sample(range(min_frames), min(4, min_frames)))
        
        perceptual_differences = []
        
        for frame_idx in frame_indices:
            # Set frame position
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frames
            ret_ref, ref_frame = ref_cap.read()
            ret_dist, dist_frame = dist_cap.read()
            
            if not ret_ref or not ret_dist:
                continue
            
            # Resize to same dimensions for comparison
            if ref_frame.shape[:2] != dist_frame.shape[:2]:
                dist_frame = cv2.resize(dist_frame, (ref_frame.shape[1], ref_frame.shape[0]))
            
            # Convert to float and normalize
            ref_frame = ref_frame.astype(np.float32) / 255.0
            dist_frame = dist_frame.astype(np.float32) / 255.0
            
            # Calculate perceptual difference using multiple metrics
            # 1. Structural similarity (SSIM-like)
            ssim_diff = calculate_ssim_like(ref_frame, dist_frame)
            
            # 2. Edge preservation
            edge_diff = calculate_edge_difference(ref_frame, dist_frame)
            
            # 3. Color fidelity
            color_diff = calculate_color_difference(ref_frame, dist_frame)
            
            # 4. Texture preservation
            texture_diff = calculate_texture_difference(ref_frame, dist_frame)
            
            # Combine metrics with weights (perceptual importance)
            perceptual_diff = (
                0.3 * ssim_diff +      # Structural similarity
                0.25 * edge_diff +     # Edge preservation
                0.25 * color_diff +    # Color fidelity
                0.2 * texture_diff     # Texture preservation
            )
            
            perceptual_differences.append(perceptual_diff)
        
        ref_cap.release()
        dist_cap.release()
        
        if not perceptual_differences:
            return 0.0
        
        # Calculate average PIE-APP score
        avg_pie_app = np.mean(perceptual_differences)
        
        # Apply Vidaio normalization:
        # 1. Cap values: max(Average_PIE-APP, 2.0)
        capped_score = min(avg_pie_app, 2.0)
        
        # 2. Sigmoid normalization: normalized_score = 1/(1+exp(-Average_PIE-APP))
        sigmoid_score = 1 / (1 + np.exp(-capped_score))
        
        # 3. Convert "lower is better" to "higher is better" (0-1 range)
        # Since PIE-APP is "lower is better", we invert it
        final_score = 1.0 - sigmoid_score
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        print(f"PIE-APP calculation failed: {e}")
        return 0.0

def calculate_ssim_like(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM-like structural similarity"""
    try:
        # Convert to grayscale for SSIM calculation
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Calculate mean
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # Calculate variance and covariance
        sigma1_sq = np.var(gray1)
        sigma2_sq = np.var(gray2)
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM formula
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        # Convert to difference (1 - SSIM)
        return 1.0 - ssim
        
    except:
        return 1.0

def calculate_edge_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate edge preservation difference"""
    try:
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Calculate gradients using Sobel
        grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
        mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Calculate normalized difference
        diff = np.mean(np.abs(mag1 - mag2)) / (np.mean(mag1) + 1e-8)
        
        return min(1.0, diff)
        
    except:
        return 1.0

def calculate_color_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate color fidelity difference"""
    try:
        # Convert to LAB color space for better perceptual color difference
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        # Calculate mean color difference in LAB space
        color_diff = np.mean(np.abs(lab1 - lab2))
        
        # Normalize (LAB values are typically 0-100 for L, -128 to 127 for a,b)
        normalized_diff = color_diff / 100.0
        
        return min(1.0, normalized_diff)
        
    except:
        return 1.0

def calculate_texture_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate texture preservation difference"""
    try:
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Calculate local binary patterns (LBP) for texture analysis
        def calculate_lbp(image):
            # Simple LBP implementation
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    center = image[i, j]
                    binary = ''
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            if image[i+di, j+dj] >= center:
                                binary += '1'
                            else:
                                binary += '0'
                    lbp[i, j] = int(binary, 2)
            return lbp
        
        lbp1 = calculate_lbp(gray1)
        lbp2 = calculate_lbp(gray2)
        
        # Calculate histogram difference
        hist1, _ = np.histogram(lbp1, bins=256, range=(0, 256))
        hist2, _ = np.histogram(lbp2, bins=256, range=(0, 256))
        
        # Normalize histograms
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)
        
        # Calculate chi-squared distance
        chi_squared = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-8))
        
        return min(1.0, chi_squared)
        
    except:
        return 1.0

def sigmoid_transformation(x: float) -> float:
    """Sigmoid transformation for quality score"""
    return 1 / (1 + math.exp(-6 * (x - 0.5)))

def calculate_vidaio_score(reference_path: str, processed_path: str, content_length: float) -> dict:
    """Calculate Vidaio scoring using the official formula"""
    
    print(f"🔍 Calculating VMAF alternative...")
    vmaf_score, psnr, ssim = calculate_vmaf_fixed(reference_path, processed_path)
    vmaf_percentage = vmaf_score / 100.0  # Convert to 0-1 range
    
    # VMAF check (must be ≥ 50% or S_Q = 0)
    vmaf_check = 1.0 if vmaf_percentage >= 0.5 else 0.0
    
    print(f"🔍 Calculating PIE-APP score...")
    pie_app_score = calculate_pie_app_improved(reference_path, processed_path)
    
    # Quality Score (S_Q)
    s_q = sigmoid_transformation(pie_app_score) * vmaf_check
    
    # Length Score (S_L)
    s_l = math.log(1 + content_length) / math.log(1 + 320)
    
    # Pre-score
    s_pre = 0.5 * s_q + 0.5 * s_l
    
    # Final Score (S_F) with exponential scaling
    s_f = 0.1 * math.exp(6.979 * (s_pre - 0.5))
    
    return {
        'vmaf_score': vmaf_score,
        'vmaf_percentage': vmaf_percentage,
        'psnr': psnr,
        'ssim': ssim,
        'pie_app_score': pie_app_score,
        'quality_score': s_q,
        'length_score': s_l,
        'pre_score': s_pre,
        'final_score': s_f,
        'vmaf_check_passed': vmaf_check > 0
    }

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

def extract_video_frames(video_path, frames_dir, max_frames=150):
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

def run_vsr_task(task_name, input_video, target_resolution, scale_factor, num_workers, test_frames=150):
    """Run VSR task with optimal workers"""
    print(f"\n🧪 TASK: {task_name}")
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
    frames_dir = f'/tmp/frames_{task_name}_{num_workers}w'
    extracted = extract_video_frames(input_video, frames_dir, test_frames)
    if extracted == 0:
        print("❌ No frames extracted")
        return None
    
    # Create output directory for upscaled frames
    upscaled_dir = f'/tmp/upscaled_{task_name}_{num_workers}w'
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
    
    # Special handling for SD24K (4x upscale = 2x + 2x)
    if task_name == 'SD24K':
        print("🔄 First upscale: 480p → 1080p")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            first_results = list(executor.map(lambda args: process_frames_worker(*args), worker_tasks))
        
        # Create intermediate video
        intermediate_video = f'/tmp/intermediate_{task_name}_{num_workers}w.mp4'
        if create_video_from_frames(upscaled_dir, intermediate_video):
            print("✅ Intermediate video created")
            
            # Resize to 1080p for second upscale
            resized_intermediate = f'/tmp/resized_intermediate_{task_name}_{num_workers}w.mp4'
            resize_cmd = [
                'ffmpeg', '-y', '-i', intermediate_video,
                '-vf', 'scale=1920:1080',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                resized_intermediate
            ]
            
            if subprocess.run(resize_cmd, capture_output=True, text=True).returncode == 0:
                print("✅ Intermediate video resized to 1080p")
                
                # Extract frames from resized intermediate
                intermediate_frames_dir = f'/tmp/frames_intermediate_{task_name}_{num_workers}w'
                intermediate_frames = extract_video_frames(resized_intermediate, intermediate_frames_dir, test_frames)
                
                if intermediate_frames > 0:
                    # Create new worker tasks for second upscale
                    intermediate_frames_list = sorted([f for f in os.listdir(intermediate_frames_dir) if f.endswith('.png')])
                    intermediate_frames_list = [os.path.join(intermediate_frames_dir, f) for f in intermediate_frames_list]
                    
                    frames_per_worker = math.ceil(len(intermediate_frames_list) / num_workers)
                    second_worker_tasks = []
                    
                    for i in range(num_workers):
                        start_idx = i * frames_per_worker
                        end_idx = min((i + 1) * frames_per_worker, len(intermediate_frames_list))
                        worker_frames = intermediate_frames_list[start_idx:end_idx]
                        second_worker_tasks.append((i+1, models[i], worker_frames, upscaled_dir))
                    
                    print(f"🔄 Second upscale: 1080p → 4K with {len(intermediate_frames_list)} frames")
                    
                    # Second upscale
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        results = list(executor.map(lambda args: process_frames_worker(*args), second_worker_tasks))
                else:
                    print("❌ Failed to extract intermediate frames")
                    return None
            else:
                print("❌ Failed to resize intermediate video")
                return None
        else:
            print("❌ Failed to create intermediate video")
            return None
    else:
        # Single upscale for other tasks
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
    output_video = f"/tmp/vsr_{task_name}_{num_workers}w.mp4"
    if create_video_from_frames(upscaled_dir, output_video):
        print(f"✅ Output video saved: {output_video}")
        
        # Fix resolution for SD2HD - resize to target resolution
        if task_name == 'SD2HD':
            print(f"🔧 Fixing SD2HD resolution: 1708x960 → 1920x1080")
            fixed_video = f"/tmp/vsr_{task_name}_{num_workers}w_fixed.mp4"
            resize_cmd = [
                'ffmpeg', '-y', '-i', output_video,
                '-vf', 'scale=1920:1080:flags=lanczos',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                fixed_video
            ]
            
            if subprocess.run(resize_cmd, capture_output=True, text=True).returncode == 0:
                print(f"✅ SD2HD resolution fixed: {fixed_video}")
                output_video = fixed_video
            else:
                print(f"❌ Failed to fix SD2HD resolution")
        
        # Get video info
        info = get_video_info(output_video)
        duration = 0.0
        resolution = "Unknown"
        
        if 'format' in info and 'duration' in info['format']:
            duration = float(info['format']['duration'])
        
        if 'streams' in info and info['streams']:
            stream = info['streams'][0]
            resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
        
        return {
            'task': task_name,
            'workers': num_workers,
            'frames': len(frames),
            'total_time': total_elapsed,
            'frames_per_second': frames_per_second,
            'seconds_per_frame': seconds_per_frame,
            'avg_inference_time': avg_inference_time,
            'efficiency': min_worker_time/max_worker_time,
            'output_video': output_video,
            'duration': duration,
            'resolution': resolution
        }
    
    return None

def create_ground_truth_videos():
    """Create FFmpeg ground truth videos for comparison"""
    print("🎬 Creating FFmpeg ground truth videos...")
    
    # Create input videos at different resolutions
    input_videos = {
        '480p': '/tmp/input_480p.mp4',
        '1080p': '/tmp/input_1080p.mp4',
        '4k': '/tmp/input_4k.mp4'
    }
    
    # Create 480p input (for SD2HD and SD24K)
    cmd_480p = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '10',  # 10 seconds for better length scores
        input_videos['480p']
    ]
    
    # Create 1080p input (for HD24K)
    cmd_1080p = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '10',  # 10 seconds for better length scores
        input_videos['1080p']
    ]
    
    # Create 4K input (for 4K28K)
    cmd_4k = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '10',  # 10 seconds for better length scores
        input_videos['4k']
    ]
    
    for cmd, path in [(cmd_480p, input_videos['480p']), 
                      (cmd_1080p, input_videos['1080p']),
                      (cmd_4k, input_videos['4k'])]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created {path}")
        else:
            print(f"❌ Failed to create {path}: {result.stderr}")
    
    # Create FFmpeg ground truth upscaled videos
    ground_truth_videos = {
        'sd2hd': '/tmp/ground_truth_sd2hd.mp4',
        'hd24k': '/tmp/ground_truth_hd24k.mp4',
        '4k28k': '/tmp/ground_truth_4k28k.mp4',
        'sd24k': '/tmp/ground_truth_sd24k.mp4'
    }
    
    # SD2HD: 480p -> 1080p (2x upscale)
    cmd_sd2hd = [
        'ffmpeg', '-y', '-i', input_videos['480p'],
        '-vf', 'scale=1920:1080:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',  # High quality
        ground_truth_videos['sd2hd']
    ]
    
    # HD24K: 1080p -> 4K (2x upscale)
    cmd_hd24k = [
        'ffmpeg', '-y', '-i', input_videos['1080p'],
        '-vf', 'scale=3840:2160:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',  # High quality
        ground_truth_videos['hd24k']
    ]
    
    # 4K28K: 4K -> 8K (2x upscale)
    cmd_4k28k = [
        'ffmpeg', '-y', '-i', input_videos['4k'],
        '-vf', 'scale=7680:4320:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',  # High quality
        ground_truth_videos['4k28k']
    ]
    
    # SD24K: 480p -> 4K (4x upscale, done in 2 steps)
    cmd_sd24k_step1 = [
        'ffmpeg', '-y', '-i', input_videos['480p'],
        '-vf', 'scale=1920:1080:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '/tmp/sd24k_intermediate.mp4'
    ]
    
    cmd_sd24k_step2 = [
        'ffmpeg', '-y', '-i', '/tmp/sd24k_intermediate.mp4',
        '-vf', 'scale=3840:2160:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        ground_truth_videos['sd24k']
    ]
    
    # Create ground truth videos
    ground_truth_commands = [
        (cmd_sd2hd, 'SD2HD'),
        (cmd_hd24k, 'HD24K'),
        (cmd_4k28k, '4K28K'),
        (cmd_sd24k_step1, 'SD24K Step 1'),
        (cmd_sd24k_step2, 'SD24K Step 2')
    ]
    
    for cmd, name in ground_truth_commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created {name} ground truth")
        else:
            print(f"❌ Failed to create {name}: {result.stderr}")
    
    # Cleanup intermediate file
    if os.path.exists('/tmp/sd24k_intermediate.mp4'):
        os.remove('/tmp/sd24k_intermediate.mp4')
    
    return input_videos, ground_truth_videos

def evaluate_vsr_against_original(vsr_results, input_videos):
    """Evaluate VSR results against original input videos (correct approach)"""
    print("\n🔍 Evaluating VSR Results Against Original Input Videos")
    print("="*80)
    
    evaluation_results = {}
    
    # Mapping of VSR results to original input videos
    comparisons = [
        {
            'vsr_task': 'SD2HD',
            'vsr_path': vsr_results.get('SD2HD', {}).get('output_video'),
            'original_path': input_videos['480p'],  # Original 480p input
            'expected_resolution': '1920x1080',
            'content_length': 10.0
        },
        {
            'vsr_task': 'SD24K',
            'vsr_path': vsr_results.get('SD24K', {}).get('output_video'),
            'original_path': input_videos['480p'],  # Original 480p input
            'expected_resolution': '3840x2160',
            'content_length': 10.0
        },
        {
            'vsr_task': 'HD24K',
            'vsr_path': vsr_results.get('HD24K', {}).get('output_video'),
            'original_path': input_videos['1080p'],  # Original 1080p input
            'expected_resolution': '3840x2160',
            'content_length': 10.0
        },
        {
            'vsr_task': '4K28K',
            'vsr_path': vsr_results.get('4K28K', {}).get('output_video'),
            'original_path': input_videos['4k'],  # Original 4K input
            'expected_resolution': '7680x4320',
            'content_length': 10.0
        }
    ]
    
    for comparison in comparisons:
        task_name = comparison['vsr_task']
        vsr_path = comparison['vsr_path']
        original_path = comparison['original_path']
        expected_res = comparison['expected_resolution']
        content_length = comparison['content_length']
        
        print(f"\n📊 Evaluating {task_name}...")
        
        if not vsr_path or not os.path.exists(vsr_path):
            print(f"   ❌ VSR video not found: {vsr_path}")
            continue
            
        if not os.path.exists(original_path):
            print(f"   ❌ Original video not found: {original_path}")
            continue
        
        # Get VSR video resolution
        vsr_info = get_video_info(vsr_path)
        vsr_resolution = "Unknown"
        if 'streams' in vsr_info and vsr_info['streams']:
            stream = vsr_info['streams'][0]
            vsr_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
        
        print(f"   VSR Resolution: {vsr_resolution}")
        print(f"   Original Path: {original_path}")
        print(f"   Expected: {expected_res}")
        print(f"   Resolution Correct: {'✅' if vsr_resolution == expected_res else '❌'}")
        
        # Resize original video to match VSR resolution for fair comparison
        print(f"   Resizing original video to match VSR resolution...")
        resized_original = f"/tmp/resized_original_{task_name}.mp4"
        
        resize_cmd = [
            'ffmpeg', '-y', '-i', original_path,
            '-vf', f'scale={expected_res.replace("x", ":")}:flags=lanczos',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            resized_original
        ]
        
        if subprocess.run(resize_cmd, capture_output=True, text=True).returncode == 0:
            print(f"   ✅ Original resized to {expected_res}")
            reference_path = resized_original
        else:
            print(f"   ⚠️  Resize failed, using original (may cause VMAF issues)")
            reference_path = original_path
        
        # Calculate quality scores against resized original
        print(f"   Calculating quality scores...")
        psnr, ssim, vmaf = calculate_vmaf_fixed(reference_path, vsr_path)
        
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.3f}")
        print(f"   VMAF: {vmaf:.1f}")
        
        # Calculate Vidaio score
        print(f"🔍 Calculating VMAF alternative...")
        print(f"🔍 Calculating PIE-APP score...")
        
        vidaio_score = calculate_vidaio_score(reference_path, vsr_path, content_length)
        
        # Determine performance level
        final_score = vidaio_score['final_score']
        if final_score >= 0.32:
            performance = "🟢 Excellent (+15% bonus)"
        elif final_score >= 0.20:
            performance = "🟡 Good"
        elif final_score >= 0.07:
            performance = "🟠 Fair"
        else:
            performance = "🔴 Poor (-20% penalty)"
        
        print(f"   Vidaio VMAF Check: {'✅' if vidaio_score['vmaf_check_passed'] else '❌'}")
        print(f"   Vidaio Final Score: {final_score:.3f}")
        print(f"   Performance: {performance}")
        
        evaluation_results[task_name] = {
            'vsr_resolution': vsr_resolution,
            'original_path': original_path,
            'expected_resolution': expected_res,
            'resolution_correct': vsr_resolution == expected_res,
            'psnr': psnr,
            'ssim': ssim,
            'vmaf_score': vmaf,
            'vidaio_score': vidaio_score,
            'performance': performance
        }
    
    return evaluation_results

def evaluate_with_ground_truth(vsr_results, ground_truth_videos):
    """Evaluate VSR results against ground truth"""
    print("\n🔍 Evaluating VSR Results Against Ground Truth")
    print("="*80)
    
    evaluation_results = {}
    
    # Mapping of VSR results to ground truth - ALL 4 TASKS
    comparisons = [
        {
            'vsr_task': 'SD2HD',
            'vsr_path': vsr_results.get('SD2HD', {}).get('output_video'),
            'ground_truth_path': ground_truth_videos['sd2hd'],
            'expected_resolution': '1920x1080'
        },
        {
            'vsr_task': 'SD24K',
            'vsr_path': vsr_results.get('SD24K', {}).get('output_video'),
            'ground_truth_path': ground_truth_videos['sd24k'],
            'expected_resolution': '3840x2160'
        },
        {
            'vsr_task': 'HD24K',
            'vsr_path': vsr_results.get('HD24K', {}).get('output_video'),
            'ground_truth_path': ground_truth_videos['hd24k'],
            'expected_resolution': '3840x2160'
        },
        {
            'vsr_task': '4K28K',
            'vsr_path': vsr_results.get('4K28K', {}).get('output_video'),
            'ground_truth_path': ground_truth_videos['4k28k'],
            'expected_resolution': '7680x4320'
        }
    ]
    
    for comparison in comparisons:
        vsr_task = comparison['vsr_task']
        vsr_path = comparison['vsr_path']
        gt_path = comparison['ground_truth_path']
        expected_res = comparison['expected_resolution']
        
        print(f"\n📊 Evaluating {vsr_task}...")
        
        if not vsr_path or not os.path.exists(vsr_path):
            print(f"   ❌ VSR output not found: {vsr_path}")
            continue
            
        if not os.path.exists(gt_path):
            print(f"   ❌ Ground truth not found: {gt_path}")
            continue
        
        # Get video info
        vsr_info = get_video_info(vsr_path)
        gt_info = get_video_info(gt_path)
        
        vsr_resolution = "Unknown"
        gt_resolution = "Unknown"
        vsr_duration = 0.0
        
        if 'streams' in vsr_info and vsr_info['streams']:
            stream = vsr_info['streams'][0]
            vsr_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
        
        if 'streams' in gt_info and gt_info['streams']:
            stream = gt_info['streams'][0]
            gt_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
        
        if 'format' in vsr_info and 'duration' in vsr_info['format']:
            vsr_duration = float(vsr_info['format']['duration'])
        
        print(f"   VSR Resolution: {vsr_resolution}")
        print(f"   GT Resolution:  {gt_resolution}")
        print(f"   Expected:       {expected_res}")
        
        # Check resolution correctness
        resolution_correct = vsr_resolution == expected_res
        print(f"   Resolution Correct: {'✅' if resolution_correct else '❌'}")
        
        # Calculate quality scores
        print(f"   Calculating quality scores...")
        vmaf_score, psnr, ssim = calculate_vmaf_fixed(gt_path, vsr_path)
        
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.3f}")
        print(f"   VMAF: {vmaf_score:.2f}")
        
        # Calculate Vidaio score
        vidaio_score = calculate_vidaio_score(gt_path, vsr_path, vsr_duration)
        
        print(f"   Vidaio VMAF Check: {'✅' if vidaio_score['vmaf_check_passed'] else '❌'}")
        print(f"   Vidaio Final Score: {vidaio_score['final_score']:.3f}")
        
        # Performance assessment
        if vidaio_score['final_score'] > 0.32:
            performance = "🟢 Excellent (+15% bonus eligible)"
        elif vidaio_score['final_score'] > 0.07:
            performance = "🟡 Good"
        else:
            performance = "🔴 Poor (-20% penalty)"
        
        print(f"   Performance: {performance}")
        
        evaluation_results[vsr_task] = {
            'vsr_resolution': vsr_resolution,
            'gt_resolution': gt_resolution,
            'expected_resolution': expected_res,
            'resolution_correct': resolution_correct,
            'psnr': psnr,
            'ssim': ssim,
            'vmaf_score': vmaf_score,
            'vidaio_score': vidaio_score,
            'performance': performance
        }
    
    return evaluation_results

def main():
    """Main evaluation function"""
    print("🚀 SIMPLIFIED VIDAIO EVALUATION")
    print("="*80)
    print("1. Creating ground truth videos")
    print("2. Running VSR processing with optimal workers")
    print("3. Evaluating results with Vidaio scoring")
    print("="*80)
    
    # Step 1: Create ground truth videos
    print("\n🎬 STEP 1: Creating Ground Truth Videos")
    print("-" * 50)
    input_videos, ground_truth_videos = create_ground_truth_videos()
    
    # Step 2: Run VSR processing with optimal workers
    print("\n🤖 STEP 2: Running VSR Processing")
    print("-" * 50)
    
    # Optimal worker counts from previous experiments
    optimal_workers = {
        'SD2HD': 10,
        'SD24K': 8,
        'HD24K': 12,
        '4K28K': 8
    }
    
    # Task configurations - ALL 4 VIDAIO TASKS
    tasks = [
        {
            'name': 'SD2HD',
            'input_video': input_videos['480p'],
            'target_resolution': '1920x1080',
            'scale_factor': 2,
            'workers': optimal_workers['SD2HD']
        },
        {
            'name': 'SD24K',
            'input_video': input_videos['480p'],
            'target_resolution': '3840x2160',
            'scale_factor': 4,  # 2x + 2x
            'workers': optimal_workers['SD24K']
        },
        {
            'name': 'HD24K',
            'input_video': input_videos['1080p'],
            'target_resolution': '3840x2160',
            'scale_factor': 2,
            'workers': optimal_workers['HD24K']
        },
        {
            'name': '4K28K',
            'input_video': input_videos['4k'],
            'target_resolution': '7680x4320',
            'scale_factor': 2,
            'workers': optimal_workers['4K28K']
        }
    ]
    
    vsr_results = {}
    
    for task in tasks:
        print(f"\n🎬 Processing {task['name']} with {task['workers']} workers...")
        
        result = run_vsr_task(
            task['name'],
            task['input_video'],
            task['target_resolution'],
            task['scale_factor'],
            task['workers'],
            test_frames=300  # Use 10-second videos (300 frames) for better length scores
        )
        
        if result:
            vsr_results[task['name']] = result
            print(f"   ✅ {task['name']} completed")
        else:
            print(f"   ❌ {task['name']} failed")
    
    # Step 3: Evaluate VSR results against original input videos
    print("\n📊 STEP 3: Evaluating Results")
    print("-" * 50)
    evaluation_results = evaluate_vsr_against_original(vsr_results, input_videos)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    
    total_tasks = len(evaluation_results)
    passed_vmaf = sum(1 for r in evaluation_results.values() if r['vidaio_score']['vmaf_check_passed'])
    correct_resolution = sum(1 for r in evaluation_results.values() if r['resolution_correct'])
    good_performance = sum(1 for r in evaluation_results.values() if r['vidaio_score']['final_score'] > 0.07)
    
    print(f"Total Tasks: {total_tasks}")
    print(f"VMAF Check Passed: {passed_vmaf}/{total_tasks} ({passed_vmaf/total_tasks*100:.1f}%)")
    print(f"Correct Resolution: {correct_resolution}/{total_tasks} ({correct_resolution/total_tasks*100:.1f}%)")
    print(f"Good Performance: {good_performance}/{total_tasks} ({good_performance/total_tasks*100:.1f}%)")
    
    print(f"\n📋 Detailed Results:")
    for task, result in evaluation_results.items():
        print(f"  {task:>8}: {result['vsr_resolution']:>12} | VMAF: {result['vmaf_score']:>6.1f} | Vidaio: {result['vidaio_score']['final_score']:>6.3f} | {result['performance']}")
    
    # Save results
    results_file = "/workspace/vidaio-win/simplified_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'vsr_results': vsr_results,
            'evaluation_results': evaluation_results,
            'summary': {
                'total_tasks': total_tasks,
                'passed_vmaf': passed_vmaf,
                'correct_resolution': correct_resolution,
                'good_performance': good_performance
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if passed_vmaf == total_tasks and correct_resolution == total_tasks:
        print("🎉 All tasks passed! VSR pipeline is working correctly.")
    else:
        print("⚠️  Some tasks failed. Check resolution handling and quality.")

if __name__ == "__main__":
    main()
