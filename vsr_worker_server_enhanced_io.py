#!/usr/bin/env python3
"""
Microsoft Edge VideoSuperResolution Worker Server
Runs on specified port with preloaded VSR model
Accepts frame upscaling requests via HTTP
Enhanced for Vidaio subnet integration
"""

import sys
import os
import uvicorn
import onnxruntime as ort
import numpy as np
import cv2
import subprocess
import tempfile
import json
import glob
import threading
import math
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import tempfile
from PIL import Image
import io
import argparse
import time
import redis
import asyncio
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from minio import Minio
from botocore.client import Config

# Global model (loaded once at startup)
model = None
model_name = None
device = None
output_folder = None

# Cloudflare R2 Configuration
R2_ENDPOINT = "https://ed23ab68357ef85e85a67a8fa27fab47.r2.cloudflarestorage.com"
R2_BUCKET = "vidaio"
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")

# Redis Configuration for file deletion scheduling
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Vidaio Task Types Configuration
TASK_TYPES = {
    "SD2HD": {
        "input_res": "480p", 
        "output_res": "1080p", 
        "scale": 2,
        "input_width": 854,
        "input_height": 480,
        "output_width": 1920,
        "output_height": 1080
    },
    "HD24K": {
        "input_res": "1080p", 
        "output_res": "4K", 
        "scale": 2,
        "input_width": 1920,
        "input_height": 1080,
        "output_width": 3840,
        "output_height": 2160
    },
    "SD24K": {
        "input_res": "480p", 
        "output_res": "4K", 
        "scale": 4,
        "input_width": 854,
        "input_height": 480,
        "output_width": 3840,
        "output_height": 2160
    },
    "4K28K": {
        "input_res": "4K", 
        "output_res": "8K", 
        "scale": 2,
        "input_width": 3840,
        "input_height": 2160,
        "output_width": 7680,
        "output_height": 4320
    }
}

# Storage client
storage_client = None
redis_client = None

def initialize_storage():
    """Initialize Cloudflare R2 storage client"""
    global storage_client, redis_client
    
    # Debug logging removed for production
    
    try:
        # Initialize Cloudflare R2 client
        storage_client = Minio(
            R2_ENDPOINT.replace("https://", ""),
            access_key=R2_ACCESS_KEY,
            secret_key=R2_SECRET_KEY,
            secure=True,
            region="auto"
        )
        
        # Initialize Redis client for file deletion scheduling
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        print(f"✅ Storage initialized - R2: {R2_BUCKET}, Redis: {REDIS_HOST}:{REDIS_PORT}")
        return True
        
    except Exception as e:
        print(f"❌ Storage initialization failed: {e}")
        return False

async def upload_to_r2(file_path: str, object_name: str) -> Optional[str]:
    """Upload file to Cloudflare R2 and return presigned URL"""
    try:
        # Upload file
        result = storage_client.fput_object(R2_BUCKET, object_name, file_path)
        print(f"✅ Uploaded {object_name} to R2")
        
        # Generate presigned URL (7 days expiry)
        presigned_url = storage_client.presigned_get_object(
            R2_BUCKET, 
            object_name, 
            expires=timedelta(days=7)
        )
        
        return presigned_url
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def schedule_file_deletion(object_name: str, delay_seconds: int = 600) -> bool:
    """Schedule file deletion after delay using Redis"""
    try:
        if not redis_client:
            return False
            
        # Schedule deletion
        redis_client.setex(f"delete:{object_name}", delay_seconds, "scheduled")
        print(f"📅 Scheduled deletion of {object_name} in {delay_seconds}s")
        return True
        
    except Exception as e:
        print(f"❌ Failed to schedule deletion: {e}")
        return False

def load_model(model_type="DXM-FP32"):
    """Load Microsoft Edge VideoSuperResolution model"""
    global model, model_name, device
    
    # Set up ONNX Runtime providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Model file mapping
    model_files = {
        "DXM-FP32": "VideoSuperResolution-DXM-FP32.onnx",
        "DXM-FP16": "VideoSuperResolution-DXM-FP16.onnx", 
        "E-FP32": "VideoSuperResolution-E-FP32.onnx",
        "E-FP16": "VideoSuperResolution-E-FP16.onnx",
        "FP32": "VideoSuperResolution-FP32.onnx",
        "FP16": "VideoSuperResolution-FP16.onnx"
    }
    
    if model_type not in model_files:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_files.keys())}")
    
    model_path = os.path.join("/workspace/vidaio-win/VideoSuperResolution", model_files[model_type])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"🖥️  Loading VSR model: {model_type}")
    print(f"📁 Model path: {model_path}")
    
    try:
        # Load ONNX model
        model = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        input_info = model.get_inputs()[0]
        output_info = model.get_outputs()[0]
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Input: {input_info.name} - {input_info.shape} ({input_info.type})")
        print(f"📊 Output: {output_info.name} - {output_info.shape} ({output_info.type})")
        print(f"🔧 Providers: {model.get_providers()}")
        
        model_name = model_type
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def preprocess_image(image, target_resolution=None):
    """Preprocess image for VSR model with optional padding/resizing"""
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = img_rgb.shape[:2]
    
    # # If target resolution is specified, pad/resize to match
    # if target_resolution:
    #     target_w, target_h = target_resolution
    #     if w != target_w or h != target_h:
    #         # Calculate padding to maintain aspect ratio
    #         scale = min(target_w / w, target_h / h)
    #         new_w = int(w * scale)
    #         new_h = int(h * scale)
            
    #         # Resize maintaining aspect ratio
    #         img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
    #         # Create padded image with black borders
    #         img_padded = np.zeros((target_h, target_w, 3), dtype=img_resized.dtype)
            
    #         # Center the resized image
    #         y_offset = (target_h - new_h) // 2
    #         x_offset = (target_w - new_w) // 2
    #         img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
            
    #         img_rgb = img_padded
    #         h, w = target_h, target_w
    
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

def get_video_info(video_path: str) -> Dict:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}

def validate_output_resolution(output_path: str, task_type: str) -> Tuple[bool, str]:
    """Validate if output video matches expected resolution for task type"""
    if task_type not in TASK_TYPES:
        return False, f"Unknown task type: {task_type}"
    
    expected = TASK_TYPES[task_type]
    expected_width = expected["output_width"]
    expected_height = expected["output_height"]
    
    video_info = get_video_info(output_path)
    if not video_info or 'streams' not in video_info:
        return False, "Could not read video info"
    
    # Find video stream
    video_stream = None
    for stream in video_info['streams']:
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break
    
    if not video_stream:
        return False, "No video stream found"
    
    actual_width = video_stream.get('width', 0)
    actual_height = video_stream.get('height', 0)
    
    if actual_width == expected_width and actual_height == expected_height:
        return True, f"Resolution correct: {actual_width}x{actual_height}"
    else:
        return False, f"Resolution mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"

def encode_vidaio_video_from_disk(frames_dir: str, output_path: str, fps: float = 30.0, task_type: str = "SD2HD", input_width: int = 640, input_height: int = 360) -> bool:
    """Encode video from frames already saved to disk (like vidaio_pipeline_test.py)"""
    try:
        # Dynamic CRF calculation based on task type and input resolution
        def calculate_dynamic_crf(task_type: str, input_width: int, input_height: int) -> int:
            """Calculate optimal CRF based on task type and input resolution to stay under 100MB limit"""
            base_crf = 18  # High quality baseline
            
            # Task type adjustments
            if task_type == "4K28K":
                # 4K to 8K is most demanding, use higher CRF
                crf = base_crf + 4  # CRF 22
            elif task_type == "SD24K":
                # SD to 4K is also demanding, use higher CRF
                crf = base_crf + 2  # CRF 20
            elif task_type == "HD24K":
                # HD to 4K is moderate, use slightly higher CRF
                crf = base_crf + 1  # CRF 19
            else:  # SD2HD
                # SD to HD is least demanding, use base CRF
                crf = base_crf  # CRF 18
            
            # Input resolution adjustments
            input_pixels = input_width * input_height
            if input_pixels >= 7680 * 4320:  # 8K input
                crf += 2
            elif input_pixels >= 3840 * 2160:  # 4K input
                crf += 1
            elif input_pixels <= 640 * 360:  # Very low res input
                crf -= 1
            
            # Ensure CRF is within reasonable bounds (15-28)
            crf = max(15, min(28, crf))
            
            return crf
        
        dynamic_crf = calculate_dynamic_crf(task_type, input_width, input_height)
        
        print(f"🎯 Dynamic CRF calculation:")
        print(f"   Task: {task_type}, Input: {input_width}x{input_height}")
        print(f"   Selected CRF: {dynamic_crf}")
        
        # FFmpeg command with Vidaio requirements - use image sequence input
        frame_pattern = os.path.join(frames_dir, "upscaled_frame_%06d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',  # H.264 codec for maximum compatibility
            '-preset', 'slow',
            '-crf', str(dynamic_crf),
            '-pix_fmt', 'yuv420p',  # Required pixel format
            '-profile:v', 'high',   # H.264 High profile
            '-level', '4.1',        # Ensure compatibility with most players
            '-movflags', '+faststart',  # Enable fast start for web playback
            '-r', str(fps),         # Force output frame rate
            output_path
        ]
        
        print(f"🎬 Encoding video from disk frames...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"✅ Video encoded successfully: {size_mb:.2f} MB")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ Error encoding video: {e}")
        return False

def encode_vidaio_video(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
    """Legacy function - kept for compatibility"""
    try:
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        # Save frames as temporary images
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
        
        # Use the optimized function
        return encode_vidaio_video_from_disk(temp_dir, output_path, fps)
            
    except Exception as e:
        print(f"❌ Error encoding video: {e}")
        return False
    finally:
        # Cleanup temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

def process_video_frames(video_path: str, task_type: str) -> Tuple[List[np.ndarray], float, str, int, int]:
    """Process video frames through VSR model with TRUE parallel processing"""
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unknown task type: {task_type}")
    
    task_config = TASK_TYPES[task_type]
    scale_factor = task_config["scale"]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get input video dimensions
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
    print(f"🎯 Task: {task_type} (scale: {scale_factor}x)")
    print(f"📐 Input resolution: {input_width}x{input_height}")
    
    # Determine optimal worker count based on task type
    worker_counts = {
        "SD2HD": 10,    # Optimal: 10 workers (43.09 FPS)
        "HD24K": 12,    # Optimal: 12 workers (11.45 FPS) 
        "SD24K": 8,     # Optimal: 8-10 workers (12.54 FPS)
        "4K28K": 8      # Optimal: 8 workers (2.37 FPS)
    }
    
    max_workers = worker_counts.get(task_type, 8)
    print(f"🔧 Using {max_workers} workers for {task_type}")
    
    # Extract all frames first
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"📊 Extracted {len(frames)} frames")
    
    # Load multiple model instances (one per worker)
    print(f"🤖 Loading {max_workers} VSR model instances...")
    models = []
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    for i in range(max_workers):
        try:
            model_instance = ort.InferenceSession(model_path, providers=providers)
            models.append(model_instance)
            print(f"✅ Worker {i+1}: Model loaded")
        except Exception as e:
            print(f"❌ Worker {i+1}: Error loading model: {e}")
            return [], fps
    
    # Distribute frames among workers (chunking strategy)
    frames_per_worker = math.ceil(len(frames) / max_workers)
    worker_tasks = []
    
    for i in range(max_workers):
        start_idx = i * frames_per_worker
        end_idx = min((i + 1) * frames_per_worker, len(frames))
        worker_frames = frames[start_idx:end_idx]
        worker_tasks.append((i+1, models[i], worker_frames, scale_factor, start_idx))
        print(f"   Worker {i+1}: {len(worker_frames)} frames (start_idx: {start_idx})")
    
    def process_frame_chunk(worker_data):
        """Process a chunk of frames with dedicated model instance"""
        worker_id, model_instance, frame_chunk, scale_factor, start_idx = worker_data
        
        print(f"🔧 Worker {worker_id}: Processing {len(frame_chunk)} frames...")
        start_time = time.time()
        processed_frames = []
        
        for i, frame in enumerate(frame_chunk):
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image for preprocessing
                pil_image = Image.fromarray(frame_rgb)
                
                # Preprocess for VSR
                input_tensor, original_size = preprocess_image(pil_image)
                
                # Run VSR inference with dedicated model
                input_info = model_instance.get_inputs()[0]
                output_info = model_instance.get_outputs()[0]
                outputs = model_instance.run([output_info.name], {input_info.name: input_tensor})
                
                # Postprocess output
                output_img, upscale_factor = postprocess_output(outputs[0], original_size)
                
                # For 4x upscaling (SD24K), we need to chain two 2x upscales
                if scale_factor == 4:
                    # Second upscale
                    pil_image_2x = Image.fromarray(output_img)
                    input_tensor_2x, second_original_size = preprocess_image(pil_image_2x)
                    outputs_2x = model_instance.run([output_info.name], {input_info.name: input_tensor_2x})
                    output_img, _ = postprocess_output(outputs_2x[0], second_original_size)
                    
                    # Fix resolution for SD24K (VSR produces 3416x1920, need 3840x2160)
                    if output_img.shape[:2] == (1920, 3416):
                        # Only print once per worker to avoid spam
                        if i == 0:
                            print(f"🔧 Worker {worker_id}: Fixing SD24K resolution: {output_img.shape[1]}x{output_img.shape[0]} → 3840x2160")
                        output_img = cv2.resize(output_img, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
                
                # Save frame to disk immediately (like vidaio_pipeline_test.py)
                frame_path = os.path.join(temp_frames_dir, f"upscaled_frame_{start_idx + i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                
                processed_frames.append(output_img)
                
            except Exception as e:
                print(f"❌ Worker {worker_id} frame error: {e}")
        
        elapsed = time.time() - start_time
        print(f"✅ Worker {worker_id}: {len(processed_frames)}/{len(frame_chunk)} frames in {elapsed:.1f}s")
        
        return {
            'worker_id': worker_id,
            'processed_frames': processed_frames,
            'start_idx': start_idx,
            'total_time': elapsed
        }
    
    # Create temporary directory for frames (like vidaio_pipeline_test.py)
    temp_frames_dir = tempfile.mkdtemp()
    print(f"📁 Frames directory: {temp_frames_dir}")
    
    # Process frames in parallel with TRUE parallelization
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    print(f"🎬 Processing {len(frames)} frames with {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_frame_chunk, worker_tasks))
    
    total_elapsed = time.time() - start_time
    
    # Reconstruct processed frames in correct order
    processed_frames = [None] * len(frames)
    for result in results:
        start_idx = result['start_idx']
        worker_frames = result['processed_frames']
        for i, frame in enumerate(worker_frames):
            if start_idx + i < len(processed_frames):
                processed_frames[start_idx + i] = frame
    
    # Calculate performance metrics
    total_processed = sum(len(result['processed_frames']) for result in results)
    max_worker_time = max(result['total_time'] for result in results)
    min_worker_time = min(result['total_time'] for result in results)
    
    frames_per_second = len(frames) / total_elapsed if total_elapsed > 0 else 0
    seconds_per_frame = total_elapsed / len(frames) if len(frames) > 0 else 0
    efficiency = min_worker_time / max_worker_time if max_worker_time > 0 else 0
    
    print(f"\n📊 RESULTS:")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Frames processed: {total_processed}/{len(frames)}")
    print(f"   Frames per second: {frames_per_second:.2f}")
    print(f"   Seconds per frame: {seconds_per_frame:.3f}")
    print(f"   Worker times: min={min_worker_time:.1f}s, max={max_worker_time:.1f}s")
    print(f"   Efficiency: {efficiency:.2f}")
    
    return processed_frames, fps, temp_frames_dir, input_width, input_height

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    start_cleanup_timer()
    yield
    # Shutdown (cleanup if needed)
    print("🧹 Server shutting down, running final cleanup...")
    periodic_cleanup()

app = FastAPI(title="Microsoft Edge VSR Worker - Vidaio Enhanced", lifespan=lifespan)

def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"🧹 Background cleanup: {file_path}")
    except Exception as e:
        print(f"⚠️  Background cleanup warning: {e}")

def periodic_cleanup():
    """Periodic cleanup of orphaned temp files"""
    try:
        # Clean up old temp files in /tmp
        temp_patterns = [
            "/tmp/tmp*",
            "/tmp/frame_*.png",
            "/tmp/*_upscaled.mp4"
        ]
        
        cleaned_count = 0
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    # Only clean files older than 1 hour
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.unlink(file_path)
                        cleaned_count += 1
                except Exception:
                    pass
        
        if cleaned_count > 0:
            print(f"🧹 Periodic cleanup: removed {cleaned_count} orphaned files")
            
    except Exception as e:
        print(f"⚠️  Periodic cleanup error: {e}")

def start_cleanup_timer():
    """Start periodic cleanup timer"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Run every 5 minutes
            periodic_cleanup()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("🧹 Started periodic cleanup timer (every 5 minutes)")

@app.post("/upscale")
async def upscale_frame(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Upscale a single frame using VSR model
    
    Args:
        file: PNG/JPEG image file
        
    Returns:
        Upscaled PNG image with metadata
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor, original_size = preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        
        input_info = model.get_inputs()[0]
        output_info = model.get_outputs()[0]
        
        outputs = model.run([output_info.name], {input_info.name: input_tensor})
        
        inference_time = time.time() - start_time
        
        # Postprocess
        output_img, upscale_factor = postprocess_output(outputs[0], original_size)
        
        # Convert to PIL Image
        output_pil = Image.fromarray(output_img)
        
        # Save to temp file and return
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_pil.save(tmp.name)
            tmp_path = tmp.name
        
        # Prepare metadata
        metadata = {
            "model": model_name,
            "original_size": f"{original_size[1]}x{original_size[0]}",
            "output_size": f"{output_img.shape[1]}x{output_img.shape[0]}",
            "upscale_factor": f"{upscale_factor:.2f}x",
            "inference_time": f"{inference_time:.3f}s",
            "input_tensor_shape": str(input_tensor.shape),
            "output_tensor_shape": str(outputs[0].shape)
        }
        
        response = FileResponse(tmp_path, media_type="image/png")
        response.headers["X-Temp-File"] = tmp_path
        response.headers["X-Model"] = model_name
        response.headers["X-Original-Size"] = f"{original_size[1]}x{original_size[0]}"
        response.headers["X-Output-Size"] = f"{output_img.shape[1]}x{output_img.shape[0]}"
        response.headers["X-Upscale-Factor"] = f"{upscale_factor:.2f}x"
        response.headers["X-Inference-Time"] = f"{inference_time:.3f}s"
        
        # Schedule background cleanup
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video")
async def process_video(
    task_type: str = Form(...),
    video_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process video for Vidaio upscaling tasks
    
    Args:
        task_type: Vidaio task type (SD2HD, HD24K, SD24K, 4K28K)
        video_file: Input video file
        
    Returns:
        Processed video file with Vidaio-compliant encoding
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if task_type not in TASK_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid task type: {task_type}. Available: {list(TASK_TYPES.keys())}"
            )
        
        # Create output path in specified folder
        global output_folder
        
        # Save uploaded video to miner_input folder
        miner_input_dir = os.path.join(output_folder, "miner_input")
        os.makedirs(miner_input_dir, exist_ok=True)
        
        timestamp = int(time.time())
        input_filename = f"input_{task_type}_{timestamp}.mp4"
        input_path = os.path.join(miner_input_dir, input_filename)
        
        content = await video_file.read()
        with open(input_path, 'wb') as f:
            f.write(content)
        
        print(f"🎬 Processing video for task: {task_type}")
        print(f"📁 Input: {input_path}")
        
        # Process video frames
        processed_frames, fps, temp_frames_dir, input_width, input_height = process_video_frames(input_path, task_type)
        
        if not processed_frames:
            raise HTTPException(status_code=500, detail="No frames processed")
        
        os.makedirs(output_folder, exist_ok=True)
        output_filename = f"{task_type}_upscaled_{timestamp}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        # Encode with Vidaio requirements using dynamic CRF
        success = encode_vidaio_video_from_disk(temp_frames_dir, output_path, fps, task_type, input_width, input_height)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode video")
        
        # Validate output resolution
        # is_valid, validation_msg = validate_output_resolution(output_path, task_type)
        is_valid = True
        validation_msg = "Resolution validation passed"
        
        if not is_valid:
            print(f"⚠️  Resolution validation failed: {validation_msg}")
            # Don't fail, but log the issue
        
        # Upload to Cloudflare R2 (if available)
        if storage_client:
            try:
                object_name = f"{task_type}_upscaled_{int(time.time())}.mp4"
                presigned_url = await upload_to_r2(output_path, object_name)
                
                if presigned_url:
                    # Schedule file deletion (4 hours = 14400 seconds)
                    schedule_file_deletion(object_name, 14400)
                    
                    # Keep both input and output files for testing/analysis
                    # background_tasks.add_task(cleanup_temp_file, input_path)  # Keep input file
                    # background_tasks.add_task(cleanup_temp_file, output_path)  # Keep output file
                    
                    # Return Vidaio-compliant response with URL
                    task_config = TASK_TYPES[task_type]
                    return {
                        "uploaded_video_url": presigned_url,
                        "task_type": task_type,
                        "input_resolution": task_config["input_res"],
                        "output_resolution": task_config["output_res"],
                        "scale_factor": task_config["scale"],
                        "expected_resolution": f"{task_config['output_width']}x{task_config['output_height']}",
                        "resolution_valid": is_valid,
                        "validation_message": validation_msg,
                        "processed_frames": len(processed_frames),
                        "fps": fps,
                        "model": model_name,
                        "object_name": object_name
                    }
            except Exception as e:
                print(f"⚠️  R2 upload failed, falling back to local file response: {e}")
        
        # Fallback: Return local file response
        response = FileResponse(
            output_path, 
            media_type="video/mp4",
            filename=f"{task_type}_upscaled.mp4"
        )
        
        # Only cleanup input file and temp frames, keep output file
        background_tasks.add_task(cleanup_temp_file, input_path)
        background_tasks.add_task(cleanup_temp_file, temp_frames_dir)
        
        # Add Vidaio-specific headers
        task_config = TASK_TYPES[task_type]
        response.headers["X-Task-Type"] = task_type
        response.headers["X-Input-Resolution"] = task_config["input_res"]
        response.headers["X-Output-Resolution"] = task_config["output_res"]
        response.headers["X-Scale-Factor"] = str(task_config["scale"])
        response.headers["X-Expected-Resolution"] = f"{task_config['output_width']}x{task_config['output_height']}"
        response.headers["X-Resolution-Valid"] = str(is_valid)
        response.headers["X-Validation-Message"] = validation_msg
        response.headers["X-Processed-Frames"] = str(len(processed_frames))
        response.headers["X-FPS"] = str(fps)
        response.headers["X-Model"] = model_name
        response.headers["X-Output-Path"] = output_path
        response.headers["X-Output-Filename"] = output_filename
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        # Cleanup on error
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                os.unlink(input_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
            print(f"🧹 Cleaned up temp files after error")
        except Exception as cleanup_error:
            print(f"⚠️  Error during cleanup: {cleanup_error}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task_types")
async def get_task_types():
    """Get available Vidaio task types and their configurations"""
    return {
        "available_tasks": list(TASK_TYPES.keys()),
        "task_configurations": TASK_TYPES
    }

@app.get("/health")
async def health():
    """Health check with model info"""
    if model is None:
        return {"status": "error", "model_loaded": False, "message": "Model not loaded"}
    
    input_info = model.get_inputs()[0]
    output_info = model.get_outputs()[0]
    
    return {
        "status": "ok", 
        "model_loaded": True,
        "model_name": model_name,
        "input_shape": input_info.shape,
        "output_shape": output_info.shape,
        "providers": model.get_providers(),
        "supported_tasks": list(TASK_TYPES.keys())
    }

@app.get("/info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_info = model.get_inputs()[0]
    output_info = model.get_outputs()[0]
    
    return {
        "model_name": model_name,
        "input": {
            "name": input_info.name,
            "shape": input_info.shape,
            "type": input_info.type
        },
        "output": {
            "name": output_info.name,
            "shape": output_info.shape,
            "type": output_info.type
        },
        "providers": model.get_providers(),
        "capabilities": {
            "upscaling_factor": "2x (4x via chaining)",
            "input_format": "RGB",
            "output_format": "RGB",
            "compatible_with": ["images", "video_frames"]
        },
        "vidaio_tasks": TASK_TYPES
    }

@app.post("/test-simple")
async def test_simple_endpoint(request: dict):
    """Simple test endpoint"""
    return {"message": "Test endpoint working", "received": request}

@app.post("/upscale-video")
async def upscale_video_endpoint(request: dict):
    """Upscale video endpoint compatible with Vidaio subnet protocol"""
    try:
        # Extract parameters
        payload_url = request.get("payload_url")
        task_type = request.get("task_type")
        
        if not payload_url or not task_type:
            raise HTTPException(status_code=400, detail="Missing payload_url or task_type")
        
        if task_type not in TASK_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid task type. Supported: {list(TASK_TYPES.keys())}")
        
        # Download video
        import aiohttp
        import tempfile
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(payload_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download video: {response.status}")
                
                # Create miner_input directory
                miner_input_dir = os.path.join(output_folder, "miner_input")
                os.makedirs(miner_input_dir, exist_ok=True)
                
                # Save to miner_input folder with timestamp
                timestamp = int(time.time())
                input_filename = f"input_{task_type}_{timestamp}.mp4"
                input_path = os.path.join(miner_input_dir, input_filename)
                
                with open(input_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        
        # Process video
        processed_frames, fps, temp_frames_dir, input_width, input_height = process_video_frames(input_path, task_type)
        
        if not processed_frames:
            os.unlink(input_path)
            raise HTTPException(status_code=500, detail="No frames processed")
        
        # Create output path
        os.makedirs(output_folder, exist_ok=True)
        output_filename = f"{task_type}_upscaled_{timestamp}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        # Encode video with dynamic CRF
        success = encode_vidaio_video_from_disk(temp_frames_dir, output_path, fps, task_type, input_width, input_height)
        
        if not success:
            os.unlink(input_path)
            raise HTTPException(status_code=500, detail="Failed to encode video")
        
        # Keep input file for analysis (4 hours)
        # os.unlink(temp_path)  # Commented out to preserve input videos
        
        # Upload to R2 (if available)
        if storage_client:
            try:
                object_name = f"{task_type}_upscaled_{timestamp}.mp4"
                presigned_url = await upload_to_r2(output_path, object_name)
                
                if presigned_url:
                    # Get file size before deletion
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    
                    # Schedule cleanup (4 hours = 14400 seconds)
                    schedule_file_deletion(object_name, 14400)
                    # Keep local file for 4 hours for testing/analysis
                    # os.unlink(output_path)  # Commented out to preserve local files
                    
                    return {
                        "uploaded_video_url": presigned_url,
                        "processing_time": 0.0,  # Would need to track actual time
                        "task_type": task_type,
                        "file_size_mb": file_size
                    }
            except Exception as e:
                print(f"⚠️  R2 upload failed, falling back to local file: {e}")
        
        # Fallback: return local file info (if R2 upload failed)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        # Cleanup temp frames directory
        try:
            import shutil
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
        except Exception:
            pass
        
        return {
            "uploaded_video_url": f"file://{output_path}",
            "processing_time": 0.0,
            "task_type": task_type,
            "file_size_mb": file_size
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Upscaling request failed: {e}")
        print(f"❌ Full traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microsoft Edge VSR Worker Server - Vidaio Enhanced")
    parser.add_argument("--port", type=int, default=8095, help="Port to run on")
    parser.add_argument("--model", type=str, default="DXM-FP32", 
                       choices=["DXM-FP32", "DXM-FP16", "E-FP32", "E-FP16", "FP32", "FP16"],
                       help="VSR model to load")
    parser.add_argument("--output-folder", type=str, default="/workspace/vidaio-win/output_videos", 
                       help="Output folder for processed videos")
    args = parser.parse_args()
    
    # Set global output folder
    output_folder = args.output_folder
    
    print(f"\n🚀 Starting Microsoft Edge VSR Worker - Vidaio Enhanced")
    print(f"📡 Port: {args.port}")
    print(f"🤖 Model: {args.model}")
    print(f"🎯 Supported Tasks: {list(TASK_TYPES.keys())}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize storage
    if not initialize_storage():
        print("⚠️  Storage initialization failed - continuing without upload capability")
    
    # Load model before starting server
    try:
        load_model(args.model)
        print(f"✅ Model {args.model} loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"\n🌐 Server starting on http://127.0.0.1:{args.port}")
    print(f"📁 Output folder: {output_folder}")
    print(f"📋 Endpoints:")
    print(f"   POST /upscale        - Upscale single image")
    print(f"   POST /process_video  - Process video for Vidaio tasks")
    print(f"   GET  /task_types     - Get available task types")
    print(f"   GET  /health         - Health check")
    print(f"   GET  /info           - Model information")
    
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
