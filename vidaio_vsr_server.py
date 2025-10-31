#!/usr/bin/env python3
"""
Vidaio VSR Server - Optimized for Subnet Integration
High-performance video super-resolution server with persistent GPU workers
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import subprocess
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import traceback
import threading
from queue import Queue, Empty
import tempfile
import shutil

# Import subnet utilities
from vidaio_subnet_core.utilities import storage_client, download_video
from services.miner_utilities.redis_utils import schedule_file_deletion

# Configuration
MAX_WORKERS = 12
MODEL_PATH = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
TEMP_DIR = "/tmp/vidaio_vsr"
OUTPUT_DIR = "/tmp/vidaio_vsr_output"

# Task type to optimal worker count mapping
TASK_WORKER_MAPPING = {
    "SD2HD": 10,
    "SD24K": 8, 
    "HD24K": 12,
    "4K28K": 8
}

# Task type to resolution mapping
TASK_RESOLUTION_MAPPING = {
    "SD2HD": ("854x480", "1920x1080", 2),
    "SD24K": ("854x480", "3840x2160", 4),  # 2x + 2x
    "HD24K": ("1920x1080", "3840x2160", 2),
    "4K28K": ("3840x2160", "7680x4320", 2)
}

class VSRWorker:
    """Individual VSR worker with persistent model loading"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.model = None
        self.is_ready = False
        self.load_model()
    
    def load_model(self):
        """Load VSR model for this worker"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model = ort.InferenceSession(MODEL_PATH, providers=providers)
            self.is_ready = True
            logger.info(f"✅ Worker {self.worker_id}: Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Worker {self.worker_id}: Failed to load model: {e}")
            self.is_ready = False
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for VSR model"""
        img_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img_cv.shape[:2]
        img_normalized = img_cv.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
        return img_tensor, (h, w)
    
    def postprocess_output(self, output_tensor: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output back to image"""
        output_img = np.transpose(output_tensor[0], (1, 2, 0))
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        return output_img
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through VSR model"""
        if not self.is_ready:
            raise Exception(f"Worker {self.worker_id} not ready")
        
        input_tensor, original_size = self.preprocess_image(frame)
        
        input_info = self.model.get_inputs()[0]
        output_info = self.model.get_outputs()[0]
        
        outputs = self.model.run([output_info.name], {input_info.name: input_tensor})
        output_img = self.postprocess_output(outputs[0], original_size)
        
        return output_img

class VSRWorkerPool:
    """Pool of persistent VSR workers"""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.workers: List[VSRWorker] = []
        self.worker_queue = Queue()
        self.initialize_workers()
    
    def initialize_workers(self):
        """Initialize all workers"""
        logger.info(f"🤖 Initializing {self.max_workers} VSR workers...")
        for i in range(self.max_workers):
            worker = VSRWorker(i + 1)
            if worker.is_ready:
                self.workers.append(worker)
                self.worker_queue.put(worker)
        
        logger.info(f"✅ Initialized {len(self.workers)} workers successfully")
    
    def get_worker(self) -> Optional[VSRWorker]:
        """Get an available worker from the pool"""
        try:
            return self.worker_queue.get(timeout=1.0)
        except Empty:
            return None
    
    def return_worker(self, worker: VSRWorker):
        """Return a worker to the pool"""
        self.worker_queue.put(worker)

class VSRProcessor:
    """Main VSR processing class"""
    
    def __init__(self):
        self.worker_pool = VSRWorkerPool()
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def resize_video(self, input_path: str, output_path: str, target_resolution: str) -> bool:
        """Resize video to target resolution using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f'scale={target_resolution}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Video resize failed: {e}")
            return False
    
    def extract_frames(self, video_path: str, max_frames: int = 150) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
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
                frames.append(frame)
                extracted += 1
                
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_frames_parallel(self, frames: List[np.ndarray], num_workers: int) -> List[np.ndarray]:
        """Process frames in parallel using worker pool"""
        upscaled_frames = []
        
        def process_frame_batch(frame_batch: List[np.ndarray]) -> List[np.ndarray]:
            worker = self.worker_pool.get_worker()
            if not worker:
                raise Exception("No available workers")
            
            try:
                batch_results = []
                for frame in frame_batch:
                    upscaled_frame = worker.process_frame(frame)
                    batch_results.append(upscaled_frame)
                return batch_results
            finally:
                self.worker_pool.return_worker(worker)
        
        # Distribute frames among workers
        frames_per_worker = math.ceil(len(frames) / num_workers)
        worker_tasks = []
        
        for i in range(num_workers):
            start_idx = i * frames_per_worker
            end_idx = min((i + 1) * frames_per_worker, len(frames))
            if start_idx < len(frames):
                worker_tasks.append(frames[start_idx:end_idx])
        
        # Process frames in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_frame_batch, worker_tasks))
        
        # Flatten results
        for batch_result in results:
            upscaled_frames.extend(batch_result)
        
        return upscaled_frames
    
    def create_video_from_frames(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
        """Create video from upscaled frames"""
        try:
            if not frames:
                return False
            
            # Save frames temporarily
            temp_frames_dir = os.path.join(TEMP_DIR, f"temp_frames_{uuid.uuid4()}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Create video using ffmpeg
            frame_pattern = os.path.join(temp_frames_dir, "frame_%06d.png")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Cleanup temp frames
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return False
    
    async def process_video(self, video_url: str, task_type: str) -> Optional[str]:
        """Main video processing pipeline"""
        try:
            logger.info(f"🎬 Processing {task_type} video from {video_url}")
            
            # Get optimal worker count for task type
            num_workers = TASK_WORKER_MAPPING.get(task_type, 8)
            input_res, output_res, scale_factor = TASK_RESOLUTION_MAPPING[task_type]
            
            # Download video
            logger.info("📥 Downloading video...")
            input_video_path = await download_video(video_url)
            
            # Resize video to input resolution if needed
            resized_video_path = os.path.join(TEMP_DIR, f"resized_{uuid.uuid4()}.mp4")
            if not self.resize_video(input_video_path, resized_video_path, input_res):
                raise Exception("Failed to resize video")
            
            # Extract frames
            logger.info("🎞️ Extracting frames...")
            frames = self.extract_frames(resized_video_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Process frames for SD24K (2x + 2x)
            if task_type == "SD24K":
                logger.info("🔄 First upscale: 480p → 1080p")
                intermediate_frames = self.process_frames_parallel(frames, num_workers)
                
                # Create intermediate video
                intermediate_video = os.path.join(TEMP_DIR, f"intermediate_{uuid.uuid4()}.mp4")
                if not self.create_video_from_frames(intermediate_frames, intermediate_video):
                    raise Exception("Failed to create intermediate video")
                
                # Resize to 1080p
                resized_intermediate = os.path.join(TEMP_DIR, f"resized_intermediate_{uuid.uuid4()}.mp4")
                if not self.resize_video(intermediate_video, resized_intermediate, "1920x1080"):
                    raise Exception("Failed to resize intermediate video")
                
                # Extract frames from intermediate video
                intermediate_frames = self.extract_frames(resized_intermediate)
                logger.info(f"🔄 Second upscale: 1080p → 4K with {len(intermediate_frames)} frames")
                
                # Second upscale
                upscaled_frames = self.process_frames_parallel(intermediate_frames, num_workers)
            else:
                # Single upscale
                logger.info(f"🔄 Upscaling: {input_res} → {output_res}")
                upscaled_frames = self.process_frames_parallel(frames, num_workers)
            
            # Create output video
            logger.info("🎬 Creating output video...")
            output_video_path = os.path.join(OUTPUT_DIR, f"upscaled_{task_type}_{uuid.uuid4()}.mp4")
            if not self.create_video_from_frames(upscaled_frames, output_video_path):
                raise Exception("Failed to create output video")
            
            # Upload to storage
            logger.info("☁️ Uploading to storage...")
            object_name = os.path.basename(output_video_path)
            await storage_client.upload_file(object_name, output_video_path)
            
            # Get presigned URL
            sharing_link = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                raise Exception("Failed to get presigned URL")
            
            # Schedule file deletion
            schedule_file_deletion(object_name)
            
            # Cleanup local files
            for file_path in [input_video_path, resized_video_path, output_video_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info(f"✅ Processing complete: {sharing_link}")
            return sharing_link
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
            traceback.print_exc()
            return None

# FastAPI Application
app = FastAPI(title="Vidaio VSR Server", version="1.0.0")

# Global processor instance
vsr_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize VSR processor on startup"""
    global vsr_processor
    logger.info("🚀 Starting Vidaio VSR Server...")
    vsr_processor = VSRProcessor()
    logger.info("✅ VSR Server ready!")

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    maximum_optimized_size_mb: int = 100

class UpscaleResponse(BaseModel):
    optimized_video_url: str
    processing_time: float
    task_type: str
    file_size_mb: float

@app.post("/upscale-video", response_model=UpscaleResponse)
async def upscale_video(request: UpscaleRequest):
    """
    Upscale video endpoint compatible with Vidaio subnet protocol
    """
    start_time = time.time()
    
    try:
        # Validate task type
        if request.task_type not in TASK_WORKER_MAPPING:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid task type. Supported: {list(TASK_WORKER_MAPPING.keys())}"
            )
        
        # Process video
        result_url = await vsr_processor.process_video(
            request.payload_url, 
            request.task_type
        )
        
        if not result_url:
            raise HTTPException(status_code=500, detail="Video processing failed")
        
        processing_time = time.time() - start_time
        
        # Get file size (approximate)
        file_size_mb = 0.0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(result_url) as response:
                    if 'content-length' in response.headers:
                        file_size_mb = int(response.headers['content-length']) / (1024 * 1024)
        except:
            pass
        
        return UpscaleResponse(
            optimized_video_url=result_url,
            processing_time=processing_time,
            task_type=request.task_type,
            file_size_mb=file_size_mb
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upscaling request failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "workers_ready": len(vsr_processor.worker_pool.workers),
        "max_workers": vsr_processor.worker_pool.max_workers,
        "supported_tasks": list(TASK_WORKER_MAPPING.keys())
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "workers_ready": len(vsr_processor.worker_pool.workers),
        "max_workers": vsr_processor.worker_pool.max_workers,
        "task_worker_mapping": TASK_WORKER_MAPPING,
        "task_resolution_mapping": TASK_RESOLUTION_MAPPING
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Vidaio VSR Server...")
    logger.info(f"📊 Max Workers: {MAX_WORKERS}")
    logger.info(f"🎯 Supported Tasks: {list(TASK_WORKER_MAPPING.keys())}")
    logger.info(f"🔧 Model Path: {MODEL_PATH}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=29115,
        log_level="info",
        access_log=True
    )





