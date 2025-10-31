#!/usr/bin/env python3
"""
Real-ESRGAN Batch Processing Worker Server
Processes batches of frames using shared storage and manifest files
"""

import sys
import os
import json
import time
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')

# Global model (loaded once at startup)
model = None
device = None
current_manifest = None
processing_status = {
    'status': 'idle',
    'total_batches': 0,
    'completed_batches': 0,
    'current_batch': None,
    'progress_percent': 0.0,
    'start_time': None,
    'end_time': None,
    'manifest_path': None
}

def load_model(scale=2):
    """Load Real-ESRGAN model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    from RealESRGAN import RealESRGAN
    
    print("🤖 Loading Real-ESRGAN model...")
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    print("✅ Model loaded")

def parse_manifest(manifest_path):
    """Parse manifest file and return batch assignments"""
    metadata = {}
    batches = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse metadata from comments
            if line.startswith('#'):
                if ':' in line:
                    key_value = line[1:].strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip().lower().replace(' ', '_')
                        value = key_value[1].strip()
                        metadata[key] = value
                continue
            
            # Parse batch assignment
            parts = line.split(',')
            if len(parts) >= 2:
                batch_id = parts[0]
                frames = parts[1:]
                batches.append({
                    'batch_id': batch_id,
                    'frames': frames,
                    'frame_count': len(frames)
                })
    
    return {
        'metadata': metadata,
        'batches': batches,
        'total_batches': len(batches)
    }

def process_batch(batch_data, input_dir, output_dir, scale=2, batch_size=8, patch_size=256):
    """Process a single batch of frames"""
    batch_id = batch_data['batch_id']
    frames = batch_data['frames']
    
    print(f"  🔧 Processing {batch_id}: {len(frames)} frames")
    
    # Load all frames in the batch
    batch_images = []
    for frame_name in frames:
        frame_path = os.path.join(input_dir, frame_name)
        if os.path.exists(frame_path):
            image = Image.open(frame_path).convert('RGB')
            batch_images.append(image)
        else:
            print(f"    ⚠️  Frame not found: {frame_path}")
            return False
    
    # Process batch with Real-ESRGAN
    try:
        print(f"    📦 Processing {len(batch_images)} images as batch...")
        
        # Process each image efficiently using the pre-loaded model
        for i, (image, frame_name) in enumerate(zip(batch_images, frames)):
            # Create output path
            output_path = os.path.join(output_dir, frame_name)
            
            # Upscale with Real-ESRGAN using optimized parameters
            # batch_size=1 for single image, patches_size for tiling optimization
            sr_image = model.predict(image, batch_size=1, patches_size=patch_size)
            sr_image.save(output_path)
            
        return True
        
    except Exception as e:
        print(f"    ❌ Error processing batch {batch_id}: {e}")
        return False

def process_manifest_async(manifest_path, worker_id):
    """Process all batches in the manifest file"""
    global processing_status, model, device
    
    try:
        print(f"🔍 Worker {worker_id}: Starting manifest processing...")
        print(f"   Manifest path: {manifest_path}")
        
        # Model should already be loaded at startup
        if model is None:
            print(f"   ❌ Model not loaded for worker {worker_id}!")
            raise Exception("Model not loaded")
        
        # Parse manifest
        manifest_data = parse_manifest(manifest_path)
        batches = manifest_data['batches']
        metadata = manifest_data['metadata']
        
        print(f"   Parsed {len(batches)} batches")
        print(f"   Metadata: {metadata}")
        
        # Get directories from metadata
        input_dir = metadata.get('input_dir', '/tmp/input_frames')
        output_dir = metadata.get('output_dir', '/tmp/output_frames')
        scale_str = metadata.get('scale', '2x')
        scale = int(scale_str.replace('x', '')) if 'x' in scale_str else int(scale_str)
        batch_size = int(metadata.get('batch_size', 8))
        patch_size = int(metadata.get('patch_size', 256))
        
        print(f"   Input dir: {input_dir}")
        print(f"   Output dir: {output_dir}")
        print(f"   Scale: {scale}, Batch size: {batch_size}, Patch size: {patch_size}")
        
        # Update status
        processing_status.update({
            'status': 'processing',
            'total_batches': len(batches),
            'completed_batches': 0,
            'current_batch': None,
            'progress_percent': 0.0,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'manifest_path': manifest_path
        })
        
        print(f"🚀 Worker {worker_id} starting batch processing...")
        print(f"   Manifest: {manifest_path}")
        print(f"   Input Dir: {input_dir}")
        print(f"   Output Dir: {output_dir}")
        print(f"   Total Batches: {len(batches)}")
        print(f"   Scale: {scale}x, Batch Size: {batch_size}, Patch Size: {patch_size}")
        
        # Process each batch
        for i, batch_data in enumerate(batches):
            batch_id = batch_data['batch_id']
            
            # Update current batch
            processing_status['current_batch'] = batch_id
            
            # Process the batch
            success = process_batch(batch_data, input_dir, output_dir, scale, batch_size, patch_size)
            
            if success:
                processing_status['completed_batches'] += 1
                processing_status['progress_percent'] = (processing_status['completed_batches'] / processing_status['total_batches']) * 100
                print(f"  ✅ {batch_id} completed ({processing_status['completed_batches']}/{processing_status['total_batches']})")
            else:
                print(f"  ❌ {batch_id} failed")
        
        # Mark as completed
        processing_status.update({
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'progress_percent': 100.0
        })
        
        print(f"🎉 Worker {worker_id} completed all batches!")
        
    except Exception as e:
        processing_status.update({
            'status': 'error',
            'end_time': datetime.now().isoformat()
        })
        print(f"❌ Worker {worker_id} error: {e}")

app = FastAPI(title="Real-ESRGAN Batch Processing Worker")

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    global model, device
    print(f"🚀 Worker starting up - loading model...")
    load_model(scale=2)
    print(f"✅ Worker ready with model loaded")

@app.post("/process_manifest")
async def process_manifest_endpoint(request: dict):
    """
    Start processing batches from manifest file
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Status response
    """
    global current_manifest, processing_status
    
    try:
        # Extract manifest path from request
        manifest_path = request.get('manifest_path')
        if not manifest_path:
            raise HTTPException(status_code=400, detail="manifest_path is required")
        
        # Validate manifest file exists
        if not os.path.exists(manifest_path):
            raise HTTPException(status_code=404, detail=f"Manifest file not found: {manifest_path}")
        
        # Check if already processing
        if processing_status['status'] == 'processing':
            return JSONResponse({
                "status": "error",
                "message": "Worker is already processing a manifest"
            })
        
        # Reset status
        processing_status = {
            'status': 'idle',
            'total_batches': 0,
            'completed_batches': 0,
            'current_batch': None,
            'progress_percent': 0.0,
            'start_time': None,
            'end_time': None,
            'manifest_path': None
        }
        
        current_manifest = manifest_path
        
        # Start processing in background
        import threading
        worker_id = os.getenv('WORKER_ID', '0')
        thread = threading.Thread(target=process_manifest_async, args=(manifest_path, worker_id))
        thread.daemon = True
        thread.start()
        
        return JSONResponse({
            "status": "started",
            "message": f"Processing started for manifest: {manifest_path}",
            "manifest_path": manifest_path
        })
        
    except Exception as e:
        print(f"❌ Error starting processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current processing status"""
    return JSONResponse(processing_status)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "processing_status": processing_status['status']
    }

@app.post("/reset")
async def reset_worker():
    """Reset worker to idle state"""
    global processing_status, current_manifest
    
    processing_status = {
        'status': 'idle',
        'total_batches': 0,
        'completed_batches': 0,
        'current_batch': None,
        'progress_percent': 0.0,
        'start_time': None,
        'end_time': None,
        'manifest_path': None
    }
    current_manifest = None
    
    return JSONResponse({"status": "reset", "message": "Worker reset to idle state"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090, help="Port to run on")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID")
    args = parser.parse_args()
    
    # Set worker ID environment variable
    os.environ['WORKER_ID'] = str(args.worker_id)
    
    print(f"\n🚀 Starting Real-ESRGAN Batch Worker {args.worker_id} on port {args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
