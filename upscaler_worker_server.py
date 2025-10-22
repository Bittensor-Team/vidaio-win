#!/usr/bin/env python3
"""
Real-ESRGAN Upscaler Worker Server
Runs on ports 8090-8094 with preloaded model
Accepts frame upscaling requests via HTTP
"""

import sys
import os
import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
from PIL import Image
import io
import argparse

sys.path.insert(0, '/workspace/vidaio-subnet/Real-ESRGAN-working-update')

# Global model (loaded once at startup)
model = None
device = None

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

app = FastAPI(title="Real-ESRGAN Upscaler Worker")

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    load_model(scale=2)

@app.post("/upscale")
async def upscale_frame(file: UploadFile = File(...)):
    """
    Upscale a single frame
    
    Args:
        file: PNG image file
        
    Returns:
        Upscaled PNG image
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Upscale
        sr_image = model.predict(image, batch_size=1, patches_size=192)
        
        # Save to temp file and return
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sr_image.save(tmp.name)
            tmp_path = tmp.name
        
        response = FileResponse(tmp_path, media_type="image/png")
        response.headers["X-Temp-File"] = tmp_path
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090, help="Port to run on")
    args = parser.parse_args()
    
    print(f"\n🚀 Starting Real-ESRGAN Worker on port {args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
