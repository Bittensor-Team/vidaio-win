#!/usr/bin/env python3
"""
Microsoft Edge VideoSuperResolution Worker Server
Runs on specified port with preloaded VSR model
Accepts frame upscaling requests via HTTP
"""

import sys
import os
import uvicorn
import onnxruntime as ort
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
from PIL import Image
import io
import argparse
import time

# Global model (loaded once at startup)
model = None
model_name = None
device = None

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

app = FastAPI(title="Microsoft Edge VSR Worker")

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    # This will be set by the main function
    pass

@app.post("/upscale")
async def upscale_frame(file: UploadFile = File(...)):
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
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        "providers": model.get_providers()
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
            "upscaling_factor": "2x",
            "input_format": "RGB",
            "output_format": "RGB",
            "compatible_with": ["images", "video_frames"]
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microsoft Edge VSR Worker Server")
    parser.add_argument("--port", type=int, default=8095, help="Port to run on")
    parser.add_argument("--model", type=str, default="DXM-FP32", 
                       choices=["DXM-FP32", "DXM-FP16", "E-FP32", "E-FP16", "FP32", "FP16"],
                       help="VSR model to load")
    args = parser.parse_args()
    
    print(f"\n🚀 Starting Microsoft Edge VSR Worker")
    print(f"📡 Port: {args.port}")
    print(f"🤖 Model: {args.model}")
    
    # Load model before starting server
    try:
        load_model(args.model)
        print(f"✅ Model {args.model} loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"\n🌐 Server starting on http://127.0.0.1:{args.port}")
    print(f"📋 Endpoints:")
    print(f"   POST /upscale - Upscale image")
    print(f"   GET  /health  - Health check")
    print(f"   GET  /info    - Model information")
    
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")





