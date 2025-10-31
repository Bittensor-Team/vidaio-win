#!/usr/bin/env python3
"""
TSD-SR Worker Server
Runs on ports 8090-8094 with preloaded model
Accepts frame upscaling requests via HTTP
"""

import sys
import os
import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
import tempfile
from PIL import Image
import io
import argparse
import json
from pathlib import Path

# Import TSD-SR functions directly from test_tsdsr.py
sys.path.append(".")
sys.path.append("./test")
import test_tsdsr as tsdsr_module
from torchvision import transforms

# Global models (loaded once at startup)
vae = None
transformer = None
unet = None
tokenizer = None
device = None
weight_dtype = None
timesteps = None
prompt_embeds = None
pooled_prompt_embeds = None

def load_models(upscale_factor=2, process_size=128):
    """Load TSD-SR and SD3 models"""
    global vae, transformer, unet, device, weight_dtype, timesteps, prompt_embeds, pooled_prompt_embeds
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"🖥️  Device: {device}")
    print(f"📊 Data type: {weight_dtype}")
    
    try:
        # Import required modules
        from diffusers import SD3Transformer2DModel
        from models.autoencoder_kl import AutoencoderKL  # Use TSD-SR's custom VAE with LoRA support
        from peft import LoraConfig
        from utils.util import load_lora_state_dict
        import safetensors.torch
        
        print("🤖 Loading Stable Diffusion 3 components...")
        
        # Load SD3 transformer
        print("  Loading transformer...")
        transformer = SD3Transformer2DModel.from_pretrained(
            'checkpoint/sd3',
            subfolder='transformer',
            torch_dtype=weight_dtype
        ).to(device)
        
        # Load VAE
        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            'checkpoint/sd3',
            subfolder='vae',
            torch_dtype=weight_dtype
        ).to(device)
        vae.eval()
        
        # Load TSD-SR LoRA weights using the proper method
        print("🎯 Loading TSD-SR LoRA weights...")
        
        # Configure LoRA for transformer
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        
        # Configure LoRA for VAE
        vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1', 'encoder.down_blocks.0.resnets.0.conv2', 'encoder.down_blocks.0.resnets.1.conv1', 
                            'encoder.down_blocks.0.resnets.1.conv2', 'encoder.down_blocks.0.downsamplers.0.conv', 'encoder.down_blocks.1.resnets.0.conv1',
                            'encoder.down_blocks.1.resnets.0.conv2', 'encoder.down_blocks.1.resnets.0.conv_shortcut', 'encoder.down_blocks.1.resnets.1.conv1', 'encoder.down_blocks.1.resnets.1.conv2', 
                            'encoder.down_blocks.1.downsamplers.0.conv', 'encoder.down_blocks.2.resnets.0.conv1', 'encoder.down_blocks.2.resnets.0.conv2',
                            'encoder.down_blocks.2.resnets.0.conv_shortcut', 'encoder.down_blocks.2.resnets.1.conv1', 'encoder.down_blocks.2.resnets.1.conv2', 'encoder.down_blocks.2.downsamplers.0.conv',
                            'encoder.down_blocks.3.resnets.0.conv1', 'encoder.down_blocks.3.resnets.0.conv2', 'encoder.down_blocks.3.resnets.1.conv1', 'encoder.down_blocks.3.resnets.1.conv2', 
                            'encoder.mid_block.attentions.0.to_q', 'encoder.mid_block.attentions.0.to_k', 'encoder.mid_block.attentions.0.to_v', 'encoder.mid_block.attentions.0.to_out.0', 
                            'encoder.mid_block.resnets.0.conv1', 'encoder.mid_block.resnets.0.conv2', 'encoder.mid_block.resnets.1.conv1', 'encoder.mid_block.resnets.1.conv2', 'encoder.conv_out', 'quant_conv']
        vae_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=vae_target_modules
        )
        vae.add_adapter(vae_lora_config)
        vae.enable_adapters()
        
        # Load LoRA state dicts
        from diffusers import StableDiffusion3Pipeline
        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict('checkpoint/tsdsr', weight_name="vae.safetensors")
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict('checkpoint/tsdsr', weight_name="transformer.safetensors")
        
        load_lora_state_dict(vae_lora_state_dict, vae)
        load_lora_state_dict(transformer_lora_state_dict, transformer)
        
        vae = vae.to(device, dtype=weight_dtype)
        transformer = transformer.to(device, dtype=weight_dtype)
        
        # Set up timesteps and prompt embeddings
        timesteps = torch.tensor([1000.], device=device, dtype=weight_dtype)
        
        # Load prompt embeddings
        prompt_embeds = torch.load('dataset/default/prompt_embeds.pt', map_location=device).to(dtype=weight_dtype)
        pooled_prompt_embeds = torch.load('dataset/default/pool_embeds.pt', map_location=device).to(dtype=weight_dtype)
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

def prepare_image_tensor(image, target_size=512):
    """Prepare image tensor for processing"""
    from torchvision import transforms
    import numpy as np
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    # Calculate scaling to fit into target_size while maintaining aspect ratio
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Pad to make divisible by 8
    new_h = (new_h + 7) // 8 * 8
    new_w = (new_w + 7) // 8 * 8
    
    image_resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    tensor = transform(image_resized).unsqueeze(0).to(device_name, dtype=dtype)
    
    return tensor, image_resized, (new_h, new_w)

# tile_sample is now imported from test.test_tsdsr

def upscale_image_tsdsr(input_image, upscale_factor=2, process_size=128):
    """Upscale image using the exact same logic as test_tsdsr.py"""
    global device, weight_dtype, vae, transformer, timesteps, prompt_embeds, pooled_prompt_embeds
    
    try:
        import time
        start_time = time.time()
        print(f"Processing image: {input_image.size}")
        
        # Convert to tensor exactly like test_tsdsr.py
        tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Calculate target size exactly like test_tsdsr.py
        ori_width, ori_height = input_image.size
        upscale = upscale_factor
        
        # Resize the image if it is not valid (same logic as test_tsdsr.py)
        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale*ori_width), int(scale*ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale*new_width, upscale*new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8
        
        print(f"  Input size: {input_image.size}")
        print(f"  Target size: {new_width}x{new_height}")
        print(f"  Upscale factor: {upscale_factor}x")
        print(f"  Process size: {process_size}")
        
        # Convert to tensor exactly like test_tsdsr.py
        tensor_start = time.time()
        pixel_values = tensor_transforms(input_image).unsqueeze(0).to(device, dtype=weight_dtype)
        print(f"  ⏱️  Tensor conversion: {time.time() - tensor_start:.2f}s")
        
        # Use the exact same logic as test_tsdsr.py main function
        # Create a mock args object with ALL the required attributes from parse_args()
        class MockArgs:
            def __init__(self):
                self.device = device
                self.upscale = upscale_factor
                self.process_size = process_size
                # Add all the other required attributes with their defaults
                self.rank = 64
                self.rank_vae = 64
                self.is_use_tile = False
                self.vae_decoder_tiled_size = 224
                self.vae_encoder_tiled_size = 1024
                self.latent_tiled_size = 64
                self.latent_tiled_overlap = 8
                self.seed = 42
                self.mixed_precision = "fp16"
                self.align_method = "wavelet"
        
        mock_args = MockArgs()
        
        # Set the global variables that test_tsdsr.py expects
        tsdsr_module.weight_dtype = weight_dtype
        tsdsr_module.vae = vae
        tsdsr_module.transformer = transformer
        tsdsr_module.timesteps = timesteps
        tsdsr_module.prompt_embeds = prompt_embeds
        tsdsr_module.pooled_prompt_embeds = pooled_prompt_embeds
        tsdsr_module.args = mock_args  # Set args for _gaussian_weights function
        
        # Copy the exact logic from test_tsdsr.py main function
        with torch.no_grad():
            # Preprocess the input image (same as test_tsdsr.py main function)
            preprocess_start = time.time()
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=(new_height, new_width), mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(device, dtype=weight_dtype).clamp(-1,1)
            print(f"  ⏱️  Preprocessing: {time.time() - preprocess_start:.2f}s")
            
            # Encode the input image (same as test_tsdsr.py)
            encode_start = time.time()
            model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            model_input = model_input.to(device, dtype=weight_dtype)
            print(f"  ⏱️  VAE encode: {time.time() - encode_start:.2f}s")

            # Predict using tile_sample (same as test_tsdsr.py)
            predict_start = time.time()
            model_pred = tsdsr_module.tile_sample(model_input, pixel_values, transformer, timesteps, prompt_embeds, pooled_prompt_embeds, weight_dtype, mock_args)
            latent_stu = model_input - model_pred
            print(f"  ⏱️  TSD-SR prediction: {time.time() - predict_start:.2f}s")
            
            # Decode the output (same as test_tsdsr.py)
            decode_start = time.time()
            image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
            print(f"  ⏱️  VAE decode: {time.time() - decode_start:.2f}s")
        
        # Convert to PIL image (same as test_tsdsr.py)
        result = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        
        # Handle resize flag (same as test_tsdsr.py)
        if resize_flag:
            result = result.resize((int(ori_width*upscale), int(ori_height*upscale)))
        
        print(f"  Output size: {result.size}")
        total_time = time.time() - start_time
        print(f"  ⏱️  Total processing time: {total_time:.2f}s")
        print("✅ Upscaling complete")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        raise

app = FastAPI(
    title="TSD-SR Upscaler Worker",
    description="Text-to-Image Diffusion Super-Resolution Worker",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    print("\n" + "="*60)
    print("TSD-SR Worker Server Startup")
    print("="*60 + "\n")
    load_models()
    print("\n" + "="*60)
    print("✅ Ready to process requests")
    print("="*60 + "\n")

@app.post("/upscale")
async def upscale_frame(
    file: UploadFile = File(...),
    upscale: int = Query(2, description="Upscale factor (2 or 4)"),
    process_size: int = Query(128, description="Processing size (64-256)")
):
    """
    Upscale a single frame using TSD-SR
    
    Args:
        file: PNG/JPEG image file
        upscale: Upscale factor (2 or 4)
        process_size: Processing size for optimization
        
    Returns:
        Upscaled PNG image
    """
    try:
        if transformer is None or vae is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Validate upscale factor
        if upscale not in [2, 4]:
            raise HTTPException(status_code=400, detail="Upscale factor must be 2 or 4")
        
        # Validate process size
        if not (64 <= process_size <= 256):
            raise HTTPException(status_code=400, detail="Process size must be between 64 and 256")
        
        print(f"\n📥 Received upscaling request")
        print(f"   Upscale: {upscale}x, Process size: {process_size}")
        
        # Read image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        print(f"   Input file: {file.filename} ({input_image.size})")
        
        # Upscale
        upscaled_image = upscale_image_tsdsr(
            input_image,
            upscale_factor=upscale,
            process_size=process_size
        )
        
        # Save to temp file and return
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            upscaled_image.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        file_size = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"   Output size: {file_size:.2f}MB")
        print(f"✅ Request completed successfully\n")
        
        response = FileResponse(tmp_path, media_type="image/png")
        response.headers["X-Temp-File"] = tmp_path
        response.headers["X-Output-Size"] = f"{upscaled_image.size[0]}x{upscaled_image.size[1]}"
        return response
        
    except HTTPException as e:
        print(f"❌ HTTP Error: {e.detail}\n")
        raise
    except Exception as e:
        print(f"❌ Error: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upscale_batch")
async def upscale_batch(
    files: list = File(...),
    upscale: int = Query(2, description="Upscale factor (2 or 4)"),
    process_size: int = Query(128, description="Processing size (64-256)")
):
    """
    Upscale multiple frames
    
    Args:
        files: List of PNG/JPEG image files
        upscale: Upscale factor (2 or 4)
        process_size: Processing size for optimization
        
    Returns:
        JSON with paths to upscaled images
    """
    try:
        if transformer is None or vae is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        results = []
        
        print(f"\n📥 Received batch upscaling request")
        print(f"   Files: {len(files)}, Upscale: {upscale}x, Process size: {process_size}")
        
        for idx, file in enumerate(files, 1):
            try:
                # Read image
                contents = await file.read()
                input_image = Image.open(io.BytesIO(contents)).convert('RGB')
                
                print(f"   [{idx}/{len(files)}] Processing {file.filename}...")
                
                # Upscale
                upscaled_image = upscale_image_tsdsr(
                    input_image,
                    upscale_factor=upscale,
                    process_size=process_size
                )
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    upscaled_image.save(tmp.name, quality=95)
                    tmp_path = tmp.name
                
                results.append({
                    "filename": file.filename,
                    "temp_path": tmp_path,
                    "output_size": f"{upscaled_image.size[0]}x{upscaled_image.size[1]}",
                    "file_size_mb": os.path.getsize(tmp_path) / (1024 * 1024)
                })
                
            except Exception as e:
                print(f"   ❌ Error processing {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        print(f"✅ Batch processing completed\n")
        return {"results": results, "total": len(files), "successful": len([r for r in results if "error" not in r])}
        
    except HTTPException as e:
        print(f"❌ HTTP Error: {e.detail}\n")
        raise
    except Exception as e:
        print(f"❌ Error: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": {
            "transformer": transformer is not None,
            "vae": vae is not None,
            "timesteps": timesteps is not None,
            "prompt_embeds": prompt_embeds is not None,
            "pooled_prompt_embeds": pooled_prompt_embeds is not None,
            "device": str(device),
            "dtype": str(weight_dtype)
        }
    }

@app.get("/info")
async def info():
    """Get server information"""
    return {
        "name": "TSD-SR Worker Server",
        "version": "1.0.0",
        "description": "Text-to-Image Diffusion Super-Resolution",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "endpoints": {
            "POST /upscale": "Upscale a single image",
            "POST /upscale_batch": "Upscale multiple images",
            "GET /health": "Health check",
            "GET /info": "Server information"
        }
    }

@app.get("/stats")
async def stats():
    """Get server statistics"""
    stats_data = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "device": str(device),
        "dtype": str(weight_dtype)
    }
    
    if torch.cuda.is_available():
        stats_data.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        })
    
    return stats_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSD-SR Worker Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to run on (default: 8090)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        choices=[2, 4],
        help="Default upscale factor (default: 2)"
    )
    parser.add_argument(
        "--process-size",
        type=int,
        default=128,
        help="Default processing size (default: 128)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🚀 Starting TSD-SR Worker Server")
    print(f"{'='*60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Default Upscale: {args.upscale}x")
    print(f"Default Process Size: {args.process_size}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
