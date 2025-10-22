#!/usr/bin/env python3
"""
Clarity Upscaler Demo Script
Standalone script to upscale images without the web UI
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """Setup the environment and initialize modules"""
    print("🔧 Setting up environment...")
    
    # Set environment variables
    os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'
    
    # Import and initialize modules
    from modules import timer
    from modules import initialize_util
    from modules import initialize
    
    startup_timer = timer.startup_timer
    startup_timer.record("launcher")
    
    initialize.imports()
    initialize.check_versions()
    initialize.initialize()
    
    print("✅ Environment setup complete")
    return startup_timer

def create_predictor():
    """Create the predictor instance with all configurations"""
    print("🤖 Initializing predictor...")
    
    from modules.api.api import Api
    from modules.call_queue import queue_lock
    from fastapi import FastAPI
    from modules.api.models import StableDiffusionImg2ImgProcessingAPI
    
    app = FastAPI()
    initialize_util.setup_middleware(app)
    
    api = Api(app, queue_lock)
    
    # Get available models
    model_response = api.get_sd_models()
    print(f"📋 Available models: {[model.title for model in model_response]}")
    
    # Initialize with default image
    file_path = Path("init.png")
    if file_path.exists():
        base64_encoded_data = base64.b64encode(file_path.read_bytes())
        base64_image = base64_encoded_data.decode('utf-8')
    else:
        # Create a simple test image if init.png doesn't exist
        test_img = Image.new('RGB', (512, 512), color='white')
        buffered = BytesIO()
        test_img.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Default payload configuration
    payload = {
        "override_settings": {
            "sd_model_checkpoint": "juggernaut_reborn.safetensors",
            "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
            "CLIP_stop_at_last_layers": 1,
        },
        "override_settings_restore_afterwards": False,
        "prompt": "office building",
        "steps": 1,
        "init_images": [base64_image],
        "denoising_strength": 0.1,
        "do_not_save_samples": True,
        "alwayson_scripts": {
            "Tiled Diffusion": {
                "args": [
                    True, "MultiDiffusion", True, True, 1, 1,
                    112, 144, 4, 8, "4x-UltraSharp", 1.1, False, 0, 0.0, 3
                ]
            },
            "Tiled VAE": {
                "args": [True, 3072, 192, True, True, True, True]
            },
            "controlnet": {
                "args": [{
                    "enabled": True,
                    "module": "tile_resample",
                    "model": "control_v11f1e_sd15_tile",
                    "weight": 0.2,
                    "image": base64_image,
                    "resize_mode": 1,
                    "lowvram": False,
                    "downsample": 1.0,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": 1,
                    "pixel_perfect": True,
                    "threshold_a": 1,
                    "threshold_b": 1,
                    "save_detected_map": False,
                    "processor_res": 512,
                }]
            }
        }
    }
    
    req = StableDiffusionImg2ImgProcessingAPI(**payload)
    api.img2imgapi(req)
    
    print("✅ Predictor initialized")
    return api, StableDiffusionImg2ImgProcessingAPI

def get_clarity_upscaler_payload(sd_model, tiling_width, tiling_height, multiplier, 
                                base64_image, resemblance, prompt, negative_prompt, 
                                num_inference_steps, dynamic, seed, scheduler, creativity):
    """Generate the upscaling payload"""
    override_settings = {
        "sd_model_checkpoint": sd_model,
        "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
        "CLIP_stop_at_last_layers": 1,
    }
    
    alwayson_scripts = {
        "Tiled Diffusion": {
            "args": [
                True, "MultiDiffusion", True, True, 1, 1,
                tiling_width, tiling_height, 4, 8, "4x-UltraSharp", 
                multiplier, False, 0, 0.0, 3
            ]
        },
        "Tiled VAE": {
            "args": [True, 2048, 128, True, True, True, True]
        },
        "controlnet": {
            "args": [{
                "enabled": True,
                "module": "tile_resample",
                "model": "control_v11f1e_sd15_tile",
                "weight": resemblance,
                "image": base64_image,
                "resize_mode": 1,
                "lowvram": False,
                "downsample": 1.0,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "control_mode": 1,
                "pixel_perfect": True,
                "threshold_a": 1,
                "threshold_b": 1,
                "save_detected_map": False,
                "processor_res": 512,
            }]
        }
    }
    
    return {
        "override_settings": override_settings,
        "override_settings_restore_afterwards": False,
        "init_images": [base64_image],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": num_inference_steps,
        "cfg_scale": dynamic,
        "seed": seed,
        "do_not_save_samples": True,
        "sampler_name": scheduler,
        "denoising_strength": creativity,
        "alwayson_scripts": alwayson_scripts,
    }

def calc_scale_factors(value):
    """Calculate multi-step scale factors"""
    lst = []
    while value >= 2:
        lst.append(2)
        value /= 2
    if value > 1:
        lst.append(value)
    return lst

def upscale_image(input_path, output_path, **kwargs):
    """Main upscaling function"""
    print(f"🖼️  Upscaling image: {input_path}")
    
    # Default parameters
    params = {
        'prompt': "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
        'negative_prompt': "(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
        'scale_factor': 2.0,
        'dynamic': 6.0,
        'creativity': 0.35,
        'resemblance': 0.6,
        'tiling_width': 112,
        'tiling_height': 144,
        'sd_model': "juggernaut_reborn.safetensors",
        'scheduler': "DPM++ 3M SDE Karras",
        'num_inference_steps': 18,
        'seed': 1337,
        'multistep_factor': 0.8,
        'sharpen': 0.0,
        'output_format': "png"
    }
    
    # Update with provided parameters
    params.update(kwargs)
    
    print(f"📊 Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Load and process image
    with open(input_path, "rb") as image_file:
        binary_image_data = image_file.read()
    
    base64_encoded_data = base64.b64encode(binary_image_data)
    base64_image = base64_encoded_data.decode('utf-8')
    
    # Calculate scale factors for multi-step upscaling
    multipliers = [params['scale_factor']]
    if params['scale_factor'] > 2:
        multipliers = calc_scale_factors(params['scale_factor'])
        print(f"🔄 Multi-step upscaling: {len(multipliers)} steps")
    
    # Initialize API
    api, StableDiffusionImg2ImgProcessingAPI = create_predictor()
    
    first_iteration = True
    current_image = base64_image
    
    # Process each scale factor
    for i, multiplier in enumerate(multipliers):
        print(f"⚡ Step {i+1}/{len(multipliers)}: Upscaling by {multiplier}x")
        
        if not first_iteration:
            params['creativity'] = params['creativity'] * params['multistep_factor']
            params['seed'] = params['seed'] + 1
        
        first_iteration = False
        
        # Create payload
        payload = get_clarity_upscaler_payload(
            params['sd_model'], params['tiling_width'], params['tiling_height'],
            multiplier, current_image, params['resemblance'], params['prompt'],
            params['negative_prompt'], params['num_inference_steps'],
            params['dynamic'], params['seed'], params['scheduler'], params['creativity']
        )
        
        # Run upscaling
        req = StableDiffusionImg2ImgProcessingAPI(**payload)
        resp = api.img2imgapi(req)
        
        current_image = resp.images[0]
    
    # Process final image
    gen_bytes = BytesIO(base64.b64decode(current_image))
    image_object = Image.open(gen_bytes)
    
    # Apply sharpening if requested
    if params['sharpen'] > 0:
        from PIL import ImageFilter
        a = -params['sharpen'] / 10
        b = 1 - 8 * a
        kernel = [a, a, a, a, b, a, a, a, a]
        kernel_filter = ImageFilter.Kernel((3, 3), kernel, scale=1, offset=0)
        image_object = image_object.filter(kernel_filter)
    
    # Save output
    if params['output_format'] in ["webp", "jpg"]:
        image_object.save(output_path, quality=95, optimize=True)
    else:
        image_object.save(output_path)
    
    print(f"✅ Upscaled image saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Clarity Upscaler Demo")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path", default=None)
    parser.add_argument("--scale", type=float, default=2.0, help="Scale factor (default: 2.0)")
    parser.add_argument("--creativity", type=float, default=0.35, help="Creativity (0.3-0.9, default: 0.35)")
    parser.add_argument("--resemblance", type=float, default=0.6, help="Resemblance (0.3-1.6, default: 0.6)")
    parser.add_argument("--dynamic", type=float, default=6.0, help="HDR/Dynamic range (3-9, default: 6.0)")
    parser.add_argument("--steps", type=int, default=18, help="Inference steps (default: 18)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337)")
    parser.add_argument("--sharpen", type=float, default=0.0, help="Sharpen amount (0-10, default: 0)")
    parser.add_argument("--prompt", default="masterpiece, best quality, highres", help="Upscaling prompt")
    parser.add_argument("--negative-prompt", default="(worst quality, low quality, normal quality:2)", help="Negative prompt")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}")
    
    print("🚀 Clarity Upscaler Demo")
    print("=" * 50)
    
    try:
        # Setup environment
        startup_timer = setup_environment()
        
        # Upscale image
        start_time = time.time()
        upscale_image(
            args.input,
            args.output,
            scale_factor=args.scale,
            creativity=args.creativity,
            resemblance=args.resemblance,
            dynamic=args.dynamic,
            num_inference_steps=args.steps,
            seed=args.seed,
            sharpen=args.sharpen,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Total processing time: {elapsed_time:.2f} seconds")
        print(f"🎉 Success! Check your upscaled image at: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
