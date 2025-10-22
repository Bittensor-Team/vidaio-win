#!/usr/bin/env python3
"""
Fixed Simple Clarity Upscaler Demo
A lightweight version that works with current torch/torchvision versions
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import torch
import numpy as np

def setup_simple_environment():
    """Setup a simple environment for Real-ESRGAN upscaling"""
    print("🔧 Setting up simple environment...")
    
    try:
        # Try to import Real-ESRGAN with compatibility fixes
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Patch torchvision before importing realesrgan
        try:
            import torchvision.transforms.functional as F
            # Check if functional_tensor exists, if not create a dummy
            if not hasattr(F, 'functional_tensor'):
                # Create a dummy functional_tensor module
                import types
                F.functional_tensor = types.ModuleType('functional_tensor')
                F.functional_tensor._is_tracing = lambda: False
        except ImportError:
            pass
        
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        print("✅ Real-ESRGAN modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Error importing Real-ESRGAN: {e}")
        print("💡 Trying alternative approach...")
        return False

def upscale_with_realesrgan_fixed(input_path, output_path, scale=2, model_name='RealESRGAN_x4plus'):
    """Upscale image using Real-ESRGAN with compatibility fixes"""
    print(f"🖼️  Upscaling image: {input_path}")
    print(f"📊 Scale factor: {scale}x")
    print(f"🤖 Model: {model_name}")
    
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Try to patch torchvision issues
        try:
            import torchvision.transforms.functional as F
            if not hasattr(F, 'functional_tensor'):
                import types
                F.functional_tensor = types.ModuleType('functional_tensor')
                F.functional_tensor._is_tracing = lambda: False
        except:
            pass
        
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        
        # Model configurations
        model_configs = {
            'RealESRGAN_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'netscale': 4,
                'file_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            },
            'RealESRGAN_x4plus_anime_6B': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                'netscale': 4,
                'file_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
            },
            'RealESRGAN_x2plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                'netscale': 2,
                'file_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            }
        }
        
        if model_name not in model_configs:
            print(f"❌ Unknown model: {model_name}")
            print(f"Available models: {list(model_configs.keys())}")
            return False
        
        config = model_configs[model_name]
        
        # Initialize upsampler with error handling
        try:
            upsampler = RealESRGANer(
                scale=config['netscale'],
                model_path=config['file_url'],
                model=config['model'],
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False  # Disable half precision to avoid compatibility issues
            )
        except Exception as e:
            print(f"⚠️  Error initializing Real-ESRGAN: {e}")
            print("🔄 Falling back to PIL upscaling...")
            return False
        
        # Load image
        img = Image.open(input_path).convert('RGB')
        img_array = np.array(img)
        
        print(f"📐 Original size: {img_array.shape[1]}x{img_array.shape[0]}")
        
        # Upscale
        start_time = time.time()
        try:
            output, _ = upsampler.enhance(img_array, outscale=scale)
        except Exception as e:
            print(f"⚠️  Error during Real-ESRGAN upscaling: {e}")
            print("🔄 Falling back to PIL upscaling...")
            return False
        
        elapsed_time = time.time() - start_time
        
        print(f"📐 Upscaled size: {output.shape[1]}x{output.shape[0]}")
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        
        # Save result
        result_img = Image.fromarray(output)
        result_img.save(output_path)
        
        print(f"✅ Upscaled image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        print("🔄 Falling back to PIL upscaling...")
        return False

def upscale_with_pil(input_path, output_path, scale=2):
    """Fallback upscaling using PIL (basic interpolation)"""
    print(f"🖼️  Basic upscaling with PIL: {input_path}")
    print(f"📊 Scale factor: {scale}x")
    
    try:
        # Load image
        img = Image.open(input_path)
        original_size = img.size
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        
        # Calculate new size
        new_size = (original_size[0] * scale, original_size[1] * scale)
        print(f"📐 Target size: {new_size[0]}x{new_size[1]}")
        
        # Upscale using LANCZOS resampling
        start_time = time.time()
        upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        
        # Save result
        upscaled.save(output_path)
        print(f"✅ Upscaled image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        return False

def upscale_with_pil_enhanced(input_path, output_path, scale=2):
    """Enhanced PIL upscaling with sharpening"""
    print(f"🖼️  Enhanced PIL upscaling: {input_path}")
    print(f"📊 Scale factor: {scale}x")
    
    try:
        # Load image
        img = Image.open(input_path)
        original_size = img.size
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        
        # Calculate new size
        new_size = (original_size[0] * scale, original_size[1] * scale)
        print(f"📐 Target size: {new_size[0]}x{new_size[1]}")
        
        start_time = time.time()
        
        # First upscale
        upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply sharpening
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Apply slight contrast enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = enhancer.enhance(1.1)
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        
        # Save result
        enhanced.save(output_path)
        print(f"✅ Enhanced upscaled image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced upscaling: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fixed Simple Image Upscaler Demo")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path", default=None)
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default: 2)")
    parser.add_argument("--model", default="RealESRGAN_x4plus", 
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus"],
                       help="Upscaling model (default: RealESRGAN_x4plus)")
    parser.add_argument("--fallback", action="store_true", help="Use PIL fallback if Real-ESRGAN fails")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced PIL upscaling")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}")
    
    print("🚀 Fixed Simple Image Upscaler Demo")
    print("=" * 50)
    
    # Try Real-ESRGAN first (unless fallback is forced)
    if not args.fallback:
        if setup_simple_environment():
            success = upscale_with_realesrgan_fixed(args.input, args.output, args.scale, args.model)
            if success:
                print(f"🎉 Success! Check your upscaled image at: {args.output}")
                return 0
            else:
                print("⚠️  Real-ESRGAN failed, trying PIL fallback...")
    
    # Fallback to PIL
    if args.enhanced:
        success = upscale_with_pil_enhanced(args.input, args.output, args.scale)
    else:
        success = upscale_with_pil(args.input, args.output, args.scale)
    
    if success:
        print(f"🎉 Success! Check your upscaled image at: {args.output}")
        return 0
    else:
        print("❌ All upscaling methods failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
