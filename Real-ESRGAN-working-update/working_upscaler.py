#!/usr/bin/env python3
"""
Working Real-ESRGAN Upscaler
Uses the clean Real-ESRGAN implementation without torchvision compatibility issues
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# Add the RealESRGAN module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(size=(256, 256), filename="test_image.png"):
    """Create a test image with patterns"""
    print(f"🎨 Creating test image: {filename}")
    
    # Create a colorful test image
    img = Image.new('RGB', size, color='white')
    pixels = np.array(img)
    
    # Add gradient pattern
    for i in range(size[0]):
        for j in range(size[1]):
            r = int(255 * (i / size[0]))
            g = int(255 * (j / size[1]))
            b = int(255 * ((i + j) / (size[0] + size[1])))
            
            # Add noise for texture
            noise = np.random.randint(-20, 20, 3)
            r = max(0, min(255, r + noise[0]))
            g = max(0, min(255, g + noise[1]))
            b = max(0, min(255, b + noise[2]))
            
            pixels[j, i] = [r, g, b]
    
    # Add geometric shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw circles
    for i in range(3):
        x = size[0] // 4 + i * size[0] // 4
        y = size[1] // 4 + i * size[1] // 4
        radius = size[0] // 8
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=(255, 0, 0, 100), outline=(0, 0, 0))
    
    img.save(filename)
    print(f"✅ Test image created: {filename} ({size[0]}x{size[1]})")
    return filename

def upscale_with_realesrgan(input_path, output_path, scale=4, batch_size=4, patch_size=192):
    """Upscale image using the working Real-ESRGAN implementation"""
    print(f"🖼️  Upscaling with Real-ESRGAN: {input_path}")
    print(f"📊 Scale factor: {scale}x")
    print(f"🔧 Batch size: {batch_size}, Patch size: {patch_size}")
    
    try:
        from RealESRGAN import RealESRGAN
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Initialize model
        print("🤖 Loading Real-ESRGAN model...")
        model = RealESRGAN(device, scale=scale)
        
        # Load weights (will download if needed)
        weights_path = f'weights/RealESRGAN_x{scale}.pth'
        model.load_weights(weights_path, download=True)
        
        # Load and process image
        print("📸 Loading input image...")
        image = Image.open(input_path).convert('RGB')
        original_size = image.size
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        
        # Upscale
        print("⚡ Processing image...")
        start_time = time.time()
        
        sr_image = model.predict(
            image, 
            batch_size=batch_size, 
            patches_size=patch_size
        )
        
        elapsed_time = time.time() - start_time
        
        # Save result
        sr_image.save(output_path)
        
        final_size = sr_image.size
        print(f"📐 Final size: {final_size[0]}x{final_size[1]}")
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Upscaled image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Real-ESRGAN upscaling: {e}")
        import traceback
        traceback.print_exc()
        return False

def upscale_with_pil_fallback(input_path, output_path, scale=4):
    """Fallback upscaling using PIL"""
    print(f"🖼️  Fallback PIL upscaling: {input_path}")
    print(f"📊 Scale factor: {scale}x")
    
    try:
        image = Image.open(input_path)
        original_size = image.size
        new_size = (original_size[0] * scale, original_size[1] * scale)
        
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        print(f"📐 Target size: {new_size[0]}x{new_size[1]}")
        
        start_time = time.time()
        upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
        elapsed_time = time.time() - start_time
        
        upscaled.save(output_path)
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Fallback upscaled image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fallback upscaling: {e}")
        return False

def compare_methods(input_path, scale=4):
    """Compare Real-ESRGAN vs PIL upscaling"""
    print(f"\n🔍 Comparing upscaling methods (scale: {scale}x)")
    print("=" * 60)
    
    methods = [
        ("realesrgan", "Real-ESRGAN AI upscaling"),
        ("pil_fallback", "PIL LANCZOS fallback")
    ]
    
    results = []
    
    for method, description in methods:
        output_path = f"output_{method}_{scale}x.png"
        
        try:
            if method == "realesrgan":
                success = upscale_with_realesrgan(input_path, output_path, scale)
            else:
                success = upscale_with_pil_fallback(input_path, output_path, scale)
            
            if success:
                file_size = os.path.getsize(output_path)
                results.append((method, description, file_size, "✅"))
            else:
                results.append((method, description, 0, "❌"))
                
        except Exception as e:
            results.append((method, description, 0, f"❌ {str(e)[:30]}"))
    
    # Print results
    print(f"\n📊 Results Summary:")
    print("-" * 60)
    print(f"{'Method':<15} {'Description':<25} {'Size (KB)':<10} {'Status'}")
    print("-" * 60)
    
    for method, description, size, status in results:
        size_kb = size // 1024 if size > 0 else 0
        print(f"{method:<15} {description:<25} {size_kb:<10} {status}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Working Real-ESRGAN Upscaler")
    parser.add_argument("input", nargs='?', help="Input image path (optional, will create test image if not provided)")
    parser.add_argument("-o", "--output", help="Output image path", default=None)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4, 8], help="Scale factor (default: 4)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (default: 4)")
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for processing (default: 192)")
    parser.add_argument("--compare", action="store_true", help="Compare Real-ESRGAN vs PIL methods")
    parser.add_argument("--test", action="store_true", help="Create and test with a sample image")
    parser.add_argument("--fallback", action="store_true", help="Use PIL fallback instead of Real-ESRGAN")
    
    args = parser.parse_args()
    
    print("🚀 Working Real-ESRGAN Upscaler")
    print("=" * 50)
    
    # Handle input
    if args.input is None or args.test:
        if not args.test and args.input is None:
            print("❌ No input image provided. Use --test to create a test image or provide an input path.")
            return 1
        
        input_path = create_test_image()
    else:
        input_path = args.input
        if not os.path.exists(input_path):
            print(f"❌ Error: Input file '{input_path}' not found")
            return 1
    
    # Set output path
    if args.output is None:
        input_path_obj = Path(input_path)
        args.output = str(input_path_obj.parent / f"{input_path_obj.stem}_upscaled_{args.scale}x{input_path_obj.suffix}")
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {args.output}")
    print(f"📊 Scale: {args.scale}x")
    
    try:
        success = True
        
        if args.compare:
            # Compare methods
            compare_methods(input_path, args.scale)
        elif args.fallback:
            # Use PIL fallback
            success = upscale_with_pil_fallback(input_path, args.output, args.scale)
        else:
            # Use Real-ESRGAN
            success = upscale_with_realesrgan(
                input_path, 
                args.output, 
                args.scale, 
                args.batch_size, 
                args.patch_size
            )
            
            # If Real-ESRGAN fails, try fallback
            if not success:
                print("⚠️  Real-ESRGAN failed, trying PIL fallback...")
                success = upscale_with_pil_fallback(input_path, args.output, args.scale)
        
        if args.compare:
            print(f"\n🎉 Comparison complete! Check the output files.")
        elif success:
            print(f"\n🎉 Success! Check your upscaled image at: {args.output}")
            
            # Show file info
            if os.path.exists(args.output):
                file_size = os.path.getsize(args.output)
                print(f"📁 Output file size: {file_size // 1024} KB")
        else:
            print("❌ All upscaling methods failed")
            return 1
        
        print(f"\n✨ Demo complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
