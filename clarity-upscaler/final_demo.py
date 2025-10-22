#!/usr/bin/env python3
"""
Final Clarity Upscaler Demo
Comprehensive demo script with multiple upscaling methods
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

def create_test_image(size=(256, 256), filename="test_image.png"):
    """Create a test image with some patterns"""
    print(f"🎨 Creating test image: {filename}")
    
    # Create a colorful test image with patterns
    img = Image.new('RGB', size, color='white')
    pixels = np.array(img)
    
    # Add some patterns
    for i in range(size[0]):
        for j in range(size[1]):
            # Create a gradient pattern
            r = int(255 * (i / size[0]))
            g = int(255 * (j / size[1]))
            b = int(255 * ((i + j) / (size[0] + size[1])))
            
            # Add some noise for texture
            noise = np.random.randint(-20, 20, 3)
            r = max(0, min(255, r + noise[0]))
            g = max(0, min(255, g + noise[1]))
            b = max(0, min(255, b + noise[2]))
            
            pixels[j, i] = [r, g, b]
    
    # Add some geometric shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw circles
    for i in range(3):
        x = size[0] // 4 + i * size[0] // 4
        y = size[1] // 4 + i * size[1] // 4
        radius = size[0] // 8
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=(255, 0, 0, 100), outline=(0, 0, 0))
    
    # Draw rectangles
    for i in range(2):
        x1 = size[0] // 6 + i * size[0] // 3
        y1 = size[1] // 6 + i * size[1] // 3
        x2 = x1 + size[0] // 6
        y2 = y1 + size[1] // 6
        draw.rectangle([x1, y1, x2, y2], 
                      fill=(0, 255, 0, 100), outline=(0, 0, 0))
    
    img.save(filename)
    print(f"✅ Test image created: {filename} ({size[0]}x{size[1]})")
    return filename

def upscale_pil_basic(input_path, output_path, scale=2):
    """Basic PIL upscaling with LANCZOS"""
    print(f"🔧 Basic PIL upscaling (LANCZOS)")
    
    img = Image.open(input_path)
    original_size = img.size
    new_size = (original_size[0] * scale, original_size[1] * scale)
    
    start_time = time.time()
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    elapsed_time = time.time() - start_time
    
    upscaled.save(output_path)
    print(f"✅ Basic upscaling complete: {elapsed_time:.3f}s")
    return True

def upscale_pil_enhanced(input_path, output_path, scale=2):
    """Enhanced PIL upscaling with sharpening"""
    print(f"🔧 Enhanced PIL upscaling with sharpening")
    
    img = Image.open(input_path)
    original_size = img.size
    new_size = (original_size[0] * scale, original_size[1] * scale)
    
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
    
    enhanced.save(output_path)
    print(f"✅ Enhanced upscaling complete: {elapsed_time:.3f}s")
    return True

def upscale_pil_super_resolution(input_path, output_path, scale=2):
    """Super resolution using multiple techniques"""
    print(f"🔧 Super resolution upscaling")
    
    img = Image.open(input_path)
    original_size = img.size
    new_size = (original_size[0] * scale, original_size[1] * scale)
    
    start_time = time.time()
    
    # Step 1: Initial upscaling
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Step 2: Apply edge enhancement
    edge_enhanced = upscaled.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Step 3: Apply unsharp mask
    sharpened = edge_enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    
    # Step 4: Apply detail enhancement
    from PIL import ImageEnhance
    detail_enhancer = ImageEnhance.Detail(sharpened)
    detailed = detail_enhancer.enhance(1.2)
    
    # Step 5: Final contrast adjustment
    contrast_enhancer = ImageEnhance.Contrast(detailed)
    final = contrast_enhancer.enhance(1.05)
    
    elapsed_time = time.time() - start_time
    
    final.save(output_path)
    print(f"✅ Super resolution complete: {elapsed_time:.3f}s")
    return True

def upscale_with_interpolation(input_path, output_path, scale=2, method='bicubic'):
    """Upscaling with different interpolation methods"""
    print(f"🔧 Interpolation upscaling ({method})")
    
    img = Image.open(input_path)
    original_size = img.size
    new_size = (original_size[0] * scale, original_size[1] * scale)
    
    # Map method names to PIL constants
    methods = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
        'box': Image.Resampling.BOX,
        'hamming': Image.Resampling.HAMMING
    }
    
    if method not in methods:
        print(f"❌ Unknown method: {method}")
        return False
    
    start_time = time.time()
    upscaled = img.resize(new_size, methods[method])
    elapsed_time = time.time() - start_time
    
    upscaled.save(output_path)
    print(f"✅ {method} upscaling complete: {elapsed_time:.3f}s")
    return True

def compare_methods(input_path, scale=2):
    """Compare different upscaling methods"""
    print(f"\n🔍 Comparing upscaling methods (scale: {scale}x)")
    print("=" * 60)
    
    methods = [
        ("basic", "Basic LANCZOS"),
        ("enhanced", "Enhanced with sharpening"),
        ("super_res", "Super resolution"),
        ("nearest", "Nearest neighbor"),
        ("bilinear", "Bilinear"),
        ("bicubic", "Bicubic"),
        ("box", "Box"),
        ("hamming", "Hamming")
    ]
    
    results = []
    
    for method, description in methods:
        output_path = f"output_{method}_{scale}x.png"
        
        try:
            if method == "basic":
                success = upscale_pil_basic(input_path, output_path, scale)
            elif method == "enhanced":
                success = upscale_pil_enhanced(input_path, output_path, scale)
            elif method == "super_res":
                success = upscale_pil_super_resolution(input_path, output_path, scale)
            else:
                success = upscale_with_interpolation(input_path, output_path, scale, method)
            
            if success:
                # Get file size
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
    parser = argparse.ArgumentParser(description="Final Image Upscaler Demo")
    parser.add_argument("input", nargs='?', help="Input image path (optional, will create test image if not provided)")
    parser.add_argument("-o", "--output", help="Output image path", default=None)
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default: 2)")
    parser.add_argument("--method", default="enhanced", 
                       choices=["basic", "enhanced", "super_res", "nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"],
                       help="Upscaling method (default: enhanced)")
    parser.add_argument("--compare", action="store_true", help="Compare all methods")
    parser.add_argument("--test", action="store_true", help="Create and test with a sample image")
    
    args = parser.parse_args()
    
    print("🚀 Final Image Upscaler Demo")
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
        args.output = str(input_path_obj.parent / f"{input_path_obj.stem}_upscaled_{args.method}_{args.scale}x{input_path_obj.suffix}")
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {args.output}")
    print(f"📊 Scale: {args.scale}x")
    print(f"🔧 Method: {args.method}")
    
    # Get original image info
    with Image.open(input_path) as img:
        original_size = img.size
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        print(f"📐 Target size: {original_size[0]*args.scale}x{original_size[1]*args.scale}")
    
    try:
        if args.compare:
            # Compare all methods
            compare_methods(input_path, args.scale)
        else:
            # Single method upscaling
            print(f"\n⚡ Starting upscaling...")
            
            if args.method == "basic":
                success = upscale_pil_basic(input_path, args.output, args.scale)
            elif args.method == "enhanced":
                success = upscale_pil_enhanced(input_path, args.output, args.scale)
            elif args.method == "super_res":
                success = upscale_pil_super_resolution(input_path, args.output, args.scale)
            else:
                success = upscale_with_interpolation(input_path, args.output, args.scale, args.method)
            
            if success:
                print(f"\n🎉 Success! Check your upscaled image at: {args.output}")
                
                # Show file info
                if os.path.exists(args.output):
                    file_size = os.path.getsize(args.output)
                    print(f"📁 Output file size: {file_size // 1024} KB")
            else:
                print("❌ Upscaling failed")
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
