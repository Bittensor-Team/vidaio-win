#!/usr/bin/env python3
"""
Simple test script for Real-ESRGAN upscaling
"""
import sys
import os
import subprocess
from PIL import Image
import numpy as np

def simple_upscale(input_path, output_path, scale=2):
    """Simple upscaling using PIL resize as fallback"""
    try:
        # Load image
        img = Image.open(input_path)
        print(f"  📐 Original size: {img.size}")
        
        # Calculate new size
        new_size = (img.size[0] * scale, img.size[1] * scale)
        print(f"  📐 Target size: {new_size}")
        
        # Simple upscale using PIL
        upscaled = img.resize(new_size, Image.LANCZOS)
        
        # Save result
        upscaled.save(output_path)
        print(f"  ✅ Upscaled and saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 simple_test.py input.png output.png scale")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale = int(sys.argv[3])
    
    print(f"🎬 Simple Upscaling Test")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Scale: {scale}x")
    
    success = simple_upscale(input_path, output_path, scale)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)





