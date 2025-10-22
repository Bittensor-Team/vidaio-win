#!/usr/bin/env python3
"""
Simplified Tier 2 Demo - Working version
"""

import os
import sys
import subprocess
import cv2
import numpy as np
from pathlib import Path

def create_test_video():
    """Create a simple test video using FFmpeg"""
    print("🎬 Creating test video...")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', 'testsrc2=duration=5:size=640x360:rate=25',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        'test_input.mp4'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists('test_input.mp4'):
        print("✅ Test video created successfully")
        return True
    else:
        print(f"❌ Failed to create test video: {result.stderr}")
        return False

def simple_upscaling():
    """Simple upscaling using OpenCV"""
    print("\n🔍 Testing Simple Upscaling...")
    
    if not os.path.exists('test_input.mp4'):
        print("❌ Input video not found")
        return False
    
    try:
        # Read video
        cap = cv2.VideoCapture('test_input.mp4')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_upscaled.mp4', fourcc, fps, (width*2, height*2))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Simple upscaling using INTER_CUBIC
            upscaled = cv2.resize(frame, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            
            # Apply simple sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            out.write(sharpened)
            frame_count += 1
            
            if frame_count % 25 == 0:
                print(f"   Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        print(f"✅ Upscaling complete: {frame_count} frames processed")
        return True
        
    except Exception as e:
        print(f"❌ Upscaling failed: {e}")
        return False

def simple_compression():
    """Simple compression using FFmpeg"""
    print("\n🔍 Testing Simple Compression...")
    
    if not os.path.exists('test_input.mp4'):
        print("❌ Input video not found")
        return False
    
    try:
        # Simple H.264 compression
        cmd = [
            'ffmpeg', '-y',
            '-i', 'test_input.mp4',
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            'test_compressed.mp4'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists('test_compressed.mp4'):
            print("✅ Compression complete")
            return True
        else:
            print(f"❌ Compression failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        return False

def test_validation():
    """Test validation using local_validation.py"""
    print("\n🔍 Testing Validation...")
    
    if not os.path.exists('test_upscaled.mp4') or not os.path.exists('test_input.mp4'):
        print("❌ Output files not found")
        return False
    
    try:
        # Test upscaling validation
        cmd = [
            'python3', 'local_validation.py',
            '--reference', 'test_input.mp4',
            '--processed', 'test_upscaled.mp4',
            '--task', 'upscaling',
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print("Upscaling Validation Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Upscaling validation successful")
            upscaling_ok = True
        else:
            print("❌ Upscaling validation failed")
            upscaling_ok = False
        
        # Test compression validation
        if os.path.exists('test_compressed.mp4'):
            cmd = [
                'python3', 'local_validation.py',
                '--reference', 'test_input.mp4',
                '--processed', 'test_compressed.mp4',
                '--task', 'compression',
                '--vmaf-threshold', '85',
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print("\nCompression Validation Output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            if result.returncode == 0:
                print("✅ Compression validation successful")
                compression_ok = True
            else:
                print("❌ Compression validation failed")
                compression_ok = False
        else:
            compression_ok = False
        
        return upscaling_ok and compression_ok
        
    except subprocess.TimeoutExpired:
        print("❌ Validation timed out")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def show_file_info():
    """Show information about generated files"""
    print("\n📊 File Information:")
    
    files = ['test_input.mp4', 'test_upscaled.mp4', 'test_compressed.mp4']
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   {file}: {size:.1f} MB")
        else:
            print(f"   {file}: Not found")

def main():
    print("======================================================================")
    print("  SIMPLIFIED TIER 2 DEMO")
    print("======================================================================")
    print("This demo shows basic video processing and validation")
    print("without the complex AI models that require specific environments.")
    print()
    
    # Step 1: Create test video
    if not create_test_video():
        print("❌ Failed to create test video")
        return 1
    
    # Step 2: Test upscaling
    upscaling_success = simple_upscaling()
    
    # Step 3: Test compression
    compression_success = simple_compression()
    
    # Step 4: Test validation
    validation_success = test_validation()
    
    # Step 5: Show results
    show_file_info()
    
    print("\n======================================================================")
    print("  RESULTS SUMMARY")
    print("======================================================================")
    print(f"Test Video Creation: ✅ PASS")
    print(f"Upscaling: {'✅ PASS' if upscaling_success else '❌ FAIL'}")
    print(f"Compression: {'✅ PASS' if compression_success else '❌ FAIL'}")
    print(f"Validation: {'✅ PASS' if validation_success else '❌ FAIL'}")
    
    if upscaling_success and compression_success and validation_success:
        print("\n🎉 DEMO SUCCESSFUL!")
        print("The basic video processing pipeline is working correctly.")
        print("This demonstrates the core functionality that would be")
        print("enhanced with the Tier 2 AI models in a proper environment.")
        return 0
    else:
        print("\n⚠️ Some components failed, but the demo shows the structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
