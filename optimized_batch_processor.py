#!/usr/bin/env python3
"""
Optimized batch processor for Real-ESRGAN
Processes multiple images more efficiently by reusing model state
"""

import os
import sys
import time
import torch
import subprocess
from PIL import Image
import numpy as np

# Add the Real-ESRGAN-working-update directory to the Python path
sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')
from RealESRGAN import RealESRGAN

class OptimizedBatchProcessor:
    def __init__(self, device, scale=2, batch_size=1, patch_size=256):
        """Initialize the optimized batch processor"""
        self.device = device
        self.scale = scale
        self.batch_size = batch_size
        self.patch_size = patch_size
        
        # Load model once
        print("🤖 Loading Real-ESRGAN model...")
        self.model = RealESRGAN(device, scale=scale)
        self.model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
        print("✅ Model loaded")
    
    def process_images_batch(self, images, output_paths):
        """Process multiple images in an optimized batch"""
        print(f"📦 Processing {len(images)} images in optimized batch...")
        
        start_time = time.time()
        results = []
        
        # Process each image (Real-ESRGAN doesn't support true multi-image batching)
        for i, (image, output_path) in enumerate(zip(images, output_paths)):
            try:
                # Process single image with optimized parameters
                sr_image = self.model.predict(
                    image, 
                    batch_size=self.batch_size, 
                    patches_size=self.patch_size
                )
                
                # Save result
                sr_image.save(output_path)
                results.append(True)
                
                if i % 4 == 0:  # Progress every 4 images
                    elapsed = time.time() - start_time
                    print(f"  📊 Processed {i+1}/{len(images)} images ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"    ❌ Error processing image {i}: {e}")
                results.append(False)
        
        total_time = time.time() - start_time
        success_count = sum(results)
        
        print(f"✅ Batch processing completed: {success_count}/{len(images)} images in {total_time:.1f}s")
        print(f"📊 Average time per image: {total_time/len(images):.3f}s")
        
        return results

def test_optimized_processing():
    """Test the optimized batch processor"""
    print("🔍 Testing Optimized Batch Processing")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create test images
    test_dir = "/tmp/optimized_test"
    input_dir = f"{test_dir}/input"
    output_dir = f"{test_dir}/output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("📸 Creating test images...")
    images = []
    output_paths = []
    
    for i in range(8):  # 8 images (1 batch)
        # Create test image
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        input_path = f"{input_dir}/frame_{i:06d}.png"
        output_path = f"{output_dir}/frame_{i:06d}.png"
        
        img.save(input_path)
        images.append(img)
        output_paths.append(output_path)
    
    # Test 1: Optimized batch processing
    print("\n📊 Test 1: Optimized Batch Processing")
    processor = OptimizedBatchProcessor(device, scale=2, batch_size=1, patch_size=256)
    
    start_time = time.time()
    results = processor.process_images_batch(images, output_paths)
    optimized_time = time.time() - start_time
    
    print(f"✅ Optimized processing: {optimized_time:.1f}s")
    
    # Test 2: Individual processing (current method)
    print("\n📊 Test 2: Individual Processing (Current Method)")
    individual_processor = OptimizedBatchProcessor(device, scale=2, batch_size=1, patch_size=256)
    
    start_time = time.time()
    individual_results = []
    
    for i, (image, output_path) in enumerate(zip(images, output_paths)):
        try:
            sr_image = individual_processor.model.predict(image, batch_size=1, patches_size=256)
            sr_image.save(output_path)
            individual_results.append(True)
        except Exception as e:
            print(f"    ❌ Error processing image {i}: {e}")
            individual_results.append(False)
    
    individual_time = time.time() - start_time
    print(f"✅ Individual processing: {individual_time:.1f}s")
    
    # Analysis
    print(f"\n📊 Performance Analysis:")
    print(f"  Optimized batch: {optimized_time:.1f}s")
    print(f"  Individual processing: {individual_time:.1f}s")
    print(f"  Difference: {abs(optimized_time - individual_time):.1f}s")
    
    if optimized_time < individual_time:
        print("  ✅ Optimized batch is faster")
    else:
        print("  ❌ No significant improvement")
    
    # Cleanup
    import subprocess
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def test_with_real_frames():
    """Test with real video frames to see the actual performance"""
    print("\n🔍 Testing with Real Video Frames")
    print("=" * 40)
    
    # Extract frames from elk.mp4
    test_dir = "/tmp/real_frames_test"
    input_dir = f"{test_dir}/input"
    output_dir = f"{test_dir}/output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("📸 Extracting frames from elk.mp4...")
    cmd = f"ffmpeg -i elk.mp4 -vf fps=29.97 {input_dir}/frame_%06d.png -y -loglevel error"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Frame extraction failed: {result.stderr}")
        return
    
    # Get frame files
    frame_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"📊 Extracted {len(frame_files)} frames")
    
    # Test with first 8 frames
    test_frames = frame_files[:8]
    print(f"📊 Testing with {len(test_frames)} frames")
    
    # Load images
    images = []
    output_paths = []
    
    for frame_file in test_frames:
        input_path = f"{input_dir}/{frame_file}"
        output_path = f"{output_dir}/{frame_file}"
        
        image = Image.open(input_path).convert('RGB')
        images.append(image)
        output_paths.append(output_path)
    
    # Test optimized processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = OptimizedBatchProcessor(device, scale=2, batch_size=1, patch_size=256)
    
    start_time = time.time()
    results = processor.process_images_batch(images, output_paths)
    real_time = time.time() - start_time
    
    print(f"✅ Real frames processing: {real_time:.1f}s")
    print(f"📊 Average time per frame: {real_time/len(images):.3f}s")
    
    # Cleanup
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run optimized batch processing tests"""
    print("🔍 Optimized Batch Processing Investigation")
    print("=" * 50)
    
    test_optimized_processing()
    test_with_real_frames()
    
    print("\n✅ Investigation complete!")

if __name__ == "__main__":
    main()
