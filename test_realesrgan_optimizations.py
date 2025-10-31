#!/usr/bin/env python3
"""
Real-ESRGAN Optimization Test
Tests batch_size, patch_size, and scale factor optimizations
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add the RealESRGAN module to path
sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')

def test_optimization(input_image, output_dir, test_name, batch_size=4, patch_size=192, scale=4):
    """Test a specific optimization configuration"""
    print(f"\n🧪 Testing {test_name}")
    print(f"   Batch size: {batch_size}, Patch size: {patch_size}, Scale: {scale}x")
    
    output_path = os.path.join(output_dir, f"{test_name}_output.png")
    
    # Run the upscaler
    cmd = [
        '/usr/bin/python3',
        '/workspace/vidaio-win/Real-ESRGAN-working-update/working_upscaler.py',
        input_image,
        '-o', output_path,
        '--batch_size', str(batch_size),
        '--patch_size', str(patch_size),
        '--scale', str(scale)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            elapsed = end_time - start_time
            print(f"   ✅ Success: {elapsed:.2f} seconds")
            return elapsed, True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return None, False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after 5 minutes")
        return None, False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, False

def main():
    print("🚀 Real-ESRGAN Optimization Test")
    print("=" * 60)
    
    # Setup
    input_image = "/workspace/vidaio-win/elk.mp4"  # We'll extract a frame first
    test_dir = "/workspace/vidaio-win/optimization_tests"
    os.makedirs(test_dir, exist_ok=True)
    
    # Extract a test frame from elk.mp4
    test_frame = os.path.join(test_dir, "test_frame.png")
    print(f"📸 Extracting test frame from elk.mp4...")
    
    extract_cmd = [
        '/usr/bin/ffmpeg', '-y', '-i', input_image,
        '-vf', 'select=eq(n\\,0)', '-vframes', '1',
        test_frame
    ]
    
    result = subprocess.run(extract_cmd, capture_output=True)
    if result.returncode != 0:
        print("❌ Failed to extract test frame")
        return
    
    print(f"✅ Test frame extracted: {test_frame}")
    
    # Test configurations
    tests = [
        # Baseline
        ("baseline", 4, 192, 4),
        
        # Batch size optimizations
        ("batch_2", 2, 192, 4),
        ("batch_8", 8, 192, 4),
        ("batch_16", 16, 192, 4),
        
        # Patch size optimizations
        ("patch_128", 4, 128, 4),
        ("patch_256", 4, 256, 4),
        ("patch_512", 4, 512, 4),
        
        # Scale optimizations
        ("scale_2x", 4, 192, 2),
        ("scale_4x", 4, 192, 4),
        
        # Combined optimizations
        ("optimized_fast", 8, 256, 2),
        ("optimized_balanced", 8, 192, 4),
        ("optimized_quality", 4, 128, 4),
    ]
    
    results = {}
    
    print(f"\n🔬 Running {len(tests)} optimization tests...")
    print("=" * 60)
    
    for test_name, batch_size, patch_size, scale in tests:
        elapsed, success = test_optimization(
            test_frame, test_dir, test_name, 
            batch_size, patch_size, scale
        )
        
        if success:
            results[test_name] = {
                'time': elapsed,
                'batch_size': batch_size,
                'patch_size': patch_size,
                'scale': scale
            }
    
    # Analysis
    print(f"\n📊 Optimization Results Analysis")
    print("=" * 60)
    
    if not results:
        print("❌ No successful tests completed")
        return
    
    baseline_time = results.get('baseline', {}).get('time')
    if baseline_time:
        print(f"📈 Baseline (4, 192, 4x): {baseline_time:.2f}s")
        print()
        
        # Sort by speed (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['time'])
        
        print("🏆 Speed Rankings:")
        for i, (name, data) in enumerate(sorted_results, 1):
            speedup = baseline_time / data['time'] if data['time'] > 0 else 0
            print(f"   {i:2d}. {name:20s}: {data['time']:6.2f}s ({speedup:.2f}x speedup)")
            print(f"       Config: batch={data['batch_size']}, patch={data['patch_size']}, scale={data['scale']}x")
        
        print(f"\n🎯 Optimization Recommendations:")
        
        # Find fastest
        fastest = sorted_results[0]
        print(f"   🚀 Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
        
        # Find best batch size
        batch_tests = [(k, v) for k, v in results.items() if 'batch_' in k]
        if batch_tests:
            best_batch = min(batch_tests, key=lambda x: x[1]['time'])
            print(f"   📦 Best batch size: {best_batch[0]} (batch_size={best_batch[1]['batch_size']})")
        
        # Find best patch size
        patch_tests = [(k, v) for k, v in results.items() if 'patch_' in k]
        if patch_tests:
            best_patch = min(patch_tests, key=lambda x: x[1]['time'])
            print(f"   🧩 Best patch size: {best_patch[0]} (patch_size={best_patch[1]['patch_size']})")
        
        # Find best scale
        scale_tests = [(k, v) for k, v in results.items() if 'scale_' in k]
        if scale_tests:
            best_scale = min(scale_tests, key=lambda x: x[1]['time'])
            print(f"   📏 Best scale: {best_scale[0]} (scale={best_scale[1]['scale']})")
    
    print(f"\n📁 Test outputs saved in: {test_dir}")
    print("✅ Optimization test completed!")

if __name__ == "__main__":
    main()





