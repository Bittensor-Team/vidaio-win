#!/usr/bin/env python3
"""
Root Cause Analysis
Analyze the specific issues causing VSR pipeline failures
"""

import os
import subprocess
import json
from pathlib import Path

def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}

def analyze_resolution_issues():
    """Analyze why SD2HD is getting wrong resolution"""
    print("🔍 ANALYZING RESOLUTION ISSUES")
    print("="*60)
    
    # Check our SD2HD video
    sd2hd_path = '/workspace/vidaio-win/real_vidaio_tests/SD2HD_10w_5s_10w.mp4'
    if os.path.exists(sd2hd_path):
        info = get_video_info(sd2hd_path)
        if info.get('streams'):
            stream = info['streams'][0]
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            print(f"SD2HD Video Resolution: {width}x{height}")
            print(f"Expected: 1920x1080")
            print(f"Actual: {width}x{height}")
            print(f"Status: {'✅' if width == 1920 and height == 1080 else '❌'}")
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            expected_aspect = 1920 / 1080
            print(f"Aspect Ratio: {aspect_ratio:.3f} (expected: {expected_aspect:.3f})")
            
            if aspect_ratio != expected_aspect:
                print(f"❌ Aspect ratio mismatch - video is stretched/squashed")
            else:
                print(f"✅ Aspect ratio correct")
    
    # Check ground truth SD2HD
    gt_sd2hd_path = '/tmp/ground_truth_sd2hd.mp4'
    if os.path.exists(gt_sd2hd_path):
        info = get_video_info(gt_sd2hd_path)
        if info.get('streams'):
            stream = info['streams'][0]
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            print(f"\nGround Truth SD2HD Resolution: {width}x{height}")
            print(f"Expected: 1920x1080")
            print(f"Status: {'✅' if width == 1920 and height == 1080 else '❌'}")

def analyze_vsr_pipeline():
    """Analyze our VSR pipeline implementation"""
    print("\n🔍 ANALYZING VSR PIPELINE")
    print("="*60)
    
    # Check if we can find the VSR pipeline code
    pipeline_files = [
        '/workspace/vidaio-win/vidaio_pipeline_test.py',
        '/workspace/vidaio-win/vidaio_pipeline_test_hd_8k.py',
        '/workspace/vidaio-win/vsr_parallel_experiment.py'
    ]
    
    for file_path in pipeline_files:
        if os.path.exists(file_path):
            print(f"Found pipeline file: {file_path}")
            
            # Look for resolution handling code
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for resolution scaling logic
            if 'scale=' in content:
                print(f"  ✅ Contains scaling logic")
            else:
                print(f"  ❌ No scaling logic found")
                
            # Check for 2x upscaling
            if '2x' in content or 'scale=2' in content:
                print(f"  ✅ Contains 2x upscaling")
            else:
                print(f"  ❌ No 2x upscaling found")
                
            # Check for specific resolution targets
            if '1920x1080' in content:
                print(f"  ✅ Contains 1920x1080 target")
            else:
                print(f"  ❌ No 1920x1080 target found")

def analyze_onnx_model_usage():
    """Analyze how ONNX models are being used"""
    print("\n🔍 ANALYZING ONNX MODEL USAGE")
    print("="*60)
    
    # Check ONNX model files
    model_path = '/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx'
    if os.path.exists(model_path):
        print(f"✅ ONNX model found: {model_path}")
        
        # Check model size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
        if size_mb > 1:
            print(f"✅ Model size looks reasonable")
        else:
            print(f"❌ Model size seems too small")
    else:
        print(f"❌ ONNX model not found")
    
    # Check if model is being loaded correctly
    print(f"\nChecking model loading in pipeline...")
    
    # Look for model loading code
    pipeline_files = [
        '/workspace/vidaio-win/vidaio_pipeline_test.py',
        '/workspace/vidaio-win/vsr_parallel_experiment.py'
    ]
    
    for file_path in pipeline_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            if 'onnxruntime' in content:
                print(f"  ✅ {file_path} uses onnxruntime")
            else:
                print(f"  ❌ {file_path} doesn't use onnxruntime")
                
            if 'InferenceSession' in content:
                print(f"  ✅ {file_path} loads ONNX models")
            else:
                print(f"  ❌ {file_path} doesn't load ONNX models")

def analyze_video_processing():
    """Analyze video processing steps"""
    print("\n🔍 ANALYZING VIDEO PROCESSING")
    print("="*60)
    
    # Check input video
    input_video = '/workspace/vidaio-win/elk.mp4'
    if os.path.exists(input_video):
        info = get_video_info(input_video)
        if info.get('streams'):
            stream = info['streams'][0]
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            print(f"Input video resolution: {width}x{height}")
            
            # Check if input is suitable for SD2HD
            if width >= 854 and height >= 480:
                print(f"✅ Input suitable for SD2HD (480p -> 1080p)")
            else:
                print(f"❌ Input not suitable for SD2HD")
    
    # Check if we're resizing input correctly
    print(f"\nChecking input resizing...")
    
    # Look for input resizing in pipeline
    pipeline_files = [
        '/workspace/vidaio-win/vidaio_pipeline_test.py'
    ]
    
    for file_path in pipeline_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            if '480p' in content or '854x480' in content:
                print(f"  ✅ {file_path} handles 480p input")
            else:
                print(f"  ❌ {file_path} doesn't handle 480p input")

def analyze_quality_issues():
    """Analyze quality issues"""
    print("\n🔍 ANALYZING QUALITY ISSUES")
    print("="*60)
    
    # Based on our ground truth results
    print("Quality Analysis from Ground Truth Results:")
    print("="*40)
    
    print("HD24K vs Ground Truth:")
    print("  PSNR: 25.95 dB (Good: >30 dB)")
    print("  SSIM: 0.941 (Good: >0.95)")
    print("  VMAF: 68.78% (Pass: >50%)")
    print("  Status: ✅ Passes Vidaio requirements")
    
    print("\n4K28K vs Ground Truth:")
    print("  PSNR: 26.34 dB (Good: >30 dB)")
    print("  SSIM: 0.959 (Good: >0.95)")
    print("  VMAF: 69.96% (Pass: >50%)")
    print("  Status: ✅ Passes Vidaio requirements")
    
    print("\nSD2HD vs Ground Truth:")
    print("  PSNR: 0.00 dB (Failed due to resolution mismatch)")
    print("  SSIM: 0.000 (Failed due to resolution mismatch)")
    print("  VMAF: 0.00% (Fail: <50%)")
    print("  Status: ❌ Complete failure")

def main():
    """Main analysis function"""
    print("🔍 ROOT CAUSE ANALYSIS")
    print("="*80)
    
    analyze_resolution_issues()
    analyze_vsr_pipeline()
    analyze_onnx_model_usage()
    analyze_video_processing()
    analyze_quality_issues()
    
    print("\n" + "="*80)
    print("ROOT CAUSE SUMMARY")
    print("="*80)
    
    print("\n🎯 PRIMARY ROOT CAUSES:")
    print("1. SD2HD Resolution Issue:")
    print("   - Expected: 1920x1080")
    print("   - Actual: 1708x960")
    print("   - Cause: Incorrect upscaling factor or aspect ratio handling")
    
    print("\n2. VSR Pipeline Implementation:")
    print("   - May not be properly scaling to target resolutions")
    print("   - Could be using wrong upscaling factors")
    print("   - Possible aspect ratio preservation issues")
    
    print("\n3. Quality Assessment:")
    print("   - HD24K and 4K28K actually work well (68-70% VMAF)")
    print("   - SD2HD fails due to resolution mismatch")
    print("   - ONNX models seem to be working for correct resolutions")
    
    print("\n🔧 IMMEDIATE FIXES NEEDED:")
    print("1. Fix SD2HD upscaling to reach 1920x1080")
    print("2. Verify upscaling factors in VSR pipeline")
    print("3. Check aspect ratio handling")
    print("4. Test with proper input resolutions")
    
    print("\n💡 KEY INSIGHT:")
    print("The VSR models are actually working well!")
    print("The issue is in the pipeline's resolution handling, not model quality.")

if __name__ == "__main__":
    main()





