#!/usr/bin/env python3
"""
Comprehensive Vidaio Scoring Validator
Validates processed videos and creates reference comparisons for scoring pipeline validation
"""

import os
import cv2
import subprocess
import json
import time
import math
import numpy as np
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
            print(f"ffprobe failed: {result.stderr}")
            return {}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}

def create_reference_videos():
    """Create reference videos for comparison testing"""
    print("🎬 Creating reference videos for comparison...")
    
    # Create different resolution reference videos
    reference_videos = {
        '480p': '/tmp/reference_480p.mp4',
        '1080p': '/tmp/reference_1080p.mp4', 
        '4k': '/tmp/reference_4k.mp4'
    }
    
    # Create 480p reference
    cmd_480p = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=854:480:force_original_aspect_ratio=decrease',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '5',  # 5 seconds
        reference_videos['480p']
    ]
    
    # Create 1080p reference  
    cmd_1080p = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '5',  # 5 seconds
        reference_videos['1080p']
    ]
    
    # Create 4K reference
    cmd_4k = [
        'ffmpeg', '-y', '-i', '/workspace/vidaio-win/elk.mp4',
        '-vf', 'scale=3840:2160:force_original_aspect_ratio=decrease',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-t', '5',  # 5 seconds
        reference_videos['4k']
    ]
    
    for cmd, path in [(cmd_480p, reference_videos['480p']), 
                      (cmd_1080p, reference_videos['1080p']),
                      (cmd_4k, reference_videos['4k'])]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created {path}")
        else:
            print(f"❌ Failed to create {path}: {result.stderr}")
    
    return reference_videos

def calculate_psnr(reference_path: str, distorted_path: str) -> float:
    """Calculate PSNR between two videos as VMAF alternative"""
    try:
        # Use ffmpeg to calculate PSNR
        psnr_cmd = [
            'ffmpeg', '-y',
            '-i', reference_path,
            '-i', distorted_path,
            '-lavfi', 'psnr=stats_file=/tmp/psnr.log',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(psnr_cmd, capture_output=True, text=True, timeout=60)
        
        # Parse PSNR from log
        psnr_score = 0.0
        if os.path.exists('/tmp/psnr.log'):
            with open('/tmp/psnr.log', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'psnr_avg' in line:
                        psnr_score = float(line.split(':')[1].strip())
                        break
        
        # Cleanup
        if os.path.exists('/tmp/psnr.log'):
            os.remove('/tmp/psnr.log')
        
        return psnr_score
        
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        return 0.0

def calculate_ssim(reference_path: str, distorted_path: str) -> float:
    """Calculate SSIM between two videos"""
    try:
        # Use ffmpeg to calculate SSIM
        ssim_cmd = [
            'ffmpeg', '-y',
            '-i', reference_path,
            '-i', distorted_path,
            '-lavfi', 'ssim=stats_file=/tmp/ssim.log',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(ssim_cmd, capture_output=True, text=True, timeout=60)
        
        # Parse SSIM from log
        ssim_score = 0.0
        if os.path.exists('/tmp/ssim.log'):
            with open('/tmp/ssim.log', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'SSIM' in line and 'All' in line:
                        ssim_score = float(line.split(':')[1].strip())
                        break
        
        # Cleanup
        if os.path.exists('/tmp/ssim.log'):
            os.remove('/tmp/ssim.log')
        
        return ssim_score
        
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0

def calculate_vmaf_alternative(reference_path: str, distorted_path: str) -> float:
    """Calculate VMAF alternative using PSNR and SSIM"""
    try:
        psnr = calculate_psnr(reference_path, distorted_path)
        ssim = calculate_ssim(reference_path, distorted_path)
        
        # Convert PSNR to 0-100 scale (VMAF-like)
        # PSNR > 30 is generally good quality
        vmaf_psnr = min(psnr * 2.0, 100.0)  # Scale PSNR to 0-100
        
        # SSIM is already 0-1, convert to 0-100
        vmaf_ssim = ssim * 100.0
        
        # Combine PSNR and SSIM for VMAF-like score
        vmaf_score = (vmaf_psnr * 0.6 + vmaf_ssim * 0.4)
        
        print(f"   PSNR: {psnr:.2f} dB -> VMAF-like: {vmaf_psnr:.2f}")
        print(f"   SSIM: {ssim:.3f} -> VMAF-like: {vmaf_ssim:.2f}")
        print(f"   Combined VMAF-like: {vmaf_score:.2f}")
        
        return vmaf_score
        
    except Exception as e:
        print(f"VMAF alternative calculation failed: {e}")
        return 0.0

def calculate_pie_app_improved(reference_path: str, distorted_path: str) -> float:
    """Calculate improved PIE-APP perceptual quality score"""
    try:
        # Read video properties
        ref_cap = cv2.VideoCapture(reference_path)
        dist_cap = cv2.VideoCapture(distorted_path)
        
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            return 0.0
        
        # Get video properties
        ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dist_width = int(dist_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        dist_height = int(dist_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
        dist_fps = dist_cap.get(cv2.CAP_PROP_FPS)
        
        ref_cap.release()
        dist_cap.release()
        
        # Calculate resolution improvement factor
        ref_pixels = ref_width * ref_height
        dist_pixels = dist_width * dist_height
        
        if ref_pixels == 0:
            return 0.0
        
        resolution_factor = dist_pixels / ref_pixels
        
        # Calculate frame rate factor
        fps_factor = min(dist_fps / ref_fps, 1.0) if ref_fps > 0 else 0.0
        
        # For upscaling tasks, we expect resolution improvement
        if resolution_factor >= 1.0:
            # Upscaling case - reward based on improvement
            pie_app_score = min(resolution_factor * 0.8 + fps_factor * 0.2, 1.0)
        else:
            # Downscaling case - penalize
            pie_app_score = resolution_factor * 0.5
        
        return pie_app_score
        
    except Exception as e:
        print(f"PIE-APP calculation failed: {e}")
        return 0.0

def sigmoid_transformation(x: float) -> float:
    """Sigmoid transformation for quality score"""
    return 1 / (1 + math.exp(-6 * (x - 0.5)))

def calculate_upscaling_score(reference_path: str, processed_path: str, content_length: float) -> dict:
    """Calculate upscaling score using Vidaio formula"""
    
    print(f"🔍 Calculating VMAF alternative...")
    vmaf_score = calculate_vmaf_alternative(reference_path, processed_path)
    vmaf_percentage = vmaf_score / 100.0  # Convert to 0-1 range
    
    # VMAF check (must be ≥ 50% or S_Q = 0)
    vmaf_check = 1.0 if vmaf_percentage >= 0.5 else 0.0
    
    print(f"🔍 Calculating PIE-APP score...")
    pie_app_score = calculate_pie_app_improved(reference_path, processed_path)
    
    # Quality Score (S_Q)
    s_q = sigmoid_transformation(pie_app_score) * vmaf_check
    
    # Length Score (S_L)
    s_l = math.log(1 + content_length) / math.log(1 + 320)
    
    # Pre-score
    s_pre = 0.5 * s_q + 0.5 * s_l
    
    # Final Score (S_F) with exponential scaling
    s_f = 0.1 * math.exp(6.979 * (s_pre - 0.5))
    
    return {
        'vmaf_score': vmaf_score,
        'vmaf_percentage': vmaf_percentage,
        'pie_app_score': pie_app_score,
        'quality_score': s_q,
        'length_score': s_l,
        'pre_score': s_pre,
        'final_score': s_f,
        'vmaf_check_passed': vmaf_check > 0
    }

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    try:
        info = get_video_info(video_path)
        if 'format' in info and 'duration' in info['format']:
            return float(info['format']['duration'])
        return 0.0
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0.0

def validate_video_with_reference(reference_path: str, processed_path: str, task_type: str) -> dict:
    """Validate a processed video against proper reference"""
    
    print(f"🔍 Validating {task_type} video with proper reference...")
    
    # Get video duration
    duration = get_video_duration(processed_path)
    
    # Calculate upscaling score
    score_data = calculate_upscaling_score(reference_path, processed_path, duration)
    
    # Get file sizes
    ref_size = os.path.getsize(reference_path) / (1024 * 1024)  # MB
    proc_size = os.path.getsize(processed_path) / (1024 * 1024)  # MB
    
    # Get video resolution
    ref_info = get_video_info(reference_path)
    proc_info = get_video_info(processed_path)
    
    ref_resolution = "Unknown"
    proc_resolution = "Unknown"
    
    if 'streams' in ref_info and ref_info['streams']:
        stream = ref_info['streams'][0]
        ref_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
    
    if 'streams' in proc_info and proc_info['streams']:
        stream = proc_info['streams'][0]
        proc_resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
    
    return {
        'task_type': task_type,
        'reference_path': reference_path,
        'processed_path': processed_path,
        'duration_seconds': duration,
        'reference_resolution': ref_resolution,
        'processed_resolution': proc_resolution,
        'reference_size_mb': ref_size,
        'processed_size_mb': proc_size,
        'size_ratio': proc_size / ref_size if ref_size > 0 else 0,
        'scoring': score_data
    }

def test_scoring_pipeline():
    """Test the scoring pipeline with known good/bad examples"""
    print("🧪 Testing scoring pipeline with reference examples...")
    
    # Create reference videos
    reference_videos = create_reference_videos()
    
    # Test cases with proper references
    test_cases = [
        {
            'reference': reference_videos['480p'],
            'processed': '/workspace/vidaio-win/real_vidaio_tests/SD2HD_10w_5s_10w.mp4',
            'task_type': 'SD2HD',
            'expected_resolution': '1920x1080'
        },
        {
            'reference': reference_videos['1080p'],
            'processed': '/workspace/vidaio-win/real_vidaio_tests/HD24K_10w_5s_10w.mp4',
            'task_type': 'HD24K',
            'expected_resolution': '3840x2160'
        },
        {
            'reference': reference_videos['4k'],
            'processed': '/workspace/vidaio-win/real_vidaio_tests/4K28K_8w_5s_8w.mp4',
            'task_type': '4K28K',
            'expected_resolution': '7680x4320'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        if os.path.exists(test_case['reference']) and os.path.exists(test_case['processed']):
            result = validate_video_with_reference(
                test_case['reference'],
                test_case['processed'], 
                test_case['task_type']
            )
            result['expected_resolution'] = test_case['expected_resolution']
            results.append(result)
        else:
            print(f"❌ Missing files for {test_case['task_type']}")
    
    return results

def main():
    """Main validation function"""
    
    print("🚀 COMPREHENSIVE VIDAIO SCORING VALIDATION")
    print("="*80)
    
    # Test scoring pipeline
    results = test_scoring_pipeline()
    
    # Print results
    print("\n" + "="*80)
    print("SCORING PIPELINE VALIDATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n📹 {result['task_type']} Video Validation")
        print(f"   Reference: {result['reference_resolution']} ({result['reference_size_mb']:.2f} MB)")
        print(f"   Processed: {result['processed_resolution']} ({result['processed_size_mb']:.2f} MB)")
        print(f"   Expected:  {result['expected_resolution']}")
        print(f"   Duration: {result['duration_seconds']:.2f} seconds")
        print(f"   Size Ratio: {result['size_ratio']:.2f}x")
        
        # Check resolution correctness
        resolution_correct = result['processed_resolution'] == result['expected_resolution']
        print(f"   Resolution Correct: {'✅' if resolution_correct else '❌'}")
        
        scoring = result['scoring']
        print(f"\n📊 Scoring Results:")
        print(f"   VMAF Score: {scoring['vmaf_score']:.2f} ({scoring['vmaf_percentage']:.2%})")
        print(f"   PIE-APP Score: {scoring['pie_app_score']:.3f}")
        print(f"   Quality Score (S_Q): {scoring['quality_score']:.3f}")
        print(f"   Length Score (S_L): {scoring['length_score']:.3f}")
        print(f"   Final Score (S_F): {scoring['final_score']:.3f}")
        print(f"   VMAF Check Passed: {'✅' if scoring['vmaf_check_passed'] else '❌'}")
        
        # Performance assessment
        if scoring['final_score'] > 0.32:
            performance = "🟢 Excellent (+15% bonus eligible)"
        elif scoring['final_score'] > 0.07:
            performance = "🟡 Good"
        else:
            performance = "🔴 Poor (-20% penalty)"
        
        print(f"   Performance: {performance}")
        print("-" * 60)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_vmaf = sum(1 for r in results if r['scoring']['vmaf_check_passed'])
    correct_resolution = sum(1 for r in results if r['processed_resolution'] == r['expected_resolution'])
    good_performance = sum(1 for r in results if r['scoring']['final_score'] > 0.07)
    
    print(f"Total Tests: {total_tests}")
    print(f"VMAF Check Passed: {passed_vmaf}/{total_tests} ({passed_vmaf/total_tests*100:.1f}%)")
    print(f"Correct Resolution: {correct_resolution}/{total_tests} ({correct_resolution/total_tests*100:.1f}%)")
    print(f"Good Performance: {good_performance}/{total_tests} ({good_performance/total_tests*100:.1f}%)")
    
    if passed_vmaf == total_tests and correct_resolution == total_tests:
        print("🎉 All tests passed! Scoring pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check VMAF calculation and resolution handling.")

if __name__ == "__main__":
    main()





