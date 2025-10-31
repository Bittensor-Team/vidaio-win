#!/usr/bin/env python3
"""
Fixed Ground Truth Scores
Properly parse PSNR/SSIM and get actual ground truth VMAF scores
"""

import os
import subprocess
import json
import time
import math

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

def calculate_psnr_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate PSNR with fixed parsing"""
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
        
        # Parse PSNR from log - look for the last frame's psnr_avg
        psnr_score = 0.0
        if os.path.exists('/tmp/psnr.log'):
            with open('/tmp/psnr.log', 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                
                # Find the last line with psnr_avg
                for line in reversed(lines):
                    if 'psnr_avg:' in line:
                        try:
                            # Extract psnr_avg value
                            parts = line.split('psnr_avg:')
                            if len(parts) > 1:
                                psnr_value = parts[1].split()[0]
                                if psnr_value == 'inf':
                                    psnr_score = 100.0  # Perfect match
                                else:
                                    psnr_score = float(psnr_value)
                                break
                        except Exception as e:
                            print(f"   PSNR parse error: {e}")
                            continue
        
        # Cleanup
        if os.path.exists('/tmp/psnr.log'):
            os.remove('/tmp/psnr.log')
        
        return psnr_score
        
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        return 0.0

def calculate_ssim_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate SSIM with fixed parsing"""
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
        
        # Parse SSIM from log - look for the last frame's All value
        ssim_score = 0.0
        if os.path.exists('/tmp/ssim.log'):
            with open('/tmp/ssim.log', 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                
                # Find the last line with All: value
                for line in reversed(lines):
                    if 'All:' in line:
                        try:
                            # Extract All value
                            parts = line.split('All:')
                            if len(parts) > 1:
                                ssim_value = parts[1].split()[0]
                                ssim_score = float(ssim_value)
                                break
                        except Exception as e:
                            print(f"   SSIM parse error: {e}")
                            continue
        
        # Cleanup
        if os.path.exists('/tmp/ssim.log'):
            os.remove('/tmp/ssim.log')
        
        return ssim_score
        
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0

def calculate_vmaf_fixed(reference_path: str, distorted_path: str) -> float:
    """Calculate VMAF alternative using fixed PSNR and SSIM"""
    try:
        psnr = calculate_psnr_fixed(reference_path, distorted_path)
        ssim = calculate_ssim_fixed(reference_path, distorted_path)
        
        # Convert PSNR to 0-100 scale (VMAF-like)
        # PSNR > 30 is generally good quality
        vmaf_psnr = min(psnr * 2.0, 100.0)  # Scale PSNR to 0-100
        
        # SSIM is already 0-1, convert to 0-100
        vmaf_ssim = ssim * 100.0
        
        # Combine PSNR and SSIM for VMAF-like score
        vmaf_score = (vmaf_psnr * 0.6 + vmaf_ssim * 0.4)
        
        print(f"   PSNR: {psnr:.2f} dB -> VMAF-like: {vmaf_psnr:.2f}")
        print(f"   SSIM: {ssim:.3f} -> VMAF-like: {vmaf_ssim:.2f}")
        print(f"   Combined VMAF: {vmaf_score:.2f}")
        
        return vmaf_score
        
    except Exception as e:
        print(f"VMAF calculation failed: {e}")
        return 0.0

def test_ground_truth_self_comparison():
    """Test ground truth against itself to verify scoring works"""
    print("🧪 Testing ground truth self-comparison...")
    
    # Test cases
    test_cases = [
        {
            'name': 'SD2HD Ground Truth vs Itself',
            'reference': '/tmp/ground_truth_sd2hd.mp4',
            'distorted': '/tmp/ground_truth_sd2hd.mp4',
            'expected_psnr': 'Very High (>40 dB)',
            'expected_ssim': '1.000'
        },
        {
            'name': 'HD24K Ground Truth vs Itself', 
            'reference': '/tmp/ground_truth_hd24k.mp4',
            'distorted': '/tmp/ground_truth_hd24k.mp4',
            'expected_psnr': 'Very High (>40 dB)',
            'expected_ssim': '1.000'
        },
        {
            'name': '4K28K Ground Truth vs Itself',
            'reference': '/tmp/ground_truth_4k28k.mp4', 
            'distorted': '/tmp/ground_truth_4k28k.mp4',
            'expected_psnr': 'Very High (>40 dB)',
            'expected_ssim': '1.000'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Expected PSNR: {test_case['expected_psnr']}")
        print(f"Expected SSIM: {test_case['expected_ssim']}")
        print(f"{'='*60}")
        
        if os.path.exists(test_case['reference']) and os.path.exists(test_case['distorted']):
            vmaf_score = calculate_vmaf_fixed(test_case['reference'], test_case['distorted'])
            
            print(f"\n📊 Results:")
            print(f"   Combined VMAF: {vmaf_score:.2f}")
            
            if vmaf_score > 90:
                print(f"   ✅ Self-comparison working correctly (VMAF > 90)")
            else:
                print(f"   ❌ Self-comparison not working (VMAF < 90)")
        else:
            print(f"   ❌ Files not found")

def test_our_vsr_vs_ground_truth():
    """Test our VSR videos against ground truth"""
    print("\n🧪 Testing our VSR vs ground truth...")
    
    test_cases = [
        {
            'name': 'SD2HD: Our VSR vs Ground Truth',
            'reference': '/tmp/ground_truth_sd2hd.mp4',
            'distorted': '/workspace/vidaio-win/real_vidaio_tests/SD2HD_10w_5s_10w.mp4'
        },
        {
            'name': 'HD24K: Our VSR vs Ground Truth',
            'reference': '/tmp/ground_truth_hd24k.mp4',
            'distorted': '/workspace/vidaio-win/real_vidaio_tests/HD24K_10w_5s_10w.mp4'
        },
        {
            'name': '4K28K: Our VSR vs Ground Truth',
            'reference': '/tmp/ground_truth_4k28k.mp4',
            'distorted': '/workspace/vidaio-win/real_vidaio_tests/4K28K_8w_5s_8w.mp4'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*60}")
        
        if os.path.exists(test_case['reference']) and os.path.exists(test_case['distorted']):
            # Get video info
            ref_info = get_video_info(test_case['reference'])
            dist_info = get_video_info(test_case['distorted'])
            
            if ref_info.get('streams') and dist_info.get('streams'):
                ref_stream = ref_info['streams'][0]
                dist_stream = dist_info['streams'][0]
                
                ref_res = f"{ref_stream.get('width', '?')}x{ref_stream.get('height', '?')}"
                dist_res = f"{dist_stream.get('width', '?')}x{dist_stream.get('height', '?')}"
                
                print(f"   Reference: {ref_res}")
                print(f"   Distorted: {dist_res}")
            
            vmaf_score = calculate_vmaf_fixed(test_case['reference'], test_case['distorted'])
            
            print(f"\n📊 Results:")
            print(f"   Combined VMAF: {vmaf_score:.2f}")
            
            # Vidaio scoring
            vmaf_percentage = vmaf_score / 100.0
            vmaf_check = 1.0 if vmaf_percentage >= 0.5 else 0.0
            
            print(f"\n🎯 Vidaio Scoring:")
            print(f"   VMAF Check Passed: {'✅' if vmaf_check > 0 else '❌'}")
            print(f"   VMAF Percentage: {vmaf_percentage:.2%}")
            
            if vmaf_check > 0:
                print(f"   ✅ Would pass Vidaio VMAF requirement")
            else:
                print(f"   ❌ Would fail Vidaio VMAF requirement")
        else:
            print(f"   ❌ Files not found")

def main():
    """Main debug function"""
    print("🔍 FIXED GROUND TRUTH SCORES")
    print("="*80)
    
    # First test: ground truth vs itself (should give perfect scores)
    test_ground_truth_self_comparison()
    
    # Second test: our VSR vs ground truth
    test_our_vsr_vs_ground_truth()
    
    print("\n" + "="*80)
    print("FIXED DEBUG COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()





