#!/usr/bin/env python3
"""
Working validation script using FFmpeg directly for VMAF
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path

def calculate_vmaf_ffmpeg(reference_path, processed_path):
    """Calculate VMAF using FFmpeg directly"""
    print("🔍 Calculating VMAF using FFmpeg...")
    
    try:
        # Create temporary file for VMAF output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vmaf_output = f.name
        
        # FFmpeg command for VMAF calculation
        cmd = [
            'ffmpeg', '-y',
            '-i', processed_path,
            '-i', reference_path,
            '-lavfi', f'libvmaf=log_fmt=json:log_path={vmaf_output}',
            '-f', 'null',
            '-'
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"   FFmpeg error: {result.stderr}")
            return 0.0
        
        # Read VMAF score from JSON output
        if os.path.exists(vmaf_output):
            with open(vmaf_output, 'r') as f:
                vmaf_data = json.load(f)
            
            # Extract VMAF score
            if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
                vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']
                print(f"   ✅ VMAF Score: {vmaf_score:.2f}")
                return vmaf_score
            else:
                print("   ⚠️ No VMAF score found in output")
                return 0.0
        else:
            print("   ⚠️ VMAF output file not created")
            return 0.0
            
    except subprocess.TimeoutExpired:
        print("   ❌ VMAF calculation timed out")
        return 0.0
    except Exception as e:
        print(f"   ❌ VMAF calculation error: {e}")
        return 0.0
    finally:
        # Clean up temporary file
        if os.path.exists(vmaf_output):
            os.unlink(vmaf_output)

def calculate_compression_ratio(reference_path, processed_path):
    """Calculate compression ratio"""
    try:
        ref_size = os.path.getsize(reference_path)
        proc_size = os.path.getsize(processed_path)
        ratio = ref_size / proc_size if proc_size > 0 else 0
        print(f"   📊 Compression Ratio: {ratio:.2f}x")
        return ratio
    except Exception as e:
        print(f"   ❌ Error calculating compression ratio: {e}")
        return 0.0

def calculate_upscaling_score(vmaf_score, vmaf_threshold=85.0):
    """Calculate upscaling score based on VMAF"""
    if vmaf_score < vmaf_threshold:
        return 0.0
    
    # Simple scoring: VMAF above threshold gets score based on how much above
    excess = vmaf_score - vmaf_threshold
    max_excess = 100.0 - vmaf_threshold  # 15 points above threshold
    score = min(excess / max_excess, 1.0)  # Normalize to 0-1
    
    print(f"   📈 Upscaling Score: {score:.4f}")
    return score

def calculate_compression_score(vmaf_score, compression_ratio, vmaf_threshold=85.0):
    """Calculate compression score based on VMAF and compression ratio"""
    if vmaf_score < vmaf_threshold:
        return 0.0
    
    # Weighted score: 70% compression ratio, 30% VMAF quality
    compression_weight = 0.7
    quality_weight = 0.3
    
    # Normalize compression ratio (higher is better, cap at 10x)
    comp_score = min(compression_ratio / 10.0, 1.0)
    
    # Normalize VMAF (higher is better)
    vmaf_score_norm = min((vmaf_score - vmaf_threshold) / (100.0 - vmaf_threshold), 1.0)
    
    final_score = (compression_weight * comp_score) + (quality_weight * vmaf_score_norm)
    
    print(f"   📈 Compression Score: {final_score:.4f}")
    return final_score

def validate_video_pair(reference_path, processed_path, task_type, vmaf_threshold=85.0):
    """Validate a video pair and return scores"""
    print(f"\n🎬 Validating {task_type.upper()}...")
    print(f"   📹 Reference: {reference_path}")
    print(f"   🎬 Processed: {processed_path}")
    
    # Check files exist
    if not os.path.exists(reference_path):
        print(f"   ❌ Reference file not found: {reference_path}")
        return False, 0.0
    
    if not os.path.exists(processed_path):
        print(f"   ❌ Processed file not found: {processed_path}")
        return False, 0.0
    
    # Calculate VMAF
    vmaf_score = calculate_vmaf_ffmpeg(reference_path, processed_path)
    
    if task_type.lower() == "upscaling":
        # Upscaling validation
        score = calculate_upscaling_score(vmaf_score, vmaf_threshold)
        success = vmaf_score >= vmaf_threshold
        
    elif task_type.lower() == "compression":
        # Compression validation
        compression_ratio = calculate_compression_ratio(reference_path, processed_path)
        score = calculate_compression_score(vmaf_score, compression_ratio, vmaf_threshold)
        success = vmaf_score >= vmaf_threshold
        
    else:
        print(f"   ❌ Unknown task type: {task_type}")
        return False, 0.0
    
    print(f"   🎯 Final Score: {score:.4f}")
    print(f"   ✅ Success: {'YES' if success else 'NO'}")
    
    return success, score

def main():
    print("======================================================================")
    print("  WORKING VIDEO VALIDATION WITH REAL SCORES")
    print("======================================================================")
    
    # Test upscaling
    print("\n🔍 Testing Upscaling Validation...")
    upscaling_success, upscaling_score = validate_video_pair(
        "test_input.mp4", 
        "test_upscaled.mp4", 
        "upscaling", 
        vmaf_threshold=85.0
    )
    
    # Test compression
    print("\n🔍 Testing Compression Validation...")
    compression_success, compression_score = validate_video_pair(
        "test_input.mp4", 
        "test_compressed.mp4", 
        "compression", 
        vmaf_threshold=85.0
    )
    
    # Results summary
    print("\n======================================================================")
    print("  VALIDATION RESULTS")
    print("======================================================================")
    print(f"Upscaling:   {'✅ PASS' if upscaling_success else '❌ FAIL'} (Score: {upscaling_score:.4f})")
    print(f"Compression: {'✅ PASS' if compression_success else '❌ FAIL'} (Score: {compression_score:.4f})")
    
    if upscaling_success and compression_success:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The Tier 2 algorithms are producing high-quality results.")
    else:
        print("\n⚠️ Some validations failed, but this shows the scoring system works.")
    
    return 0 if (upscaling_success and compression_success) else 1

if __name__ == "__main__":
    sys.exit(main())
