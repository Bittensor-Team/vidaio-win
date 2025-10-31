#!/usr/bin/env python3
"""
Vidaio Scoring Validator
Validates processed videos using the Vidaio subnet scoring system
"""

import os
import cv2
import numpy as np
import subprocess
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil
from loguru import logger

class VidaioScoringValidator:
    """Validates videos using Vidaio subnet scoring formulas"""
    
    def __init__(self):
        self.temp_dir = "/tmp/vidaio_scoring"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_video_info(self, video_path: str) -> Dict:
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
                logger.error(f"ffprobe failed: {result.stderr}")
                return {}
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def calculate_vmaf(self, reference_path: str, distorted_path: str) -> float:
        """Calculate VMAF score between reference and distorted videos"""
        try:
            # Create temporary files for VMAF calculation
            ref_yuv = os.path.join(self.temp_dir, "ref.yuv")
            dist_yuv = os.path.join(self.temp_dir, "dist.yuv")
            
            # Convert to YUV format for VMAF calculation
            ref_cmd = [
                'ffmpeg', '-y', '-i', reference_path,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease',
                '-pix_fmt', 'yuv420p', '-f', 'rawvideo', ref_yuv
            ]
            
            dist_cmd = [
                'ffmpeg', '-y', '-i', distorted_path,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease', 
                '-pix_fmt', 'yuv420p', '-f', 'rawvideo', dist_yuv
            ]
            
            # Convert videos
            subprocess.run(ref_cmd, capture_output=True, check=True)
            subprocess.run(dist_cmd, capture_output=True, check=True)
            
            # Calculate VMAF using ffmpeg
            vmaf_cmd = [
                'ffmpeg', '-y',
                '-i', reference_path,
                '-i', distorted_path,
                '-lavfi', 'libvmaf=log_path=/tmp/vmaf.json:log_fmt=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(vmaf_cmd, capture_output=True, text=True)
            
            # Parse VMAF score from logs
            vmaf_score = 0.0
            if os.path.exists('/tmp/vmaf.json'):
                with open('/tmp/vmaf.json', 'r') as f:
                    vmaf_data = json.load(f)
                    if 'frames' in vmaf_data and vmaf_data['frames']:
                        vmaf_score = vmaf_data['frames'][0].get('metrics', {}).get('vmaf', 0.0)
            
            # Cleanup
            for file_path in [ref_yuv, dist_yuv, '/tmp/vmaf.json']:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return vmaf_score
            
        except Exception as e:
            logger.error(f"VMAF calculation failed: {e}")
            return 0.0
    
    def calculate_pie_app(self, reference_path: str, distorted_path: str) -> float:
        """Calculate PIE-APP perceptual quality score"""
        try:
            # For now, use a simplified PIE-APP approximation
            # In production, you'd use the actual PIE-APP implementation
            
            # Read reference and distorted videos
            ref_cap = cv2.VideoCapture(reference_path)
            dist_cap = cv2.VideoCapture(distorted_path)
            
            if not ref_cap.isOpened() or not dist_cap.isOpened():
                return 0.0
            
            # Get video properties
            ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
            dist_fps = dist_cap.get(cv2.CAP_PROP_FPS)
            
            # Simple quality approximation based on resolution and frame rate
            ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dist_width = int(dist_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            dist_height = int(dist_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
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
            
            # Simplified PIE-APP score (0-1 range)
            pie_app_score = min(resolution_factor * fps_factor, 1.0)
            
            return pie_app_score
            
        except Exception as e:
            logger.error(f"PIE-APP calculation failed: {e}")
            return 0.0
    
    def sigmoid_transformation(self, x: float) -> float:
        """Sigmoid transformation for quality score"""
        return 1 / (1 + math.exp(-6 * (x - 0.5)))
    
    def calculate_upscaling_score(self, reference_path: str, processed_path: str, 
                                 content_length: float) -> Dict:
        """Calculate upscaling score using Vidaio formula"""
        
        # Calculate VMAF
        vmaf_score = self.calculate_vmaf(reference_path, processed_path)
        vmaf_percentage = vmaf_score / 100.0  # Convert to 0-1 range
        
        # VMAF check (must be ≥ 50% or S_Q = 0)
        vmaf_check = 1.0 if vmaf_percentage >= 0.5 else 0.0
        
        # Calculate PIE-APP score
        pie_app_score = self.calculate_pie_app(reference_path, processed_path)
        
        # Quality Score (S_Q)
        s_q = self.sigmoid_transformation(pie_app_score) * vmaf_check
        
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
    
    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            info = self.get_video_info(video_path)
            if 'format' in info and 'duration' in info['format']:
                return float(info['format']['duration'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting duration: {e}")
            return 0.0
    
    def validate_video(self, reference_path: str, processed_path: str, 
                      task_type: str) -> Dict:
        """Validate a processed video against reference"""
        
        logger.info(f"🔍 Validating {task_type} video...")
        
        # Get video duration
        duration = self.get_video_duration(processed_path)
        
        # Calculate upscaling score
        score_data = self.calculate_upscaling_score(reference_path, processed_path, duration)
        
        # Get file sizes
        ref_size = os.path.getsize(reference_path) / (1024 * 1024)  # MB
        proc_size = os.path.getsize(processed_path) / (1024 * 1024)  # MB
        
        # Get video resolution
        ref_info = self.get_video_info(reference_path)
        proc_info = self.get_video_info(processed_path)
        
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

def main():
    """Main validation function"""
    validator = VidaioScoringValidator()
    
    # Test with our processed videos
    test_cases = [
        {
            'reference': '/workspace/vidaio-win/elk.mp4',
            'processed': '/workspace/vidaio-win/real_vidaio_tests/SD2HD_10w_5s_10w.mp4',
            'task_type': 'SD2HD'
        },
        {
            'reference': '/workspace/vidaio-win/elk.mp4', 
            'processed': '/workspace/vidaio-win/real_vidaio_tests/HD24K_12w_5s_12w.mp4',
            'task_type': 'HD24K'
        },
        {
            'reference': '/workspace/vidaio-win/elk.mp4',
            'processed': '/workspace/vidaio-win/real_vidaio_tests/4K28K_8w_5s_8w.mp4', 
            'task_type': '4K28K'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        if os.path.exists(test_case['reference']) and os.path.exists(test_case['processed']):
            result = validator.validate_video(
                test_case['reference'],
                test_case['processed'], 
                test_case['task_type']
            )
            results.append(result)
        else:
            logger.warning(f"Missing files for {test_case['task_type']}")
    
    # Print results
    print("\n" + "="*80)
    print("VIDAIO SCORING VALIDATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n📹 {result['task_type']} Video Validation")
        print(f"   Reference: {result['reference_resolution']} ({result['reference_size_mb']:.2f} MB)")
        print(f"   Processed: {result['processed_resolution']} ({result['processed_size_mb']:.2f} MB)")
        print(f"   Duration: {result['duration_seconds']:.2f} seconds")
        print(f"   Size Ratio: {result['size_ratio']:.2f}x")
        
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

if __name__ == "__main__":
    main()





