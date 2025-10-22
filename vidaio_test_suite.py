#!/usr/bin/env python3
"""
Vidaio Subnet Test Suite
Tests all task types with proper validation and scoring
"""

import os
import sys
import cv2
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Add local validation to path
sys.path.insert(0, '/workspace/vidaio-subnet')

class VidaioTestSuite:
    def __init__(self, reference_video="/workspace/vidaio-subnet/elk.mp4"):
        self.reference_video = reference_video
        self.output_dir = "/workspace/vidaio-subnet/vidaio_tests"
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def crop_video(self, input_path, output_path, start_sec=0, duration_sec=5):
        """Crop video to specific duration"""
        self.log(f"✂️  Cropping video: {duration_sec}s from {start_sec}s")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-c', 'copy',  # Fast copy without re-encoding
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            self.log(f"✅ Cropped video created: {os.path.getsize(output_path)/1024/1024:.2f} MB")
            return True
        else:
            self.log(f"❌ Cropping failed: {result.stderr}")
            return False
    
    def get_video_info(self, video_path):
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }
    
    def upscale_video(self, input_path, output_path, target_width, target_height, task_name):
        """Upscale video to target resolution"""
        self.log(f"🖼️  Upscaling {task_name}: {input_path}")
        
        # Get input info
        info = self.get_video_info(input_path)
        self.log(f"   Input: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
        self.log(f"   Target: {target_width}x{target_height}")
        
        # Use FFmpeg for upscaling (simplified for testing)
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', f'scale={target_width}:{target_height}:flags=lanczos',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',  # High quality
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            self.log(f"✅ Upscaled: {size_mb:.2f} MB in {elapsed:.1f}s")
            return True
        else:
            self.log(f"❌ Upscaling failed: {result.stderr}")
            return False
    
    def compress_video(self, input_path, output_path, vmaf_target=90):
        """Compress video with target VMAF"""
        self.log(f"🗜️  Compressing with VMAF target: {vmaf_target}")
        
        # Map VMAF target to CRF
        crf_map = {80: 45, 85: 38, 90: 32, 95: 25}
        crf = crf_map.get(vmaf_target, 32)
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',  # Using H.264 for speed
            '-crf', str(crf),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(output_path):
            ref_size = os.path.getsize(input_path)
            comp_size = os.path.getsize(output_path)
            ratio = ref_size / comp_size
            
            self.log(f"✅ Compressed: {comp_size/1024/1024:.2f} MB, ratio: {ratio:.1f}x in {elapsed:.1f}s")
            return True
        else:
            self.log(f"❌ Compression failed: {result.stderr}")
            return False
    
    def validate_scores(self, reference_path, test_path, task_type):
        """Calculate VMAF and PIE-APP scores"""
        try:
            from local_validation import vmaf_metric, pie_app_metric
            
            self.log(f"📊 Calculating scores for {task_type}...")
            
            # VMAF calculation
            vmaf_result = vmaf_metric(reference_path, test_path)
            if not vmaf_result or vmaf_result['vmaf_score'] <= 0:
                self.log("❌ VMAF calculation failed")
                return None
            
            # PIE-APP calculation
            pie_result = pie_app_metric(reference_path, test_path)
            if not pie_result or pie_result['pie_app_score'] <= 0:
                self.log("❌ PIE-APP calculation failed")
                return None
            
            scores = {
                'vmaf': vmaf_result['vmaf_score'],
                'pie_app': pie_result['pie_app_score'],
                'task_type': task_type
            }
            
            self.log(f"   VMAF: {scores['vmaf']:.2f}")
            self.log(f"   PIE-APP: {scores['pie_app']:.4f}")
            
            return scores
            
        except Exception as e:
            self.log(f"❌ Validation error: {e}")
            return None
    
    def calculate_final_score(self, scores, task_type, duration_sec):
        """Calculate final Vidaio score"""
        if not scores:
            return 0
        
        vmaf = scores['vmaf']
        
        if task_type.startswith('UPSCALING'):
            # Upscaling score formula
            quality_score = (vmaf / 100) * 100
            length_bonus = min(50, 10 * (1 + 0.1 * min(duration_sec / 320, 1)))
            final_score = quality_score + length_bonus
            
        elif task_type.startswith('COMPRESSION'):
            # Compression score formula
            ref_size = scores.get('ref_size_mb', 10)
            comp_size = scores.get('comp_size_mb', 5)
            ratio = ref_size / comp_size
            
            quality_comp = max(0, (vmaf - 85) / 15) * 50
            compression_comp = min(50, (ratio - 1) / 3 * 50)
            final_score = quality_comp + compression_comp
            
        else:
            final_score = vmaf
        
        return final_score
    
    def test_upscaling_tasks(self, input_video):
        """Test all upscaling tasks"""
        self.log("\n" + "="*80)
        self.log("TESTING UPSCALING TASKS")
        self.log("="*80)
        
        info = self.get_video_info(input_video)
        self.log(f"📹 Input video: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
        self.log(f"⏱️  Duration: {info['duration']:.1f} seconds")
        
        upscaling_tasks = [
            ('SD2HD', 1920, 1080, 85),      # 2x upscale, VMAF ≥ 85
            ('HD24K', 3840, 2160, 85),      # 2x upscale, VMAF ≥ 85  
            ('SD24K', 3840, 2160, 85),      # 4x upscale, VMAF ≥ 85
            ('4K28K', 7680, 4320, 85),      # 2x upscale, VMAF ≥ 85
        ]
        
        results = {}
        
        for task_name, target_w, target_h, vmaf_threshold in upscaling_tasks:
            self.log(f"\n🎯 Testing {task_name}...")
            
            output_path = os.path.join(self.output_dir, f"{task_name.lower()}_{self.timestamp}.mp4")
            
            if self.upscale_video(input_video, output_path, target_w, target_h, task_name):
                scores = self.validate_scores(input_video, output_path, f"UPSCALING_{task_name}")
                if scores:
                    final_score = self.calculate_final_score(scores, f"UPSCALING_{task_name}", info['duration'])
                    scores['final_score'] = final_score
                    scores['vmaf_threshold'] = vmaf_threshold
                    scores['meets_threshold'] = scores['vmaf'] >= vmaf_threshold
                    results[task_name] = scores
                    
                    self.log(f"   Final Score: {final_score:.2f}")
                    self.log(f"   Meets VMAF {vmaf_threshold}: {'✅' if scores['meets_threshold'] else '❌'}")
        
        return results
    
    def test_compression_tasks(self, input_video):
        """Test compression tasks with different VMAF targets"""
        self.log("\n" + "="*80)
        self.log("TESTING COMPRESSION TASKS")
        self.log("="*80)
        
        info = self.get_video_info(input_video)
        ref_size_mb = os.path.getsize(input_video) / 1024 / 1024
        
        compression_tasks = [
            ('COMPRESSION_VMAF80', 80, 15),   # Aggressive compression
            ('COMPRESSION_VMAF85', 85, 10),   # Balanced
            ('COMPRESSION_VMAF90', 90, 5),    # Conservative
            ('COMPRESSION_VMAF95', 95, 2),    # Near-lossless
        ]
        
        results = {}
        
        for task_name, vmaf_target, expected_ratio in compression_tasks:
            self.log(f"\n🎯 Testing {task_name}...")
            
            output_path = os.path.join(self.output_dir, f"{task_name.lower()}_{self.timestamp}.mp4")
            
            if self.compress_video(input_video, output_path, vmaf_target):
                scores = self.validate_scores(input_video, output_path, task_name)
                if scores:
                    comp_size_mb = os.path.getsize(output_path) / 1024 / 1024
                    ratio = ref_size_mb / comp_size_mb
                    
                    scores['ref_size_mb'] = ref_size_mb
                    scores['comp_size_mb'] = comp_size_mb
                    scores['compression_ratio'] = ratio
                    scores['expected_ratio'] = expected_ratio
                    
                    final_score = self.calculate_final_score(scores, task_name, info['duration'])
                    scores['final_score'] = final_score
                    scores['meets_threshold'] = scores['vmaf'] >= vmaf_target
                    
                    results[task_name] = scores
                    
                    self.log(f"   Compression Ratio: {ratio:.1f}x (expected: {expected_ratio}x)")
                    self.log(f"   Final Score: {final_score:.2f}")
                    self.log(f"   Meets VMAF {vmaf_target}: {'✅' if scores['meets_threshold'] else '❌'}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        self.log("🚀 VIDAIO SUBNET COMPREHENSIVE TEST SUITE")
        self.log("="*80)
        
        # Test 1: Original 10s video
        self.log(f"\n📹 Testing original video: {self.reference_video}")
        info = self.get_video_info(self.reference_video)
        self.log(f"   Duration: {info['duration']:.1f}s, Resolution: {info['width']}x{info['height']}")
        
        upscaling_results_10s = self.test_upscaling_tasks(self.reference_video)
        compression_results_10s = self.test_compression_tasks(self.reference_video)
        
        # Test 2: 5s cropped video
        self.log(f"\n✂️  Testing 5s cropped video...")
        cropped_video = os.path.join(self.output_dir, f"cropped_5s_{self.timestamp}.mp4")
        
        if self.crop_video(self.reference_video, cropped_video, 0, 5):
            upscaling_results_5s = self.test_upscaling_tasks(cropped_video)
            compression_results_5s = self.test_compression_tasks(cropped_video)
        else:
            upscaling_results_5s = {}
            compression_results_5s = {}
        
        # Compile results
        all_results = {
            'test_info': {
                'timestamp': self.timestamp,
                'reference_video': self.reference_video,
                'original_duration': info['duration'],
                'original_resolution': f"{info['width']}x{info['height']}",
                'original_fps': info['fps']
            },
            'upscaling_10s': upscaling_results_10s,
            'compression_10s': compression_results_10s,
            'upscaling_5s': upscaling_results_5s,
            'compression_5s': compression_results_5s
        }
        
        # Save results
        results_file = os.path.join(self.output_dir, f"vidaio_test_results_{self.timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results):
        """Print test summary"""
        self.log("\n" + "="*80)
        self.log("VIDAIO TEST SUITE SUMMARY")
        self.log("="*80)
        
        for test_type in ['upscaling_10s', 'compression_10s', 'upscaling_5s', 'compression_5s']:
            if test_type in results and results[test_type]:
                self.log(f"\n📊 {test_type.upper()}:")
                for task, scores in results[test_type].items():
                    status = "✅" if scores.get('meets_threshold', False) else "❌"
                    self.log(f"   {task}: VMAF={scores['vmaf']:.1f}, Score={scores['final_score']:.2f} {status}")
        
        self.log(f"\n💾 Full results saved to: {self.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Vidaio Subnet Test Suite")
    parser.add_argument("--video", default="/workspace/vidaio-subnet/elk.mp4", help="Input video path")
    
    args = parser.parse_args()
    
    suite = VidaioTestSuite(reference_video=args.video)
    suite.run_comprehensive_test()

if __name__ == "__main__":
    main()
