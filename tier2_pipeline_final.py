#!/usr/bin/env python3
"""
Tier 2 Advanced Pipeline - FINAL WORKING VERSION
Uses woman.mp4 as reference and working upscaler
"""

import os
import sys
import cv2
import subprocess
import time
from pathlib import Path
import json
import tempfile
from datetime import datetime

# Add working upscaler to path
sys.path.insert(0, '/workspace/vidaio-subnet/Real-ESRGAN-working-update')

class Tier2Pipeline:
    def __init__(self, reference_video="/workspace/vidaio-subnet/woman.mp4"):
        self.reference_video = reference_video
        self.output_dir = "/workspace/vidaio-subnet/tier2_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def step_1_extract_frames(self):
        """Extract frames from reference video for upscaling"""
        self.log("="*80)
        self.log("STEP 1: EXTRACT FRAMES FOR UPSCALING")
        self.log("="*80)
        
        # Create temp directory for frames
        self.frames_dir = os.path.join(self.output_dir, f"frames_{self.timestamp}")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Get video info
        cap = cv2.VideoCapture(self.reference_video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"📹 Video: {self.width}x{self.height} @ {self.fps} fps")
        self.log(f"📊 Total frames: {self.total_frames}")
        self.log(f"⏱️  Duration: {self.total_frames / self.fps:.1f} seconds")
        
        # Sample every Nth frame to balance speed and quality
        sample_rate = max(1, self.total_frames // 100)
        frame_count = 0
        extracted = 0
        
        self.log(f"🎬 Sampling every {sample_rate} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_path = os.path.join(self.frames_dir, f"frame_{extracted:06d}.png")
                cv2.imwrite(frame_path, frame)
                extracted += 1
                
                if extracted % 10 == 0:
                    self.log(f"   ✓ Extracted {extracted} frames")
            
            frame_count += 1
        
        cap.release()
        self.log(f"✅ Extracted {extracted} sample frames to {self.frames_dir}")
        self.results['extracted_frames'] = extracted
        return True
    
    def step_2_upscale_frames_realesrgan(self, scale=2):
        """Upscale frames using Real-ESRGAN"""
        self.log("\n" + "="*80)
        self.log("STEP 2: UPSCALE FRAMES WITH REAL-ESRGAN")
        self.log("="*80)
        
        upscaled_dir = os.path.join(self.output_dir, f"upscaled_{self.timestamp}")
        os.makedirs(upscaled_dir, exist_ok=True)
        
        try:
            from RealESRGAN import RealESRGAN
            import torch
            from PIL import Image
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log(f"🖥️  Device: {device}")
            
            # Load model
            self.log("🤖 Loading Real-ESRGAN model...")
            model = RealESRGAN(device, scale=scale)
            model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
            self.log("✅ Model loaded")
            
            # Process frames
            frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            total_frames = len(frames)
            
            self.log(f"🎬 Upscaling {total_frames} frames...")
            start_time = time.time()
            
            upscaled_files = []
            for idx, frame_file in enumerate(frames):
                frame_path = os.path.join(self.frames_dir, frame_file)
                output_path = os.path.join(upscaled_dir, f"upscaled_{idx:06d}.png")
                
                # Upscale
                image = Image.open(frame_path).convert('RGB')
                sr_image = model.predict(image, batch_size=1, patches_size=192)
                sr_image.save(output_path)
                upscaled_files.append(output_path)
                
                if (idx + 1) % 5 == 0:
                    self.log(f"   ✓ {idx + 1}/{total_frames} frames upscaled")
            
            elapsed = time.time() - start_time
            self.log(f"✅ Upscaling complete in {elapsed:.1f}s")
            self.log(f"   Average: {elapsed/total_frames:.2f}s per frame")
            
            self.upscaled_dir = upscaled_dir
            self.results['upscaled_frames'] = len(upscaled_files)
            self.results['upscaling_time_sec'] = elapsed
            return True
            
        except Exception as e:
            self.log(f"❌ ESRGAN Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_3_reconstruct_video_upscaled(self):
        """Reconstruct upscaled video from frames"""
        self.log("\n" + "="*80)
        self.log("STEP 3: RECONSTRUCT UPSCALED VIDEO")
        self.log("="*80)
        
        self.upscaled_video = os.path.join(self.output_dir, f"upscaled_{self.timestamp}.mp4")
        
        # Get upscaled frame info
        upscaled_frames = sorted([f for f in os.listdir(self.upscaled_dir) if f.endswith('.png')])
        
        if not upscaled_frames:
            self.log("❌ No upscaled frames found")
            return False
        
        # Get frame dimensions
        import cv2
        first_frame = cv2.imread(os.path.join(self.upscaled_dir, upscaled_frames[0]))
        h, w = first_frame.shape[:2]
        
        self.log(f"📐 Upscaled resolution: {w}x{h}")
        self.log(f"📊 Frames: {len(upscaled_frames)}")
        
        # Use ffmpeg to create video from frames
        self.log("🎬 Creating video from frames...")
        
        frame_pattern = os.path.join(self.upscaled_dir, "upscaled_%06d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            self.upscaled_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(self.upscaled_video):
            size_mb = os.path.getsize(self.upscaled_video) / 1024 / 1024
            self.log(f"✅ Upscaled video created: {self.upscaled_video}")
            self.log(f"   Size: {size_mb:.2f} MB")
            self.results['upscaled_video_size_mb'] = size_mb
            return True
        else:
            self.log(f"❌ Failed to create video: {result.stderr}")
            return False
    
    def step_4_compress_video_av1(self, crf=30, preset=4):
        """Compress upscaled video with AV1"""
        self.log("\n" + "="*80)
        self.log("STEP 4: COMPRESS VIDEO WITH AV1")
        self.log("="*80)
        
        self.compressed_video = os.path.join(self.output_dir, f"compressed_av1_{self.timestamp}.mp4")
        
        # Use upscaled video as input (simulating the workflow)
        input_video = self.upscaled_video if os.path.exists(self.upscaled_video) else self.reference_video
        
        self.log(f"📹 Input: {input_video}")
        self.log(f"🔧 AV1 CRF: {crf}, Preset: {preset}")
        
        # Detect scene cuts for better encoding
        self.log("🔍 Detecting scene cuts...")
        try:
            cmd = [
                'ffmpeg', '-i', input_video,
                '-vf', 'select=gt(scene\\,0.4),metadata=print_file:-',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            keyframes = []
            for line in result.stderr.split('\n'):
                if 'Parsed_select' in line and 'pts_time' in line:
                    try:
                        time_str = line.split('pts_time:')[1].split(' ')[0]
                        frame_num = int(float(time_str) * self.fps)
                        keyframes.append(frame_num)
                    except:
                        pass
            
            self.log(f"   Found {len(keyframes)} scene cuts")
        except Exception as e:
            self.log(f"   Scene detection failed, using auto keyframes")
            keyframes = []
        
        # Compress with AV1
        self.log("⚡ Compressing with AV1...")
        start_time = time.time()
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-c:v', 'libaom-av1',
            '-crf', str(crf),
            '-preset', str(preset),
            '-c:a', 'aac',
            '-b:a', '128k',
            self.compressed_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(self.compressed_video):
            ref_size = os.path.getsize(input_video)
            comp_size = os.path.getsize(self.compressed_video)
            ratio = ref_size / comp_size
            
            self.log(f"✅ Compression complete in {elapsed:.1f}s")
            self.log(f"   Reference: {ref_size/1024/1024:.2f} MB")
            self.log(f"   Compressed: {comp_size/1024/1024:.2f} MB")
            self.log(f"   Ratio: {ratio:.2f}x")
            
            self.results['compression_ratio'] = ratio
            self.results['compression_time_sec'] = elapsed
            self.results['compressed_size_mb'] = comp_size / 1024 / 1024
            return True
        else:
            self.log(f"❌ Compression failed")
            return False
    
    def step_5_validate_scores(self):
        """Calculate VMAF and PIE-APP scores"""
        self.log("\n" + "="*80)
        self.log("STEP 5: VALIDATE SCORES")
        self.log("="*80)
        
        try:
            # Import local validation
            sys.path.insert(0, '/workspace/vidaio-subnet')
            from local_validation import vmaf_metric, pie_app_metric, calculate_score
            
            # For upscaled video, scale to reference resolution for comparison
            if os.path.exists(self.upscaled_video):
                self.log("📊 Comparing UPSCALING...")
                
                # Scale upscaled video back to reference resolution
                upscaled_scaled = os.path.join(self.output_dir, f"upscaled_scaled_{self.timestamp}.mp4")
                
                self.log(f"   Scaling upscaled video to {self.width}x{self.height}...")
                cmd = [
                    'ffmpeg', '-y', '-i', self.upscaled_video,
                    '-vf', f'scale={self.width}:{self.height}',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    upscaled_scaled
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log("   ✓ Calculating VMAF...")
                    vmaf_result = vmaf_metric(self.reference_video, upscaled_scaled)
                    
                    if vmaf_result and vmaf_result['vmaf_score'] > 0:
                        self.results['upscaling_vmaf'] = vmaf_result['vmaf_score']
                        self.log(f"   ✅ VMAF: {vmaf_result['vmaf_score']:.2f}")
                    
                    self.log("   ✓ Calculating PIE-APP...")
                    pie_result = pie_app_metric(self.reference_video, upscaled_scaled)
                    if pie_result and pie_result['pie_app_score'] > 0:
                        self.results['upscaling_pie_app'] = pie_result['pie_app_score']
                        self.log(f"   ✅ PIE-APP: {pie_result['pie_app_score']:.4f}")
            
            # For compressed video
            if os.path.exists(self.compressed_video):
                self.log("📊 Comparing COMPRESSION...")
                
                self.log("   ✓ Calculating VMAF...")
                vmaf_result = vmaf_metric(self.reference_video, self.compressed_video)
                
                if vmaf_result and vmaf_result['vmaf_score'] > 0:
                    self.results['compression_vmaf'] = vmaf_result['vmaf_score']
                    self.log(f"   ✅ VMAF: {vmaf_result['vmaf_score']:.2f}")
                
                self.log("   ✓ Calculating PIE-APP...")
                pie_result = pie_app_metric(self.reference_video, self.compressed_video)
                if pie_result and pie_result['pie_app_score'] > 0:
                    self.results['compression_pie_app'] = pie_result['pie_app_score']
                    self.log(f"   ✅ PIE-APP: {pie_result['pie_app_score']:.4f}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Validation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_6_calculate_final_scores(self):
        """Calculate final scores per Vidaio formula"""
        self.log("\n" + "="*80)
        self.log("STEP 6: CALCULATE FINAL SCORES")
        self.log("="*80)
        
        # Upscaling score
        if 'upscaling_vmaf' in self.results:
            vmaf = self.results['upscaling_vmaf']
            duration_sec = self.total_frames / self.fps
            
            # Quality component (0-100 based on VMAF)
            quality_score = (vmaf / 100) * 100
            
            # Length bonus (logarithmic)
            length_bonus = min(50, 10 * (1 + 0.1 * min(duration_sec / 320, 1)))
            
            upscaling_final = quality_score + length_bonus
            self.results['upscaling_final_score'] = upscaling_final
            
            self.log(f"📈 UPSCALING FINAL SCORE: {upscaling_final:.2f}")
            self.log(f"   - Quality: {quality_score:.2f} (VMAF: {vmaf:.2f})")
            self.log(f"   - Length Bonus: {length_bonus:.2f}")
        
        # Compression score
        if 'compression_vmaf' in self.results and 'compression_ratio' in self.results:
            vmaf = self.results['compression_vmaf']
            ratio = self.results['compression_ratio']
            
            # Quality component
            quality_comp = max(0, (vmaf - 85) / 15) * 50
            
            # Compression component
            compression_comp = min(50, (ratio - 1) / 3 * 50)
            
            compression_final = quality_comp + compression_comp
            self.results['compression_final_score'] = compression_final
            
            self.log(f"📈 COMPRESSION FINAL SCORE: {compression_final:.2f}")
            self.log(f"   - Quality: {quality_comp:.2f} (VMAF: {vmaf:.2f})")
            self.log(f"   - Compression: {compression_comp:.2f} (Ratio: {ratio:.2f}x)")
    
    def print_summary(self):
        """Print final summary"""
        self.log("\n" + "="*80)
        self.log("FINAL SUMMARY - TIER 2 ADVANCED PIPELINE")
        self.log("="*80)
        
        self.log(f"\n📊 RESULTS:")
        for key, value in self.results.items():
            if isinstance(value, float):
                self.log(f"   {key}: {value:.2f}")
            else:
                self.log(f"   {key}: {value}")
        
        # Save results to JSON
        results_file = os.path.join(self.output_dir, f"results_{self.timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log(f"\n💾 Results saved to: {results_file}")
        
        self.log(f"\n📂 Output directory: {self.output_dir}")
    
    def run(self):
        """Run the full pipeline"""
        self.log("🚀 TIER 2 ADVANCED PIPELINE - STARTING")
        self.log(f"📹 Reference: {self.reference_video}")
        self.log(f"📂 Output: {self.output_dir}")
        
        try:
            if not self.step_1_extract_frames():
                raise Exception("Frame extraction failed")
            
            if not self.step_2_upscale_frames_realesrgan(scale=2):
                raise Exception("Upscaling failed")
            
            if not self.step_3_reconstruct_video_upscaled():
                raise Exception("Video reconstruction failed")
            
            if not self.step_4_compress_video_av1(crf=30, preset=4):
                raise Exception("Compression failed")
            
            self.step_5_validate_scores()
            self.step_6_calculate_final_scores()
            
            self.print_summary()
            self.log("\n✅ PIPELINE COMPLETE!")
            
        except Exception as e:
            self.log(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    pipeline = Tier2Pipeline()
    pipeline.run()
