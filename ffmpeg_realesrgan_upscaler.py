#!/usr/bin/env python3
"""
FFmpeg + Real-ESRGAN Video Upscaler
Alternative to Video2X for Vidaio subnet
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

# Add Real-ESRGAN to path
sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')

class FFmpegRealESRGANUpscaler:
    def __init__(self):
        self.realesrgan_path = "/workspace/vidaio-win/Real-ESRGAN-working-update"
        
    def upscale_video(self, input_path, output_path, scale=2, task_type="SD2HD"):
        """
        Upscale video using FFmpeg + Real-ESRGAN frame-by-frame processing
        """
        
        # Map task types to scale factors
        scale_map = {
            "SD2HD": 2,
            "HD24K": 2,
            "SD24K": 4, 
            "4K28K": 2
        }
        
        if task_type in scale_map:
            scale = scale_map[task_type]
        
        print(f"🎬 Upscaling video: {input_path}")
        print(f"📊 Scale factor: {scale}x")
        print(f"📁 Output: {output_path}")
        
        try:
            # Get video info
            video_info = self.get_video_info(input_path)
            if not video_info:
                return False
            
            # Extract video properties
            video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
            fps = eval(video_stream['r_frame_rate'])
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            print(f"📐 Original resolution: {width}x{height}")
            print(f"🎞️  FPS: {fps}")
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                upscaled_frames_dir = os.path.join(temp_dir, "upscaled_frames")
                os.makedirs(frames_dir)
                os.makedirs(upscaled_frames_dir)
                
                # Extract frames
                print("📸 Extracting frames...")
                subprocess.run([
                    "ffmpeg", "-i", input_path, "-vf", "fps=30",
                    os.path.join(frames_dir, "frame_%06d.png")
                ], check=True, capture_output=True)
                
                # Upscale frames with Real-ESRGAN
                print("⚡ Upscaling frames with Real-ESRGAN...")
                from working_upscaler import upscale_with_realesrgan
                
                frame_files = sorted(Path(frames_dir).glob("*.png"))
                for i, frame_file in enumerate(frame_files):
                    output_frame = os.path.join(upscaled_frames_dir, f"upscaled_{i:06d}.png")
                    
                    success = upscale_with_realesrgan(
                        str(frame_file), 
                        output_frame, 
                        scale=scale,
                        batch_size=1,
                        patch_size=192
                    )
                    
                    if not success:
                        print(f"❌ Failed to upscale frame {i}")
                        return False
                    
                    if i % 10 == 0:
                        print(f"📊 Processed {i}/{len(frame_files)} frames")
                
                # Reconstruct video
                print("🎬 Reconstructing video...")
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(upscaled_frames_dir, "upscaled_%06d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "28",
                    "-pix_fmt", "yuv420p", output_path
                ], check=True)
                
                print(f"✅ Video upscaled successfully: {output_path}")
                return True
                
        except Exception as e:
            print(f"❌ Error during upscaling: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_video_info(self, video_path):
        """Get video information using FFprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
            
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return None

# Example usage
if __name__ == "__main__":
    upscaler = FFmpegRealESRGANUpscaler()
    
    # Test with a sample video
    input_video = "test_video.mp4"
    output_video = "test_video_upscaled_2x.mp4"
    
    if os.path.exists(input_video):
        success = upscaler.upscale_video(input_video, output_video, scale=2, task_type="SD2HD")
        if success:
            print("🎉 Video upscaling completed successfully!")
        else:
            print("❌ Video upscaling failed!")
    else:
        print(f"❌ Input video not found: {input_video}")





