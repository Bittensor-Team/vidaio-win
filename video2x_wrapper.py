#!/usr/bin/env python3
"""
Video2X Wrapper for Vidaio Subnet
Integrates Video2X with Real-ESRGAN for video upscaling
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

class Video2XUpscaler:
    def __init__(self, video2x_path="~/video2x/Video2X-x86_64.AppImage", 
                 realesrgan_path="/workspace/vidaio-win/Real-ESRGAN-working-update"):
        self.video2x_path = os.path.expanduser(video2x_path)
        self.realesrgan_path = realesrgan_path
        
    def upscale_video(self, input_path, output_path, scale=2, task_type="SD2HD"):
        """
        Upscale video using Video2X + Real-ESRGAN
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            scale: Scale factor (2 or 4)
            task_type: Task type (SD2HD, HD24K, SD24K, 4K28K)
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
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input = os.path.join(temp_dir, "input.mp4")
                temp_output = os.path.join(temp_dir, "output.mp4")
                
                # Copy input to temp directory
                subprocess.run(["cp", input_path, temp_input], check=True)
                
                # Prepare Video2X command
                cmd = [
                    self.video2x_path,
                    "--input", temp_input,
                    "--output", temp_output,
                    "--driver", "realesrgan",
                    "--scale", str(scale),
                    "--format", "mp4",
                    "--quality", "28",
                    "--codec", "libx264",
                    "--preset", "slow"
                ]
                
                print(f"🚀 Running command: {' '.join(cmd)}")
                
                # Run Video2X
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"❌ Video2X failed: {result.stderr}")
                    return False
                
                # Copy output to final destination
                subprocess.run(["cp", temp_output, output_path], check=True)
                
                print(f"✅ Video upscaled successfully: {output_path}")
                return True
                
        except subprocess.TimeoutExpired:
            print("❌ Video2X timed out")
            return False
        except Exception as e:
            print(f"❌ Error during upscaling: {e}")
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
    upscaler = Video2XUpscaler()
    
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





