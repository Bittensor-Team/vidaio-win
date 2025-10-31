#!/usr/bin/env python3
"""
Create Working Video
Create a working video from the VSR frames
"""

import os
import subprocess
import tempfile
import cv2
import numpy as np

def create_working_video():
    """Create a working video from VSR frames"""
    
    # Use the existing processed video as input
    input_video = "/workspace/vidaio-win/downloaded_SD2HD_upscaled_1761364600.mp4"
    output_video = "/workspace/vidaio-win/working_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    print(f"🔧 Creating working video...")
    print(f"📹 Input: {input_video}")
    print(f"📹 Output: {output_video}")
    
    # Extract frames and recreate video with proper settings
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Temp directory: {temp_dir}")
    
    try:
        # Extract frames
        print("🎬 Extracting frames...")
        extract_cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-vf', 'fps=30',  # Force 30 FPS
            os.path.join(temp_dir, 'frame_%06d.png')
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Frame extraction failed: {result.stderr}")
            return
        
        # Count frames
        frame_files = [f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')]
        print(f"📊 Extracted {len(frame_files)} frames")
        
        if not frame_files:
            print("❌ No frames extracted")
            return
        
        # Create video with proper settings
        print("🎬 Creating video with proper settings...")
        create_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',  # Higher quality
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level', '4.1',
            '-movflags', '+faststart',
            '-r', '30',  # Force output frame rate
            output_video
        ]
        
        print(f"🎬 Running: {' '.join(create_cmd)}")
        result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Video created successfully!")
            
            # Check the video
            print("\n🔍 Checking created video...")
            check_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', output_video
            ]
            
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
            
            if check_result.returncode == 0:
                import json
                data = json.loads(check_result.stdout)
                
                if 'streams' in data and data['streams']:
                    video_stream = data['streams'][0]
                    print(f"✅ Video format:")
                    print(f"   📺 Codec: {video_stream.get('codec_name', 'unknown')}")
                    print(f"   📐 Resolution: {video_stream.get('width', '?')}x{video_stream.get('height', '?')}")
                    print(f"   ⏱️  Duration: {video_stream.get('duration', '?')} seconds")
                    print(f"   🎬 FPS: {video_stream.get('r_frame_rate', '?')}")
                    print(f"   📊 Bitrate: {video_stream.get('bit_rate', '?')} bps")
                    print(f"   🎨 Pixel format: {video_stream.get('pix_fmt', '?')}")
                    
                    # Test playability
                    print(f"\n🎮 Testing playability...")
                    play_cmd = [
                        'ffplay', '-v', 'quiet', '-t', '3', '-autoexit', output_video
                    ]
                    
                    play_result = subprocess.run(play_cmd, capture_output=True, text=True, timeout=15)
                    
                    if play_result.returncode == 0:
                        print("✅ Video is playable!")
                        print(f"📁 Working video saved: {output_video}")
                    else:
                        print("❌ Video still not playable")
                        print(f"Error: {play_result.stderr}")
                else:
                    print("❌ No video streams found")
            else:
                print(f"❌ Failed to check video: {check_result.stderr}")
                
        else:
            print(f"❌ Video creation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    create_working_video()



