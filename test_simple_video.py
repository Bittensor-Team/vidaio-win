#!/usr/bin/env python3
"""
Test Simple Video
Create a simple test video to verify the encoding works
"""

import os
import subprocess
import tempfile
import cv2
import numpy as np

def create_simple_test_video():
    """Create a simple test video to verify encoding works"""
    
    output_video = "/workspace/vidaio-win/simple_test_video.mp4"
    temp_dir = tempfile.mkdtemp()
    
    print(f"🎬 Creating simple test video...")
    print(f"📁 Temp directory: {temp_dir}")
    
    try:
        # Create simple frames (gradient)
        print("🎨 Creating test frames...")
        for i in range(90):  # 3 seconds at 30 FPS
            # Create a simple gradient frame
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
            
            # Add a moving gradient
            for y in range(480):
                for x in range(854):
                    frame[y, x] = [
                        int(255 * (x / 854)),  # Red gradient
                        int(255 * (y / 480)),  # Green gradient
                        int(255 * (i / 90))    # Blue changes over time
                    ]
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{i:06d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        print(f"📊 Created 90 test frames")
        
        # Create video
        print("🎬 Creating video...")
        create_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-movflags', '+faststart',
            output_video
        ]
        
        result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Simple test video created!")
            
            # Test playability
            print("🎮 Testing playability...")
            play_cmd = [
                'ffplay', '-v', 'quiet', '-t', '2', '-autoexit', output_video
            ]
            
            play_result = subprocess.run(play_cmd, capture_output=True, text=True, timeout=10)
            
            if play_result.returncode == 0:
                print("✅ Simple test video is playable!")
                print("📁 This means the encoding process works correctly")
                print("❌ The issue is likely with the VSR model output")
            else:
                print("❌ Simple test video is not playable")
                print("❌ There's an issue with the video encoding process")
                print(f"Error: {play_result.stderr}")
        else:
            print(f"❌ Video creation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    create_simple_test_video()



