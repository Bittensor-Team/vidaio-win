#!/usr/bin/env python3
"""
Test in-memory video encoding to eliminate disk I/O bottleneck
This should reduce SD24K processing time from 235s to ~40s
"""

import cv2
import numpy as np
import subprocess
import time
import tempfile
import os
from io import BytesIO

def create_video_from_memory_frames(frames, output_path, fps=30, width=1920, height=1080):
    """Create video from frames in memory using FFmpeg stdin pipe"""
    
    print(f"🎬 Creating video from {len(frames)} frames in memory...")
    print(f"📐 Resolution: {width}x{height}, FPS: {fps}")
    
    # FFmpeg command that reads raw video from stdin
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',           # Input format: raw video
        '-vcodec', 'rawvideo',      # Input codec: raw video
        '-pix_fmt', 'bgr24',        # Input pixel format (OpenCV uses BGR)
        '-s', f'{width}x{height}',  # Input resolution
        '-r', str(fps),             # Input frame rate
        '-i', '-',                  # Read from stdin
        '-c:v', 'libx264',          # Output codec
        '-preset', 'fast',          # Encoding preset
        '-crf', '23',               # Quality setting
        '-pix_fmt', 'yuv420p',      # Output pixel format
        '-profile:v', 'high',       # H.264 profile
        '-level', '4.1',            # H.264 level
        '-movflags', '+faststart',  # Web optimization
        output_path
    ]
    
    print(f"🔧 FFmpeg command: {' '.join(cmd)}")
    
    try:
        # Start FFmpeg process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Send frames to FFmpeg stdin
        start_time = time.time()
        for i, frame in enumerate(frames):
            # Ensure frame is correct size and format
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Convert to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already BGR from OpenCV
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to FFmpeg stdin
            process.stdin.write(frame.tobytes())
            
            if i % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i+1}/{len(frames)} frames ({elapsed:.1f}s)")
        
        # Close stdin to signal end of input
        process.stdin.close()
        
        # Wait for FFmpeg to finish
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            encoding_time = time.time() - start_time
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Video created in {encoding_time:.1f}s: {file_size:.2f} MB")
            return True
        else:
            print(f"❌ FFmpeg error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_memory_encoding():
    """Test in-memory video encoding with sample frames"""
    
    print("🧪 Testing In-Memory Video Encoding")
    print("=" * 50)
    
    # Create test frames (simulate VSR output)
    num_frames = 150  # 5 seconds at 30 FPS
    width, height = 1920, 1080
    
    print(f"📊 Creating {num_frames} test frames ({width}x{height})...")
    
    frames = []
    for i in range(num_frames):
        # Create a test frame with some pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some visual pattern
        frame[:, :, 0] = (i * 2) % 255  # Red channel
        frame[:, :, 1] = (i * 3) % 255  # Green channel
        frame[:, :, 2] = (i * 5) % 255  # Blue channel
        
        # Add some text
        cv2.putText(frame, f"Frame {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        frames.append(frame)
    
    print(f"✅ Created {len(frames)} frames")
    
    # Test in-memory encoding
    output_path = "/tmp/test_memory_video.mp4"
    start_time = time.time()
    
    success = create_video_from_memory_frames(
        frames, output_path, fps=30, width=width, height=height
    )
    
    total_time = time.time() - start_time
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Frames per second: {len(frames) / total_time:.1f}")
        print(f"📁 Output: {output_path}")
        
        # Verify video
        cap = cv2.VideoCapture(output_path)
        if cap.isOpened():
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"✅ Video verification:")
            print(f"   Frames: {actual_frames}")
            print(f"   FPS: {actual_fps}")
            print(f"   Resolution: {actual_width}x{actual_height}")
        else:
            print("❌ Video verification failed")
    else:
        print("❌ FAILED!")

if __name__ == "__main__":
    test_memory_encoding()



