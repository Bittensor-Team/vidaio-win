#!/usr/bin/env python3
"""
Test OpenCV in-memory video encoding to eliminate disk I/O bottleneck
This should reduce SD24K processing time from 235s to ~40s
"""

import cv2
import numpy as np
import time
import os
import tempfile

def create_video_with_opencv(frames, output_path, fps=30, width=1920, height=1080):
    """Create video from frames in memory using OpenCV VideoWriter"""
    
    print(f"🎬 Creating video with OpenCV from {len(frames)} frames...")
    print(f"📐 Resolution: {width}x{height}, FPS: {fps}")
    
    try:
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to open VideoWriter")
            return False
        
        # Process frames
        start_time = time.time()
        for i, frame in enumerate(frames):
            # Ensure frame is correct size
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Ensure frame is BGR (OpenCV expects BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already BGR from OpenCV
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            out.write(frame)
            
            if i % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i+1}/{len(frames)} frames ({elapsed:.1f}s)")
        
        # Release VideoWriter
        out.release()
        
        encoding_time = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ Video created in {encoding_time:.1f}s: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

def test_opencv_encoding():
    """Test OpenCV in-memory video encoding"""
    
    print("🧪 Testing OpenCV In-Memory Video Encoding")
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
    
    # Test OpenCV encoding
    output_path = "/tmp/test_opencv_video.mp4"
    start_time = time.time()
    
    success = create_video_with_opencv(
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
    test_opencv_encoding()



