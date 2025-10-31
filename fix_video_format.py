#!/usr/bin/env python3
"""
Fix Video Format
Fix the video format issues
"""

import subprocess
import os

def fix_video_format():
    """Fix the video format to make it playable"""
    
    input_video = "/workspace/vidaio-win/downloaded_SD2HD_upscaled_1761364600.mp4"
    output_video = "/workspace/vidaio-win/fixed_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    print(f"🔧 Fixing video format...")
    print(f"📹 Input: {input_video}")
    print(f"📹 Output: {output_video}")
    
    # Fix the video with proper FPS and encoding
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-c:v', 'libx264',  # Use H.264 instead of HEVC for better compatibility
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-r', '30',  # Force 30 FPS
        '-movflags', '+faststart',
        output_video
    ]
    
    print(f"🎬 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Video fixed successfully!")
            
            # Check the fixed video
            print("\n🔍 Checking fixed video...")
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
                    print(f"✅ Fixed video format:")
                    print(f"   📺 Codec: {video_stream.get('codec_name', 'unknown')}")
                    print(f"   📐 Resolution: {video_stream.get('width', '?')}x{video_stream.get('height', '?')}")
                    print(f"   ⏱️  Duration: {video_stream.get('duration', '?')} seconds")
                    print(f"   🎬 FPS: {video_stream.get('r_frame_rate', '?')}")
                    print(f"   📊 Bitrate: {video_stream.get('bit_rate', '?')} bps")
                    print(f"   🎨 Pixel format: {video_stream.get('pix_fmt', '?')}")
                    
                    # Test playability
                    print(f"\n🎮 Testing playability...")
                    play_cmd = [
                        'ffplay', '-v', 'quiet', '-t', '2', '-autoexit', output_video
                    ]
                    
                    play_result = subprocess.run(play_cmd, capture_output=True, text=True, timeout=10)
                    
                    if play_result.returncode == 0:
                        print("✅ Fixed video is playable!")
                    else:
                        print("❌ Fixed video still not playable")
                        print(f"Error: {play_result.stderr}")
                else:
                    print("❌ No video streams found in fixed video")
            else:
                print(f"❌ Failed to check fixed video: {check_result.stderr}")
                
        else:
            print(f"❌ Video fixing failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Video fixing timed out")
    except Exception as e:
        print(f"❌ Error fixing video: {e}")

if __name__ == "__main__":
    fix_video_format()



