#!/usr/bin/env python3
"""
Download and Check R2 Video
Download video from R2 and check its format/playability
"""

import os
import subprocess
from minio import Minio
from datetime import datetime, timedelta

# R2 credentials
R2_ACCESS_KEY = "eacd80b3fcf1a6d0c98572134a05680c"
R2_SECRET_KEY = "b5a4e900e924f029dd804f096a765c9cc590d997bb25aa7dd0e0fa78ca36ed5f"
R2_ENDPOINT = "ed23ab68357ef85e85a67a8fa27fab47.r2.cloudflarestorage.com"
R2_BUCKET = "vidaio"

def download_and_check_video():
    """Download video from R2 and check its format"""
    
    # Initialize R2 client
    client = Minio(R2_ENDPOINT, 
                   access_key=R2_ACCESS_KEY, 
                   secret_key=R2_SECRET_KEY)
    
    # List objects in R2 bucket
    print("🔍 Checking R2 bucket contents...")
    objects = list(client.list_objects(R2_BUCKET, prefix='SD2HD_upscaled_'))
    
    if not objects:
        print("❌ No SD2HD videos found in R2 bucket")
        return
    
    # Get the latest video
    latest_obj = max(objects, key=lambda obj: obj.last_modified)
    print(f"📹 Latest video: {latest_obj.object_name}")
    print(f"📊 Size: {latest_obj.size} bytes ({latest_obj.size / (1024*1024):.2f} MB)")
    print(f"📅 Modified: {latest_obj.last_modified}")
    
    # Download the video
    local_path = f"/workspace/vidaio-win/downloaded_{latest_obj.object_name}"
    print(f"⬇️  Downloading to: {local_path}")
    
    try:
        client.fget_object(R2_BUCKET, latest_obj.object_name, local_path)
        print("✅ Download successful")
        
        # Check if file exists and get size
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"📊 Downloaded file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
            
            # Check video format with ffprobe
            print("\n🔍 Checking video format with ffprobe...")
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_format', '-show_streams', local_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    
                    if 'streams' in data and data['streams']:
                        video_stream = data['streams'][0]
                        print("✅ Video format analysis:")
                        print(f"   📺 Codec: {video_stream.get('codec_name', 'unknown')}")
                        print(f"   📐 Resolution: {video_stream.get('width', '?')}x{video_stream.get('height', '?')}")
                        print(f"   ⏱️  Duration: {video_stream.get('duration', '?')} seconds")
                        print(f"   🎬 FPS: {video_stream.get('r_frame_rate', '?')}")
                        print(f"   📊 Bitrate: {video_stream.get('bit_rate', '?')} bps")
                        print(f"   🎨 Pixel format: {video_stream.get('pix_fmt', '?')}")
                        print(f"   📱 Profile: {video_stream.get('profile', '?')}")
                        
                        # Check if it's playable
                        print(f"\n🎮 Testing playability...")
                        play_result = subprocess.run([
                            'ffplay', '-v', 'quiet', '-t', '2', '-autoexit', local_path
                        ], capture_output=True, text=True, timeout=10)
                        
                        if play_result.returncode == 0:
                            print("✅ Video is playable!")
                        else:
                            print("❌ Video playback failed")
                            print(f"Error: {play_result.stderr}")
                    else:
                        print("❌ No video streams found")
                else:
                    print(f"❌ ffprobe failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("⏰ ffprobe timed out")
            except Exception as e:
                print(f"❌ Error checking video: {e}")
                
        else:
            print("❌ Downloaded file not found")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    download_and_check_video()



