#!/usr/bin/env python3
"""
Simple upscaling service test - uploads video directly
"""

import sys
import subprocess
from pathlib import Path

# Create a simple test video
test_video = Path("test_video_simple.mp4")

print("Creating test video...")
cmd = [
    'ffmpeg', '-y', '-f', 'lavfi',
    '-i', 'testsrc=duration=5:size=640x480:rate=30',
    '-pix_fmt', 'yuv420p',
    '-c:v', 'libx264',
    str(test_video)
]

subprocess.run(cmd, capture_output=True, check=True)
print(f"✅ Created: {test_video}")

# Get video info
print("\nGetting video info...")
cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(test_video)]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[:200])

# Test Video2X directly
print("\n" + "="*80)
print("Testing Video2X directly...")
print("="*80)

output_video = Path("test_output_upscaled.mp4")

video2x_cmd = [
    'video2x',
    '-i', str(test_video),
    '-o', str(output_video),
    '-p', 'realesrgan',
    '-s', '2',
    '-c', 'libx264',
    '-e', 'preset=slow',
    '-e', 'crf=28'
]

print(f"Running: {' '.join(video2x_cmd)}")

try:
    result = subprocess.run(video2x_cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print(f"✅ Video2X SUCCESS!")
        print(f"✅ Output: {output_video}")
        
        # Get output info
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(output_video)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        import json
        data = json.loads(result.stdout)
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        print(f"✅ Output resolution: {video_stream['width']}x{video_stream['height']}")
        
    else:
        print(f"✘ Video2X FAILED")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print(f"✘ Video2X TIMEOUT (>60s)")
    sys.exit(1)
except FileNotFoundError:
    print(f"✘ Video2X not found - is it installed?")
    print(f"Install with: wget https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb && sudo dpkg -i video2x-linux-ubuntu2404-amd64.deb")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - Video2X upscaling is working!")
print("="*80)


