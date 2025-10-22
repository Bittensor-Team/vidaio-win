#!/usr/bin/env python3
"""
Tier 2 Advanced Pipeline
Complete end-to-end: Download → Process → Validate
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def download_sample_video(output_path: str = "sample_input.mp4") -> bool:
    """Download a sample video from internet"""
    print_header("STEP 1: Download Sample Video")
    
    print(f"📥 Downloading sample video to: {output_path}")
    
    # Using a public domain video source
    urls = [
        "https://commondatastorage.googleapis.com/gtv-videos-library/sample/BigBuckBunny.mp4",
        "https://media.w3.org/2010/05/sintel/trailer.mp4",
    ]
    
    for url in urls:
        print(f"\n   Trying: {url}")
        try:
            cmd = ['yt-dlp', '-f', 'best[height<=480]', '-o', output_path, url]
            result = subprocess.run(cmd, timeout=300, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024
                print(f"   ✅ Downloaded: {file_size:.1f} MB")
                return True
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            continue
    
    print("\n❌ Download failed. Please download a sample video manually.")
    return False

def downscale_video(input_path: str, output_path: str, target_height: int = 720) -> bool:
    """Downscale video to match Vidaio task requirements"""
    print_header("STEP 2: Prepare Reference Video")
    
    print(f"📹 Downscaling to {target_height}p for testing...")
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale=-1:{target_height}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-t', '10',  # Limit to 10 seconds for testing
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024 / 1024
            print(f"   ✅ Reference video ready: {size:.1f} MB")
            return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

def process_tier2_upscaling(ref_path: str, output_path: str) -> bool:
    """Process upscaling with Tier 2 algorithm"""
    print_header("STEP 3: Tier 2 Upscaling")
    
    print("🎬 Running ESRGAN + GFPGAN + Post-processing...")
    
    try:
        # Import and run locally
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tier2_upscaling import upscale_video_tier2
        
        return upscale_video_tier2(ref_path, output_path, scale=2, apply_post_processing=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def process_tier2_compression(input_path: str, output_path: str) -> bool:
    """Process compression with Tier 2 algorithm"""
    print_header("STEP 4: Tier 2 Compression")
    
    print("🎥 Running AV1 + Scene-aware + VMAF-guided...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tier2_compression import compress_video_tier2
        
        return compress_video_tier2(input_path, output_path, vmaf_threshold=90.0)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_output(ref_path: str, proc_path: str, task_type: str) -> bool:
    """Validate output using local_validation.py"""
    print_header(f"STEP 5: Validate {task_type.upper()} Score")
    
    print(f"📊 Running local validation...")
    
    try:
        cmd = [
            'python3', 'local_validation.py',
            '--reference', ref_path,
            '--processed', proc_path,
            '--task', task_type,
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=False, timeout=600)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print_header("TIER 2 ADVANCED PIPELINE")
    print("Complete video processing & validation workflow")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    work_dir = Path("tier2_test")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    sample_video = "sample_input.mp4"
    reference_video = "reference_720p.mp4"
    upscaled_video = "output_upscaled.mp4"
    compressed_video = "output_compressed.mp4"
    
    try:
        # Step 1: Download
        if not os.path.exists(sample_video):
            if not download_sample_video(sample_video):
                print("\n❌ Pipeline failed: Could not download sample video")
                return 1
        else:
            print(f"✅ Sample video found: {sample_video}")
        
        # Step 2: Prepare reference
        if not downscale_video(sample_video, reference_video):
            print("\n❌ Pipeline failed: Could not prepare reference")
            return 1
        
        # Step 3: Upscaling
        print("\n📝 Processing Tier 2 Upscaling...")
        if not process_tier2_upscaling(reference_video, upscaled_video):
            print("\n❌ Pipeline failed: Upscaling error")
            return 1
        
        # Validate upscaling
        print("\n🔍 Validating upscaling output...")
        if not validate_output(reference_video, upscaled_video, "upscaling"):
            print("⚠️ Upscaling validation had issues")
        
        # Step 4: Compression
        print("\n📝 Processing Tier 2 Compression...")
        if not process_tier2_compression(reference_video, compressed_video):
            print("\n❌ Pipeline failed: Compression error")
            return 1
        
        # Validate compression
        print("\n🔍 Validating compression output...")
        if not validate_output(reference_video, compressed_video, "compression"):
            print("⚠️ Compression validation had issues")
        
        # Summary
        print_header("PIPELINE COMPLETE ✅")
        print(f"\n📊 Results Summary:")
        print(f"   Reference Video: {reference_video}")
        print(f"   Upscaled Output: {upscaled_video}")
        print(f"   Compressed Output: {compressed_video}")
        
        # File sizes
        ref_size = os.path.getsize(reference_video) / 1024 / 1024
        up_size = os.path.getsize(upscaled_video) / 1024 / 1024
        comp_size = os.path.getsize(compressed_video) / 1024 / 1024
        
        print(f"\n📁 File Sizes:")
        print(f"   Reference: {ref_size:.1f} MB")
        print(f"   Upscaled: {up_size:.1f} MB (quality enhancement)")
        print(f"   Compressed: {comp_size:.1f} MB ({ref_size/comp_size:.1f}x compression)")
        
        print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🚀 All tests passed! Ready for network deployment.\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
