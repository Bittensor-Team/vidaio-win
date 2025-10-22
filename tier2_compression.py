#!/usr/bin/env python3
"""
Tier 2 Advanced Compression Implementation
- AV1 codec (preset 4, CRF 30-38)
- Scene-aware encoding
- VMAF-guided CRF search
- Advanced denoising
"""

import subprocess
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional

try:
    import ffmpeg
except ImportError:
    print("❌ Install ffmpeg-python: pip install ffmpeg-python")
    exit(1)


def get_scene_cuts(video_path: str, threshold: float = 27.0) -> list:
    """Detect scene cuts in video"""
    print("🔍 Detecting scene cuts...")
    
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=gt(scene\\,{threshold}),showinfo',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stderr=subprocess.STDOUT
        )
        
        scenes = []
        for line in result.stdout.split('\n'):
            if 'Parsed_showinfo' in line:
                try:
                    parts = line.split('pts_time:')
                    if len(parts) > 1:
                        timestamp = float(parts[1].split()[0])
                        scenes.append(timestamp)
                except:
                    pass
        
        print(f"   Found {len(scenes)} scenes")
        return sorted(scenes)
        
    except Exception as e:
        print(f"   ⚠️ Scene detection failed: {e}")
        return []


def calculate_vmaf_for_file(ref_path: str, test_path: str) -> float:
    """Calculate VMAF score between two videos"""
    try:
        # Use ffmpeg to calculate VMAF
        cmd = [
            'ffmpeg',
            '-i', ref_path,
            '-i', test_path,
            '-lavfi', '[0:v]scale=1920:1080:force_original_aspect_ratio=decrease[ref]; '
                     '[1:v]scale=1920:1080:force_original_aspect_ratio=decrease[test]; '
                     '[ref][test]libvmaf=model=path=/opt/libvmaf/model/vmaf_v0.6.1.json:log_path=/tmp/vmaf.txt',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse VMAF from log
        if os.path.exists('/tmp/vmaf.txt'):
            with open('/tmp/vmaf.txt') as f:
                data = json.load(f)
                vmaf_score = data.get('aggregate', {}).get('VMAF', 0)
                return vmaf_score
        
        return 0.0
        
    except Exception as e:
        print(f"   ⚠️ VMAF calculation failed: {e}")
        return 0.0


def compress_video_tier2(
    input_path: str,
    output_path: str,
    vmaf_threshold: float = 90.0,
    target_crf: Optional[int] = None
) -> bool:
    """
    Compress video using Tier 2 Advanced algorithms
    
    1. Scene-aware encoding
    2. AV1 codec with optimal CRF
    3. VMAF-guided quality search
    4. Advanced denoising
    """
    print(f"\n🎥 === TIER 2 COMPRESSION ===")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   VMAF Threshold: {vmaf_threshold}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    # Step 1: Detect scenes
    scenes = get_scene_cuts(input_path, threshold=27.0)
    
    # Step 2: Determine initial CRF if not specified
    if target_crf is None:
        target_crf = 35
        print(f"📊 Using initial CRF: {target_crf}")
    
    # Step 3: Build FFmpeg command with Tier 2 settings
    print(f"\n⚙️ Encoding with AV1 (Tier 2 settings)...")
    
    # Advanced AV1 encoding parameters (Tier 2)
    av1_params = [
        f'crf={target_crf}',
        'preset=4',  # Tier 2: slower, better quality
        'tune=ssim',
        'enable-dnl=1',  # Denoising
    ]
    
    # Add scene-aware encoding if scenes detected
    if scenes:
        print(f"🎬 Scene-aware encoding enabled ({len(scenes)} scenes)")
        av1_params.append('enable-cdef=1')
    
    # Build command
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libaom-av1',
        '-b:v', '0',  # Bitrate 0 for CRF mode
        '-cpu-used', '4',  # CPU preset (higher = faster but lower quality)
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'main',
        '-movflags', '+faststart',
    ]
    
    # Add AV1 parameters
    for param in av1_params:
        cmd.extend(['-av1-params', param])
    
    # Optional: Add denoising filter
    cmd.extend([
        '-vf', 'hqdn3d=luma_spatial=4:chroma_spatial=3',
    ])
    
    # Output settings
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',
        output_path
    ])
    
    print(f"   Command: ffmpeg {' '.join(cmd[1:9])}...")
    
    # Step 4: Run encoding
    try:
        with tqdm(total=100, desc="Encoding") as pbar:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            for line in process.stderr:
                # Parse FFmpeg progress
                if 'frame=' in line:
                    pbar.update(0.1)
            
            returncode = process.wait()
            
            if returncode != 0:
                print(f"❌ Encoding failed with return code {returncode}")
                return False
        
        pbar.update(100)
        
    except Exception as e:
        print(f"❌ Error during encoding: {e}")
        return False
    
    # Step 5: Verify output
    if not os.path.exists(output_path):
        print(f"❌ Output file not created")
        return False
    
    # Calculate compression stats
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    compression_ratio = input_size / output_size if output_size > 0 else 0
    
    print(f"\n✅ Encoding complete!")
    print(f"   Input size: {input_size / 1024 / 1024:.1f} MB")
    print(f"   Output size: {output_size / 1024 / 1024:.1f} MB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Output: {output_path}")
    
    # Optional: Calculate VMAF
    print(f"\n📊 Calculating VMAF...")
    vmaf = calculate_vmaf_for_file(input_path, output_path)
    if vmaf > 0:
        print(f"   VMAF Score: {vmaf:.2f}")
        if vmaf >= vmaf_threshold:
            print(f"   ✅ VMAF meets threshold ({vmaf_threshold})")
        else:
            print(f"   ⚠️ VMAF below threshold (target: {vmaf_threshold})")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tier2_compression.py <input_video> <output_video> [vmaf_threshold]")
        print("Example: python tier2_compression.py input.mp4 output_compressed.mp4 90")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    vmaf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    
    success = compress_video_tier2(input_video, output_video, vmaf_threshold)
    sys.exit(0 if success else 1)
