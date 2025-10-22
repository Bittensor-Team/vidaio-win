#!/usr/bin/env python3
"""
Tier 2 Advanced Upscaling Implementation
- ESRGAN for general upscaling
- GFPGAN for face super-resolution
- Post-processing filters for quality enhancement
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

try:
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
except ImportError:
    print("❌ Install required packages: pip install realesrgan gfpgan")
    exit(1)


def download_models():
    """Download required model files"""
    print("📥 Checking models...")
    model_dir = Path.home() / '.cache' / 'realesrgan'
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Models will be cached in: {model_dir}")


def initialize_upscalers(scale: int = 2):
    """Initialize ESRGAN and GFPGAN upscalers"""
    print(f"🔧 Initializing Tier 2 upscalers (scale={scale}x)...")
    
    # ESRGAN - General upscaling
    upname = f'RealESRGAN_x{scale}plus'
    upsampler = RealESRGANer(
        scale=scale,
        model_name=upname,
        model_path=None,
        upsampler_bg_upsampler=None,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    print(f"   ✅ ESRGAN ({upname}) loaded")
    
    # GFPGAN - Face super-resolution
    face_upsampler = GFPGANer(
        scale=scale,
        model_path=None,
        upscale=scale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )
    print(f"   ✅ GFPGAN (face SR) loaded")
    
    return upsampler, face_upsampler


def apply_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply unsharp masking for sharpness enhancement"""
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_denoise(image: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering for noise reduction"""
    return cv2.bilateralFilter(image, 9, 75, 75)


def process_frame_tier2(
    frame: np.ndarray,
    upsampler,
    face_upsampler,
    apply_sharpening_filter: bool = True,
    apply_denoise_filter: bool = True
) -> np.ndarray:
    """
    Process single frame with Tier 2 pipeline
    
    1. ESRGAN upscaling
    2. GFPGAN face enhancement
    3. Sharpening (optional)
    4. Denoising (optional)
    """
    # Step 1: General upscaling with ESRGAN
    upscaled, _ = upsampler.enhance(frame, outscale=2, alpha_upsampler=None)
    
    # Step 2: Face enhancement with GFPGAN
    _, _, face_enhanced = face_upsampler.enhance(
        upscaled,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.7
    )
    
    # Step 3: Post-processing
    result = face_enhanced
    
    if apply_denoise_filter:
        result = apply_denoise(result)
    
    if apply_sharpening_filter:
        result = apply_sharpening(result, strength=0.3)
    
    return result


def upscale_video_tier2(
    input_path: str,
    output_path: str,
    scale: int = 2,
    apply_post_processing: bool = True
) -> bool:
    """
    Upscale video using Tier 2 Advanced algorithms
    """
    print(f"\n🎬 === TIER 2 UPSCALING ===")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Scale: {scale}x")
    print(f"   Post-processing: {'Yes' : 'No'}")
    
    # Initialize upscalers
    upsampler, face_upsampler = initialize_upscalers(scale)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video properties
    output_width = width * scale
    output_height = height * scale
    
    print(f"\n📊 Video Info:")
    print(f"   Resolution: {width}x{height} → {output_width}x{output_height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {frame_count}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"❌ Failed to create output video writer")
        cap.release()
        return False
    
    # Process video frame by frame
    print(f"\n⏳ Processing frames...")
    processed = 0
    
    try:
        with tqdm(total=frame_count, desc="Upscaling") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with Tier 2 pipeline
                upscaled_frame = process_frame_tier2(
                    frame,
                    upsampler,
                    face_upsampler,
                    apply_sharpening_filter=apply_post_processing,
                    apply_denoise_filter=apply_post_processing
                )
                
                # Write to output
                out.write(upscaled_frame)
                processed += 1
                pbar.update(1)
        
        print(f"\n✅ Processing complete!")
        print(f"   Frames processed: {processed}")
        print(f"   Output: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
        
    finally:
        cap.release()
        out.release()
        print("🧹 Resources released")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python tier2_upscaling.py <input_video> <output_video> [scale]")
        print("Example: python tier2_upscaling.py input.mp4 output_upscaled.mp4 2")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    success = upscale_video_tier2(input_video, output_video, scale)
    sys.exit(0 if success else 1)
