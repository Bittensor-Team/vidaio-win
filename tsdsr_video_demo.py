#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import argparse

# Add TSD-SR to path
sys.path.append('/workspace/vidaio-win/TSD-SR')

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def get_video_info(video_path):
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
        duration = float(data['format']['duration'])
        total_frames = int(duration * fps)
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames
        }
    except Exception as e:
        log(f"Error getting video info: {e}")
        return None

def extract_frames(video_path, output_dir, max_frames=None):
    """Extract frames from video"""
    log(f"Extracting frames from {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Could not open video {video_path}")
        return False
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        frame_pil.save(frame_path)
        extracted_frames.append(frame_path)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            log(f"Extracted {frame_count} frames...")
    
    cap.release()
    log(f"Extracted {frame_count} frames to {output_dir}")
    return extracted_frames

def upscale_frame_tsdsr(frame_path, output_path, upscale=2, process_size=128):
    """Upscale single frame using TSD-SR"""
    try:
        # Import TSD-SR modules
        from test.test_tsdsr import main as tsdsr_main
        import test.test_tsdsr as tsdsr_module
        
        # Load image
        image = Image.open(frame_path).convert('RGB')
        
        # Convert to tensor
        tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        pixel_values = tensor_transforms(image).unsqueeze(0)
        
        # Calculate target size
        ori_width, ori_height = image.size
        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale*ori_width), int(scale*ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale*new_width, upscale*new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8
        
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.device = 'cuda'
                self.upscale = upscale
                self.process_size = process_size
                self.rank = 64
                self.rank_vae = 64
                self.is_use_tile = False
                self.vae_decoder_tiled_size = 224
                self.vae_encoder_tiled_size = 1024
                self.latent_tiled_size = 64
                self.latent_tiled_overlap = 8
                self.seed = 42
                self.mixed_precision = "fp16"
                self.align_method = "wavelet"
        
        mock_args = MockArgs()
        
        # Set global variables
        tsdsr_module.weight_dtype = torch.bfloat16
        tsdsr_module.vae = tsdsr_module.vae
        tsdsr_module.transformer = tsdsr_module.transformer
        tsdsr_module.timesteps = tsdsr_module.timesteps
        tsdsr_module.prompt_embeds = tsdsr_module.prompt_embeds
        tsdsr_module.pooled_prompt_embeds = tsdsr_module.pooled_prompt_embeds
        tsdsr_module.args = mock_args
        
        # Run TSD-SR
        with torch.no_grad():
            # Preprocess
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=(new_height, new_width), mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to('cuda', dtype=torch.bfloat16).clamp(-1,1)
            
            # Encode
            model_input = tsdsr_module.vae.encode(pixel_values).latent_dist.sample() * tsdsr_module.vae.config.scaling_factor
            model_input = model_input.to('cuda', dtype=torch.bfloat16)

            # Predict
            model_pred = tsdsr_module.tile_sample(model_input, pixel_values, tsdsr_module.transformer, tsdsr_module.timesteps, tsdsr_module.prompt_embeds, tsdsr_module.pooled_prompt_embeds, torch.bfloat16, mock_args)
            latent_stu = model_input - model_pred
            
            # Decode
            image = tsdsr_module.vae.decode(latent_stu / tsdsr_module.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
        
        # Convert to PIL
        result = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        
        # Handle resize flag
        if resize_flag:
            result = result.resize((int(ori_width*upscale), int(ori_height*upscale)))
        
        # Save result
        result.save(output_path)
        return True
        
    except Exception as e:
        log(f"Error upscaling frame {frame_path}: {e}")
        return False

def create_video_from_frames(frame_dir, output_video, fps=30):
    """Create video from upscaled frames"""
    log(f"Creating video from frames in {frame_dir}")
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not frame_files:
        log("No frame files found!")
        return False
    
    # Get first frame to determine size
    first_frame = Image.open(os.path.join(frame_dir, frame_files[0]))
    width, height = first_frame.size
    
    # Create video using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'upscaled_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_video
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log(f"Video created: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Error creating video: {e}")
        log(f"ffmpeg stderr: {e.stderr}")
        return False

def process_video_tsdsr(input_video, output_video, upscale=2, process_size=128, max_frames=None):
    """Process video with TSD-SR"""
    log(f"Processing video: {input_video}")
    
    # Get video info
    video_info = get_video_info(input_video)
    if not video_info:
        return False
    
    log(f"Video info: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS, {video_info['total_frames']} frames")
    
    # Create temp directories
    frames_dir = f"/tmp/frames_{int(time.time())}"
    upscaled_dir = f"/tmp/upscaled_{int(time.time())}"
    
    try:
        # Extract frames
        extracted_frames = extract_frames(input_video, frames_dir, max_frames)
        if not extracted_frames:
            return False
        
        # Process frames
        log(f"Upscaling {len(extracted_frames)} frames...")
        os.makedirs(upscaled_dir, exist_ok=True)
        
        processed_count = 0
        start_time = time.time()
        
        for i, frame_path in enumerate(extracted_frames):
            frame_idx = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
            output_path = os.path.join(upscaled_dir, f"upscaled_{frame_idx:06d}.png")
            
            frame_start = time.time()
            success = upscale_frame_tsdsr(frame_path, output_path, upscale, process_size)
            frame_time = time.time() - frame_start
            
            if success:
                processed_count += 1
                if i % 10 == 0 or i == len(extracted_frames) - 1:
                    avg_time = (time.time() - start_time) / (i + 1)
                    log(f"Processed {i+1}/{len(extracted_frames)} frames, avg: {avg_time:.2f}s/frame")
            else:
                log(f"Failed to process frame {frame_idx}")
        
        total_time = time.time() - start_time
        log(f"Processed {processed_count}/{len(extracted_frames)} frames in {total_time:.2f}s")
        log(f"Average time per frame: {total_time/len(extracted_frames):.2f}s")
        
        # Create output video
        if processed_count > 0:
            success = create_video_from_frames(upscaled_dir, output_video, video_info['fps'])
            if success:
                log(f"✅ Video processing complete: {output_video}")
                return True
        
        return False
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        if os.path.exists(upscaled_dir):
            shutil.rmtree(upscaled_dir)

def main():
    parser = argparse.ArgumentParser(description='Process video with TSD-SR')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--upscale', '-u', type=int, default=2, help='Upscale factor')
    parser.add_argument('--process_size', '-ps', type=int, default=128, help='Process size')
    parser.add_argument('--max_frames', '-mf', type=int, help='Maximum frames to process (for testing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        log(f"Error: Input video {args.input} not found!")
        return
    
    # Check if TSD-SR is available
    try:
        sys.path.append('/workspace/vidaio-win/TSD-SR')
        import test.test_tsdsr as tsdsr_module
        log("TSD-SR module loaded successfully")
    except Exception as e:
        log(f"Error loading TSD-SR: {e}")
        log("Please make sure TSD-SR is properly set up")
        return
    
    # Process video
    success = process_video_tsdsr(
        args.input, 
        args.output, 
        upscale=args.upscale, 
        process_size=args.process_size,
        max_frames=args.max_frames
    )
    
    if success:
        log("✅ Video processing completed successfully!")
    else:
        log("❌ Video processing failed!")

if __name__ == "__main__":
    main()



