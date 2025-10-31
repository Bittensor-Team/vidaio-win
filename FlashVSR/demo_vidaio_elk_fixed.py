#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

from diffsynth import ModelManager, FlashVSRFullPipeline
from examples.WanVSR.utils.utils import Buffer_LQ4x_Proj

# Import ParaAttention optimizations
try:
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
    PARA_ATTN_AVAILABLE = True
    print("✅ ParaAttention optimizations available")
except ImportError:
    PARA_ATTN_AVAILABLE = False
    print("⚠️  ParaAttention not available, using standard FlashVSR")

def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = int(w0 * scale), int(h0 * scale)
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = int(w0 * scale), int(h0 * scale)
    # 先放大
    up = img.resize((sW, sH), Image.BICUBIC)
    # 中心裁剪
    l = max(0, (sW - tW) // 2); t = max(0, (sH - tH) // 2)
    return up.crop((l, t, l + tW, t + tH))

def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16, device='cuda', max_frames=None):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)   
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))             
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)             
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n); n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        # FIXED: Don't limit frames for full video processing
        if max_frames is not None:
            total = min(max_frames, total)
            print(f"[{os.path.basename(path)}] Limited to {total} frames for memory optimization")
        else:
            print(f"[{os.path.basename(path)}] Processing ALL {total} frames")
        
        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)   # 1 C F H W
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")

def init_pipeline():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        print(f"Available GPU Memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB reserved, {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB allocated")
    else:
        print("CUDA not available, using CPU mode")
    
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./examples/WanVSR/FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors",
        "./examples/WanVSR/FlashVSR/Wan2.1_VAE.pth",
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    LQ_proj_in_path = "./examples/WanVSR/FlashVSR/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location=device), strict=True)

    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to(device)
    if device == "cuda":
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
    
    # Apply ParaAttention optimizations if available
    if PARA_ATTN_AVAILABLE:
        print("🔧 Applying ParaAttention optimizations...")
        try:
            # Apply first block cache for memory efficiency
            apply_cache_on_pipe(
                pipe,
                # residual_diff_threshold=0.0,  # Uncomment for more aggressive caching
            )
            print("✅ First block cache applied")
            
            # Parallelize VAE for better memory management
            parallelize_vae(pipe.vae)
            print("✅ VAE parallelization applied")
            
        except Exception as e:
            print(f"⚠️  ParaAttention optimization failed: {e}")
            print("Continuing with standard FlashVSR...")
    
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit","vae"])
    
    if torch.cuda.is_available():
        print(f"After model loading - GPU Memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB reserved, {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB allocated")
    else:
        print("Models loaded on CPU")
    return pipe

def process_elk_vidaio_tasks():
    """Process elk video with different Vidaio task types"""
    print("🦌 Elk Video Vidaio Task Types Demo")
    print("="*80)
    
    # Create output directory
    output_dir = Path("./results/elk_vidaio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Input video
    input_video = Path("/workspace/vidaio-win/elk.mp4")
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Vidaio task types to test
    vidaio_tasks = {
        "SD2HD": 2,   # 1080p → 2160p (2x upscale)
        "HD24K": 2,   # 1080p → 2160p (2x upscale) 
    }
    
    # Initialize FlashVSR pipeline
    print("\n🚀 Initializing FlashVSR pipeline...")
    try:
        pipe = init_pipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    results = []
    
    for task_type, scale_factor in vidaio_tasks.items():
        print(f"\n{'='*80}")
        print(f"Processing Elk Video: {task_type}")
        print(f"{'='*80}")
        
        result = {
            'task_type': task_type,
            'scale_factor': scale_factor,
            'success': False,
            'error': None,
            'processing_time': 0,
            'output_path': None
        }
        
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            print("Preparing input tensor...")
            LQ, th, tw, F, fps = prepare_input_tensor(
                str(input_video), 
                scale=scale_factor, 
                dtype=torch.bfloat16, 
                device='cuda',
                max_frames=None  # Process ALL frames
            )
            
            print(f"Processing video with FlashVSR...")
            print(f"  - Input resolution: {th}x{tw}")
            print(f"  - Number of frames: {F}")
            print(f"  - Scale factor: {scale_factor}x")
            print(f"  - FPS: {fps}")
            
            start_time = time.time()
            
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0, 
                tiled=True,  # Enable tiling for lower memory consumption
                LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=2.0*768*1280/(th*tw), 
                kv_ratio=3.0,
                local_range=11,
                color_fix = True,
            )
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            print(f"Average time per frame: {processing_time/F:.2f} seconds")
            
            print("Converting to video format...")
            video = tensor2video(video)
            
            # Save output video
            output_path = output_dir / f"FlashVSR_{task_type}_elk.mp4"
            print(f"Saving video to: {output_path}")
            save_video(video, str(output_path), fps=fps)
            
            result['output_path'] = str(output_path)
            result['success'] = True
            
            print("✅ FlashVSR processing complete!")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        results.append(result)
        
        # Brief pause between tasks
        time.sleep(2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 ELK VIDEO VIDAIO TASKS DEMO SUMMARY")
    print("="*80)
    
    successful_tasks = sum(1 for r in results if r['success'])
    total_tasks = len(results)
    
    print(f"\nOverall: {successful_tasks}/{total_tasks} tasks completed successfully")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} {result['task_type']}:")
        
        if result['success']:
            print(f"  - Processing time: {result['processing_time']:.2f}s")
            print(f"  - Output file: {result['output_path']}")
        else:
            print(f"  - Error: {result['error']}")
    
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    from pathlib import Path
    process_elk_vidaio_tasks()


