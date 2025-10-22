#!/usr/bin/env python3
"""
Tier 2 Advanced Pipeline - CHUNKED DISTRIBUTED VERSION
Each worker gets a specific chunk of frames to process independently
"""

import os
import sys
import cv2
import subprocess
import time
import requests
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import math

class ChunkedDistributedTier2Pipeline:
    def __init__(self, reference_video="/workspace/vidaio-subnet/elk.mp4", num_workers=5, use_previous=False):
        self.reference_video = reference_video
        self.output_dir = "/workspace/vidaio-subnet/tier2_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.num_workers = num_workers
        self.worker_ports = [8090 + i for i in range(num_workers)]
        self.worker_urls = [f"http://127.0.0.1:{port}" for port in self.worker_ports]
        self.use_previous = use_previous
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def find_latest_frames_dir(self):
        """Find the most recent frames directory"""
        frame_dirs = []
        for item in os.listdir(self.output_dir):
            if item.startswith("frames_") and os.path.isdir(os.path.join(self.output_dir, item)):
                frame_dirs.append(item)
        
        if not frame_dirs:
            return None
        
        frame_dirs.sort(reverse=True)
        latest_dir = os.path.join(self.output_dir, frame_dirs[0])
        
        frames = [f for f in os.listdir(latest_dir) if f.endswith('.png')]
        if frames:
            self.log(f"📁 Found previous frames: {latest_dir} ({len(frames)} frames)")
            return latest_dir
        
        return None
    
    def check_worker_health(self, url):
        """Check if worker is healthy"""
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("model_loaded", False)
            return False
        except Exception as e:
            print(f"Health check failed for {url}: {e}")
            return False
    
    def process_chunk(self, args):
        """Process a chunk of frames on a specific worker"""
        worker_id, worker_url, frame_files, frames_dir, upscaled_dir = args
        
        self.log(f"🔧 Worker {worker_id+1} processing {len(frame_files)} frames...")
        start_time = time.time()
        
        processed = 0
        failed = 0
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame_idx = int(frame_file.split('_')[1].split('.')[0])
            output_path = os.path.join(upscaled_dir, f"upscaled_{frame_idx:06d}.png")
            
            try:
                with open(frame_path, 'rb') as f:
                    files = {'file': (frame_file, f, 'image/png')}
                    resp = requests.post(f"{worker_url}/upscale", files=files, timeout=120)
                    
                    if resp.status_code == 200:
                        with open(output_path, 'wb') as out:
                            out.write(resp.content)
                        processed += 1
                    else:
                        failed += 1
                        self.log(f"   ⚠️  Worker {worker_id+1} frame {frame_idx} failed: Status {resp.status_code}")
                        
            except Exception as e:
                failed += 1
                self.log(f"   ⚠️  Worker {worker_id+1} frame {frame_idx} failed: {str(e)[:50]}")
        
        elapsed = time.time() - start_time
        self.log(f"✅ Worker {worker_id+1} completed: {processed}/{len(frame_files)} frames in {elapsed:.1f}s")
        
        return (worker_id, processed, failed, elapsed)
    
    def step_1_extract_frames(self):
        """Extract frames from reference video or use previous frames"""
        self.log("="*80)
        self.log("STEP 1: EXTRACT FRAMES FOR UPSCALING")
        self.log("="*80)
        
        if self.use_previous:
            previous_frames_dir = self.find_latest_frames_dir()
            if previous_frames_dir:
                self.frames_dir = previous_frames_dir
                frames = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
                self.log(f"♻️  Using previous frames: {len(frames)} frames")
                
                cap = cv2.VideoCapture(self.reference_video)
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                self.results['extracted_frames'] = len(frames)
                return True
            else:
                self.log("⚠️  No previous frames found, extracting new ones...")
        
        # Extract new frames
        self.frames_dir = os.path.join(self.output_dir, f"frames_{self.timestamp}")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.reference_video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"📹 Video: {self.width}x{self.height} @ {self.fps} fps")
        self.log(f"📊 Total frames: {self.total_frames}")
        self.log(f"⏱️  Duration: {self.total_frames / self.fps:.1f} seconds")
        
        sample_rate = max(1, self.total_frames // 100)
        frame_count = 0
        extracted = 0
        
        self.log(f"🎬 Sampling every {sample_rate} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_path = os.path.join(self.frames_dir, f"frame_{extracted:06d}.png")
                cv2.imwrite(frame_path, frame)
                extracted += 1
                
                if extracted % 10 == 0:
                    self.log(f"   ✓ Extracted {extracted} frames")
            
            frame_count += 1
        
        cap.release()
        self.log(f"✅ Extracted {extracted} sample frames")
        self.results['extracted_frames'] = extracted
        return True
    
    def step_2_upscale_frames_chunked(self):
        """Upscale frames using chunked distributed workers"""
        self.log("\n" + "="*80)
        self.log(f"STEP 2: UPSCALE FRAMES (CHUNKED DISTRIBUTED - {self.num_workers} workers)")
        self.log("="*80)
        
        upscaled_dir = os.path.join(self.output_dir, f"upscaled_{self.timestamp}")
        os.makedirs(upscaled_dir, exist_ok=True)
        
        try:
            # Check worker health
            self.log(f"🏥 Checking {self.num_workers} worker servers...")
            healthy_workers = []
            for i, url in enumerate(self.worker_urls):
                if self.check_worker_health(url):
                    healthy_workers.append((i, url))
            
            if not healthy_workers:
                self.log("❌ No healthy workers found")
                return False
            
            self.log(f"✅ {len(healthy_workers)}/{self.num_workers} workers healthy")
            
            # Get frames and split into chunks
            frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            total_frames = len(frames)
            
            # Split frames into chunks for each worker
            chunk_size = math.ceil(total_frames / len(healthy_workers))
            chunks = []
            
            for i, (worker_id, worker_url) in enumerate(healthy_workers):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_frames)
                chunk_frames = frames[start_idx:end_idx]
                chunks.append((worker_id, worker_url, chunk_frames, self.frames_dir, upscaled_dir))
                self.log(f"   Worker {worker_id+1}: frames {start_idx}-{end_idx-1} ({len(chunk_frames)} frames)")
            
            self.log(f"🎬 Processing {total_frames} frames in {len(chunks)} chunks...")
            start_time = time.time()
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=len(healthy_workers)) as executor:
                results = list(executor.map(self.process_chunk, chunks))
            
            # Collect results
            total_processed = 0
            total_failed = 0
            max_worker_time = 0
            
            for worker_id, processed, failed, elapsed in results:
                total_processed += processed
                total_failed += failed
                max_worker_time = max(max_worker_time, elapsed)
            
            total_elapsed = time.time() - start_time
            
            self.log(f"✅ Chunked upscaling complete!")
            self.log(f"   Total time: {total_elapsed:.1f}s")
            self.log(f"   Slowest worker: {max_worker_time:.1f}s")
            self.log(f"   Processed: {total_processed}/{total_frames}")
            self.log(f"   Speedup: ~{max_worker_time/total_elapsed:.1f}x vs single worker")
            
            self.upscaled_dir = upscaled_dir
            self.results['upscaled_frames'] = total_processed
            self.results['upscaling_time_sec'] = total_elapsed
            self.results['upscaling_workers'] = len(healthy_workers)
            self.results['slowest_worker_time'] = max_worker_time
            
            return total_processed == total_frames
            
        except Exception as e:
            self.log(f"❌ Chunked upscaling error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_3_reconstruct_video(self):
        """Reconstruct video from upscaled frames"""
        self.log("\n" + "="*80)
        self.log("STEP 3: RECONSTRUCT UPSCALED VIDEO")
        self.log("="*80)
        
        self.upscaled_video = os.path.join(self.output_dir, f"upscaled_{self.timestamp}.mp4")
        
        upscaled_frames = sorted([f for f in os.listdir(self.upscaled_dir) if f.endswith('.png')])
        
        if not upscaled_frames:
            self.log("❌ No upscaled frames found")
            return False
        
        first_frame = cv2.imread(os.path.join(self.upscaled_dir, upscaled_frames[0]))
        h, w = first_frame.shape[:2]
        
        self.log(f"📐 Upscaled resolution: {w}x{h}")
        self.log(f"📊 Frames: {len(upscaled_frames)}")
        self.log("🎬 Creating video from frames...")
        
        frame_pattern = os.path.join(self.upscaled_dir, "upscaled_%06d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            self.upscaled_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(self.upscaled_video):
            size_mb = os.path.getsize(self.upscaled_video) / 1024 / 1024
            self.log(f"✅ Upscaled video created: {size_mb:.2f} MB")
            self.results['upscaled_video_size_mb'] = size_mb
            return True
        else:
            self.log(f"❌ Failed to create video")
            return False
    
    def step_4_compress_video_h264(self, crf=23, preset='medium'):
        """Compress video with H.264 (fallback since AV1 might not be available)"""
        self.log("\n" + "="*80)
        self.log("STEP 4: COMPRESS VIDEO WITH H.264")
        self.log("="*80)
        
        self.compressed_video = os.path.join(self.output_dir, f"compressed_h264_{self.timestamp}.mp4")
        input_video = self.upscaled_video if os.path.exists(self.upscaled_video) else self.reference_video
        
        self.log(f"📹 Input: {os.path.basename(input_video)}")
        self.log(f"🔧 H.264 CRF: {crf}, Preset: {preset}")
        self.log("⚡ Compressing with H.264...")
        
        start_time = time.time()
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', 'aac',
            '-b:a', '128k',
            self.compressed_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(self.compressed_video):
            ref_size = os.path.getsize(input_video)
            comp_size = os.path.getsize(self.compressed_video)
            ratio = ref_size / comp_size
            
            self.log(f"✅ Compression complete in {elapsed:.1f}s")
            self.log(f"   Reference: {ref_size/1024/1024:.2f} MB")
            self.log(f"   Compressed: {comp_size/1024/1024:.2f} MB")
            self.log(f"   Ratio: {ratio:.2f}x")
            
            self.results['compression_ratio'] = ratio
            self.results['compression_time_sec'] = elapsed
            self.results['compressed_size_mb'] = comp_size / 1024 / 1024
            return True
        else:
            self.log(f"❌ Compression failed: {result.stderr}")
            return False
    
    def print_summary(self):
        """Print summary"""
        self.log("\n" + "="*80)
        self.log("FINAL SUMMARY - CHUNKED DISTRIBUTED TIER 2 PIPELINE")
        self.log("="*80)
        
        self.log(f"\n📊 RESULTS:")
        for key, value in self.results.items():
            if isinstance(value, float):
                self.log(f"   {key}: {value:.2f}")
            else:
                self.log(f"   {key}: {value}")
        
        results_file = os.path.join(self.output_dir, f"results_chunked_{self.timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log(f"\n💾 Results saved: {results_file}")
    
    def run(self):
        """Run full pipeline"""
        self.log("🚀 TIER 2 ADVANCED PIPELINE - CHUNKED DISTRIBUTED")
        self.log(f"📹 Reference: {self.reference_video}")
        self.log(f"📂 Output: {self.output_dir}")
        self.log(f"⚙️  Workers: {self.num_workers} (chunked processing)")
        
        try:
            if not self.step_1_extract_frames():
                raise Exception("Frame extraction failed")
            
            if not self.step_2_upscale_frames_chunked():
                raise Exception("Chunked upscaling failed")
            
            if not self.step_3_reconstruct_video():
                raise Exception("Video reconstruction failed")
            
            if not self.step_4_compress_video_h264(crf=23, preset='medium'):
                raise Exception("Compression failed")
            
            self.print_summary()
            self.log("\n✅ PIPELINE COMPLETE!")
            
        except Exception as e:
            self.log(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tier 2 Chunked Distributed Pipeline")
    parser.add_argument("--video", default="/workspace/vidaio-subnet/elk.mp4", help="Input video path")
    parser.add_argument("--workers", type=int, default=5, help="Number of worker servers")
    parser.add_argument("--use-previous", action="store_true", help="Use previously extracted frames")
    
    args = parser.parse_args()
    
    pipeline = ChunkedDistributedTier2Pipeline(
        reference_video=args.video,
        num_workers=args.workers,
        use_previous=args.use_previous
    )
    pipeline.run()

if __name__ == "__main__":
    main()
