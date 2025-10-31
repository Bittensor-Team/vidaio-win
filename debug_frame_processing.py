#!/usr/bin/env python3
"""
Debug script to investigate frame processing issues
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def debug_single_worker():
    """Debug single worker frame processing"""
    print("🔍 Debugging Single Worker Frame Processing")
    print("=" * 50)
    
    # Start one worker
    print("🚀 Starting single worker...")
    cmd = "WORKER_ID=0 /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port 8090 --worker-id 0"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for worker to be ready
    print("⏳ Waiting for worker to be ready...")
    for attempt in range(30):
        try:
            response = requests.get("http://127.0.0.1:8090/health", timeout=2)
            if response.status_code == 200:
                print("✅ Worker ready")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ Worker not ready")
        return
    
    # Create test setup with elk.mp4 frames
    print("📸 Extracting frames from elk.mp4...")
    test_dir = "/tmp/debug_frames"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Extract frames from elk.mp4
    cmd = f"ffmpeg -i elk.mp4 -vf fps=29.97 {input_dir}/frame_%06d.png -y -loglevel error"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Frame extraction failed: {result.stderr}")
        return
    
    # Count extracted frames
    frame_files = list(Path(input_dir).glob("*.png"))
    print(f"📊 Extracted {len(frame_files)} frames")
    
    # Generate manifest for single worker
    print("📋 Generating manifest...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} {len(frame_files)} 1 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Manifest generation failed: {result.stderr}")
        return
    
    # Check manifest content
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            content = f.read()
            print(f"📄 Manifest content (first 500 chars):")
            print(content[:500])
            print("...")
            
            # Count batches
            batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
            print(f"📊 Manifest has {len(batch_lines)} batches")
    
    # Send manifest to worker
    print("⚡ Sending manifest to worker...")
    response = requests.post("http://127.0.0.1:8090/process_manifest",
                           json={"manifest_path": manifest_path}, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Failed to send manifest: {response.text}")
        return
    
    print("✅ Manifest sent")
    
    # Monitor progress
    print("⏳ Monitoring progress...")
    start_time = time.time()
    
    while True:
        try:
            response = requests.get("http://127.0.0.1:8090/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                elapsed = time.time() - start_time
                
                print(f"  📊 Status: {status['status']}, Progress: {status['progress_percent']:.1f}%, Elapsed: {elapsed:.1f}s")
                
                if status['status'] in ['completed', 'error']:
                    total_time = time.time() - start_time
                    print(f"✅ Processing completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(2)
    
    # Check output frames
    output_frames = list(Path(output_dir).glob("*.png"))
    print(f"📊 Output frames: {len(output_frames)}")
    
    if len(output_frames) > 0:
        print("📋 First few output frames:")
        for i, frame in enumerate(sorted(output_frames)[:5]):
            print(f"  {frame.name}")
    
    # Cleanup
    process.terminate()
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def debug_manifest_generation():
    """Debug manifest generation process"""
    print("\n🔍 Debugging Manifest Generation")
    print("=" * 40)
    
    # Create test frames
    test_dir = "/tmp/debug_manifest"
    input_dir = f"{test_dir}/input_frames"
    
    os.makedirs(input_dir, exist_ok=True)
    
    # Create 10 test frames
    from PIL import Image
    for i in range(1, 11):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img.save(frame_path)
    
    print(f"📸 Created 10 test frames in {input_dir}")
    
    # Test manifest generation with different worker counts
    for num_workers in [1, 2, 6]:
        print(f"\n📋 Testing {num_workers} workers:")
        cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 10 {num_workers} 8"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Manifest generation successful")
            
            # Check manifest files
            manifest_dir = f"{input_dir}/../manifests"
            if os.path.exists(manifest_dir):
                manifest_files = list(Path(manifest_dir).glob("*.txt"))
                print(f"  📄 Created {len(manifest_files)} manifest files")
                
                # Show content of first manifest
                if manifest_files:
                    with open(manifest_files[0], 'r') as f:
                        content = f.read()
                        print(f"  📝 First manifest content:")
                        print(f"     {content[:200]}...")
        else:
            print(f"  ❌ Manifest generation failed: {result.stderr}")
    
    # Cleanup
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run debug tests"""
    print("🔍 Frame Processing Debug Suite")
    print("=" * 40)
    
    debug_manifest_generation()
    debug_single_worker()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()





