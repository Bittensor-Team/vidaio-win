#!/usr/bin/env python3
"""
Test elk.mp4 with single worker to check performance
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

def test_elk_single_worker():
    """Test elk.mp4 with single worker"""
    print("🔍 Testing elk.mp4 with Single Worker")
    print("=" * 50)
    
    # Check if worker is ready
    try:
        response = requests.get("http://127.0.0.1:8090/health", timeout=2)
        if response.status_code != 200:
            print("❌ Worker not ready")
            return
        print("✅ Worker ready")
    except:
        print("❌ Worker not responding")
        return
    
    # Create test setup with elk.mp4 frames
    print("📸 Extracting frames from elk.mp4...")
    test_dir = "/tmp/elk_single_test"
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
            batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
            print(f"📊 Manifest has {len(batch_lines)} batches")
    
    # Test processing with detailed timing
    print("⚡ Testing elk.mp4 processing...")
    start_time = time.time()
    
    # Send manifest
    response = requests.post("http://127.0.0.1:8090/process_manifest",
                           json={"manifest_path": manifest_path}, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Failed to send manifest: {response.text}")
        return
    
    print("✅ Manifest sent")
    
    # Monitor progress with detailed timing
    print("⏳ Monitoring progress...")
    last_progress = 0
    last_time = time.time()
    batch_times = []
    
    while True:
        try:
            response = requests.get("http://127.0.0.1:8090/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                current_time = time.time()
                elapsed = current_time - start_time
                
                if status['progress_percent'] != last_progress:
                    progress_time = current_time - last_time
                    if last_progress > 0:
                        batch_time = progress_time / (status['progress_percent'] - last_progress) * 100
                        batch_times.append(batch_time)
                    
                    print(f"  📊 Progress: {status['progress_percent']:.1f}% (elapsed: {elapsed:.1f}s, delta: {progress_time:.1f}s)")
                    last_progress = status['progress_percent']
                    last_time = current_time
                
                if status['status'] in ['completed', 'error']:
                    total_time = current_time - start_time
                    print(f"✅ elk.mp4 processing completed in {total_time:.1f}s")
                    break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break
        
        time.sleep(5)
    
    # Check results
    output_frames = list(Path(output_dir).glob("*.png"))
    print(f"📊 Output frames: {len(output_frames)}/{len(frame_files)}")
    
    # Analyze performance
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"📊 Average batch time: {avg_batch_time:.1f}s")
        print(f"📊 Expected time for {len(batch_lines)} batches: {avg_batch_time * len(batch_lines):.1f}s")
        print(f"📊 Actual time: {total_time:.1f}s")
        
        if total_time > avg_batch_time * len(batch_lines) * 1.5:  # 50% tolerance
            print("⚠️  Performance degradation detected!")
        else:
            print("✅ Performance is within expected range")
    
    # Cleanup
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run elk.mp4 single worker test"""
    print("🔍 elk.mp4 Single Worker Test")
    print("=" * 40)
    
    test_elk_single_worker()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()





