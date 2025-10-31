#!/usr/bin/env python3
"""
Test script for batch processing setup
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def test_worker_health():
    """Test if all workers are healthy"""
    print("🔍 Testing worker health...")
    
    worker_urls = [f"http://127.0.0.1:{8090 + i}" for i in range(6)]
    healthy_workers = []
    
    for i, url in enumerate(worker_urls):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Worker {i+1} ({url}): {data['status']}")
                healthy_workers.append(url)
            else:
                print(f"  ❌ Worker {i+1} ({url}): HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ Worker {i+1} ({url}): {e}")
    
    print(f"\n📊 Healthy workers: {len(healthy_workers)}/6")
    return healthy_workers

def test_manifest_generation():
    """Test manifest generation"""
    print("\n📋 Testing manifest generation...")
    
    # Create test input directory
    test_dir = "/tmp/test_batch_processing"
    input_dir = f"{test_dir}/input_frames"
    os.makedirs(input_dir, exist_ok=True)
    
    # Create dummy frame files
    for i in range(1, 25):  # 24 frames
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        # Create a small dummy PNG file
        os.system(f"convert -size 100x100 xc:white {frame_path} 2>/dev/null || echo 'dummy' > {frame_path}")
    
    print(f"  📁 Created test frames in {input_dir}")
    
    # Generate manifests
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 24 6 8"
    result = os.system(cmd)
    
    if result == 0:
        print("  ✅ Manifest generation successful")
        
        # Check if manifests were created
        manifest_dir = f"{test_dir}/manifests"
        if os.path.exists(manifest_dir):
            manifest_files = list(Path(manifest_dir).glob("*.txt"))
            print(f"  📄 Created {len(manifest_files)} manifest files")
            
            # Show first manifest content
            if manifest_files:
                with open(manifest_files[0], 'r') as f:
                    content = f.read()
                    print(f"  📝 Sample manifest content:")
                    print(f"     {content[:200]}...")
        else:
            print("  ❌ Manifest directory not found")
    else:
        print("  ❌ Manifest generation failed")
    
    return test_dir

def test_worker_processing():
    """Test worker processing with a manifest"""
    print("\n⚡ Testing worker processing...")
    
    # Create test manifest
    test_dir = "/tmp/test_worker_processing"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create a simple test manifest for worker 0
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("# Worker 0 Test Manifest\n")
        f.write("# Input Dir: /tmp/test_worker_processing/input_frames\n")
        f.write("# Output Dir: /tmp/test_worker_processing/output_frames\n")
        f.write("# Scale: 2x\n")
        f.write("# Batch Size: 8\n")
        f.write("# Patch Size: 256\n")
        f.write("# Total Frames: 8\n")
        f.write("# Worker Batches: [0]\n")
        f.write("# Total Batches: 1\n")
        f.write("\n")
        f.write("batch_0,frame_000001.png,frame_000002.png,frame_000003.png,frame_000004.png,frame_000005.png,frame_000006.png,frame_000007.png,frame_000008.png\n")
    
    # Create dummy input frames
    for i in range(1, 9):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        os.system(f"convert -size 100x100 xc:white {frame_path} 2>/dev/null || echo 'dummy' > {frame_path}")
    
    print(f"  📁 Created test setup in {test_dir}")
    
    # Send manifest to worker 0
    worker_url = "http://127.0.0.1:8090"
    try:
        response = requests.post(f"{worker_url}/process_manifest", 
                               json={"manifest_path": manifest_path},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Manifest sent to worker: {data['status']}")
            
            # Monitor progress
            print("  ⏳ Monitoring progress...")
            for i in range(10):  # Check for 10 iterations
                status_response = requests.get(f"{worker_url}/status", timeout=5)
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"    Status: {status['status']}, Progress: {status['progress_percent']:.1f}%")
                    
                    if status['status'] in ['completed', 'error']:
                        break
                
                time.sleep(2)
            
            # Check if output frames were created
            output_frames = list(Path(output_dir).glob("*.png"))
            print(f"  📊 Output frames created: {len(output_frames)}")
            
        else:
            print(f"  ❌ Failed to send manifest: HTTP {response.status_code}")
            print(f"     Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Error testing worker processing: {e}")

def main():
    """Run all tests"""
    print("🧪 Batch Processing Test Suite")
    print("=============================")
    
    # Test 1: Worker health
    healthy_workers = test_worker_health()
    
    if len(healthy_workers) == 0:
        print("\n❌ No healthy workers found! Please start workers first:")
        print("   bash start_workers_batch.sh")
        return
    
    # Test 2: Manifest generation
    test_dir = test_manifest_generation()
    
    # Test 3: Worker processing
    test_worker_processing()
    
    print("\n✅ Test suite completed!")
    print(f"\n🧹 Cleaning up test directory: {test_dir}")
    os.system(f"rm -rf {test_dir}")

if __name__ == "__main__":
    main()





