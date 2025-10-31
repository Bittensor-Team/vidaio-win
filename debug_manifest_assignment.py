#!/usr/bin/env python3
"""
Debug manifest assignment to see if there's a mismatch
"""

import os
import sys
import subprocess

def debug_manifest_assignment():
    """Debug manifest assignment"""
    print("🔍 Debugging Manifest Assignment")
    print("=" * 40)
    
    # Create test setup
    test_dir = "/tmp/manifest_debug"
    input_dir = f"{test_dir}/input_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create 24 test frames
    from PIL import Image
    for i in range(1, 25):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        img.save(frame_path)
    
    # Generate manifests for 6 workers
    print("📋 Generating manifests for 6 workers...")
    cmd = f"python3 /workspace/vidaio-win/batch_manifest_generator.py {input_dir} 24 6 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Check what manifests were created
    print("📊 Checking created manifests...")
    manifest_files = []
    for i in range(6):
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                content = f.read()
                batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
                print(f"  worker_{i}_manifest.txt: {len(batch_lines)} batches")
                if batch_lines:
                    print(f"    First batch: {batch_lines[0]}")
            manifest_files.append(manifest_path)
        else:
            print(f"  worker_{i}_manifest.txt: NOT FOUND")
    
    print(f"\n📊 Total manifest files created: {len(manifest_files)}")
    
    # Simulate the wrapper script logic
    print("\n🔍 Simulating wrapper script logic...")
    WORKER_URLS = ["http://127.0.0.1:8090", "http://127.0.0.1:8091", "http://127.0.0.1:8092", 
                   "http://127.0.0.1:8093", "http://127.0.0.1:8094", "http://127.0.0.1:8095"]
    
    for i in range(len(WORKER_URLS)):
        worker_url = WORKER_URLS[i]
        manifest_path = f"{manifest_dir}/worker_{i}_manifest.txt"
        
        print(f"  Worker {i}: {worker_url}")
        print(f"    Manifest: {manifest_path}")
        print(f"    Exists: {os.path.exists(manifest_path)}")
        
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                content = f.read()
                batch_lines = [line for line in content.split('\n') if line.startswith('batch_')]
                print(f"    Batches: {len(batch_lines)}")
        print()
    
    # Cleanup
    subprocess.run(f"rm -rf {test_dir}", shell=True)

def main():
    """Run manifest assignment debug"""
    print("🔍 Manifest Assignment Debug")
    print("=" * 30)
    
    debug_manifest_assignment()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main()





