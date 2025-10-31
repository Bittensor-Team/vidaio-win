#!/usr/bin/env python3
"""
Direct test of the batch worker functionality
"""

import sys
import os
sys.path.insert(0, '/workspace/vidaio-win/Real-ESRGAN-working-update')

from upscaler_worker_batch import process_manifest_async, parse_manifest
import time

def test_worker_direct():
    """Test worker functionality directly"""
    print("🧪 Testing worker functionality directly...")
    
    # Create test manifest
    test_dir = "/tmp/test_worker_direct"
    input_dir = f"{test_dir}/input_frames"
    output_dir = f"{test_dir}/output_frames"
    manifest_dir = f"{test_dir}/manifests"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Create a simple test manifest
    manifest_path = f"{manifest_dir}/worker_0_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("# Worker 0 Test Manifest\n")
        f.write("# Input Dir: /tmp/test_worker_direct/input_frames\n")
        f.write("# Output Dir: /tmp/test_worker_direct/output_frames\n")
        f.write("# Scale: 2x\n")
        f.write("# Batch Size: 8\n")
        f.write("# Patch Size: 256\n")
        f.write("# Total Frames: 2\n")
        f.write("# Worker Batches: [0]\n")
        f.write("# Total Batches: 1\n")
        f.write("\n")
        f.write("batch_0,frame_000001.png,frame_000002.png\n")
    
    # Create real test frames (not dummy files)
    from PIL import Image
    for i in range(1, 3):
        frame_path = f"{input_dir}/frame_{i:06d}.png"
        # Create a real 100x100 image
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))  # Red image
        img.save(frame_path)
        print(f"  Created test frame: {frame_path}")
    
    print(f"  Created manifest: {manifest_path}")
    
    # Test manifest parsing
    print("\n📋 Testing manifest parsing...")
    manifest_data = parse_manifest(manifest_path)
    print(f"  Parsed {len(manifest_data['batches'])} batches")
    print(f"  Metadata: {manifest_data['metadata']}")
    
    # Test worker processing
    print("\n⚡ Testing worker processing...")
    try:
        process_manifest_async(manifest_path, 0)
        print("  ✅ Worker processing completed")
        
        # Check output
        output_frames = os.listdir(output_dir)
        print(f"  📊 Output frames: {output_frames}")
        
    except Exception as e:
        print(f"  ❌ Worker processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_worker_direct()





