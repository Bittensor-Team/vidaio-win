#!/usr/bin/env python3
"""
Batch Manifest Generator
Creates manifest files for distributed batch processing
"""

import os
import json
from pathlib import Path

def create_batch_manifests(input_dir, output_dir, manifest_dir, total_frames, 
                          num_workers=6, batch_size=8, scale=2, patch_size=256):
    """
    Create manifest files for each worker with batch assignments
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory where output frames will be written
        manifest_dir: Directory to store manifest files
        total_frames: Total number of frames to process
        num_workers: Number of worker processes
        batch_size: Number of frames per batch
        scale: Scale factor (2x or 4x)
        patch_size: Patch size for Real-ESRGAN
    
    Returns:
        List of manifest file paths
    """
    
    # Calculate total batches
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    print(f"📊 Batch Assignment Calculation:")
    print(f"   Total Frames: {total_frames}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Total Batches: {total_batches}")
    print(f"   Num Workers: {num_workers}")
    print(f"   Scale: {scale}x")
    print(f"   Patch Size: {patch_size}")
    print()
    
    manifest_paths = []
    
    # Create manifest for each worker
    for worker_id in range(num_workers):
        manifest_path = os.path.join(manifest_dir, f"worker_{worker_id}_manifest.txt")
        
        # Assign batches to this worker (round-robin)
        worker_batches = []
        for batch_id in range(worker_id, total_batches, num_workers):
            worker_batches.append(batch_id)
        
        print(f"Worker {worker_id}: {len(worker_batches)} batches - {worker_batches[:5]}{'...' if len(worker_batches) > 5 else ''}")
        
        with open(manifest_path, 'w') as f:
            # Header with metadata
            f.write(f"# Worker {worker_id} Manifest\n")
            f.write(f"# Input Dir: {input_dir}\n")
            f.write(f"# Output Dir: {output_dir}\n")
            f.write(f"# Scale: {scale}x\n")
            f.write(f"# Batch Size: {batch_size}\n")
            f.write(f"# Patch Size: {patch_size}\n")
            f.write(f"# Total Frames: {total_frames}\n")
            f.write(f"# Worker Batches: {worker_batches}\n")
            f.write(f"# Total Batches: {len(worker_batches)}\n")
            f.write(f"\n")
            
            # Write batch assignments
            for batch_id in worker_batches:
                start_frame = batch_id * batch_size + 1
                end_frame = min(start_frame + batch_size - 1, total_frames)
                
                # List frame filenames for this batch
                frame_list = []
                for frame_num in range(start_frame, end_frame + 1):
                    frame_list.append(f"frame_{frame_num:06d}.png")
                
                # Write batch line: batch_id,frame1,frame2,...
                f.write(f"batch_{batch_id},{','.join(frame_list)}\n")
        
        manifest_paths.append(manifest_path)
    
    print(f"\n✅ Created {len(manifest_paths)} manifest files in {manifest_dir}")
    return manifest_paths


def parse_manifest(manifest_path):
    """
    Parse a manifest file
    
    Returns:
        dict with metadata and batch assignments
    """
    metadata = {}
    batches = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse metadata from comments
            if line.startswith('#'):
                if ':' in line:
                    key_value = line[1:].strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip().lower().replace(' ', '_')
                        value = key_value[1].strip()
                        metadata[key] = value
                continue
            
            # Parse batch assignment
            parts = line.split(',')
            if len(parts) >= 2:
                batch_id = parts[0]
                frames = parts[1:]
                batches.append({
                    'batch_id': batch_id,
                    'frames': frames,
                    'frame_count': len(frames)
                })
    
    return {
        'metadata': metadata,
        'batches': batches,
        'total_batches': len(batches)
    }


def generate_status_template(worker_id, manifest_path, status_dir):
    """
    Generate initial status file for a worker
    
    Args:
        worker_id: Worker ID
        manifest_path: Path to worker's manifest file
        status_dir: Directory to store status files
    
    Returns:
        Path to status file
    """
    manifest_data = parse_manifest(manifest_path)
    
    status = {
        'worker_id': worker_id,
        'status': 'idle',
        'total_batches': manifest_data['total_batches'],
        'completed_batches': 0,
        'current_batch': None,
        'progress_percent': 0.0,
        'start_time': None,
        'end_time': None,
        'estimated_completion': None,
        'manifest_path': manifest_path
    }
    
    status_path = os.path.join(status_dir, f"worker_{worker_id}_status.json")
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    
    return status_path


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python batch_manifest_generator.py <input_dir> <total_frames> [num_workers] [batch_size]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    total_frames = int(sys.argv[2])
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    
    # Use the same directory structure as the video2x wrapper
    base_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(base_dir, "output_frames")
    manifest_dir = os.path.join(base_dir, "manifests")
    status_dir = os.path.join(base_dir, "status")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(status_dir, exist_ok=True)
    
    # Create manifests
    manifest_paths = create_batch_manifests(
        input_dir, output_dir, manifest_dir,
        total_frames, num_workers, batch_size
    )
    
    # Create initial status files
    print(f"\n📝 Creating status files...")
    for worker_id, manifest_path in enumerate(manifest_paths):
        status_path = generate_status_template(worker_id, manifest_path, status_dir)
        print(f"   Worker {worker_id}: {status_path}")
    
    print(f"\n✅ Setup complete!")

