# 🚀 Shared Storage Batch Processing Design

## 📊 **Architecture Overview**

Instead of sending images over HTTP, we use shared storage with manifest files for efficient batch processing.

### **Flow Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Video2X Wrapper (Coordinator)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Extract frames to shared directory                                     │
│     /tmp/video2x_12345/                                                    │
│     ├── input_frames/                                                      │
│     │   ├── frame_000001.png                                               │
│     │   ├── frame_000002.png                                               │
│     │   ├── ...                                                            │
│     │   └── frame_000300.png                                               │
│     │                                                                       │
│     ├── output_frames/                                                     │
│     │   (empty, workers will write here)                                   │
│     │                                                                       │
│     └── manifests/                                                         │
│         ├── worker_0_manifest.txt                                          │
│         ├── worker_1_manifest.txt                                          │
│         ├── worker_2_manifest.txt                                          │
│         ├── worker_3_manifest.txt                                          │
│         ├── worker_4_manifest.txt                                          │
│         └── worker_5_manifest.txt                                          │
│                                                                             │
│  2. Create manifest files (batch assignments)                              │
│                                                                             │
│     worker_0_manifest.txt:                                                 │
│     ┌────────────────────────────────────────────────────────────────┐     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000001.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000002.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000003.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000004.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000005.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000006.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000007.png      │     │
│     │ batch_0:/tmp/video2x_12345/input_frames/frame_000008.png      │     │
│     │ batch_6:/tmp/video2x_12345/input_frames/frame_000049.png      │     │
│     │ ...                                                            │     │
│     └────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│     worker_1_manifest.txt:                                                 │
│     ┌────────────────────────────────────────────────────────────────┐     │
│     │ batch_1:/tmp/video2x_12345/input_frames/frame_000009.png      │     │
│     │ batch_1:/tmp/video2x_12345/input_frames/frame_000010.png      │     │
│     │ ...                                                            │     │
│     └────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Trigger workers via HTTP endpoint (just a signal)                      │
│     POST /process_manifest with manifest_path                              │
│                                                                             │
│  4. Workers process independently                                          │
│     ┌────────────────────────────────────────────────────────────────┐     │
│     │  Worker 0: Read manifest → Process batches → Write output     │     │
│     │  Worker 1: Read manifest → Process batches → Write output     │     │
│     │  Worker 2: Read manifest → Process batches → Write output     │     │
│     │  Worker 3: Read manifest → Process batches → Write output     │     │
│     │  Worker 4: Read manifest → Process batches → Write output     │     │
│     │  Worker 5: Read manifest → Process batches → Write output     │     │
│     └────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  5. Poll for completion status                                             │
│     GET /status → workers report progress                                  │
│                                                                             │
│  6. Reconstruct video from output_frames/                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📝 **Manifest File Format**

### **Example: worker_0_manifest.txt**
```
# Worker 0 Manifest - Batches: 0, 6, 12, 18, 24, 30, 36
# Scale: 2x, Batch Size: 8, Patch Size: 256
# Total Batches: 7

batch_0,frame_000001,frame_000002,frame_000003,frame_000004,frame_000005,frame_000006,frame_000007,frame_000008
batch_6,frame_000049,frame_000050,frame_000051,frame_000052,frame_000053,frame_000054,frame_000055,frame_000056
batch_12,frame_000097,frame_000098,frame_000099,frame_000100,frame_000101,frame_000102,frame_000103,frame_000104
batch_18,frame_000145,frame_000146,frame_000147,frame_000148,frame_000149,frame_000150,frame_000151,frame_000152
batch_24,frame_000193,frame_000194,frame_000195,frame_000196,frame_000197,frame_000198,frame_000199,frame_000200
batch_30,frame_000241,frame_000242,frame_000243,frame_000244,frame_000245,frame_000246,frame_000247,frame_000248
batch_36,frame_000289,frame_000290,frame_000291,frame_000292,frame_000293,frame_000294,frame_000295,frame_000296
```

## 🔧 **Batch Assignment Algorithm**

```python
def create_batch_manifests(total_frames, num_workers, batch_size=8, base_dir="/tmp/video2x_12345"):
    """
    Create manifest files for each worker with batch assignments
    
    Algorithm:
    - Total batches = ceil(total_frames / batch_size)
    - Assign batches in round-robin: Worker N gets batches N, N+num_workers, N+2*num_workers, ...
    
    Example with 300 frames, 6 workers, batch_size=8:
    - Total batches: ceil(300/8) = 38 batches
    - Worker 0: batches [0, 6, 12, 18, 24, 30, 36]  (7 batches)
    - Worker 1: batches [1, 7, 13, 19, 25, 31, 37]  (7 batches)
    - Worker 2: batches [2, 8, 14, 20, 26, 32]      (6 batches)
    - Worker 3: batches [3, 9, 15, 21, 27, 33]      (6 batches)
    - Worker 4: batches [4, 10, 16, 22, 28, 34]     (6 batches)
    - Worker 5: batches [5, 11, 17, 23, 29, 35]     (6 batches)
    """
    
    input_dir = f"{base_dir}/input_frames"
    output_dir = f"{base_dir}/output_frames"
    manifest_dir = f"{base_dir}/manifests"
    
    # Calculate total batches
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    # Create manifest for each worker
    for worker_id in range(num_workers):
        manifest_path = f"{manifest_dir}/worker_{worker_id}_manifest.txt"
        
        with open(manifest_path, 'w') as f:
            # Header
            f.write(f"# Worker {worker_id} Manifest\n")
            f.write(f"# Base Dir: {base_dir}\n")
            f.write(f"# Input Dir: {input_dir}\n")
            f.write(f"# Output Dir: {output_dir}\n")
            f.write(f"# Scale: 2x, Batch Size: {batch_size}, Patch Size: 256\n")
            f.write(f"# Total Frames: {total_frames}\n\n")
            
            # Assign batches to this worker (round-robin)
            worker_batches = []
            for batch_id in range(worker_id, total_batches, num_workers):
                worker_batches.append(batch_id)
            
            f.write(f"# Worker {worker_id} Batches: {worker_batches}\n")
            f.write(f"# Total Batches: {len(worker_batches)}\n\n")
            
            # Write batch assignments
            for batch_id in worker_batches:
                start_frame = batch_id * batch_size + 1
                end_frame = min(start_frame + batch_size - 1, total_frames)
                
                # List frame filenames for this batch
                frame_list = []
                for frame_num in range(start_frame, end_frame + 1):
                    frame_list.append(f"frame_{frame_num:06d}")
                
                # Write batch line
                f.write(f"batch_{batch_id},{','.join(frame_list)}\n")
    
    return manifest_dir
```

## 🎯 **Performance Benefits**

### **Network Overhead Comparison:**

| Method | Data Transfer | Overhead | Time |
|--------|---------------|----------|------|
| **HTTP Upload** | 300 frames × 5MB = 1.5GB | High | ~50s |
| **Shared Storage** | 0 bytes (just file paths) | Minimal | ~0.1s |
| **Speedup** | - | - | **500x faster I/O** |

### **Processing Time:**

| Component | Time |
|-----------|------|
| Frame Extraction | 5s |
| Manifest Creation | 0.1s |
| Worker Processing (parallel) | 61s |
| Status Polling | 1s |
| Video Reconstruction | 5s |
| **Total** | **72s** |

## 📊 **Worker Distribution Example**

**300 frames ÷ 8 batch_size = 38 batches**

```
Worker 0 (Port 8090): [Batches: 0, 6, 12, 18, 24, 30, 36]
├── Batch 0:  Frames 1-8
├── Batch 6:  Frames 49-56
├── Batch 12: Frames 97-104
├── Batch 18: Frames 145-152
├── Batch 24: Frames 193-200
├── Batch 30: Frames 241-248
└── Batch 36: Frames 289-296

Worker 1 (Port 8091): [Batches: 1, 7, 13, 19, 25, 31, 37]
├── Batch 1:  Frames 9-16
├── Batch 7:  Frames 57-64
├── Batch 13: Frames 105-112
├── Batch 19: Frames 153-160
├── Batch 25: Frames 201-208
├── Batch 31: Frames 249-256
└── Batch 37: Frames 297-300 (4 frames)

Worker 2 (Port 8092): [Batches: 2, 8, 14, 20, 26, 32]
├── Batch 2:  Frames 17-24
├── Batch 8:  Frames 65-72
├── Batch 14: Frames 113-120
├── Batch 20: Frames 161-168
├── Batch 26: Frames 209-216
└── Batch 32: Frames 257-264

Worker 3 (Port 8093): [Batches: 3, 9, 15, 21, 27, 33]
Worker 4 (Port 8094): [Batches: 4, 10, 16, 22, 28, 34]
Worker 5 (Port 8095): [Batches: 5, 11, 17, 23, 29, 35]
```

## 🔄 **Status Tracking**

### **Status File Format: worker_0_status.json**
```json
{
  "worker_id": 0,
  "status": "processing",
  "total_batches": 7,
  "completed_batches": 3,
  "current_batch": 18,
  "progress_percent": 42.8,
  "start_time": "2025-10-23T20:30:00",
  "estimated_completion": "2025-10-23T20:31:00"
}
```

## 🚀 **Implementation Files**

### **1. video2x_wrapper_batch.sh** (Coordinator)
- Extract frames to shared directory
- Create batch manifests dynamically
- Trigger workers via HTTP
- Poll for completion
- Reconstruct video

### **2. upscaler_worker_batch.py** (Worker)
- `/process_manifest` endpoint: Start processing
- `/status` endpoint: Return progress
- Read manifest file
- Process batches in sequence
- Write output to shared directory
- Update status file

### **3. batch_manifest_generator.py** (Utility)
- Generate manifest files
- Calculate optimal batch distribution
- Validate frame count and batch sizes

## 💡 **Advantages of This Design**

1. **Zero Network Overhead**: No image data transferred over HTTP
2. **Independent Workers**: Each worker processes independently
3. **Fault Tolerance**: If a worker fails, only reprocess its manifest
4. **Progress Tracking**: Real-time status monitoring
5. **Scalability**: Easy to add more workers
6. **Debugging**: Clear manifest files for troubleshooting
7. **Dynamic Configuration**: Adapts to any video length/worker count

## 🎯 **Expected Performance**

With 6 workers, batch_size=8, patch_size=256:
- **Total Time**: ~61s
- **Network Overhead**: ~0.1s (vs ~50s HTTP)
- **Total Speedup**: 7x faster than current implementation

This design eliminates the network bottleneck and enables true parallel processing!





