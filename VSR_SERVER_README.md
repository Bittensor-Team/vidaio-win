# Vidaio VSR Server

High-performance video super-resolution server optimized for the Vidaio subnet, featuring persistent GPU workers and efficient file handling.

## 🚀 Features

- **Persistent GPU Workers**: 12 pre-loaded VSR models for maximum performance
- **Optimal Worker Mapping**: Task-specific worker counts based on performance analysis
- **Efficient File Handling**: Uses presigned URLs for large video files (>20MB)
- **Subnet Integration**: Compatible with Vidaio subnet protocol
- **Parallel Processing**: Multi-threaded frame processing
- **Memory Management**: Automatic cleanup and resource optimization

## 📊 Performance Specifications

| Task Type | Input Resolution | Output Resolution | Optimal Workers | Expected FPS | File Size |
|-----------|------------------|-------------------|-----------------|--------------|-----------|
| **SD2HD** | 480p | 1080p | 10 workers | 43.09 | ~4.8MB |
| **SD24K** | 480p | 4K | 8 workers | 12.54 | ~15MB |
| **HD24K** | 1080p | 4K | 12 workers | 11.45 | ~16.7MB |
| **4K28K** | 4K | 8K | 8 workers | 2.37 | ~48.3MB |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (RTX 4090+ recommended)
- ONNX Runtime with CUDA support
- FFmpeg
- Vidaio subnet dependencies

### Setup

1. **Install dependencies**:
   ```bash
   conda activate vidaio
   pip install fastapi uvicorn aiohttp opencv-python onnxruntime-gpu
   ```

2. **Download VSR model**:
   ```bash
   # Ensure the DXM-FP32 model is in the correct location
   ls /workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx
   ```

3. **Set up directories**:
   ```bash
   mkdir -p /tmp/vidaio_vsr
   mkdir -p /tmp/vidaio_vsr_output
   ```

## 🚀 Usage

### Start the Server

```bash
# Using the startup script (recommended)
./start_vsr_server.sh

# Or directly with Python
python vidaio_vsr_server.py
```

The server will start on `http://localhost:29115`

### API Endpoints

#### Health Check
```bash
curl http://localhost:29115/health
```

#### Server Statistics
```bash
curl http://localhost:29115/stats
```

#### Video Upscaling
```bash
curl -X POST http://localhost:29115/upscale-video \
  -H "Content-Type: application/json" \
  -d '{
    "payload_url": "https://example.com/video.mp4",
    "task_type": "HD24K",
    "maximum_optimized_size_mb": 100
  }'
```

### Test the Server

```bash
python test_vsr_server.py
```

## 🔧 Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0)
- `OMP_NUM_THREADS`: CPU thread count (default: 4)

### Server Configuration

Edit `vidaio_vsr_server.py` to modify:

- `MAX_WORKERS`: Maximum number of GPU workers (default: 12)
- `MODEL_PATH`: Path to VSR model file
- `TEMP_DIR`: Temporary file directory
- `OUTPUT_DIR`: Output file directory

## 📈 Performance Optimization

### Worker Pool Management

The server maintains a pool of persistent VSR workers:

- **Pre-loaded Models**: All workers load models at startup
- **Queue-based Distribution**: Frames distributed efficiently across workers
- **Resource Isolation**: Each worker processes independently
- **Automatic Cleanup**: Workers returned to pool after processing

### File Transfer Strategy

For videos >20MB, the server uses an optimized approach:

1. **Download**: Video downloaded to local storage
2. **Process**: Frames processed using GPU workers
3. **Upload**: Result uploaded to MinIO storage
4. **Return**: Presigned URL returned to client
5. **Cleanup**: Local files automatically deleted

This approach minimizes network overhead and provides better performance for large files.

### Memory Management

- **GPU Memory**: ~1GB per worker for 4K processing
- **CPU Memory**: Efficient frame batching
- **Storage**: Automatic cleanup of temporary files
- **Queue Management**: Prevents memory leaks

## 🔍 Monitoring

### Health Endpoint

```json
{
  "status": "healthy",
  "workers_ready": 12,
  "max_workers": 12,
  "supported_tasks": ["SD2HD", "SD24K", "HD24K", "4K28K"]
}
```

### Statistics Endpoint

```json
{
  "workers_ready": 12,
  "max_workers": 12,
  "task_worker_mapping": {
    "SD2HD": 10,
    "SD24K": 8,
    "HD24K": 12,
    "4K28K": 8
  },
  "task_resolution_mapping": {
    "SD2HD": ["854x480", "1920x1080", 2],
    "SD24K": ["854x480", "3840x2160", 4],
    "HD24K": ["1920x1080", "3840x2160", 2],
    "4K28K": ["3840x2160", "7680x4320", 2]
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   - Reduce `MAX_WORKERS` if running out of GPU memory
   - Check GPU memory usage with `nvidia-smi`

2. **Model Loading Failures**:
   - Verify model file exists and is accessible
   - Check ONNX Runtime installation
   - Ensure CUDA drivers are properly installed

3. **Processing Timeouts**:
   - Increase timeout values in FFmpeg commands
   - Check disk space for temporary files
   - Monitor CPU and GPU usage

### Logs

The server provides detailed logging:

- **Worker Status**: Model loading and readiness
- **Processing Progress**: Frame extraction and processing
- **Performance Metrics**: Processing times and throughput
- **Error Handling**: Detailed error messages and stack traces

## 🔄 Integration with Vidaio Subnet

The server is designed to integrate seamlessly with the Vidaio subnet:

1. **Protocol Compatibility**: Uses `VideoUpscalingProtocol`
2. **Task Type Support**: Handles all required task types
3. **File Management**: Compatible with MinIO storage
4. **Error Handling**: Proper error responses for subnet validation

### Miner Integration

Update your miner's `miner_utils.py` to use the VSR server:

```python
async def video_upscaler(payload_url: str, task_type: str) -> str | None:
    url = f"http://localhost:29115/upscale-video"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "task_type": task_type,
        "maximum_optimized_size_mb": 100
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("optimized_video_url")
    return None
```

## 📝 License

This project is part of the Vidaio subnet ecosystem.

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style
2. Tests pass for all task types
3. Performance benchmarks are maintained
4. Documentation is updated

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review server logs
3. Test with the provided test client
4. Verify GPU and model setup





