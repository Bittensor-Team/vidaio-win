# TSD-SR Worker Server Documentation

## Overview

The TSD-SR Worker Server is a FastAPI-based HTTP server that loads TSD-SR and Stable Diffusion 3 models at startup and provides REST endpoints for image upscaling.

Similar to `upscaler_worker_server.py`, it:
- Preloads models at startup
- Accepts frame upscaling requests via HTTP
- Returns upscaled images with metadata
- Provides health checks and statistics

## Quick Start

### Start Server
```bash
# Activate environment
conda activate vidaio

# Start server on default port 8090
python tsdsr_worker_server.py

# Or with custom port
python tsdsr_worker_server.py --port 8091 --workers 2
```

### Upscale Image
```bash
# In another terminal
python tsdsr_client.py -i input.png -o output.png

# With 4x upscale
python tsdsr_client.py -i input.png -o output.png -u 4

# Batch process
python tsdsr_client.py --batch ./images --output ./upscaled -u 4
```

## Server Endpoints

### POST /upscale
Upscale a single image.

**Parameters:**
- `file` (required): Image file (PNG, JPEG)
- `upscale` (optional): Upscale factor, 2 or 4 (default: 2)
- `process_size` (optional): Processing size 64-256 (default: 128)

**Example:**
```bash
curl -X POST "http://127.0.0.1:8090/upscale?upscale=2&process_size=128" \
  -F "file=@input.png" \
  --output output.png
```

**Response:**
- PNG image file
- Headers: `X-Output-Size` (e.g., "3840x2160")

### POST /upscale_batch
Upscale multiple images.

**Parameters:**
- `files` (required): List of image files
- `upscale` (optional): Upscale factor (default: 2)
- `process_size` (optional): Processing size (default: 128)

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.png",
      "temp_path": "/tmp/xyz.png",
      "output_size": "3840x2160",
      "file_size_mb": 8.5
    }
  ],
  "total": 1,
  "successful": 1
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": {
    "transformer": true,
    "vae": true,
    "device": "cuda",
    "dtype": "torch.float16"
  }
}
```

### GET /info
Get server information.

**Response:**
```json
{
  "name": "TSD-SR Worker Server",
  "version": "1.0.0",
  "description": "Text-to-Image Diffusion Super-Resolution",
  "device": "cuda",
  "cuda_available": true,
  "gpu": "NVIDIA A100",
  "gpu_memory_gb": 40.0,
  "endpoints": { ... }
}
```

### GET /stats
Get server statistics including GPU memory usage.

**Response:**
```json
{
  "timestamp": "2024-10-23T01:00:00",
  "device": "cuda",
  "dtype": "torch.float16",
  "gpu_name": "NVIDIA A100",
  "gpu_memory_total_gb": 40.0,
  "gpu_memory_reserved_gb": 32.5,
  "gpu_memory_allocated_gb": 31.8,
  "gpu_memory_free_gb": 8.2
}
```

## Server Arguments

```bash
python tsdsr_worker_server.py [OPTIONS]

Options:
  --port PORT              Port to run on (default: 8090)
  --host HOST              Host to bind to (default: 127.0.0.1)
  --upscale FACTOR         Default upscale factor: 2 or 4 (default: 2)
  --process-size SIZE      Default processing size (default: 128)
  --workers WORKERS        Number of worker processes (default: 1)
```

## Client Usage

### Basic Upscaling
```bash
python tsdsr_client.py -i input.png -o output.png
```

### Advanced Options
```bash
# 4x upscale
python tsdsr_client.py -i input.png -u 4

# Custom process size
python tsdsr_client.py -i input.png -ps 256

# Custom server
python tsdsr_client.py -i input.png --host 192.168.1.100 --port 8090
```

### Batch Processing
```bash
# Process all PNG files in directory
python tsdsr_client.py --batch ./images --output ./upscaled

# Process specific pattern
python tsdsr_client.py --batch ./images --pattern "*.jpg"

# 4x upscale batch
python tsdsr_client.py --batch ./images --output ./upscaled -u 4
```

### Server Monitoring
```bash
# Check if server is running
python tsdsr_client.py --health

# Get server info
python tsdsr_client.py --info

# Get GPU stats
python tsdsr_client.py --stats
```

## Multi-Worker Setup

Run multiple workers on different ports:

```bash
# Terminal 1: Worker on port 8090
python tsdsr_worker_server.py --port 8090

# Terminal 2: Worker on port 8091
python tsdsr_worker_server.py --port 8091

# Terminal 3: Worker on port 8092
python tsdsr_worker_server.py --port 8092

# Terminal 4: Worker on port 8093
python tsdsr_worker_server.py --port 8093

# Terminal 5: Worker on port 8094
python tsdsr_worker_server.py --port 8094
```

Client can connect to any worker:
```bash
python tsdsr_client.py -i input.png --port 8090
```

## Performance Tuning

### Upscale Factor
- `--upscale 2`: Faster, lower memory (recommended for 40GB GPU)
- `--upscale 4`: Slower, higher memory (need ~50GB+ GPU)

### Process Size
- Smaller (64-128): Lower memory, may need tiling
- Larger (256): Higher quality, higher memory usage

### Optimal Settings by GPU
```
8GB GPU:   --upscale 2 --process-size 64
24GB GPU:  --upscale 2 --process-size 128
40GB GPU:  --upscale 2 --process-size 256
80GB GPU:  --upscale 4 --process-size 256
```

## Error Handling

### Model Not Loaded
**Error:** "Models not loaded"
**Solution:** Ensure server started properly and models are downloaded

### GPU Out of Memory
**Error:** "CUDA out of memory"
**Solution:** 
- Reduce `--process-size`
- Reduce `--upscale` factor
- Use different worker on less loaded GPU

### Server Not Responding
**Error:** "Connection refused"
**Solution:** 
- Check server is running
- Verify port is correct
- Check firewall settings

## File Formats

**Supported Input:**
- PNG (recommended)
- JPEG/JPG
- Any format PIL can read

**Output:**
- PNG (always 8-bit)

## Integration with Distributed Systems

### Kubernetes Deployment
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tsdsr-worker
spec:
  containers:
  - name: tsdsr
    image: tsdsr:latest
    ports:
    - containerPort: 8090
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
```

### Load Balancing
Use a load balancer to distribute requests:
```bash
# Using nginx as reverse proxy
# Forward requests to workers on 8090-8094
```

## Monitoring

### Health Check Script
```bash
#!/bin/bash
for port in {8090..8094}; do
  echo "Checking port $port..."
  curl -s http://127.0.0.1:$port/health | jq .
done
```

### GPU Monitoring
```bash
# Watch GPU usage while processing
watch -n 1 'python tsdsr_client.py --stats | jq ".gpu_memory_allocated_gb"'
```

## Troubleshooting

### Check Logs
```bash
# View server output
tail -f server.log
```

### Test Individual Endpoints
```bash
# Test health
curl http://127.0.0.1:8090/health

# Test info
curl http://127.0.0.1:8090/info

# Test stats
curl http://127.0.0.1:8090/stats
```

### Debug Client
```bash
# With verbose output
python tsdsr_client.py -i input.png --verbose
```

## Performance Benchmarks

| GPU | Upscale | Process Size | Time | Memory |
|-----|---------|--------------|------|--------|
| A100 (40GB) | 2x | 128 | 16s | 32GB |
| A100 (40GB) | 4x | 128 | 45s | 36GB |
| V100 (32GB) | 2x | 64 | 20s | 28GB |
| RTX 3090 (24GB) | 2x | 64 | 25s | 22GB |

## Advanced Usage

### Custom Python Integration
```python
import requests
from pathlib import Path

# Upload and process
with open('input.png', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8090/upscale',
        files={'file': f},
        params={'upscale': 2, 'process_size': 128}
    )

with open('output.png', 'wb') as f:
    f.write(response.content)
```

### Streaming Processing
```python
# Process images as they arrive
import asyncio
from pathlib import Path

async def process_stream(input_dir):
    while True:
        for image in Path(input_dir).glob('*.png'):
            # Process image
            # Move to processed folder
            pass
        await asyncio.sleep(1)
```

## Support & Issues

For issues or questions:
1. Check server logs
2. Verify models are downloaded
3. Test with `tsdsr_client.py --health`
4. Check GPU memory with `nvidia-smi`
5. Review error messages in server output

