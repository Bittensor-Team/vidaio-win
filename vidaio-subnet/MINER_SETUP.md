# Vidaio Miner Setup & Running Guide

This guide provides complete instructions for running the Vidaio miner with the conda environment already configured.

---

## Quick Start Summary

**Wallet**: `lucky_key`  
**Hotkey**: `lucky_hotkey1`  
**Network**: Finney (testnet)  
**Subnet**: 85  
**Python**: 3.11+ via conda env `vidaio`

---

## Prerequisites & Setup Status

✓ **Conda environment `vidaio` created** with Python 3.11.13  
✓ **All dependencies installed** including torch, tensorflow, video processing libraries  
✓ **Package installed** in editable mode  
✓ **.env configuration file created** with placeholder credentials  

### System Requirements

- **OS**: Ubuntu 24.04 LTS (recommended)
- **GPU**: NVIDIA RTX 4090 or higher (required for video upscaling)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ SSD (for video processing and models)

---

## Environment Activation

Always activate the conda environment before running any miner commands:

```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
```

Verify activation:
```bash
python --version  # Should show 3.11.13
which python      # Should show /workspace/miniconda/envs/vidaio/bin/python
```

---

## Step 1: Configure Environment Variables

Edit `.env` file with your actual credentials:

```bash
cd /workspace/vidaio-subnet
nano .env  # or vim, or your editor of choice
```

**Required for Miners**:
- `BUCKET_TYPE`: Bucket provider (`backblaze`, `amazon_s3`, `cloudflare`, or `hippius`)
- `BUCKET_NAME`: Your bucket name
- `BUCKET_COMPATIBLE_ENDPOINT`: S3-compatible endpoint URL
- `BUCKET_COMPATIBLE_ACCESS_KEY`: Access key/API key
- `BUCKET_COMPATIBLE_SECRET_KEY`: Secret key/token

Example for Backblaze B2:
```env
BUCKET_TYPE="backblaze"
BUCKET_NAME="my-vidaio-bucket"
BUCKET_COMPATIBLE_ENDPOINT="s3.us-west-000.backblazeb2.com"
BUCKET_COMPATIBLE_ACCESS_KEY="000abc1234567890"
BUCKET_COMPATIBLE_SECRET_KEY="K000/ABC+==aBcDeFgHiJkLmNoPqRsTu"
```

---

## Step 2: Install System Dependencies

Install required system packages (run once):

```bash
sudo apt update
sudo apt install -y \
  npm \
  redis-server \
  ffmpeg \
  jq

# Install PM2 process manager
sudo npm install -g pm2
pm2 update

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

---

## Step 3: Install Video2X (Required for Upscaling)

Choose one method:

### Option A: Download Pre-built Debian Package (Fastest)

```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet

mkdir -p services/upscaling/models
wget -P services/upscaling/models \
  https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb

sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb
sudo apt-get install -f -y
```

### Option B: Build from Source

```bash
cd /workspace/vidaio-subnet

# Install build tools
sudo apt-get install -y cargo
cargo install just --version=1.39.0
export PATH="$HOME/.cargo/bin:$PATH"

# Clone and build
git clone --recurse-submodules https://github.com/vidAio-subnet/video2x video2x-build
cd video2x-build
just ubuntu2404

# Install
sudo dpkg -i video2x-linux-ubuntu-amd64.deb
sudo apt-get install -f -y
```

Verify installation:
```bash
video2x --version
```

---

## Step 4: Starting Services with PM2

All services must run in order. Execute these commands:

### 4a. Video Upscaling Service

```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet

pm2 start "python services/upscaling/server.py" --name video-upscaler
pm2 logs video-upscaler  # View logs
```

### 4b. Video Compression Service

```bash
pm2 start "python services/compress/server.py" --name video-compressor
pm2 logs video-compressor
```

### 4c. File Deletion Service

```bash
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter
pm2 logs video-deleter
```

### 4d. **Miner Process** (Main)

```bash
pm2 start "python3 neurons/miner.py \
  --wallet.name lucky_key \
  --wallet.hotkey lucky_hotkey1 \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port 8091 \
  --logging.debug" \
  --name video-miner

pm2 logs video-miner  # Monitor real-time logs
```

**Optional parameters**:
- `--axon.port`: Change port if 8091 is unavailable
- `--logging.debug`: Enable debug logging (recommended for troubleshooting)
- `--logging.level`: Set to `trace`, `debug`, `info`, `warning`, `error`, or `critical`

---

## Step 5: Managing PM2 Processes

### View All Running Processes

```bash
pm2 list
```

### View Real-time Logs

```bash
pm2 logs video-miner          # Main miner logs
pm2 logs video-upscaler       # Upscaling logs
pm2 logs video-compressor     # Compression logs
pm2 logs video-deleter        # File deletion logs

# View all logs with timestamps
pm2 logs --timestamp
```

### Control Processes

```bash
# Restart a service
pm2 restart video-miner

# Stop a service
pm2 stop video-miner

# Start a stopped service
pm2 start video-miner

# Delete a process
pm2 delete video-miner

# Restart all services
pm2 restart all
```

### Setup Auto-Start on Reboot

```bash
pm2 startup
pm2 save
```

---

## Step 6: Health Check

Verify all services are running:

```bash
# Check PM2 processes
pm2 list

# Check Redis
redis-cli ping  # Should return PONG

# Check Video2X
video2x --version

# Check CUDA/GPU (if available)
nvidia-smi
```

Expected output should show:
- ✓ All PM2 processes with status `online`
- ✓ Redis connection working
- ✓ Video2X version information
- ✓ NVIDIA GPU information (if GPU available)

---

## Complete Startup Script

For convenience, you can use this single command to start all services:

```bash
eval "$(conda shell.bash hook)" && \
conda activate vidaio && \
cd /workspace/vidaio-subnet && \
pm2 start "python services/upscaling/server.py" --name video-upscaler 2>/dev/null || true && \
pm2 start "python services/compress/server.py" --name video-compressor 2>/dev/null || true && \
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter 2>/dev/null || true && \
pm2 start "python3 neurons/miner.py --wallet.name lucky_key --wallet.hotkey lucky_hotkey1 --subtensor.network finney --netuid 85 --axon.port 8091 --logging.debug" --name video-miner 2>/dev/null || true && \
pm2 list
```

---

## Troubleshooting

### Issue: "Connection refused" for Bittensor network

**Solution**: Ensure internet connectivity and firewall allows outbound connections on ports 8091+.

```bash
ping google.com
netstat -tuln | grep 8091  # Check if port is open
```

### Issue: GPU not detected by Video2X

**Ensure CUDA & drivers are installed**:
```bash
nvidia-smi              # Should show GPU info
nvcc --version          # Should show CUDA compiler version
```

If CUDA not installed:
```bash
# Ubuntu 24.04
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit nvidia-utils
```

### Issue: Redis connection errors

```bash
# Check if Redis is running
sudo systemctl status redis-server

# Restart Redis
sudo systemctl restart redis-server

# Verify Redis connection
redis-cli ping
```

### Issue: S3 bucket authentication failures

**Check .env file**:
```bash
cat .env  # Verify all BUCKET_* variables are set correctly
```

**Test connection**:
```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
python -c "from minio import Minio; print('MinIO imported successfully')"
```

### Issue: Out of disk space

Video processing creates large temporary files. Monitor disk usage:

```bash
df -h /  # Check total available space
du -sh /workspace/vidaio-subnet  # Check project size
```

Clear old processed videos:
```bash
# Check file deletion service logs
pm2 logs video-deleter
```

### Issue: High memory usage

Set memory limit for PM2 processes:
```bash
pm2 delete all
pm2 start "python services/upscaling/server.py" --name video-upscaler --max-memory-restart 8G
```

---

## Monitoring & Logging

### PM2 Web Dashboard (Optional)

```bash
pm2 web  # Start web dashboard on http://localhost:9615
```

### View Historical Logs

```bash
pm2 logs video-miner --lines 100      # Last 100 lines
pm2 logs video-miner --format json    # JSON format for parsing
```

### System Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Monitor CPU and memory
top  # Press 'q' to exit
```

---

## Network & Wallet Management

### Check Wallet Status (Requires Bittensor CLI)

```bash
# Install if not present
pip install bittensor

# List wallets
btcli wallet list

# Get wallet balance
btcli wallet balance --wallet.name lucky_key

# View hotkey info
btcli wallet balance --wallet.name lucky_key --wallet.hotkey lucky_hotkey1
```

### Update Wallet/Hotkey

To use different credentials:

```bash
pm2 delete video-miner
pm2 start "python3 neurons/miner.py \
  --wallet.name your_new_wallet \
  --wallet.hotkey your_new_hotkey \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port 8091" \
  --name video-miner
```

---

## Performance Optimization

### GPU Memory Optimization

Edit `services/upscaling/server.py` to adjust batch sizes and memory allocation based on your GPU VRAM.

### CPU Optimization

For high CPU usage:
```bash
# Limit CPU affinity (use cores 0-7)
taskset -c 0-7 pm2 start "python3 neurons/miner.py ..." --name video-miner
```

### Network Optimization

Ensure adequate bandwidth for:
- Downloading benchmark videos from validators (~50-500MB per task)
- Uploading processed results (~100MB-1GB per task)

---

## Regular Maintenance

### Weekly

- Check `pm2 list` for crashed processes
- Review `pm2 logs` for errors
- Verify S3 bucket connectivity

### Monthly

- Update package: `pip install -e . --upgrade`
- Review disk usage and clean old files
- Check NVIDIA driver and CUDA versions

### Restart Services (Optional but Recommended)

```bash
# Safe restart (saves PM2 state)
pm2 restart all

# Full restart
pm2 kill
pm2 start ...  # Re-start all services
```

---

## Environment Info

```bash
# Print system info
eval "$(conda shell.bash hook)"
conda activate vidaio
python -c "
import sys
import torch
import tensorflow as tf
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
"
```

---

## Need Help?

1. **Check logs**: `pm2 logs video-miner`
2. **Check status**: `pm2 list`
3. **Check disk/memory**: `df -h && free -h`
4. **Visit docs**: [Vidaio Documentation](https://vidaio.io)
5. **Bittensor docs**: [Bittensor Network](https://docs.bittensor.com)

---

## Stopping Everything

```bash
pm2 stop all
pm2 list  # Verify all stopped
```

Or completely remove PM2 processes:
```bash
pm2 delete all
pm2 list  # Should be empty
```

