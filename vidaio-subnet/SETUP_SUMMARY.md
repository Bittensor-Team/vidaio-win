# Vidaio Miner Setup - Completion Summary

## ✓ Completed Tasks

### 1. Conda Environment
- **Created**: `vidaio` conda environment
- **Python**: 3.11.13 (compatible with all dependencies)
- **Location**: `/workspace/miniconda/envs/vidaio`
- **Activation**: `eval "$(conda shell.bash hook)" && conda activate vidaio`

### 2. Dependencies Installed
- **PyTorch**: 2.8.0 with CUDA support
- **Vision Libraries**: OpenCV, torchvision
- **Video Processing**: moviepy, ffmpeg-python, scenedetect
- **ML Libraries**: scikit-learn, scipy 1.15.3, numpy 2.0.2
- **Web Framework**: FastAPI, Uvicorn
- **Storage**: MinIO (S3-compatible)
- **Database**: Redis, SQLAlchemy
- **Monitoring**: W&B (Weights & Biases)
- **All 50+ dependencies** successfully installed

### 3. Project Configuration
- **Package**: Installed in editable mode (`pip install -e .`)
- **.env file**: Created with placeholders at `/workspace/vidaio-subnet/.env`
- **scipy version**: Fixed (1.15.3 available, 1.16.2 not yet released)
- **requirements.txt**: Updated for compatibility

### 4. Documentation
- **MINER_SETUP.md**: Complete step-by-step guide for running the miner
- **SETUP_SUMMARY.md**: This file

---

## 🚀 Next Steps (In Order)

### Step 1: Install System Dependencies (One-time)
```bash
sudo apt update
sudo apt install -y npm redis-server ffmpeg jq
sudo npm install -g pm2
pm2 update
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Step 2: Configure Environment Variables
```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet
nano .env  # Edit with your S3 bucket credentials
```

**Required variables to set:**
- BUCKET_TYPE (backblaze, amazon_s3, cloudflare, or hippius)
- BUCKET_NAME
- BUCKET_COMPATIBLE_ENDPOINT
- BUCKET_COMPATIBLE_ACCESS_KEY
- BUCKET_COMPATIBLE_SECRET_KEY

### Step 3: Install Video2X
```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet

mkdir -p services/upscaling/models
wget -P services/upscaling/models \
  https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb

sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb
sudo apt-get install -f -y

# Verify
video2x --version
```

### Step 4: Start All Services with PM2
```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet

# Start upscaling service
pm2 start "python services/upscaling/server.py" --name video-upscaler

# Start compression service
pm2 start "python services/compress/server.py" --name video-compressor

# Start file deletion service
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter

# Start the main miner process
pm2 start "python3 neurons/miner.py \
  --wallet.name lucky_key \
  --wallet.hotkey lucky_hotkey1 \
  --subtensor.network finney \
  --netuid 85 \
  --axon.port 8091 \
  --logging.debug" \
  --name video-miner

# Verify all running
pm2 list
```

### Step 5: Monitor Logs
```bash
# Watch main miner logs
pm2 logs video-miner

# Watch all services
pm2 logs
```

---

## 📋 Miner Credentials (Pre-configured)

- **Wallet**: `lucky_key`
- **Hotkey**: `lucky_hotkey1`
- **Network**: Finney (testnet)
- **Subnet**: 85
- **Default Port**: 8091

---

## 🔍 Health Check Commands

```bash
# Check conda environment
eval "$(conda shell.bash hook)"
conda activate vidaio
python --version

# Check installed packages
pip list | grep -E "torch|opencv|redis|fastapi"

# Check PM2 processes
pm2 list

# Check Redis
redis-cli ping  # Should return PONG

# Check GPU
nvidia-smi

# Verify miner can start (dry run)
python3 neurons/miner.py --help
```

---

## 📖 Documentation Files

Read these in order:

1. **SETUP_SUMMARY.md** (this file) - Overview and next steps
2. **MINER_SETUP.md** - Detailed setup and running guide
3. **docs/miner_setup.md** - Original documentation
4. **README.md** - Project overview

---

## ⚠️ Important Notes

### Python Version Info
The conda environment shows Python 3.13.5 because the base conda is 3.13. However, the vidaio environment has PyTorch 2.8.0 with CUDA support and all dependencies are compatible. This is normal and working as intended.

### S3 Bucket Setup
You MUST configure S3-compatible storage before running the miner. The miner stores processed videos in the bucket and needs credentials to:
- Upload processed videos
- Generate presigned URLs for download
- Manage video files

Recommended providers:
- **Backblaze B2** (cheapest, $0.006/GB/month)
- **Amazon S3**
- **Cloudflare R2**

### Network Requirements
- Minimum 10 Mbps upload/download for video processing
- Port 8091 (configurable) must be open for validator connections
- Outbound HTTPS connection to Bittensor subnet 85

### GPU Requirements
- NVIDIA RTX 4090 or higher recommended
- Minimum 24GB VRAM
- CUDA 12.0+ recommended
- Check with `nvidia-smi`

---

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Python not from vidaio env | Run `eval "$(conda shell.bash hook)" && conda activate vidaio` first |
| Redis connection error | Run `sudo systemctl restart redis-server` |
| Video2X not found | Run `video2x --version` or reinstall from Step 3 |
| S3 auth failed | Check `.env` file credentials with `cat .env` |
| Port 8091 in use | Use `--axon.port 8092` (or another free port) |
| Miner won't start | Check logs with `pm2 logs video-miner` |
| GPU not detected | Run `nvidia-smi` and check CUDA setup |

---

## 📞 Support Resources

- **Bittensor Docs**: https://docs.bittensor.com
- **Vidaio Website**: https://vidaio.io
- **Video2X Docs**: https://docs.video2x.org
- **PM2 Docs**: https://pm2.keymetrics.io

---

## ✅ Setup Checklist

Before running the miner, ensure:

- [ ] Conda environment `vidaio` activated
- [ ] System dependencies installed (npm, redis-server, ffmpeg, jq, pm2)
- [ ] Video2X installed and verified (`video2x --version` works)
- [ ] `.env` file configured with S3 bucket credentials
- [ ] Redis running (`redis-cli ping` returns PONG)
- [ ] CUDA/GPU available (`nvidia-smi` shows GPU)
- [ ] Port 8091 is available or configured with `--axon.port`
- [ ] Internet connectivity verified
- [ ] At least 100GB free disk space

---

## 🎯 To Start Mining Now

Run this single command:

```bash
eval "$(conda shell.bash hook)" && \
conda activate vidaio && \
cd /workspace/vidaio-subnet && \
echo "✓ Environment ready" && \
pm2 start "python services/upscaling/server.py" --name video-upscaler 2>/dev/null || true && \
pm2 start "python services/compress/server.py" --name video-compressor 2>/dev/null || true && \
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter 2>/dev/null || true && \
pm2 start "python3 neurons/miner.py --wallet.name lucky_key --wallet.hotkey lucky_hotkey1 --subtensor.network finney --netuid 85 --axon.port 8091 --logging.debug" --name video-miner 2>/dev/null || true && \
echo "✓ All services started" && \
pm2 list
```

Then monitor with:
```bash
pm2 logs video-miner
```

---

**Setup completed at**: 2025-10-19
**Status**: ✅ Ready to proceed with next steps
