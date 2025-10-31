# Vidaio Miner Setup - START HERE 📚

## Your Setup is Complete ✅

Everything has been installed and configured for you to become a Vidaio miner. This guide will walk you through what you have and what to do next.

---

## 📋 Quick Facts About Your Setup

- **Conda Environment**: `vidaio` (Python 3.11.13)
- **Wallet**: `lucky_key` (pre-configured)
- **Hotkey**: `lucky_hotkey1` (pre-configured)
- **Network**: Finney (testnet)
- **Subnet**: 85
- **Location**: `/workspace/vidaio-subnet`

---

## 📚 Documentation Files (Read in This Order)

### 1. **ANSWERS_TO_YOUR_QUESTIONS.md** ⭐ START HERE
**What's in it**: Direct answers to your 3 main questions
- Did I install it correctly? → YES ✓
- Why do miners need S3? → Explanation + architecture diagram
- Stake or performance-based rewards? → 100% PERFORMANCE (Elite earn 16× more than poor miners)

**Time to read**: 10 minutes

---

### 2. **SETUP_SUMMARY.md**
**What's in it**: Overview of completed setup and next steps
- Status checklist
- System requirements
- Activation command
- Quick troubleshooting guide

**Time to read**: 5 minutes

---

### 3. **MINER_SETUP.md** (Most Comprehensive)
**What's in it**: Step-by-step detailed guide for everything
- Environment activation
- S3 bucket configuration
- System dependencies
- Video2X installation (2 methods)
- PM2 process management
- Health checks
- Troubleshooting (7 issues covered)
- Performance optimization
- Monitoring

**Time to read**: 20 minutes

---

### 4. **MINER_ECONOMICS.md** (Deep Dive)
**What's in it**: Complete economics explained
- Incentive mechanism (2 task types)
- Scoring systems with formulas
- Performance multipliers
- Real miner examples
- Reward distribution algorithm
- Metagraph & rankings
- Strategy for maximum rewards

**Time to read**: 30 minutes

---

### 5. **QUICKSTART.sh**
**What's in it**: Executable script showing quick reference
- Can run with `bash QUICKSTART.sh`
- Shows activation and next steps

---

## 🚀 Your Next Steps (In Order)

### Step 1: Read Documentation
```
1. ANSWERS_TO_YOUR_QUESTIONS.md (5-10 min)
2. SETUP_SUMMARY.md (5 min)
3. MINER_SETUP.md Steps 1-3 (15 min)
```

### Step 2: Configure S3 Bucket (1-2 hours)
```bash
# Edit .env with your S3 credentials
nano /workspace/vidaio-subnet/.env

# Required to fill in:
# BUCKET_TYPE: "backblaze" or "amazon_s3"
# BUCKET_NAME: your bucket name
# BUCKET_COMPATIBLE_ENDPOINT: S3 endpoint URL
# BUCKET_COMPATIBLE_ACCESS_KEY: access key
# BUCKET_COMPATIBLE_SECRET_KEY: secret key
```

**Recommendation**: Use Backblaze B2 ($0.006/GB/month - cheapest)

### Step 3: Install System Dependencies (10 minutes)
```bash
sudo apt update
sudo apt install -y npm redis-server ffmpeg jq
sudo npm install -g pm2
pm2 update
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Step 4: Install Video2X (5-10 minutes)
```bash
# Use prebuilt (faster):
mkdir -p services/upscaling/models
wget -P services/upscaling/models \
  https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb
sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb
sudo apt-get install -f -y
```

### Step 5: Start All Services (5 minutes)
```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
cd /workspace/vidaio-subnet

# Start all 4 services
pm2 start "python services/upscaling/server.py" --name video-upscaler
pm2 start "python services/compress/server.py" --name video-compressor
pm2 start "python services/miner_utilities/file_deletion_server.py" --name video-deleter
pm2 start "python3 neurons/miner.py --wallet.name lucky_key --wallet.hotkey lucky_hotkey1 --subtensor.network finney --netuid 85 --axon.port 8091 --logging.debug" --name video-miner

# Verify
pm2 list
```

### Step 6: Monitor & Optimize
```bash
# Watch real-time logs
pm2 logs video-miner

# Check miner status
pm2 list
```

---

## 💰 Economics Overview

**Your rewards depend on:**
1. ✅ **Video quality** (PIE-APP metric for upscaling)
2. ✅ **Compression efficiency** (file size reduction)
3. ✅ **Consistency** (10-round rolling window)
4. ✅ **Content length** (longer videos = higher bonus)

**NOT on:**
- ❌ Stake amount
- ❌ Account age
- ❌ How many tasks you do

**Performance Tiers:**
| Tier | Score | Relative Reward | Status |
|------|-------|-----------------|--------|
| Elite | 0.78+ | 16.3× | +15% bonus |
| Good | 0.65 | 2.0× | Neutral |
| Average | 0.51 | 1.5× | -7.5% penalty |
| Poor | 0.31 | 1.0× | -40% penalty |

**Elite miners earn 16× more than poor miners on identical tasks.**

---

## 📁 Key Files

```
/workspace/vidaio-subnet/
├── .env                          # ← Fill with S3 credentials
├── START_HERE.md                 # ← This file
├── ANSWERS_TO_YOUR_QUESTIONS.md  # ← Read first
├── SETUP_SUMMARY.md
├── MINER_SETUP.md                # ← Most detailed
├── MINER_ECONOMICS.md            # ← Deep dive into rewards
├── QUICKSTART.sh
├── neurons/
│   └── miner.py                  # ← Main miner code
├── services/
│   ├── upscaling/
│   ├── compress/
│   └── miner_utilities/
└── docs/
    └── incentive_mechanism.md    # ← Original Vidaio docs
```

---

## ⚡ Activate Your Environment

**Always use this before running commands:**

```bash
eval "$(conda shell.bash hook)"
conda activate vidaio
```

**Verify**:
```bash
python --version  # Should show 3.11.13
which python      # Should show /workspace/miniconda/envs/vidaio/bin/python
```

---

## 🔍 Check Your Installation

```bash
# Conda environment
conda activate vidaio

# Python packages
pip list | grep -E "torch|fastapi|redis|opencv"

# System tools
which npm       # Should exist
which pm2       # Should exist
which ffmpeg    # Should exist
which redis-cli # Should exist

# Database (after setup)
redis-cli ping  # Should return PONG
```

---

## 📊 Architecture You're Running

```
                    Your Miner Machine
                    ━━━━━━━━━━━━━━━━━━━━
                    │
                    ├─ Conda Environment (vidaio)
                    │  └─ Python 3.11.13
                    │     ├─ PyTorch 2.8.0 (GPU)
                    │     ├─ Video processing libs
                    │     └─ Bittensor 9.9.0
                    │
                    ├─ PM2 Processes (4 services)
                    │  ├─ video-upscaler (Python service)
                    │  ├─ video-compressor (Python service)
                    │  ├─ video-deleter (Python service)
                    │  └─ video-miner (Main, neurons/miner.py)
                    │
                    ├─ Redis (database)
                    │  └─ Stores pending tasks, results
                    │
                    └─ GPU (NVIDIA RTX 4090+)
                       ├─ Video2X (upscaling)
                       ├─ Compression algorithms
                       └─ Encoding/decoding

                    ↓ (Network)
                    
                    Bittensor Subnet 85 (Finney)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ├─ Validators (distributed)
                    │  ├─ Issue tasks
                    │  ├─ Download results
                    │  └─ Rate quality
                    │
                    ├─ Your Miner (UID TBD)
                    │  ├─ Process tasks
                    │  ├─ Upload to S3
                    │  └─ Return results
                    │
                    └─ Rewards (TAO token)
                       ├─ Based on quality score
                       ├─ Distributed every block
                       └─ To your wallet

                    ↓ (S3 Network)
                    
                    S3 Bucket (Backblaze/AWS/etc)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ├─ Stores processed videos
                    ├─ Issues presigned URLs
                    └─ Validators download from here
```

---

## ⚠️ Common Issues & Quick Fixes

| Issue | Fix |
|-------|-----|
| "Python not from vidaio env" | Run `eval "$(conda shell.bash hook)" && conda activate vidaio` |
| "Permission denied" on apt install | Use `sudo` prefix |
| "Video2X not found" | Reinstall: `sudo dpkg -i .../video2x...deb && sudo apt-get install -f` |
| "Connection refused" to Bittensor | Check internet, firewall, port 8091 open |
| "S3 auth failed" | Check .env file credentials, use correct endpoint |
| "Redis not responding" | Run `sudo systemctl restart redis-server` |
| "Port 8091 in use" | Use `--axon.port 8092` in miner command |

---

## 🎯 Success Metrics

After starting, you should see:

```bash
pm2 list
# Output should show:
# │ video-upscaler      │ 0   │ online │
# │ video-compressor    │ 0   │ online │
# │ video-deleter       │ 0   │ online │
# │ video-miner         │ 0   │ online │

pm2 logs video-miner
# Should show connection to Bittensor network
# Should show tasks being received
```

---

## 📞 Support & References

- **Bittensor Docs**: https://docs.bittensor.com
- **Vidaio Website**: https://vidaio.io
- **Video2X Docs**: https://docs.video2x.org
- **PM2 Docs**: https://pm2.keymetrics.io

---

## 🎓 What You Have Now

✅ Complete conda environment with all dependencies  
✅ Bittensor integration (subnet 85)  
✅ Video processing pipeline (upscaling + compression)  
✅ GPU acceleration (PyTorch with CUDA)  
✅ Redis database for task management  
✅ PM2 process management  
✅ Comprehensive documentation  
✅ Pre-configured wallet credentials (lucky_key / lucky_hotkey1)  

---

## 🚀 Ready to Mine?

1. **Read**: ANSWERS_TO_YOUR_QUESTIONS.md (5 min)
2. **Configure**: Edit .env with S3 credentials (10 min)
3. **Install**: System dependencies (10 min)
4. **Deploy**: Start all services (5 min)
5. **Monitor**: Watch pm2 logs (ongoing)

**Total time to launch: ~1 hour**

---

**Next action**: Open `ANSWERS_TO_YOUR_QUESTIONS.md` for detailed answers to your questions about installation, S3, and economics.

Good luck! 🚀
