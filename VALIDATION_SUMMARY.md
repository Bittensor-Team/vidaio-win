# Vidaio Subnet - Complete Overview & Local Validation Setup

## 📚 Table of Contents

1. [Repository Overview](#repository-overview)
2. [What Miners Need to Know](#what-miners-need-to-know)
3. [Scoring System Deep Dive](#scoring-system-deep-dive)
4. [Local Validation Usage](#local-validation-usage)
5. [Miner Preparation Checklist](#miner-preparation-checklist)

---

## Repository Overview

### Vidaio Subnet Mission

**Democratize video enhancement through decentralization, AI, and blockchain technology.**

This Bittensor subnet enables miners to earn rewards by processing videos (upscaling and compression) and validators to allocate rewards based on quality metrics.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDAIO SUBNET                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  VALIDATORS                    MINERS                        │
│  ──────────                    ──────                        │
│  • Score outputs              • Process videos              │
│  • Test with synthetic data   • Implement best models       │
│  • Test with organic data     • Optimize for speed/quality  │
│  • Allocate rewards           • Submit results              │
│  • Track performance          • Handle upscaling/compress   │
│                                                               │
│  KEY METRICS:                                                │
│  - VMAF (frame quality)                                      │
│  - PIE-APP (perceptual similarity for upscaling)            │
│  - Compression Ratio (for compression)                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
vidaio-subnet/
├── local_validation.py           ← YOUR LOCAL VALIDATION TOOL
├── LOCAL_VALIDATION_README.md    ← FULL DOCUMENTATION
├── QUICK_START.md                ← QUICK REFERENCE
│
├── services/                     ← Core services
│   ├── scoring/                  ← Scoring functions
│   │   ├── server.py            ← Scoring endpoints
│   │   ├── scoring_function.py  ← Compression scoring logic
│   │   ├── vmaf_metric.py       ← VMAF calculation
│   │   └── pieapp_metric.py     ← PIE-APP calculation
│   │
│   ├── upscaling/               ← Upscaling service
│   │   ├── server.py            ← Upscaling API
│   │   └── models/              ← Model files
│   │
│   └── compress/                ← Compression service
│       ├── server.py            ← Compression API
│       ├── encoder.py           ← Encoding logic
│       └── vmaf_calculator.py   ← VMAF calculations
│
├── neurons/
│   ├── miner.py                 ← Miner neuron
│   └── validator.py             ← Validator neuron
│
├── docs/
│   ├── miner_setup.md           ← Miner installation guide
│   ├── validator_setup.md       ← Validator installation guide
│   └── incentive_mechanism.md   ← Scoring details
│
└── vidaio_subnet_core/          ← Core utilities
    └── CONFIG                   ← Configuration
```

---

## What Miners Need to Know

### Two Primary Tasks

#### 1️⃣ Upscaling
- **Goal**: Enhance video quality and increase resolution
- **Input**: Low-resolution video
- **Output**: High-resolution, high-quality video
- **Primary Metric**: PIE-APP (Perceptual Image-Error Assessment)
- **Secondary Metric**: VMAF (must be ≥ 85)
- **Scoring**: Quality + Content Length rewards

#### 2️⃣ Compression
- **Goal**: Reduce file size while maintaining quality
- **Input**: High-quality video
- **Output**: Smaller compressed video
- **Primary Metric**: Compression Ratio + VMAF
- **Secondary Metric**: VMAF (threshold typically 80-95)
- **Scoring**: Compression efficiency + Quality preservation

### Mining Workflow

```
1. Receive Task from Validator
   ├─ Task Type (upscaling or compression)
   ├─ Input Video
   ├─ Requirements (VMAF threshold, content length, etc.)
   └─ Deadline

2. Process Video
   ├─ Apply your AI models
   ├─ Optimize for speed and quality
   └─ Encode output

3. VALIDATE LOCALLY (NEW!)
   ├─ Run: python3 local_validation.py
   ├─ Check score against benchmarks
   └─ Debug if needed

4. Submit to Network
   ├─ Upload processed video
   └─ Wait for validator scoring

5. Receive Rewards
   ├─ Rewards based on local score calculation
   ├─ Track historical performance
   └─ Repeat
```

---

## Scoring System Deep Dive

### Upscaling Scoring

#### Step 1: Quality Score (S_Q)
- Calculated from **PIE-APP metric**
- Measures perceptual similarity between original and upscaled
- **Requirement**: VMAF ≥ 85 (or validation fails)
- **Formula**: Complex sigmoid + logarithmic transformation
- **Range**: 0.0 to 1.0

#### Step 2: Length Score (S_L)
- Based on video duration
- **Formula**: `log(1 + content_length) / log(1 + 320)`
- **Examples**:
  - 5s video: 0.3105 (31%)
  - 10s video: 0.4155 (42%)
  - 50s video: 0.6799 (68%)
  - 320s video: 1.0000 (100%)

#### Step 3: Preliminary Score (S_pre)
- **Formula**: `0.5 × S_Q + 0.5 × S_L`
- Combines quality and length equally

#### Step 4: Final Score (S_F)
- **Formula**: `0.1 × e^(6.979 × (S_pre - 0.5))`
- **Effect**: Exponential transformation amplifies good performance
- **Examples**:
  - S_pre 0.30 → S_F 0.025 (Poor)
  - S_pre 0.50 → S_F 0.100 (Below Average)
  - S_pre 0.60 → S_F 0.201 (Good)
  - S_pre 0.72 → S_F 0.464 (Very Good)
  - S_pre 0.90 → S_F 1.631 (Excellent)

### Compression Scoring

#### Key Calculation
```python
if vmaf_score < vmaf_threshold:
    score = 0  # Hard failure

else:
    # Soft zone (within 5 points of threshold)
    if vmaf_score < vmaf_threshold:
        quality_factor = 0.7 * ((vmaf_score - hard_cutoff) / 5.0) ** 2
        compression_component = compression_function(ratio)
        final_score = compression_component * quality_factor
    
    # Above threshold (full scoring)
    else:
        quality_component = 0.7 + 0.3 * (vmaf_excess / 15)
        compression_component = compression_function(ratio)
        final_score = 0.7 * compression_component + 0.3 * quality_component
```

#### Compression Rate Function
- **1x compression** (no compression): Score = 0.0
- **5x compression**: Score ≈ 0.65
- **10x compression**: Score ≈ 0.89
- **15x compression**: Score ≈ 0.97 (peak)
- **20x+ compression**: Logarithmic bonus up to 1.3x cap

#### Default Weights
- **Compression Weight**: 0.70 (70% emphasis)
- **Quality Weight**: 0.30 (30% emphasis)

### Metrics Explained

#### VMAF (Video Multimethod Assessment Fusion)
- **What**: Frame-by-frame quality comparison
- **Range**: 0-100 (higher is better)
- **Calculation**: Harmonic mean of individual frame scores
- **Why Harmonic Mean**: Emphasizes poor frames to prevent gaming
- **Sample Size**: 5 random frames from video
- **Threshold**: Typically 85 for upscaling, 80-95 for compression

#### PIE-APP (Perceptual Image-Error Assessment)
- **What**: Deep learning-based perceptual similarity
- **Range**: 0-5+ (lower is better, normalized)
- **Purpose**: Upscaling-specific quality metric
- **Process**: Compares frame features perceptually
- **Sigmoid Transform**: Converts to 0-1 scale
- **Sample Size**: 4 random frames from video

#### Compression Rate
- **Formula**: `compressed_size / original_size`
- **Range**: 0-1 (lower is better)
- **Example**: 0.25 = 4x compression (75% reduction)

---

## Local Validation Usage

### Quick Command Reference

```bash
# Validate upscaling pair
python3 local_validation.py \
    --reference ref.mp4 \
    --processed proc.mp4 \
    --task upscaling

# Validate compression pair
python3 local_validation.py \
    --reference orig.mp4 \
    --processed comp.mp4 \
    --task compression

# Batch validate folder
python3 local_validation.py \
    --folder ./videos \
    --task upscaling \
    --output scores.json
```

### Output Interpretation

```
🎬 Starting Single Video Pair Validation...
   📹 Task: UPSCALING
   📊 Reference: video.mp4
   🎬 Processed: video_upscaled.mp4
   
   🎯 VMAF Score: 88.45           ← Frame quality (must be ≥85)
   🎨 PIE-APP Score: 0.8234       ← Perceptual quality
   📈 Quality Score: 0.7123       ← Normalized (0-1)
   📏 Length Score: 0.4155        ← 10 second content = 42%
   ✅ Final Score: 0.2856         ← Your score vs network
   📝 Reason: success
```

### Batch Mode Output

```
============================================================
BATCH SUMMARY
============================================================
✅ Successful: 5/5 videos validated
❌ Failed: 0/5 videos

📊 Score Statistics:
   Average: 0.3124  ← Compare with benchmarks
   Min: 0.2145
   Max: 0.4567
   Std Dev: 0.0823
```

### Benchmark Comparison

#### Upscaling Scores
- **Top 10%**: 0.35 - 0.50 (Elite miners)
- **Top 50%**: 0.20 - 0.35 (Good miners)
- **Average**: 0.10 - 0.20 (Standard)
- **Poor**: < 0.10 (Below competitive)

#### Compression Scores
- **Top 10%**: 0.85 - 1.00 (Elite miners)
- **Top 50%**: 0.60 - 0.85 (Good miners)
- **Average**: 0.30 - 0.60 (Standard)
- **Poor**: < 0.30 (Below competitive)

---

## Miner Preparation Checklist

### ✅ Hardware Setup

- [ ] NVIDIA GPU (RTX 4090 or equivalent)
- [ ] Ubuntu 24.04 LTS or later
- [ ] Minimum 16GB RAM
- [ ] At least 500GB free storage
- [ ] Gigabit internet connection

### ✅ Software Installation

```bash
# 1. System dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip ffmpeg

# 2. Project setup
git clone https://github.com/vidaio-subnet/vidaio-subnet.git
cd vidaio-subnet
python3 -m venv venv
source venv/bin/activate
pip install -e .

# 3. Video2X (for upscaling)
wget -P services/upscaling/models \
    https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb
sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb

# 4. Process management
sudo npm install -g pm2
```

- [ ] Python 3.10+ installed
- [ ] FFmpeg and ffprobe installed
- [ ] CUDA 12.0+ configured
- [ ] Video2X installed
- [ ] PM2 installed
- [ ] Redis running
- [ ] Dependencies installed with `pip install -e .`

### ✅ Environment Configuration

```bash
# 1. Create environment file
cp .env.template .env

# 2. Add credentials
BUCKET_NAME="your-s3-bucket"
BUCKET_COMPATIBLE_ENDPOINT="endpoint"
BUCKET_COMPATIBLE_ACCESS_KEY="key"
BUCKET_COMPATIBLE_SECRET_KEY="secret"
PEXELS_API_KEY="api-key"
WANDB_API_KEY="wandb-key"
```

- [ ] `.env` file created and configured
- [ ] S3 bucket credentials added
- [ ] API keys added (Pexels, WANDB optional)
- [ ] Bucket permissions verified

### ✅ Model Preparation

For **Upscaling**:
- [ ] Video2X properly installed
- [ ] GPU acceleration working
- [ ] Test upscaling on sample video

For **Compression**:
- [ ] AV1 codec available
- [ ] FFmpeg supports AV1
- [ ] Encoding parameters tuned
- [ ] Test compression on sample video

### ✅ Test Before Mining

```bash
# 1. Test upscaling
python services/upscaling/server.py

# 2. Test compression
python services/compress/server.py

# 3. Validate both locally
python3 local_validation.py \
    --reference test_ref.mp4 \
    --processed test_out.mp4 \
    --task upscaling

# 4. Check scores
# Expect: Score > 0.20 for upscaling, > 0.60 for compression
```

- [ ] Service endpoints respond correctly
- [ ] Video processing works end-to-end
- [ ] Local validation runs without errors
- [ ] Scores are in reasonable range

### ✅ Network Setup

```bash
# Bittensor configuration
python3 neurons/miner.py \
    --wallet.name YOUR_WALLET \
    --wallet.hotkey YOUR_HOTKEY \
    --subtensor.network finney \
    --netuid 85 \
    --axon.port YOUR_PORT \
    --logging.debug
```

- [ ] Bittensor wallet created
- [ ] Hotkey generated
- [ ] Netuid confirmed (85 for Vidaio)
- [ ] Port open and forwarded
- [ ] Network connectivity tested

### ✅ Start Mining

```bash
# Using PM2 for process management
pm2 start "python services/upscaling/server.py" --name upscaler
pm2 start "python services/compress/server.py" --name compressor
pm2 start "python neurons/miner.py \
    --wallet.name YOUR_WALLET \
    --wallet.hotkey YOUR_HOTKEY \
    --subtensor.network finney \
    --netuid 85 \
    --axon.port YOUR_PORT" --name vidaio-miner

# Monitor
pm2 logs vidaio-miner
pm2 status
```

- [ ] All services running with PM2
- [ ] Logs show successful startup
- [ ] No errors in process logs
- [ ] Receiving tasks from validators

### ✅ Ongoing Optimization

1. **Monitor Performance**
   - [ ] Track score distribution
   - [ ] Identify failure patterns
   - [ ] Compare with benchmarks

2. **Optimize Quality**
   - [ ] Adjust model parameters
   - [ ] Try different architectures
   - [ ] Benchmark regularly

3. **Optimize Speed**
   - [ ] Reduce processing time
   - [ ] Batch process when possible
   - [ ] Monitor GPU utilization

4. **Track Earnings**
   - [ ] Monitor reward allocation
   - [ ] Correlate with local scores
   - [ ] Adjust strategy accordingly

---

## Performance Targets

### For Competitive Mining

**Upscaling:**
- VMAF: 90+ (above minimum 85)
- PIE-APP: < 0.5 (high perceptual quality)
- Processing: < 2 minutes per 5-second clip
- Target Score: 0.30+ (aim for 0.40+)

**Compression:**
- VMAF: 92+ (maintain quality)
- Compression Ratio: 5x-10x (5:1 to 10:1)
- Processing: < 1 minute per 5-second clip
- Target Score: 0.70+ (aim for 0.90+)

---

## Next Steps

1. **Read**: `LOCAL_VALIDATION_README.md` for full documentation
2. **Quick Start**: Use `QUICK_START.md` for immediate commands
3. **Setup**: Follow miner preparation checklist
4. **Test**: Run local validation on your first outputs
5. **Optimize**: Use local scores to improve quality
6. **Mine**: Start earning on Vidaio Subnet!

---

**Ready to mine? Let's go! 🚀**
