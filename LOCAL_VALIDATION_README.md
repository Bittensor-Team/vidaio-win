# Local Validation Script - Vidaio Subnet Miner Guide

## Overview

The `local_validation.py` script allows miners to locally validate their video processing outputs **before pushing results to the network**. This enables you to:

- ✅ Check scores before submission
- 📊 Compare performance with top miners locally
- 🔍 Debug quality issues early
- 📁 Batch validate multiple videos
- 💾 Export results as JSON

---

## What This Repo Is About

**Vidaio Subnet** is a decentralized video processing network built on Bittensor that enables:

- **Video Upscaling**: Enhance video quality and resolution using AI models
- **Video Compression**: Reduce file sizes while maintaining quality

### Key Components:

| Component | Role |
|-----------|------|
| **Miners** | Process videos (upscaling/compression) and submit results |
| **Validators** | Score miner outputs and allocate rewards |
| **Scoring System** | Uses VMAF and PIE-APP metrics for evaluation |

---

## Miner Preparation Checklist

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 or higher (recommended)
- **OS**: Ubuntu 24.04 LTS or higher
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: At least 500GB free space

### Software Requirements
- Python 3.10+
- FFmpeg and ffprobe
- CUDA 12.0+
- Video2X (for upscaling)

### Setup Steps

1. **Clone and Install**
   ```bash
   git clone https://github.com/vidaio-subnet/vidaio-subnet.git
   cd vidaio-subnet
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install ffmpeg redis-server -y
   npm install -g pm2
   ```

3. **Install Video2X** (for upscaling miners)
   ```bash
   wget -P services/upscaling/models https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb
   sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb
   ```

4. **Configure Environment**
   ```bash
   cp .env.template .env
   # Edit .env with your S3 credentials
   ```

5. **Start Services** (if running as miner)
   ```bash
   pm2 start "python services/upscaling/server.py" --name video-upscaler
   pm2 start "python services/compress/server.py" --name video-compressor
   ```

---

## Scoring System Overview

### Upscaling Scoring Formula

```
Quality Score (S_Q) = PIE-APP normalized to [0, 1]
Length Score (S_L) = log(1 + content_length) / log(1 + 320)
Preliminary Score (S_pre) = 0.5 × S_Q + 0.5 × S_L
Final Score (S_F) = 0.1 × e^(6.979 × (S_pre - 0.5))
```

**Key Points:**
- VMAF must be ≥ 85 (configurable threshold) for non-zero score
- PIE-APP measures perceptual similarity (lower is better, normalized)
- Content length reward: 5s = 31%, 10s = 42%, 320s = 100%
- Exponential transform amplifies good performance

### Compression Scoring Formula

```
If VMAF < threshold: Score = 0
Else:
  Compression Component = f(compression_ratio) with diminishing returns
  Quality Component = g(VMAF_excess_above_threshold)
  Final Score = 0.7 × Compression + 0.3 × Quality
```

**Key Points:**
- VMAF threshold must be met (typically 80-95)
- Compression ratio: 1x = 0 score, 15x = peak score (~1.0)
- >20x compression: logarithmic bonus up to 1.3x
- Soft zone: VMAF within 5 points of threshold gets partial credit

### Metrics Used

| Metric | Range | Purpose | Task |
|--------|-------|---------|------|
| **VMAF** | 0-100 | Frame-by-frame quality | Both |
| **PIE-APP** | 0-5+ (capped at 2.0) | Perceptual similarity | Upscaling |
| **Compression Rate** | 0-1 | File size ratio | Compression |

---

## Installation & Setup

### Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev

# Activate venv (if using one)
source venv/bin/activate
```

### Install Required Packages

The script uses existing packages from the repo, but ensure these are installed:

```bash
pip install opencv-python numpy torch torchvision pyiqa loguru
```

---

## Usage Guide

### Single Video Pair Validation

#### Upscaling Task
```bash
python3 local_validation.py \
    --reference path/to/reference_video.mp4 \
    --processed path/to/processed_video.mp4 \
    --task upscaling \
    --content-length 10 \
    --verbose
```

#### Compression Task
```bash
python3 local_validation.py \
    --reference path/to/original.mp4 \
    --processed path/to/compressed.mp4 \
    --task compression \
    --vmaf-threshold 85 \
    --verbose
```

### Batch Folder Validation

The `--folder` option scores all video pairs in a directory following naming conventions:

```bash
python3 local_validation.py \
    --folder ./output_videos \
    --task upscaling \
    --content-length 5 \
    --output results.json
```

#### Expected Folder Structure

For upscaling:
```
output_videos/
├── video1_ref.mp4
├── video1_proc.mp4
├── video2_reference.mp4
├── video2_processed.mp4
└── ...
```

For compression:
```
output_videos/
├── video1_ref.mp4
├── video1_proc.mp4
├── video2_reference.mp4
├── video2_processed.mp4
└── ...
```

**Naming Conventions:**
- Reference: `*_ref.mp4` or `*_reference.mp4`
- Processed: `*_proc.mp4` or `*_processed.mp4`

### Full Command Examples

#### Compare multiple upscaled videos
```bash
python3 local_validation.py \
    --folder ./upscaled_outputs \
    --task upscaling \
    --content-length 20 \
    --output upscale_scores.json \
    --verbose
```

#### Batch validate compressions with custom threshold
```bash
python3 local_validation.py \
    --folder ./compressed_videos \
    --task compression \
    --vmaf-threshold 90 \
    --output compression_results.json
```

---

## Output Interpretation

### Single Video Output

```
🎬 Starting Single Video Pair Validation...
   📹 Task: UPSCALING

   📹 Task: UPSCALING
   📊 Reference: original.mp4
   🎬 Processed: upscaled.mp4
   🎯 VMAF Score: 88.45
   🎨 PIE-APP Score: 0.8234
   📈 Quality Score: 0.7123
   📏 Length Score: 0.4155
   ✅ Final Score: 0.2856
   📝 Reason: success

✅ Score: 0.2856
```

### Batch Output Summary

```
============================================================
BATCH SUMMARY
============================================================
✅ Successful: 5/5
❌ Failed: 0/5

📊 Score Statistics:
   Average: 0.3124
   Min: 0.2145
   Max: 0.4567
   Std Dev: 0.0823

💾 Results saved to: results.json
```

### JSON Output Format

```json
{
  "task_type": "upscaling",
  "reference": "/path/to/ref.mp4",
  "processed": "/path/to/proc.mp4",
  "vmaf_score": 88.45,
  "pieapp_score": 0.8234,
  "quality_score": 0.7123,
  "length_score": 0.4155,
  "final_score": 0.2856,
  "reason": "success",
  "success": true
}
```

---

## Comparison with Top Miners

After running local validation, compare your scores:

### For Upscaling
- **Top Tier**: Final score > 0.40
- **Mid Tier**: Final score 0.25 - 0.40
- **Entry Level**: Final score 0.10 - 0.25

### For Compression
- **Top Tier**: Final score > 0.90
- **Mid Tier**: Final score 0.60 - 0.90
- **Entry Level**: Final score 0.30 - 0.60

---

## Troubleshooting

### Issue: "Invalid reference video"
**Solution**: Ensure video is valid MP4 with H.264/H.265 codec
```bash
ffprobe -v error -show_entries stream=codec_type your_video.mp4
```

### Issue: "Video too short (ref: X, proc: Y)"
**Solution**: Videos must have minimum 10 frames
```bash
ffmpeg -i your_video.mp4 -vf "select='isnan(prev_selected_t)+gte(t\,0)',setpts='(RTCSTART-RTSTART)/(TB)*N[expr_pts_start]+PTS" output.mp4
```

### Issue: "VMAF calculation error"
**Solution**: Ensure videos have same resolution
```bash
ffprobe -select_streams v:0 -show_entries stream=width,height your_video.mp4
```

### Issue: "PIE-APP calculation error" / CUDA out of memory
**Solution**: Reduce batch size or use CPU
```bash
# Script automatically falls back to CPU if CUDA unavailable
```

---

## Advanced Usage

### Custom VMAF Threshold
```bash
python3 local_validation.py \
    --reference ref.mp4 \
    --processed proc.mp4 \
    --task compression \
    --vmaf-threshold 90  # Stricter threshold
```

### Custom Content Length (for upscaling rewards)
```bash
python3 local_validation.py \
    --reference ref.mp4 \
    --processed proc.mp4 \
    --task upscaling \
    --content-length 50  # Longer content = higher potential score
```

### Export Results for Analysis
```bash
python3 local_validation.py \
    --folder ./videos \
    --task upscaling \
    --output results.json

# Analyze with Python
import json
with open('results.json') as f:
    results = json.load(f)
    avg_score = sum(r['final_score'] for r in results) / len(results)
    print(f"Average score: {avg_score:.4f}")
```

---

## Miner Best Practices

1. **Validate Before Submit**: Always run local validation before pushing results
2. **Track Performance**: Keep results.json files to monitor trends
3. **Target 85+ VMAF**: This is the baseline for upscaling rewards
4. **For Upscaling**: Focus on perceptual quality (PIE-APP), not just resolution
5. **For Compression**: Balance file size reduction with quality preservation
6. **Batch Test**: Use --folder to validate multiple outputs at once
7. **Monitor Logs**: Use --verbose to debug issues

---

## Script Arguments Reference

```
--reference PATH          Path to reference video file
--processed PATH          Path to processed video file
--task {upscaling,compression}  Task type (default: upscaling)
--vmaf-threshold FLOAT    VMAF threshold (default: 85.0)
--content-length FLOAT    Content length in seconds (default: 5.0)
--folder PATH             Path to folder with video pairs
--output PATH             Output JSON file for batch results
--verbose                 Enable verbose output
--help                    Show this help message
```

---

## Integration with Miner Workflow

```bash
# 1. Generate output videos (your processing logic)
python3 your_miner.py --input input.mp4 --output output.mp4

# 2. Validate locally
python3 local_validation.py \
    --reference input.mp4 \
    --processed output.mp4 \
    --task upscaling

# 3. If score is good, submit to network
# 4. Track results for performance analysis
python3 local_validation.py \
    --folder ./daily_outputs \
    --task upscaling \
    --output daily_report.json
```

---

## Performance Tips

- **GPU Acceleration**: Script uses CUDA if available (3-5x faster)
- **Batch Processing**: Use --folder to score 10+ videos efficiently
- **Memory**: Each VMAF calculation uses ~500MB RAM
- **Time**: Single upscaling validation: ~2-5 minutes
- **Time**: Single compression validation: ~1-3 minutes

---

## Questions & Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs with --verbose flag
3. Ensure all dependencies are installed
4. Validate input videos are properly formatted

---

**Happy mining! 🚀**
