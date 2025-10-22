# Vidaio Subnet - Miner Algorithms & Task Types Guide

## 📋 Quick Reference

**Current Upscaling Tasks:**
- `SD2HD`: 1920×1080 (2x upscale)
- `HD24K`: 3840×2160 (2x upscale)
- `SD24K`: 3840×2160 (4x upscale)
- `4K28K`: 7680×4320 (4x upscale) - 8K Ultra

**Current Compression Task:**
- Single compression task with configurable VMAF thresholds (80-95)

**Content Lengths Supported:**
- Currently: 5s and 10s
- Future: 20s, 40s, 80s, 160s, 320s

---

## Task Types Explained

### Upscaling Tasks

#### 1. SD2HD (Standard Definition to HD)
```
Input:  1280 × 720  (SD)
Output: 1920 × 1080 (Full HD)
Scale:  2x
Use Cases:
  - Old movies/TV shows restoration
  - Legacy content enhancement
  - Social media upconversion
VMAF Threshold: ≥ 85 (soft requirement)
Expected VMAF: 88-92+
```

#### 2. HD24K (HD to 4K)
```
Input:  1920 × 1080 (Full HD)
Output: 3840 × 2160 (4K UHD)
Scale:  2x
Use Cases:
  - Streaming content enhancement
  - Documentary upscaling
  - Professional footage improvement
VMAF Threshold: ≥ 85 (soft requirement)
Expected VMAF: 90-95+
```

#### 3. SD24K (SD to 4K Direct)
```
Input:  1280 × 720  (SD)
Output: 3840 × 2160 (4K UHD)
Scale:  4x (aggressive)
Use Cases:
  - Maximum quality improvement
  - Archive digitization
  - Content restoration
VMAF Threshold: ≥ 85 (harder to achieve)
Expected VMAF: 85-90+ (more challenging)
```

#### 4. 4K28K (4K to 8K)
```
Input:  3840 × 2160 (4K UHD)
Output: 7680 × 4320 (8K)
Scale:  2x
Use Cases:
  - Premium 8K content creation
  - Cinematic applications
  - Ultra-high quality
VMAF Threshold: ≥ 85 (very challenging)
Expected VMAF: 88-92+ (high difficulty)
GPU Memory: 24GB+ recommended
```

### Compression Task

#### Compression (Single Task Type)
```
Input:  Any resolution video (reference quality)
Output: Smaller file size with quality preservation
Codec:  AV1 (VP9/HEVC also supported)
VMAF Threshold: Configurable per request (typically 80-95)

Quality Tiers:
- VMAF 80-85: Aggressive compression (10-20x)
- VMAF 85-90: Balanced (5-10x)
- VMAF 90-95: Conservative (2-5x)
- VMAF 95+: Near-lossless (1-2x)
```

---

## Validator Expectations

### What Validators Check

#### 1. **Task Warranty Protocol**
Validators query miners to confirm capabilities:
```python
class TaskWarrantProtocol:
    warrant_task: TaskType  # UPSCALING or COMPRESSION
```

**Your Miner Must Declare:**
- ✅ Can you handle upscaling? → TaskType.UPSCALING
- ✅ Can you handle compression? → TaskType.COMPRESSION
- ✅ Mix of both supported

### 2. **Length Check Protocol**
Validators verify content processing capability:
```python
class LengthCheckProtocol:
    max_content_length: ContentLength  # 5 or 10 seconds
```

**Your Miner Must Declare:**
- ✅ Can process 5-second videos: ContentLength.FIVE
- ✅ Can process 10-second videos: ContentLength.TEN

### 3. **Task Payload Requirements**

#### Upscaling Payload
```python
class UpscalingMinerPayload:
    reference_video_url: str           # URL to low-res video
    maximum_optimized_size_mb: int     # Output size limit (typically 100MB)
    task_type: str                     # "SD2HD", "HD24K", "SD24K", "4K28K"
```

**Your Miner Must:**
- Download the reference video
- Apply appropriate upscaling for task_type
- Keep output ≤ maximum_optimized_size_mb
- Return URL to processed video
- Processing time: < 2 min (5s clip recommended)

#### Compression Payload
```python
class CompressionMinerPayload:
    reference_video_url: str           # URL to high-quality video
    vmaf_threshold: float              # 80-95 (quality target)
```

**Your Miner Must:**
- Download the reference video
- Compress while maintaining VMAF ≥ vmaf_threshold
- Maximize compression ratio
- Return URL to compressed video
- Use AV1 codec (Main profile, yuv420p)
- Processing time: < 1 min (5s clip recommended)

### 4. **VMAF Quality Thresholds**

Current network thresholds in validator:
```python
VMAF_QUALITY_THRESHOLDS = [
    85,  # Low quality requirement
    90,  # Medium quality requirement
    95,  # High quality requirement
]
```

**Scoring Impact:**
- Below 85: Usually receives 0 score (hard fail)
- 85-90: Partial credit (soft zone)
- 90-95: Full credit tier
- 95+: Bonus rewards possible

---

## Latest Algorithms (2024-2025)

### For Upscaling

#### 1. **ESRGAN (Enhanced Super-Resolution GAN)**
✅ **Status**: Industry standard, widely adopted
- **What**: Deep learning GAN for image/video super-resolution
- **Advantages**:
  - Excellent detail preservation
  - Handles diverse image types
  - Good temporal consistency for video
  - Fast inference
- **Performance**:
  - VMAF: 88-94 typically
  - Speed: 2-5x real-time on RTX 4090
  - Memory: 2-4GB per frame
- **Implementation**:
  ```python
  # RealESRGAN or BasicSR libraries
  from realesrgan import RealESRGANer
  upsampler = RealESRGANer(scale=2, model_name='RealESRGAN_x2plus')
  ```

#### 2. **Pixop Super Resolution**
✅ **Status**: Commercial/Research, high quality
- **What**: ML-based super-resolution trained on large datasets
- **Advantages**:
  - Up to 4x upscaling accuracy
  - Robust and predictable
  - Handles video efficiently
- **Performance**:
  - VMAF: 89-95
  - Speed: Real-time capable
  - Quality: Best for extreme scales (4x)

#### 3. **BSRGAN (Blind Super-Resolution GAN)**
✅ **Status**: Advanced, handles real degradation
- **What**: GAN that handles real-world image degradation
- **Advantages**:
  - Handles various compression types
  - Better generalization
  - Good for organic/user video
- **Performance**:
  - VMAF: 87-92
  - Speed: 3-8x real-time
  - Best for: Mixed quality inputs

#### 4. **Face Super-Resolution (Optional)**
🔧 **Status**: Specialized enhancement
- **What**: Face-specific SR for facial clarity
- **Best For**: Upscaling videos with faces
- **Combination**: ESRGAN + Face SR for best results
- **Performance Impact**:
  - VMAF boost: +2-3 points for face-heavy content
  - Processing time: +20-30% overhead

### For Compression

#### 1. **AV1 Codec with Advanced Settings**
✅ **Status**: Network standard, preferred
- **What**: State-of-the-art video codec
- **Advantages**:
  - Best compression ratio (20-30% better than HEVC)
  - Quality preservation at low bitrates
  - Required by validator spec
- **Settings for Vidaio**:
  ```bash
  # Key parameters for balance
  -c:v libaom-av1              # AV1 encoder
  -crf 30-45                   # Quality (lower = better)
  -preset 4-6                  # 4=slow/quality, 6=faster
  -pix_fmt yuv420p             # Required format
  -profile:v main              # Required profile
  ```
- **Performance**:
  - Compression: 8-15x typical
  - Quality: VMAF 90+ achievable
  - Speed: 10-50% real-time (preset dependent)

#### 2. **HEVC/H.265 (Alternative)**
⚠️ **Status**: Supported but slower than AV1
- **What**: Older but proven codec
- **Advantages**:
  - Faster encoding than AV1
  - Widely compatible
  - Good quality/size ratio
- **Performance**:
  - Compression: 5-10x typical
  - Quality: VMAF 88+ achievable
  - Speed: 50%+ real-time

#### 3. **Bitrate Optimization Algorithms**
🔧 **Status**: Critical for score maximization
- **Approach**: Two-pass encoding
  ```bash
  # Pass 1: Analyze
  ffmpeg -i input.mp4 -c:v libaom-av1 -pass 1 -f null /dev/null
  
  # Pass 2: Encode with optimal bitrate
  ffmpeg -i input.mp4 -c:v libaom-av1 -pass 2 output.mp4
  ```

#### 4. **Scene-Aware Compression**
🔧 **Status**: Advanced optimization
- **What**: Adjust encoding per scene complexity
- **Approach**:
  1. Detect scene changes
  2. Adjust CRF per scene (complex: lower CRF, static: higher CRF)
  3. Maintain VMAF threshold across all frames
- **Tools**: FFmpeg scenedetect, custom analysis
- **Benefit**: +5-10% compression vs fixed bitrate

#### 5. **Perceptual Quality Optimization**
🔧 **Status**: Advanced, experimental
- **What**: Use VMAF feedback to optimize
- **Approach**:
  ```
  1. Encode with initial CRF
  2. Calculate VMAF on sample frames
  3. If VMAF < threshold: reduce CRF, retry
  4. If VMAF > threshold + margin: increase CRF, retry
  5. Binary search for optimal CRF
  ```
- **Tools**: FFmpeg + custom VMAF loop

---

## Task Difficulty Rankings

### Upscaling Difficulty vs Potential Reward

| Task | Difficulty | VMAF Challenge | Potential Score | Notes |
|------|-----------|-----------------|-----------------|-------|
| SD2HD | ⭐ Easy | Lower artifact risk | 0.35-0.45 | Good entry point |
| HD24K | ⭐⭐ Medium | Moderate complexity | 0.30-0.40 | Most common |
| SD24K | ⭐⭐⭐ Hard | 4x scale challenge | 0.20-0.35 | Rare, high reward |
| 4K28K | ⭐⭐⭐⭐ Very Hard | 8K extremely difficult | 0.15-0.30 | Premium miners only |

### Compression Difficulty vs Potential Reward

| VMAF Threshold | Difficulty | Compression Target | Potential Score | Notes |
|----------------|-----------|-------------------|-----------------|-------|
| 80 | ⭐ Easy | 15-20x | 0.85-1.00 | Aggressive |
| 85 | ⭐⭐ Medium | 10-15x | 0.70-0.90 | Balanced |
| 90 | ⭐⭐⭐ Hard | 5-10x | 0.50-0.75 | Careful |
| 95 | ⭐⭐⭐⭐ Very Hard | 2-5x | 0.30-0.60 | Conservative |

---

## Recommended Algorithm Stack for Vidaio

### Tier 1: Competitive (Top 50%)
**Upscaling:**
- ESRGAN (2x upscaling)
- Quality optimization
- Frame interpolation (optional)

**Compression:**
- AV1 codec (preset 6, CRF 35-42)
- Two-pass encoding
- Basic bitrate optimization

**Expected Scores:**
- Upscaling: 0.28-0.40
- Compression: 0.65-0.85

### Tier 2: Advanced (Top 10%)
**Upscaling:**
- ESRGAN + Face Super-Resolution
- Temporal consistency enforcement
- Post-processing filters

**Compression:**
- AV1 codec (preset 4, CRF 30-38)
- Scene-aware encoding
- VMAF-guided CRF search
- Advanced denoising

**Expected Scores:**
- Upscaling: 0.38-0.50
- Compression: 0.80-1.00

### Tier 3: Elite (Top 1%)
**Upscaling:**
- Custom trained models on Vidaio data
- Ensemble methods (ESRGAN + BSRGAN + custom)
- Advanced temporal interpolation
- Context-aware upscaling

**Compression:**
- Custom AV1 parameters per content type
- Neural network bitrate prediction
- Perceptual loss functions
- Adaptive frame allocation

**Expected Scores:**
- Upscaling: 0.45-0.55+
- Compression: 0.90-1.00+

---

## Network Quality Levels (No F4 in Current Code)

⚠️ **Note**: No "F4" task found in codebase. Current tasks are:
- SD2HD (1080p target)
- HD24K (4K target)
- SD24K (4K target)
- 4K28K (8K target)

**If F4 Mentioned Elsewhere:**
- Likely refers to "Frame Rate 4x" interpolation (future feature)
- Currently not implemented in validators
- Future roadmap feature for Phase 5 (Live Streaming)

---

## Validator Protocol Requirements (from core code)

### @vidaio_subnet_core/protocol.py
```python
# Task types validators send
class TaskType(IntEnum):
    COMPRESSION = 1
    UPSCALING = 2

# What validators expect back
class UpscalingMinerPayload:
    task_type: str  # "SD2HD", "HD24K", "SD24K", "4K28K"
    maximum_optimized_size_mb: int
    
class CompressionMinerPayload:
    vmaf_threshold: float  # 80-95

# Response format
class MinerResponse:
    optimized_video_url: str  # Must be valid URL to processed video
```

### Validator Scoring Protocol (from neurons/validator.py)
```python
# Validators check VMAF thresholds
VMAF_QUALITY_THRESHOLDS = [85, 90, 95]

# Validators measure:
# 1. VMAF score (frame quality)
# 2. PIE-APP score (upscaling perceptual quality)
# 3. Compression rate (file size reduction)
# 4. Processing time (implicit penalty if too slow)

# Historical tracking in database:
# - MinerMetadata.processing_task_type
# - MinerMetadata.average_vmaf
# - MinerPerformanceHistory.processed_task_type
```

### Video Quality Constraints (from services/video_scheduler/worker.py)
```python
RESOLUTIONS = {
    "HD24K": (3840, 2160),    # 4K
    "SD2HD": (1920, 1080),    # Full HD
    "SD24K": (3840, 2160),    # 4K
    # "4K28K": (7680, 4320),  # 8K (commented out)
}

# Weights for task selection (prevalence on network)
task_weights = {
    "HD24K": config.weight_hd_to_4k,
    "SD2HD": config.weight_sd_to_hd + config.weight_hd_to_4k,
    "SD24K": config.weight_sd_to_4k + config.weight_sd_to_hd,
    "4K28K": config.weight_4k_to_8k + ...,  # Rare
}
```

---

## Implementation Checklist

### Upscaling Miner Setup
- [ ] Install ESRGAN or RealESRGAN library
- [ ] Verify CUDA support for GPU acceleration
- [ ] Test on all 4 task types (SD2HD, HD24K, SD24K, 4K28K)
- [ ] Validate VMAF scores ≥ 85 on test videos
- [ ] Measure processing time (target: < 2 min for 5s clip)
- [ ] Set warrant_task = TaskType.UPSCALING
- [ ] Declare max_content_length capability
- [ ] Test with local_validation.py

### Compression Miner Setup
- [ ] Configure AV1 encoder (libaom-av1)
- [ ] Implement two-pass encoding
- [ ] Test VMAF threshold compliance (80-95)
- [ ] Achieve target compression ratios (5-15x)
- [ ] Measure processing time (target: < 1 min for 5s clip)
- [ ] Set warrant_task = TaskType.COMPRESSION
- [ ] Verify AV1 Main profile + yuv420p format
- [ ] Test with local_validation.py

### Network Integration
- [ ] Create Bittensor wallet
- [ ] Register on subnet netuid 85
- [ ] Start services with PM2
- [ ] Monitor logs for validator requests
- [ ] Track local scores vs network scores
- [ ] Optimize based on feedback

---

## Performance Monitoring

Track these metrics locally:
```bash
# After each video process
python3 local_validation.py \
    --reference ref.mp4 \
    --processed output.mp4 \
    --task upscaling/compression

# Key metrics to monitor
- VMAF score (target: 90+)
- PIE-APP score (upscaling, target: < 0.5)
- Compression ratio (compression, target: 8-15x)
- Processing time (target: < 2 min/clip)
- Final score (compare vs benchmarks)
```

---

## Summary

✅ **What Validators Expect:**
1. Declare your task capability (upscaling or compression)
2. Declare your content length capability (5s or 10s)
3. Process videos to the specified task_type or quality level
4. Achieve VMAF ≥ 85 (soft) to ≥ 90-95 (hard)
5. Return valid URL to processed video
6. Return within time limits

✅ **Best Algorithm Choices (2024-2025):**
- **Upscaling**: ESRGAN (baseline), Pixop (extreme 4x), BSRGAN (robustness)
- **Compression**: AV1 codec with scene-aware optimization and VMAF-guided search

✅ **Next Steps:**
1. Read validators/managing code for scoring logic
2. Review incentive_mechanism.md for scoring formulas
3. Implement chosen algorithms
4. Validate locally with provided script
5. Start mining! 🚀
