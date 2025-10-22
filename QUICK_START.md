# Quick Start - Local Validation

## One-Minute Overview

```bash
cd /workspace/vidaio-subnet

# Validate a single upscaling pair
python3 local_validation.py \
    --reference reference.mp4 \
    --processed output.mp4 \
    --task upscaling

# Validate a single compression pair
python3 local_validation.py \
    --reference original.mp4 \
    --processed compressed.mp4 \
    --task compression

# Batch validate all videos in folder
python3 local_validation.py \
    --folder ./video_outputs \
    --task upscaling \
    --output results.json
```

## Common Use Cases

### 1. Before Submitting to Network
```bash
# Validate your output before pushing
python3 local_validation.py \
    --reference /path/to/input.mp4 \
    --processed /path/to/output.mp4 \
    --task upscaling \
    --verbose
```

### 2. Compare Multiple Outputs
```bash
# Score all your daily outputs against top miners
python3 local_validation.py \
    --folder ./daily_outputs \
    --task upscaling \
    --output daily_scores.json
```

### 3. Debug Quality Issues
```bash
# Use verbose mode to see detailed breakdown
python3 local_validation.py \
    --reference ref.mp4 \
    --processed proc.mp4 \
    --task upscaling \
    --verbose
```

### 4. Custom Parameters
```bash
# For compression with strict VMAF requirement
python3 local_validation.py \
    --reference orig.mp4 \
    --processed compressed.mp4 \
    --task compression \
    --vmaf-threshold 90

# For upscaling with longer content
python3 local_validation.py \
    --reference ref.mp4 \
    --processed proc.mp4 \
    --task upscaling \
    --content-length 50
```

## Expected Scores

### Upscaling
- **Excellent** (Top 10%): 0.35 - 0.50
- **Good** (Top 50%): 0.20 - 0.35
- **Average**: 0.10 - 0.20
- **Poor**: < 0.10

### Compression
- **Excellent** (Top 10%): 0.85 - 1.00
- **Good** (Top 50%): 0.60 - 0.85
- **Average**: 0.30 - 0.60
- **Poor**: < 0.30

## Required Metrics

Both tasks calculate:
- **VMAF**: Frame-by-frame quality (0-100)
- **Compression Rate**: For compression task (0-1, lower is better)
- **PIE-APP**: For upscaling task (0-2.0, lower is better)

## File Naming for Batch Mode

Place video pairs in a folder with this naming:

```
videos/
├── video_01_ref.mp4        # Reference video
├── video_01_proc.mp4       # Processed output
├── video_02_reference.mp4  # Or use _reference
├── video_02_processed.mp4  # Or use _processed
```

Then run:
```bash
python3 local_validation.py --folder ./videos --task upscaling --output scores.json
```

## Output Files

Single validation: Prints to console

Batch validation with `--output`:
```bash
# results.json contains:
# - All scores for each video
# - VMAF, PIE-APP, compression rates
# - Final scores
# - Success/failure reasons
```

Parse results:
```python
import json
with open('results.json') as f:
    data = json.load(f)
    for item in data:
        print(f"{item['reference']}: {item['final_score']:.4f}")
```

## Troubleshooting

**"Invalid video"** → Check codec: `ffprobe your_video.mp4`

**"Video too short"** → Min 10 frames required

**"VMAF calculation error"** → Ensure same resolution as reference

**Slow performance** → First run is slower; uses GPU if available

## Next Steps

1. ✅ Validate locally before every submission
2. 📊 Track scores in results.json files
3. 🎯 Compare with benchmark scores above
4. 🚀 Submit when confident
5. 📈 Monitor performance trends

For full documentation, see `LOCAL_VALIDATION_README.md`
