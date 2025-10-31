# Vidaio Scoring System - Complete Explanation

## Overview

The Vidaio subnet uses a sophisticated scoring system that rewards miners based on **video processing quality** and **consistency**. The system is **100% performance-based** - stake only affects request priority, not rewards.

## Exact Scoring Formulas

### 1. Upscaling Tasks

#### Quality Score (S_Q)
```
S_Q = f(PIE-APP_score, VMAF_score)

Where:
- VMAF threshold check: VMAF_score/100 >= 0.5 (50%)
- If VMAF fails: S_Q = 0
- If VMAF passes: S_Q = sigmoid_transformation(PIE-APP_score)

PIE-APP transformation:
1. sigmoid_normalized = 1 / (1 + exp(-PIE-APP_score))
2. original_value = (1 - (log10(sigmoid_normalized + 1) / log10(3.5))) ** 2.5
3. S_Q = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
```

#### Length Score (S_L)
```
S_L = log(1 + content_length) / log(1 + 320)

Where:
- content_length = video duration in seconds
- Maximum S_L = 1.0 (at 320 seconds)
```

#### Final Score (S_F)
```
S_pre = 0.5 × S_Q + 0.5 × S_L
S_F = 0.1 × exp(6.979 × (S_pre - 0.5))

This exponential transformation amplifies differences:
- S_pre = 0.5 → S_F = 0.1
- S_pre = 0.6 → S_F = 0.18
- S_pre = 0.7 → S_F = 0.32
- S_pre = 0.8 → S_F = 0.58
```

### 2. Compression Tasks

#### Compression Score (S_f)
```
S_f = w_c × compression_component + w_q × quality_component

Where:
- w_c = 0.8 (compression weight)
- w_q = 0.2 (quality weight)

Quality component:
- If VMAF < threshold - 5: 0 (hard cutoff)
- If VMAF < threshold: gradual recovery zone
- If VMAF >= threshold: full scoring with bonus

Compression component:
- If compression_rate >= 0.95: 0 (no compression)
- If compression_rate >= 0.80: poor compression penalty
- Otherwise: exponential reward curve
```

## Performance Multipliers

### Rolling 10-Round Window
All multipliers are calculated using the **last 10 rounds** of performance:

### Upscaling Multipliers
```
Bonus Multiplier:
- Count rounds where S_f > 0.32
- bonus_multiplier = 1.0 + (bonus_count/10) × 0.15
- Maximum: 1.15 (+15%)

Performance Penalty:
- Count rounds where S_f < 0.07
- penalty_f_multiplier = 1.0 - (penalty_f_count/10) × 0.20
- Maximum penalty: -20%

Quality Penalty:
- Count rounds where S_q < 0.5
- penalty_q_multiplier = 1.0 - (penalty_q_count/10) × 0.25
- Maximum penalty: -25%

Total Multiplier = bonus_multiplier × penalty_f_multiplier × penalty_q_multiplier
```

### Compression Multipliers
```
Bonus Multiplier:
- Count rounds where S_f > 0.74
- bonus_multiplier = 1.0 + (bonus_count/10) × 0.15

Performance Penalty:
- Count rounds where S_f < 0.4
- penalty_f_multiplier = 1.0 - (penalty_f_count/10) × 0.20

Quality Penalty:
- Count rounds where VMAF < threshold
- penalty_q_multiplier = 1.0 - (penalty_q_count/10) × 0.30
```

## Performance Tiers

Based on **Accumulated Score** (not stake):

| Tier | Multiplier | Description |
|------|------------|-------------|
| **Elite** | 16.3× | Top performers with consistent high quality |
| **Good** | 2.0× | Solid performance with good consistency |
| **Average** | 1.5× | Decent performance with some penalties |
| **Poor** | 1.0× | Inconsistent performance with penalties |

## Stake vs Performance

### What Stake Affects
- **Request Priority**: Higher stake = more likely to get tasks
- **Rate Limiting**: Higher stake = more requests per round
- **Validator Selection**: Stake affects validator selection probability

### What Stake Does NOT Affect
- **Rewards**: 100% performance-based
- **Scoring**: Same formulas for all miners
- **Multipliers**: Based only on performance history
- **Tier Assignment**: Based only on accumulated score

## Accumulated Score Calculation

```
Accumulated_Score = decay_factor × old_score + (1 - decay_factor) × new_score

Where:
- decay_factor = 0.9 (from CONFIG.score.decay_factor)
- new_score = S_f × total_multiplier
```

## Local Validation Tools

### 1. local_validation.py
```bash
# Score upscaling task
python local_validation.py --task upscaling --reference video.mp4 --processed upscaled.mp4 --content-length 30

# Score compression task
python local_validation.py --task compression --reference video.mp4 --processed compressed.mp4 --threshold 85

# Calculate multipliers from history
python local_validation.py --calculate-multipliers --history-folder ./performance_history
```

### 2. calculate_multipliers.py
```bash
# Analyze performance history
python calculate_multipliers.py --history-folder ./performance_history --trends

# Save results
python calculate_multipliers.py --history-folder ./performance_history --output results.json
```

### 3. analyze_subnet.py
```bash
# Analyze subnet data
python analyze_subnet.py --top-n 20

# Analyze specific miner
python analyze_subnet.py --miner-uid 123

# Save analysis
python analyze_subnet.py --output subnet_analysis.json
```

## Key Insights

### 1. Quality is Paramount
- VMAF threshold must be met (50% for upscaling)
- PIE-APP scores directly affect quality component
- Poor quality = immediate failure

### 2. Consistency is Rewarded
- 10-round rolling window prevents gaming
- Consistent performance = higher multipliers
- Inconsistent performance = penalties

### 3. Exponential Scaling
- Small improvements in S_pre lead to large S_f increases
- Elite miners earn 16× more than poor miners
- Performance gaps compound over time

### 4. No Stake Advantage
- All miners use same scoring formulas
- Rewards purely based on video quality
- Meritocratic system

## Example Calculations

### Upscaling Example
```
Input:
- VMAF_score = 85 (passes 50% threshold)
- PIE-APP_score = 0.8 (good quality)
- content_length = 30 seconds

Calculations:
- S_Q = 0.75 (from PIE-APP transformation)
- S_L = 0.65 (from length formula)
- S_pre = 0.5 × 0.75 + 0.5 × 0.65 = 0.70
- S_F = 0.1 × exp(6.979 × (0.70 - 0.5)) = 0.32

Result: Good performance (S_F > 0.32 threshold)
```

### Compression Example
```
Input:
- VMAF_score = 90 (above 85 threshold)
- compression_rate = 0.1 (10x compression)

Calculations:
- compression_component = 0.95 (excellent compression)
- quality_component = 0.85 (good VMAF)
- S_f = 0.8 × 0.95 + 0.2 × 0.85 = 0.93

Result: Excellent performance (S_f > 0.74 threshold)
```

## Performance Optimization Tips

### 1. Focus on Quality
- Ensure VMAF scores consistently above threshold
- Optimize PIE-APP scores through better upscaling
- Use high-quality reference videos

### 2. Maintain Consistency
- Avoid processing failures
- Keep S_f scores above penalty thresholds
- Monitor performance trends

### 3. Optimize Processing
- Use efficient compression algorithms
- Balance quality vs compression ratio
- Process videos of appropriate length

### 4. Monitor Performance
- Use local validation tools
- Track multiplier changes
- Analyze performance trends

## Conclusion

The Vidaio scoring system is designed to reward **quality** and **consistency** above all else. While stake affects task allocation, rewards are purely meritocratic. Elite miners who consistently produce high-quality video processing can earn 16× more rewards than poor performers, making this a highly competitive and rewarding system for skilled miners.

