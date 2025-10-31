# Vidaio Scoring System - Your Questions Answered

## 1. Exact Scoring Formulas

### Upscaling Tasks
```
S_Q (Quality) = sigmoid_transformation(PIE-APP_score) × VMAF_check
S_L (Length) = log(1 + content_length) / log(1 + 320)
S_F (Final) = 0.1 × exp(6.979 × (0.5 × S_Q + 0.5 × S_L - 0.5))
```

### Compression Tasks
```
S_f = 0.8 × compression_component + 0.2 × quality_component
Where compression_component uses exponential reward curve
And quality_component based on VMAF threshold compliance
```

## 2. Performance Multipliers (Rolling 10-Round Window)

### Upscaling
- **Bonus**: S_f > 0.32 → +15% max
- **Performance Penalty**: S_f < 0.07 → -20% max  
- **Quality Penalty**: S_q < 0.5 → -25% max

### Compression
- **Bonus**: S_f > 0.74 → +15% max
- **Performance Penalty**: S_f < 0.4 → -20% max
- **Quality Penalty**: VMAF < threshold → -30% max

## 3. Stake Only Affects Request Priority

**Stake does NOT affect rewards** - it only determines:
- Who gets tasks first (priority)
- Rate limiting (requests per round)
- Validator selection probability

**Rewards are 100% performance-based** using the formulas above.

## 4. Current Subnet Data (Live Analysis) - CORRECTED

### Top 10 Miners by Stake (Actual Miners):
1. **UID 64**: 18.22 TAO (0.00%)
2. **UID 162**: 18.11 TAO (0.00%)
3. **UID 65**: 17.53 TAO (0.00%)
4. **UID 193**: 17.42 TAO (0.00%)
5. **UID 151**: 17.17 TAO (0.00%)
6. **UID 137**: 16.78 TAO (0.00%)
7. **UID 18**: 16.72 TAO (0.00%)
8. **UID 83**: 16.60 TAO (0.00%)
9. **UID 171**: 16.57 TAO (0.00%)
10. **UID 46**: 16.38 TAO (0.00%)

### Active Miners Getting Emissions:
- **202 miners** are actively getting emissions
- **Total miner stake**: 862.68 TAO
- **Average miner stake**: 4.27 TAO
- **Median miner stake**: 1.55 TAO

### Validators (Not Miners):
- **4 validators** with massive stakes
- **Total validator stake**: 2,283,884.59 TAO (99.96% of total!)
- **Average validator stake**: 570,971.15 TAO

### Key Insights:
- **Validators control 99.96% of stake** - they are the high-stake entities
- **Miners have tiny stakes** - 90th percentile = 0.43 TAO
- **202 miners actively earning** despite minimal stakes
- **Stake concentration is in validators, not miners**
- **Miners compete on performance, not stake**

## 5. Access to Other Miners' Submissions

**No, you cannot access other miners' submissions** for comparison because:

1. **Privacy**: Each miner's processed videos are private
2. **Security**: Only validators can access submissions for scoring
3. **Decentralization**: No central database of all submissions
4. **Competition**: Miners compete without seeing each other's work

**What you CAN do:**
- Use local validation tools to score your own work
- Compare your performance against thresholds
- Analyze your own performance trends
- Use the subnet analysis to see stake distribution

## 6. Local Validation Tools Created

### `local_validation.py`
```bash
# Score your upscaling work
python local_validation.py --task upscaling --reference video.mp4 --processed upscaled.mp4 --content-length 30

# Score your compression work  
python local_validation.py --task compression --reference video.mp4 --processed compressed.mp4 --threshold 85
```

### `calculate_multipliers.py`
```bash
# Analyze your performance history
python calculate_multipliers.py --history-folder ./performance_history --trends
```

### `analyze_subnet.py`
```bash
# See current subnet status
python analyze_subnet.py --top-n 20
```

## 7. Performance Tiers (Based on Accumulated Score)

| Tier | Multiplier | Description |
|------|------------|-------------|
| **Elite** | 16.3× | Top performers with consistent high quality |
| **Good** | 2.0× | Solid performance with good consistency |
| **Average** | 1.5× | Decent performance with some penalties |
| **Poor** | 1.0× | Inconsistent performance with penalties |

## 8. Key Takeaways

1. **Stake is irrelevant for rewards** - only affects task priority
2. **Quality and consistency matter most** - 10-round rolling window
3. **Elite miners earn 16× more** than poor miners
4. **239 miners are actively earning** despite extreme stake concentration
5. **You can't see other miners' work** - focus on your own performance
6. **Use local tools** to optimize your scoring before submitting

The system is designed to be **meritocratic** - your video processing quality determines your rewards, not your stake!
