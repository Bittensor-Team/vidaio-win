# Vidaio Miner Economics & Incentives

## ✅ Installation Status

**npm/pm2 ARE installed correctly** - despite the dependency warnings shown during apt install, npm and pm2 are functional:
```bash
which npm    # /usr/bin/npm ✓
which pm2    # /usr/local/bin/pm2 ✓
```

The warnings are from apt's dependency resolver but the packages work fine. You can safely ignore those warnings.

---

## 🤔 Why Do Miners Need S3 Bucket?

### Architecture Overview

The Vidaio subnet uses a **centralized storage model** for processed videos:

```
Validator  -->  Miner  -->  S3 Bucket  <--  Validator (download & evaluate)
(sends task)    (process)   (store result)    (rate miner)
```

### Miner's Workflow

1. **Receive Task**: Validator sends video + parameters (upscale/compress)
2. **Process**: Miner uses GPU to process the video
3. **Upload to S3**: Store the processed result in your S3 bucket
4. **Return URL**: Send presigned URL to validator
5. **Validator Evaluates**: Validator downloads from S3 and rates quality
6. **Rewards**: Based on quality scores

### Why NOT Store Locally?

- **Validators are distributed** - they can't access your local server
- **Network efficiency** - S3 with presigned URLs is faster than P2P transfers
- **Scalability** - Multiple validators can download simultaneously
- **Permanence** - Videos stay available for future reference/appeals
- **Decoupling** - Miner doesn't need to stay online after upload

### S3 Bucket Requirements

Your bucket must support:
- ✓ File upload (PUT)
- ✓ Presigned URLs for download (GET with time limit)
- ✓ Public accessibility (for validators to download)
- ✓ Compatible with MinIO S3 API

**Recommended Providers** (all support MinIO API):
- **Backblaze B2**: $0.006/GB/month (cheapest)
- **Amazon S3**: $0.023/GB/month
- **Cloudflare R2**: $0.015/GB/month
- **Wasabi**: $0.004/GB/month (very cheap)

---

## 💰 How Are Emissions & Rewards Calculated?

### TL;DR: **Performance-Based NOT Stake-Based**

**Emissions Distribution:**
- 41% → All Miners (distributed by performance scores)
- 41% → All Validators (distributed by validation quality)
- 18% → Subnet Creator (Vidaio)

Within miners, rewards are **purely based on:**
1. **Accumulated Score** (from processing quality)
2. **Task Type Allocation** (60% compression, 40% upscaling)
3. **Performance Multipliers** (bonuses/penalties from history)

**Stake is NOT used for reward distribution** - only for request priority.

---

## 📊 Vidaio's Incentive Mechanism

### Two Task Types with Different Scoring

#### **1. UPSCALING TASKS** (40% of rewards)

**Goal**: Improve video quality by increasing resolution

**Scoring Components:**

```
Quality Score (S_Q) = PIE-APP Assessment
- Uses deep learning to measure perceptual quality
- Range: 0-1 (higher = better)
- Checks VMAF threshold first (safety)

Content Length Score (S_L) = log(1 + length) / log(1 + 320)
- Rewards processing longer videos
- Range: 0-1
- 5s = 31%, 10s = 41%, 320s = 100%

Preliminary Score = 0.5 × S_Q + 0.5 × S_L

Final Score (S_F) = 0.1 × e^(6.979 × (S_pre - 0.5))
- EXPONENTIAL transformation
- Creates massive rewards for high performance
- Range: 0.002 to 40+ (theoretical)
```

**Example Performance Tiers:**
| Score | Performance Tier | Relative Reward |
|-------|-----------------|-----------------|
| 0.30 | Poor | 0.25× |
| 0.60 | Good | 2.01× |
| 0.72 | Very High | 4.64× |
| 0.84 | Outstanding | 10.73× |
| 0.90 | Elite | 16.31× |

**Key Insight**: Elite miners get **16× more rewards** than poor miners on same task!

---

#### **2. COMPRESSION TASKS** (60% of rewards)

**Goal**: Reduce file size while maintaining quality

**Scoring Components:**

```
Compression Rate = compressed_size / original_size
- Lower is better (more compression)

Quality Check = VMAF score must exceed threshold
- If fails: Score = 0 (automatic failure)

Final Score (S_f) = 0.8 × (1 - C^1.5) + 0.2 × (VMAF - threshold) / (100 - threshold)
- 80% weight on compression efficiency
- 20% weight on quality above threshold
- Enforces quality-first approach
```

**Example Results:**
| Compression | VMAF | Result | Performance |
|------------|------|--------|-------------|
| 30% reduced | 85 | 0.769 | Excellent |
| 50% reduced | 80 | 0.584 | Good |
| 70% reduced | 75 | 0.364 | Average |
| 40% reduced | 65 | 0.000 | **FAILED** (quality below threshold) |

---

### Historical Performance Multipliers (Rolling 10-Round Window)

#### **Bonus System** (Excellence Rewards)

```
For Upscaling: If S_F > 0.32 in a round
For Compression: If S_f > 0.74 in a round

bonus_multiplier = 1.0 + (bonus_count / 10) × 0.15

Maximum Bonus: +15% (all 10 rounds excellent)
```

**Example:** 7/10 excellent rounds = +10.5% bonus multiplier

---

#### **Performance Penalty** (Consistency Enforcement)

```
For Upscaling: If S_F < 0.20 in a round
For Compression: If S_f < 0.4 in a round

penalty_f_multiplier = 1.0 - (penalty_f_count / 10) × 0.20

Maximum Penalty: -20%
```

**Example:** 4/10 poor rounds = -8% penalty multiplier

---

#### **Quality Penalty** (Strictest)

```
For Upscaling: If S_Q < 0.25 in a round
For Compression: If VMAF < threshold + 5 margin

penalty_q_multiplier = 1.0 - (penalty_q_count / 10) × 0.25

Maximum Penalty: -25% (strictest penalty)
```

**Why strictest?** Quality is non-negotiable - users care most about output quality.

---

### Real Miner Examples

#### Upscaling Miners:

| Miner Type | Avg S_F | Avg S_Q | Bonus | Penalties | Final Multiplier | Net Effect |
|-----------|---------|---------|-------|-----------|-----------------|-----------|
| **Elite** | 0.854 | 0.717 | 10/10 | 0 | **1.150×** | **+15.0%** |
| **Good** | 0.653 | 0.571 | 0/10 | 0 | **1.000×** | **±0.0%** |
| **Average** | 0.511 | 0.505 | 0/10 | 3/10 | **0.925×** | **-7.5%** |
| **Poor** | 0.311 | 0.411 | 0/10 | 10/10 | **0.600×** | **-40.0%** |

#### Compression Miners:

| Miner Type | Avg S_f | VMAF Margin | Bonus | Penalties | Final Multiplier | Net Effect |
|-----------|---------|-----------|-------|-----------|-----------------|-----------|
| **Elite Compressor** | 0.654 | +15 | 10/10 | 0 | **1.150×** | **+15.0%** |
| **Good Compressor** | 0.453 | +8 | 0/10 | 0 | **1.000×** | **±0.0%** |
| **Average** | 0.311 | +3 | 0/10 | 0 | **0.910×** | **-9.0%** |
| **Poor Compressor** | 0.211 | -2 | 0/10 | 10/10 | **0.490×** | **-51.0%** |

---

## 📈 Reward Distribution Algorithm

### Step 1: Collect Scores by Task Type

```
Compression Miners:
  - UID1: accumulate_score = 847.5
  - UID3: accumulate_score = 625.3
  - UID7: accumulate_score = 412.1
  Total: 1885.0

Upscaling Miners:
  - UID2: accumulate_score = 523.8
  - UID5: accumulate_score = 411.2
  Total: 935.0
```

### Step 2: Allocate Rewards by Task Type

```
Total Pool = 100 TAO (per round example)

Compression Allocation: 60% = 60 TAO
Upscaling Allocation: 40% = 40 TAO
```

### Step 3: Distribute Within Each Task Type

```
COMPRESSION (60 TAO total):
- UID1: (847.5 / 1885.0) × 60 = 27.0 TAO
- UID3: (625.3 / 1885.0) × 60 = 19.9 TAO
- UID7: (412.1 / 1885.0) × 60 = 13.1 TAO

UPSCALING (40 TAO total):
- UID2: (523.8 / 935.0) × 40 = 22.4 TAO
- UID5: (411.2 / 935.0) × 40 = 17.6 TAO
```

### Step 4: Apply Performance Multipliers

```
UID1: 27.0 TAO × 1.150 (elite bonus) = 31.05 TAO ✓ +15% from excellence
UID2: 22.4 TAO × 1.000 (neutral) = 22.4 TAO
UID3: 19.9 TAO × 0.925 (penalty) = 18.41 TAO ✗ -7.5% from inconsistency
UID5: 17.6 TAO × 0.600 (poor) = 10.56 TAO ✗ -40% from poor performance
UID7: 13.1 TAO × 1.000 (neutral) = 13.1 TAO
```

---

## 🔗 Metagraph & Rankings

### What is the Metagraph?

The Bittensor metagraph is the **on-chain state** that tracks:

```python
metagraph.S[uid]              # Stake of each UID
metagraph.validator_permit[uid]  # Is this UID a validator? (bool)
metagraph.hotkeys[uid]        # Hotkey string for each UID
```

### Vidaio's Mining Metagraph

**In Vidaio subnet 85:**
- Only **validators** can query miners
- Only **validators** get to set weights (distribute rewards)
- **Miners** are ranked by their accumulated scores
- **Rankings update every block** (~12 seconds)

### Where You Can Check Rankings

```bash
# 1. Use Bittensor CLI
btcli subnet metagraph --netuid 85 --subtensor.network finney

# 2. Visit Web Dashboard
https://taostats.io/subnets/85/  # Shows all miners ranked by stake

# 3. Query via Python
import bittensor as bt
subtensor = bt.subtensor("finney")
metagraph = subtensor.metagraph(85)
print(metagraph.S)  # All stakes
print(metagraph.validator_permit)  # Validator flags
```

### Performance Ranking in Vidaio

**Stored in Database** (not on-chain):

```
MinerMetadata table contains:
- uid
- accumulate_score (your total performance score)
- bonus_multiplier (excellence rewards)
- penalty_f_multiplier (performance penalties)
- penalty_q_multiplier (quality penalties)
- total_multiplier (combined effect)
- avg_s_q, avg_s_l, avg_s_f (average scores)
- performance_tier (Elite/Good/Average/Poor)
- success_rate
- longest_content_processed
```

**You can check this via:**

```bash
# Direct database query (needs access)
SELECT * FROM miner_metadata ORDER BY accumulate_score DESC LIMIT 10;

# Via validator API (once implemented)
curl https://validator.vidaio.io/metrics/rankings
```

---

## 🎯 Key Takeaways for Miners

### 1. **Performance > Stake**
- Your rewards are **100% based on output quality**
- Stake only matters for request priority (not rewards)
- A high-performing miner with low stake beats a low-performing miner with high stake

### 2. **Exponential Rewards for Excellence**
- Small performance improvements = massive reward gains
- Elite miners earn **16× more** than poor miners
- Incentive system aggressively rewards excellence

### 3. **Consistency Matters**
- Bonus/penalty multipliers use **rolling 10-round windows**
- Sustained excellence = +15% ongoing bonus
- One bad round doesn't hurt much, but 10 bad rounds = -40% to -51%

### 4. **Quality is Paramount**
- Quality penalties are strongest (-25% max)
- VMAF threshold is a hard floor (score = 0 if violated)
- 5-point safety margin enforced (VMAF_score > threshold + 5)

### 5. **Task Allocation**
- **60% of rewards** go to compression tasks (more demand)
- **40% of rewards** go to upscaling tasks
- Pick your specialization based on hardware capabilities

### 6. **S3 is Non-Negotiable**
- Validators cannot evaluate your work without accessing uploaded files
- S3 bucket acts as your "submission portal"
- Must be reliable and fast (validators wait for downloads)

---

## 📋 Strategy for Maximum Rewards

### For New Miners:

1. **Phase 1: Get Baseline** (Rounds 1-10)
   - Focus on consistency (avoid penalties)
   - Target S_pre > 0.5 for upscaling
   - Maintain VMAF > threshold + 10 for compression
   - No bonus yet, focus on avoiding penalties

2. **Phase 2: Build Excellence** (Rounds 11-20)
   - Push for S_pre > 0.6 (upscaling)
   - Optimize compression rate without quality loss
   - Start accumulating bonus history
   - Aim for 7/10 excellent rounds = +10.5% bonus

3. **Phase 3: Elite Status** (Rounds 21+)
   - Target S_pre > 0.7 consistently (upscaling)
   - Target S_f > 0.74 consistently (compression)
   - Maintain 10/10 excellent rounds = +15% bonus
   - **16× multiplier on base scores**

### GPU Optimization Tips:

```
For Upscaling:
- Invest in model quality (better AI = higher PIE-APP scores)
- Process longer videos when possible (higher S_L scores)
- Focus on quality over speed (S_Q is half the score)

For Compression:
- Implement smart compression algorithms
- Maintain quality buffer above threshold (not minimum)
- Test extensively before deploying (failures = 0 score)
```

---

## 🔄 Emission Cycle Details

### Per Mining Round:
1. **Validators issue tasks** (~100-500 per round depending on network load)
2. **Miners process videos** (5-10 seconds each, parallel processing)
3. **Validators download results** and evaluate using VMAF/PIE-APP
4. **Scores recorded** in database
5. **Weights calculated** based on accumulated scores + multipliers
6. **Rewards distributed** on-chain (next block)

### Annual Emission (Example):
```
Assuming:
- 100 TAO per round
- 35 rounds per day
- 365 days per year

Total Annual: 100 × 35 × 365 = 1,277,500 TAO emitted to subnet

Split:
- 41% to miners: 523,475 TAO
- 41% to validators: 523,475 TAO
- 18% to creator: 229,950 TAO

If you're top miner (elite rank):
- Your share of miner pool depends on your score
- With 10% of total miner score:
  523,475 × 10% = ~52,347 TAO/year
  = ~143 TAO/day (at current prices, ~$600-800/day)
```

---

## ⚠️ What You CANNOT Rely On

**These DO NOT guarantee rewards:**
- ✗ Stake amount (even validators with high stake get low rewards if validation quality is poor)
- ✗ Account age (new miners can earn more than old ones)
- ✗ Hotkey registration (only performance matters)
- ✗ Processing speed (slower but higher quality beats fast but lower quality)

---

## ✅ What DOES Guarantee Rewards

**These ARE the determinants:**
- ✓ **Quality Score (PIE-APP)** - Primary for upscaling
- ✓ **Compression Rate + Quality** - Primary for compression
- ✓ **Content Length** - Secondary for upscaling (video duration)
- ✓ **Consistency** - Rolling 10-round multipliers
- ✓ **VMAF Threshold Compliance** - Must exceed minimum

---

## 🔍 References in Codebase

**Incentive Logic:**
```
miner_manager.py:1132-1164  # Weight calculation
  - 60% for compression miners
  - 40% for upscaling miners
  - Proportional by accumulated_score
```

**Priority (NOT Rewards):**
```
miner.py:138-151, 202-215  # Priority calculation
  - priority = metagraph.S[uid]  # Stake-based only
  - Higher stake = higher priority in queue
  - Does NOT affect rewards
```

**Database Schema:**
```
sql_schemas.py:1-58  # MinerMetadata
  - accumulate_score
  - bonus_multiplier, penalty_f_multiplier, penalty_q_multiplier
  - total_multiplier
  - performance_tier
  - avg_s_q, avg_s_l, avg_s_f
```

---

**Bottom Line:** Vidaio is a **pure performance-meritocracy**. Your rewards depend entirely on the quality of videos you process. Stake is irrelevant. Consistency is rewarded. Excellence is aggressively incentivized. S3 is essential for validators to evaluate your work.

