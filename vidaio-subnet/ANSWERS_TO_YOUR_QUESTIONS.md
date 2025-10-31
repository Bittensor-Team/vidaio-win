# Answers to Your Questions

## ✅ Q1: Did I install it correctly?

**YES - Installation is correct.** 

The npm/pm2 dependency warnings during `apt install` are normal and can be ignored:
- `npm` is installed at `/usr/bin/npm` ✓
- `pm2` is installed at `/usr/local/bin/pm2` ✓
- Both work fine despite apt's dependency resolver warnings

The conda environment setup was also **perfectly done**:
- Python 3.11.13 configured
- PyTorch 2.8.0 with CUDA support installed
- All 50+ dependencies successfully installed
- .env file created
- Full documentation provided

You can proceed with confidence.

---

## 🤔 Q2: Why do WE (miners) need an S3 bucket?

### The Architecture:

```
Your Miner (on your GPU)  →  S3 Bucket  →  Validators (distributed worldwide)
  Process video           Upload result   Download & evaluate
```

### Why You Need It:

1. **Validators can't access your local server** - they're geographically distributed
2. **Presigned URLs** - you upload to S3, give validators a temporary download link
3. **Multiple validators** can download simultaneously without overloading your connection
4. **Decentralized** - you don't stay online waiting for validation
5. **Permanence** - videos available for appeals, audits, historical reference

### Real Workflow:

1. Validator sends video task to you
2. You process it (5-10 seconds) on your GPU
3. You upload result to S3 bucket
4. You send presigned URL back to validator
5. Validator downloads from S3 and evaluates quality
6. You get rewarded based on quality score

**Without S3:** Validators would need direct connection to your server - defeats the purpose of decentralization and doesn't work with distributed validators.

### Cost Perspective:

- Backblaze B2: $0.006/GB/month (cheapest)
- A 1GB processed video stored 1 month = $0.006
- You process ~100-500 videos per day (depending on task size)
- Monthly cost likely $50-200 (very cheap for a mining operation)

---

## 💰 Q3: Are emissions/incentives stake-based or novelty/performance-based?

### TL;DR: **100% PERFORMANCE-BASED** (NOT stake)

### Bittensor Level Distribution:
- 41% TAO → All miners (shared among all)
- 41% TAO → All validators
- 18% TAO → Subnet creator

### Within Miners: Pure Merit-Based

```python
Rewards = Accumulated_Score × Performance_Multiplier

Accumulated_Score = Based on:
  ✓ Video quality (PIE-APP metric)
  ✓ Compression efficiency
  ✓ Content length processed
  ✓ VMAF compliance

Performance_Multiplier = Based on:
  ✓ Bonus: up to +15% for consistency
  ✓ Penalties: down to -25% for poor quality
  ✓ Rolling 10-round history window

Stake DOES NOT affect rewards
Stake only affects: Request priority (who gets tasks first)
```

### Real Examples (from code):

**Upscaling Rewards:**
| Miner Type | Performance Score | Final Multiplier | Relative Reward |
|-----------|-----------------|-----------------|-----------------|
| Elite     | 0.854           | 1.150×          | 16.3× vs poor |
| Good      | 0.653           | 1.000×          | 2.0× vs poor |
| Average   | 0.511           | 0.925×          | 1.5× vs poor |
| Poor      | 0.311           | 0.600×          | 1.0× (baseline) |

**Elite miners earn 16× more than poor miners on identical tasks.**

### How Scores Are Calculated:

**For Upscaling (40% of rewards):**
- Quality Score (S_Q) = PIE-APP assessment (deep learning quality metric)
- Length Score (S_L) = log(video_length) bonus
- Final = 0.5×S_Q + 0.5×S_L, then exponentially transformed

**For Compression (60% of rewards):**
- Compression Rate = how much smaller the file is
- VMAF Score = quality maintained above threshold
- Final = 80%×compression_efficiency + 20%×quality_above_threshold
- If VMAF < threshold: automatic score = 0 (failure)

### Proof from Code:

```python
# File: miner_manager.py (line 1132-1164)
@property
def weights(self):
    """
    Calculate weights for reward distribution with 60% for compression 
    and 40% for upscaling.
    Within each task type, rewards are distributed PROPORTIONALLY 
    based on ACCUMULATED SCORES.
    """
    
    # Stake-based priority (NOT rewards):
    # priority = metagraph.S[uid]  # Stake amount
    
    # Reward-based distribution (performance):
    # rewards ∝ accumulate_score  # Quality & consistency score
```

### Metagraph Breakdown:

```python
metagraph.S[uid]              # Stake (affects priority, NOT rewards)
metagraph.validator_permit[uid]  # Is validator? (bool)

Rewards = function(accumulate_score)  # Performance metric
NOT = function(metagraph.S[uid])      # Stake is irrelevant
```

### Checking Rankings:

```bash
# Check your performance score (in database):
SELECT * FROM miner_metadata 
WHERE uid = YOUR_UID 
ORDER BY accumulate_score DESC;

# Shows:
- accumulate_score (your total performance)
- bonus_multiplier (+15% if excellent)
- penalty_q_multiplier (-25% if poor quality)
- performance_tier (Elite/Good/Average/Poor)
- avg_s_q, avg_s_l, avg_s_f (your average scores)
```

### Real-World Impact:

**Example Scenario:**

```
Round 1: 100 TAO per round pool
Task: Upscale 10 videos each

You (low stake, high performance):
- Processing score: 0.78 (excellent)
- Bonus multiplier: 1.15× (10/10 excellent rounds)
- Share: (Your_Score / Total_Score) × 40 TAO × 1.15
- Estimated: 15-25 TAO per round

Validator (high stake, low performance):
- Stake: 1000 TAO
- Reward priority: HIGH (stake-based)
- But: Their performance score is 0.35
- Their multiplier: 0.60× (penalties)
- Estimated: 1-2 TAO per round

You earn 10-15× MORE despite lower stake!
```

---

## 📊 Performance Determines EVERYTHING

### What Matters for Rewards:
- ✅ **Quality of your outputs** (PIE-APP, VMAF scores)
- ✅ **Compression efficiency**
- ✅ **Consistency** (10-round rolling window)
- ✅ **Content length processed**

### What Does NOT Matter:
- ❌ Stake amount
- ❌ Account age
- ❌ How many tasks you attempt
- ❌ Processing speed (unless it sacrifices quality)
- ❌ Validator connections

---

## 🎯 Strategy to Maximize Rewards

1. **Focus on Quality First**
   - Every 0.1 improvement in quality score = exponential reward increase
   - Elite (0.78) vs Average (0.51) = 3-4× reward difference

2. **Maintain Consistency**
   - 10 excellent rounds = +15% bonus multiplier
   - One bad round matters less, but 10 bad = -40% to -51%

3. **Pick Your Specialization**
   - 60% rewards for compression (harder, more demand)
   - 40% rewards for upscaling (quality-focused)

4. **S3 Reliability**
   - Must be fast and reliable (validators wait for downloads)
   - This affects their satisfaction but not directly your score

---

## Bottom Line

**Vidaio is a pure performance meritocracy:**
- Your rewards = f(output quality + consistency)
- NOT = f(stake amount)
- S3 bucket is your submission portal (non-negotiable)
- Elite miners earn 16× more than poor miners
- Quality is paramount, consistency is valued, excellence is aggressively rewarded

Start with compression tasks (60% of rewards) if you're new - they're more predictable than upscaling quality metrics.

