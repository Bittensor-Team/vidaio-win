# Vidaio Subnet - Complete Documentation Index

## 📚 What You Have Access To

This repository now contains comprehensive documentation and tools to help you become a successful Vidaio Subnet miner.

---

## 🎯 Start Here

### 1. **For New Miners** → Read `VALIDATION_SUMMARY.md`
- Repository overview and mission
- What miners need to know
- Mining workflow explained
- Miner preparation checklist
- Performance targets

**Time to read:** 20-30 minutes

### 2. **For Quick Commands** → Read `QUICK_START.md`
- One-minute overview
- Common use cases
- Expected score ranges
- Troubleshooting quick tips

**Time to read:** 5-10 minutes

### 3. **For Algorithm Selection** → Read `MINER_ALGORITHMS_GUIDE.md`
- Task types explained (SD2HD, HD24K, SD24K, 4K28K)
- Validator expectations from protocol
- Latest algorithms (ESRGAN, AV1, etc.)
- Recommended algorithm stacks
- Implementation checklist

**Time to read:** 25-35 minutes

### 4. **For Local Validation** → Read `LOCAL_VALIDATION_README.md`
- Full validation script documentation
- Single video validation
- Batch folder validation
- Output interpretation
- Integration with mining workflow

**Time to read:** 15-20 minutes

---

## 📋 Documentation by Use Case

### Setting Up as a Miner
1. Read: `VALIDATION_SUMMARY.md` → Miner Preparation Checklist
2. Read: `MINER_ALGORITHMS_GUIDE.md` → Implementation Checklist
3. Read: `docs/miner_setup.md` (from repo)
4. Action: Follow setup steps

### Understanding Tasks & Scoring
1. Read: `VALIDATION_SUMMARY.md` → Scoring System Deep Dive
2. Read: `MINER_ALGORITHMS_GUIDE.md` → Task Types Explained & Validator Expectations
3. Read: `docs/incentive_mechanism.md` (from repo)
4. Reference: `vidaio_subnet_core/protocol.py` (code)

### Choosing Algorithms
1. Read: `MINER_ALGORITHMS_GUIDE.md` → Latest Algorithms
2. Reference: `MINER_ALGORITHMS_GUIDE.md` → Recommended Algorithm Stacks
3. Compare: Tier 1 (Competitive), Tier 2 (Advanced), Tier 3 (Elite)
4. Action: Implement and test locally

### Validating Your Work
1. Read: `LOCAL_VALIDATION_README.md` → Usage Guide
2. Read: `QUICK_START.md` → Command Examples
3. Action: Run `python3 local_validation.py`
4. Interpret: Compare scores with benchmarks

### Debugging Issues
1. Read: `LOCAL_VALIDATION_README.md` → Troubleshooting
2. Use: `python3 local_validation.py --verbose`
3. Reference: `MINER_ALGORITHMS_GUIDE.md` → Task Difficulty Rankings
4. Check: Score vs expected range

---

## 🛠️ Tools Available

### Local Validation Script
**File:** `local_validation.py` (476 lines, executable)

**Purpose:** Score your video outputs locally before submitting to network

**Commands:**

```bash
# Single upscaling validation
python3 local_validation.py \
    --reference input.mp4 \
    --processed output.mp4 \
    --task upscaling

# Single compression validation
python3 local_validation.py \
    --reference original.mp4 \
    --processed compressed.mp4 \
    --task compression

# Batch validation of folder
python3 local_validation.py \
    --folder ./videos \
    --task upscaling \
    --output results.json
```

**Key Features:**
- ✅ Scores upscaling (PIE-APP + VMAF + quality + length)
- ✅ Scores compression (VMAF + compression ratio)
- ✅ Single or batch mode
- ✅ JSON export for analysis
- ✅ GPU acceleration (CUDA auto-detect)
- ✅ Comprehensive error handling

---

## 📊 Repository Structure

```
/workspace/vidaio-subnet/
├── Documentation (YOU ARE HERE)
│   ├── DOCUMENTATION_INDEX.md         ← This file
│   ├── VALIDATION_SUMMARY.md          ← Overview & checklist
│   ├── LOCAL_VALIDATION_README.md     ← Validation tool guide
│   ├── MINER_ALGORITHMS_GUIDE.md      ← Algorithms & tasks
│   └── QUICK_START.md                 ← Quick reference
│
├── Tools
│   └── local_validation.py            ← Local scoring script
│
├── Original Repo (existing)
│   ├── neurons/
│   │   ├── miner.py                   ← Miner implementation
│   │   └── validator.py               ← Validator implementation
│   │
│   ├── services/
│   │   ├── scoring/                   ← Scoring functions
│   │   ├── upscaling/                 ← Upscaling service
│   │   └── compress/                  ← Compression service
│   │
│   ├── vidaio_subnet_core/
│   │   ├── protocol.py                ← Protocol definitions
│   │   ├── validating/                ← Validation logic
│   │   └── configs/                   ← Configuration
│   │
│   └── docs/
│       ├── miner_setup.md             ← Installation guide
│       ├── validator_setup.md         ← Validator guide
│       └── incentive_mechanism.md     ← Scoring details
```

---

## 🎓 Learning Path

### Day 1: Foundation
- [ ] Read `VALIDATION_SUMMARY.md` (complete overview)
- [ ] Understand repository mission and architecture
- [ ] Review miner preparation checklist
- [ ] Estimated time: 45 minutes

### Day 2: Deep Dive
- [ ] Read `MINER_ALGORITHMS_GUIDE.md` (task types & algorithms)
- [ ] Understand validator expectations from protocol
- [ ] Research algorithms (ESRGAN, AV1, etc.)
- [ ] Choose algorithm tier for your setup
- [ ] Estimated time: 60 minutes

### Day 3: Setup
- [ ] Follow `VALIDATION_SUMMARY.md` → Miner Preparation Checklist
- [ ] Install dependencies and tools
- [ ] Test hardware (GPU, CUDA)
- [ ] Get to "Test Before Mining" step
- [ ] Estimated time: 120+ minutes

### Day 4: Validation
- [ ] Read `LOCAL_VALIDATION_README.md` (full guide)
- [ ] Test with `python3 local_validation.py` on sample videos
- [ ] Understand scoring output
- [ ] Batch validate multiple videos
- [ ] Compare with benchmarks
- [ ] Estimated time: 60 minutes

### Day 5: Optimization
- [ ] Read scoring documentation (incentive_mechanism.md)
- [ ] Analyze local validation results
- [ ] Identify improvement opportunities
- [ ] Iterate algorithm tuning
- [ ] Re-validate and track improvements
- [ ] Estimated time: 120+ minutes

### Day 6+: Go Live
- [ ] Complete network setup (wallet, hotkey, etc.)
- [ ] Start mining with PM2
- [ ] Monitor validator requests
- [ ] Track local vs network scores
- [ ] Continuously optimize

---

## 🎯 Quick Reference by Role

### Upscaling Specialist
1. Read: `VALIDATION_SUMMARY.md` → Upscaling Scoring
2. Read: `MINER_ALGORITHMS_GUIDE.md` → For Upscaling section
3. Focus: ESRGAN, BSRGAN, Pixop algorithms
4. Target Score: 0.35-0.50
5. Key Metric: VMAF ≥ 90, PIE-APP < 0.5

### Compression Specialist
1. Read: `VALIDATION_SUMMARY.md` → Compression Scoring
2. Read: `MINER_ALGORITHMS_GUIDE.md` → For Compression section
3. Focus: AV1 codec optimization
4. Target Score: 0.70-1.00
5. Key Metric: VMAF ≥ 90, Compression 8-15x

### Generalist (Both Tasks)
1. Read all documentation
2. Implement both algorithm stacks
3. Target competitive scores in both
4. Switch between tasks based on validator demand

---

## 📈 Performance Benchmarks

### Upscaling Scores
- **Poor** (< 0.10): Score significantly below network average
- **Average** (0.10-0.20): Standard miner performance
- **Good** (0.20-0.35): Top 50% of miners
- **Excellent** (0.35-0.50): Top 10% of miners
- **Elite** (0.45-0.55+): Top 1% of miners

### Compression Scores
- **Poor** (< 0.30): Score significantly below network average
- **Average** (0.30-0.60): Standard miner performance
- **Good** (0.60-0.85): Top 50% of miners
- **Excellent** (0.85-1.00): Top 10% of miners
- **Elite** (0.90-1.00+): Top 1% of miners

---

## 🔧 Troubleshooting Guide

### Issue: "Local score is much lower than expected"
**Steps:**
1. Check VMAF score ≥ 85 (minimum requirement)
2. Read `LOCAL_VALIDATION_README.md` → Output Interpretation
3. Run with `--verbose` flag for details
4. Compare compression ratio or PIE-APP scores
5. Read `MINER_ALGORITHMS_GUIDE.md` → Task Difficulty Rankings

### Issue: "Validation fails with 'Invalid video'"
**Steps:**
1. Check video format: `ffprobe your_video.mp4`
2. Ensure MP4 container with H.264/H.265 codec
3. Minimum 10 frames required
4. Same resolution as reference (for compression)

### Issue: "Processing is too slow"
**Steps:**
1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA is being used (not CPU fallback)
3. Reduce batch size if applicable
4. Consider Tier 1 algorithms if using Tier 3

### Issue: "Can't decide which algorithms to use"
**Steps:**
1. Read `MINER_ALGORITHMS_GUIDE.md` → Recommended Algorithm Stacks
2. Start with **Tier 1** (Competitive) algorithms
3. Implement and validate locally
4. Upgrade to Tier 2+ if hardware allows
5. Track performance improvements

---

## 📞 What to Read When...

| Situation | Read This |
|-----------|-----------|
| "I'm new to Vidaio" | `VALIDATION_SUMMARY.md` |
| "How do I validate locally?" | `LOCAL_VALIDATION_README.md` |
| "What algorithms should I use?" | `MINER_ALGORITHMS_GUIDE.md` |
| "Show me example commands" | `QUICK_START.md` |
| "What does validator expect?" | `MINER_ALGORITHMS_GUIDE.md` → Validator Expectations |
| "How is score calculated?" | `VALIDATION_SUMMARY.md` → Scoring System Deep Dive |
| "I need installation steps" | `docs/miner_setup.md` (from repo) |
| "I'm stuck on a problem" | `LOCAL_VALIDATION_README.md` → Troubleshooting |

---

## ✅ Checklist for Success

Before you start mining, ensure you have:

- [ ] Read all relevant documentation
- [ ] Understood the two task types (upscaling/compression)
- [ ] Chosen algorithm tier for your hardware
- [ ] Installed all dependencies
- [ ] Tested hardware (GPU, CUDA, memory)
- [ ] Run local_validation.py on test videos
- [ ] Achieved target scores (0.25+ upscaling, 0.60+ compression)
- [ ] Created Bittensor wallet
- [ ] Set up environment variables
- [ ] Tested services locally (upscaling/compress/scoring)
- [ ] Reviewed miner.py and validator.py code
- [ ] Planned monitoring/optimization strategy

---

## 🚀 Next Steps

1. **Start Reading**: Pick your starting document based on your role
2. **Learn Architecture**: Understand tasks, scoring, and validation
3. **Choose Algorithms**: Select appropriate tier for your hardware
4. **Setup Hardware**: Install dependencies and test CUDA
5. **Implement & Test**: Code algorithms and validate locally
6. **Go Live**: Start mining on network
7. **Optimize**: Monitor scores and improve continuously

---

## 📊 Documentation Stats

| Document | Lines | Focus | Audience |
|----------|-------|-------|----------|
| VALIDATION_SUMMARY.md | 478 | Overview, scoring, checklist | All miners |
| LOCAL_VALIDATION_README.md | 398 | Validation tool, usage guide | All miners |
| MINER_ALGORITHMS_GUIDE.md | 519 | Algorithms, tasks, validator | Developers |
| QUICK_START.md | 142 | Quick commands, examples | Quick reference |
| local_validation.py | 476 | Scoring script implementation | Python developers |

**Total Documentation**: ~2,000 lines of comprehensive guides
**Total Code**: 476 lines of production-ready validation script

---

## 🎓 Learning Resources

### In This Repository
- Comprehensive algorithm guides
- Local validation scripts
- Scoring documentation
- Protocol specifications
- Incentive mechanism details

### External Resources
- VMAF Documentation: https://github.com/Netflix/vmaf
- ESRGAN: https://github.com/xinntao/Real-ESRGAN
- AV1 Codec: https://aomedia.org/
- Bittensor Docs: https://docs.bittensor.com/
- Video2X: https://docs.video2x.org/

---

## 💡 Pro Tips

1. **Start Simple**: Begin with Tier 1 algorithms to understand the system
2. **Validate Everything**: Always run local validation before submitting
3. **Track Metrics**: Keep results.json files to monitor improvements
4. **Optimize Gradually**: Improve one aspect at a time
5. **Read Code**: The validator and miner code reveal scoring logic
6. **Test Often**: Use local_validation.py frequently during development
7. **Monitor Performance**: Compare your scores with network benchmarks
8. **Join Community**: Connect with other miners for insights

---

## 📝 Version Info

- **Documentation Created**: October 21, 2025
- **Repository**: Vidaio Subnet
- **Netuid**: 85
- **Protocol Version**: Latest (check vidaio_subnet_core/protocol.py)

---

**You're all set! Pick a document and start your mining journey. Good luck! 🚀**
