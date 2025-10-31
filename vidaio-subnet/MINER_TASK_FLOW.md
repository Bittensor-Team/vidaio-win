# Vidaio Miner Task Serving Flow

This document explains how miners receive, process, and respond to tasks from validators in the Vidaio subnet.

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VIDAIO SUBNET (netuid 85)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐                                    ┌──────────────┐   │
│  │  VALIDATOR   │                                    │    MINER     │   │
│  │              │                                    │              │   │
│  │  - Synthetic │───────(1) Task Warrant Request───▶│  - Upscale   │   │
│  │    Tasks     │                                    │  - Compress  │   │
│  │  - Organic   │◀──────(2) Warrant Response────────│  - Storage   │   │
│  │    Tasks     │     (I can handle UPSCALING)      │  - GPU Proc  │   │
│  │  - Scoring   │                                    │              │   │
│  │              │───────(3) Video Task Request──────▶│              │   │
│  │              │   (reference_video_url + params)   │              │   │
│  │              │                                    │              │   │
│  │              │◀──────(4) Processed Response──────│              │   │
│  │              │   (optimized_video_url)           │              │   │
│  │              │                                    │              │   │
│  │              │───────(5) Download & Score────────▶│              │   │
│  │              │                                    │              │   │
│  │              │───────(6) Update Weights──────────▶│              │   │
│  └──────────────┘                                    └──────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Miner Task Warrant System

Before validators send tasks, they query miners to see what tasks they can handle.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TASK WARRANT HANDSHAKE                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Validator queries all miners
┌──────────────┐
│  VALIDATOR   │
└──────┬───────┘
       │
       │ TaskWarrantProtocol()
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
┌──────────────┐                      ┌──────────────┐
│  Miner #42   │                      │  Miner #99   │
│              │                      │              │
└──────┬───────┘                      └──────┬───────┘
       │                                      │
       │ warrant_task = UPSCALING             │ warrant_task = COMPRESSION
       │                                      │
       └──────────────────┬───────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  VALIDATOR    │
                  │  Groups miners│
                  │  by task type │
                  └───────────────┘

RESULT: Validator knows which miners handle which tasks
- Miner #42 → upscaling_miners list
- Miner #99 → compression_miners list
```

**Code Reference:**
```python
# neurons/miner.py - Lines 108-124
async def forward_task_warrant_requests(self, synapse: TaskWarrantProtocol):
    # Miner declares what task type it can handle
    synapse.warrant_task = TaskType.UPSCALING  # or COMPRESSION
    return synapse
```

---

## 3. Detailed Task Flow: Upscaling Task

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UPSCALING TASK PROCESSING FLOW                        │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────┐
│ VALIDATOR  │
│            │
│ Round #123 │
└─────┬──────┘
      │
      │ (1) SELECT MINER from upscaling_miners
      │     Priority = f(stake, availability)
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VideoUpscalingProtocol                                              │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ miner_payload:                                                  │ │
│ │   - reference_video_url: "https://s3.../original.mp4"          │ │
│ │   - task_type: "SD2HD" or "HD24K" or "SD24K"                  │ │
│ │   - round_id: 123                                              │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
      │
      │ (2) MINER RECEIVES
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MINER: neurons/miner.py::forward_upscaling_requests()              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: Download Reference Video                            │ │
│  │  └─▶ download_video(reference_video_url)                    │ │
│  │      └─▶ saves to: /workspace/.../videos/uuid.mp4           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: Send to Upscaling Service (HTTP POST)               │ │
│  │  └─▶ POST http://localhost:29115/upscale-video              │ │
│  │      {                                                       │ │
│  │        "payload_url": "...",                                 │ │
│  │        "task_type": "SD2HD"                                  │ │
│  │      }                                                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ UPSCALING SERVICE: services/upscaling/server.py              │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  A. Duplicate last 2 frames (fix video2x bug)               │ │
│  │     ffmpeg -vf tpad=stop_mode=clone:...                     │ │
│  │                                                              │ │
│  │  B. Run Video2X upscaling                                   │ │
│  │     video2x -i input.mp4 -o output.mp4                      │ │
│  │            -p realesrgan                                     │ │
│  │            -s 2  (or 4 for SD24K)                           │ │
│  │            -c libx264 -e crf=28                             │ │
│  │                                                              │ │
│  │  C. Upload to S3/MinIO storage                              │ │
│  │     storage_client.upload_file(...)                         │ │
│  │                                                              │ │
│  │  D. Generate presigned URL (10 min expiry)                  │ │
│  │     storage_client.get_presigned_url(...)                   │ │
│  │                                                              │ │
│  │  E. Schedule deletion (10 minutes via Redis)                │ │
│  │     schedule_file_deletion(object_name)                     │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: Return Response to Validator                        │ │
│  │  synapse.miner_response.optimized_video_url =               │ │
│  │    "https://s3.../processed_uuid.mp4?signature=..."         │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
      │
      │ (3) VALIDATOR RECEIVES
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VALIDATOR SCORING PHASE                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Download processed video from miner's URL                       │
│  2. Send to scoring service (http://localhost:29090)               │
│  3. Calculate metrics:                                              │
│     - PIE-APP score (perceptual quality)                            │
│     - VMAF score (video quality)                                    │
│     - Content length                                                │
│  4. Compute final score with performance multipliers                │
│  5. Update miner's accumulated score in database                    │
│  6. Set weights on-chain periodically                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Task Flow: Compression Task

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  COMPRESSION TASK PROCESSING FLOW                        │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────┐
│ VALIDATOR  │
└─────┬──────┘
      │
      │ (1) SELECT MINER from compression_miners
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VideoCompressionProtocol                                            │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ miner_payload:                                                  │ │
│ │   - reference_video_url: "https://s3.../original.mp4"          │ │
│ │   - vmaf_threshold: 85, 90, or 95                              │ │
│ │   - round_id: 456                                              │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MINER: neurons/miner.py::forward_compression_requests()             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: Download Reference Video                            │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: Send to Compression Service                         │ │
│  │  └─▶ POST http://localhost:29116/compress-video             │ │
│  │      {                                                       │ │
│  │        "payload_url": "...",                                 │ │
│  │        "vmaf_threshold": 90                                  │ │
│  │      }                                                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ COMPRESSION SERVICE: services/compress/server.py             │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  PIPELINE (5 stages):                                        │ │
│  │                                                              │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ 1. VIDEO PREPROCESSING                                 │ │ │
│  │  │    - Extract metadata (resolution, fps, duration)      │ │ │
│  │  │    - Validate video quality                            │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                       │                                      │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ 2. SCENE DETECTION                                     │ │ │
│  │  │    - Split video into scenes                           │ │ │
│  │  │    - Detect motion, complexity per scene               │ │ │
│  │  │    - Use fast_scene_detect.py (PySceneDetect)         │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                       │                                      │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ 3. AI ENCODING (PER SCENE)                             │ │ │
│  │  │    - ML model predicts optimal CRF per scene           │ │ │
│  │  │    - Encode: ffmpeg -crf [predicted] scene.mp4         │ │ │
│  │  │    - Model: scene_classifier_model.pth (PyTorch)      │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                       │                                      │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ 4. VMAF CALCULATION                                    │ │ │
│  │  │    - Calculate VMAF per scene vs original              │ │ │
│  │  │    - Retry with lower CRF if VMAF < threshold          │ │ │
│  │  │    - Use calculate_vmaf_adv.py (ffmpeg-quality-metrics)│ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                       │                                      │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ 5. MERGE & UPLOAD                                      │ │ │
│  │  │    - Concatenate all compressed scenes                 │ │ │
│  │  │    - Upload to S3/MinIO                                │ │ │
│  │  │    - Generate presigned URL                            │ │ │
│  │  │    - Schedule deletion                                 │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: Return Response                                     │ │
│  │  synapse.miner_response.optimized_video_url = "..."         │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VALIDATOR SCORING (Compression)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. Download compressed video                                       │
│  2. Calculate:                                                      │
│     - VMAF score vs original                                        │
│     - Compression rate (compressed_size / original_size)            │
│     - Content length                                                │
│  3. Apply scoring formula:                                          │
│     score = w_c × compression_component + w_q × quality_component   │
│     where w_c=0.70, w_q=0.30                                        │
│  4. Update miner performance metrics                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Miner Architecture: Service Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MINER INTERNAL ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │   BITTENSOR AXON     │
                        │   (neurons/miner.py) │
                        │   Port: 8091         │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ forward_upscaling   │ │ forward_compression │ │ forward_length_check│
│ _requests()         │ │ _requests()         │ │ _requests()         │
└──────────┬──────────┘ └──────────┬──────────┘ └─────────────────────┘
           │                       │
           │ HTTP POST             │ HTTP POST
           │ localhost:29115       │ localhost:29116
           │                       │
           ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐
│ VIDEO UPSCALER      │ │ VIDEO COMPRESSOR    │
│ FastAPI Service     │ │ FastAPI Service     │
│ Port: 29115         │ │ Port: 29116         │
├─────────────────────┤ ├─────────────────────┤
│ • Download video    │ │ • Scene detection   │
│ • Video2X process   │ │ • AI encoding       │
│ • S3 upload         │ │ • VMAF validation   │
│ • URL generation    │ │ • S3 upload         │
└──────────┬──────────┘ └──────────┬──────────┘
           │                       │
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │   STORAGE (S3)      │
           │   MinIO/Backblaze   │
           │                     │
           │ Bucket: vidaio-*    │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   REDIS QUEUE       │
           │   Port: 6379        │
           │                     │
           │ File deletion tasks │
           │ (10 min TTL)        │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ FILE DELETION SVC   │
           │ Port: 29117         │
           │                     │
           │ Cleans up old files │
           └─────────────────────┘

ALL SERVICES MANAGED BY PM2:
  pm2 list
  ├─ video-miner       (neurons/miner.py)
  ├─ video-upscaler    (services/upscaling/server.py)
  ├─ video-compressor  (services/compress/server.py)
  └─ video-deleter     (services/miner_utilities/file_deletion_server.py)
```

---

## 6. Request Priority & Task Assignment

Validators prioritize miners based on **stake** when assigning tasks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      VALIDATOR MINER SELECTION                           │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Query all miners for task warrant
        ↓
STEP 2: Filter by task type (upscaling vs compression)
        ↓
STEP 3: Calculate priority for each miner
        
        priority = stake_amount
        
        Example:
        - Miner #42: 1000 TAO stake → priority = 1000.0
        - Miner #99: 500 TAO stake  → priority = 500.0
        - Miner #7:  10 TAO stake   → priority = 10.0
        ↓
STEP 4: Sort miners by priority (highest first)
        ↓
STEP 5: Assign task to highest priority available miner

┌──────────────────────────────────────────────────────────────────┐
│ IMPORTANT: Stake ONLY affects task priority                     │
│            NOT reward amount!                                    │
│                                                                  │
│ Rewards are based on:                                           │
│  ✓ Video quality scores (PIE-APP, VMAF)                         │
│  ✓ Compression efficiency                                       │
│  ✓ Content length processed                                     │
│  ✓ Performance multipliers (consistency bonuses/penalties)      │
│                                                                  │
│ Higher stake = More tasks = More chances to earn                │
└──────────────────────────────────────────────────────────────────┘
```

**Code Reference:**
```python
# neurons/miner.py - Lines 138-151
async def priority_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> float:
    caller_uid: int = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
    priority: float = float(self.metagraph.S[caller_uid])  # S = stake
    return priority
```

---

## 7. Complete End-to-End Timeline

```
Time    Event
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T+0s    Validator starts synthetic epoch
        └─▶ Queries all miners with TaskWarrantProtocol

T+1s    Miners respond with task capabilities
        - 50% respond: UPSCALING
        - 30% respond: COMPRESSION  
        - 20% no response / unknown

T+2s    Validator groups miners by task type
        └─▶ Selects top 10 highest-stake upscaling miners

T+3s    Validator sends VideoUpscalingProtocol to Miner #42
        payload: reference_video_url, task_type=SD2HD

T+4s    Miner #42 receives request
        └─▶ Downloads reference video (5 seconds for 100MB)

T+9s    Miner calls upscaling service (HTTP POST)
        └─▶ Video2X processing begins

T+10s   Video2X: Duplicate last 2 frames (1 second)

T+11s   Video2X: Start upscaling with RealESRGAN
        └─▶ GPU processing: ~30 seconds for 30-second video

T+41s   Video2X: Encoding finished
        └─▶ Upload to S3 (10 seconds for 300MB)

T+51s   Generate presigned URL
        Schedule deletion in Redis (10 min TTL)

T+52s   Miner returns response with optimized_video_url

T+53s   Validator receives response
        └─▶ Downloads processed video (15 seconds)

T+68s   Validator sends to scoring service
        └─▶ Calculate PIE-APP, VMAF (20 seconds)

T+88s   Scoring complete
        └─▶ Final score computed with multipliers
        └─▶ Database updated with new performance metrics

T+89s   Process complete
        └─▶ Miner's accumulated_score increased
        └─▶ Weights will be set on-chain in next epoch

T+600s  Redis worker deletes processed files from S3
        (10 minutes after upload)
```

---

## 8. Error Handling & Timeouts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ERROR SCENARIOS                                 │
└─────────────────────────────────────────────────────────────────────────┘

SCENARIO 1: Miner doesn't respond
┌────────────┐
│ VALIDATOR  │─── (timeout: 40s for upscaling) ───▶ ✘ No response
└────────────┘                                       └─▶ Score = 0
                                                         No penalty beyond
                                                         missed opportunity


SCENARIO 2: Miner returns None/empty URL
┌────────────┐
│ VALIDATOR  │◀─── optimized_video_url = None ───── MINER
└────────────┘
       │
       └─▶ Score = 0
           └─▶ Counts toward penalty_q_count in rolling window


SCENARIO 3: Download fails (bad URL)
┌────────────┐
│ VALIDATOR  │─── GET miner's URL ───▶ ✘ 404 / 403 / timeout
└────────────┘
       │
       └─▶ Score = 0
           └─▶ Penalty applied


SCENARIO 4: Processing service down
┌────────────┐                    ┌─────────────────┐
│   MINER    │─── POST ───▶       │ Upscaler: DOWN  │
└────────────┘             ✘      └─────────────────┘
       │
       └─▶ Returns synapse with None
           └─▶ Validator sees failure


SCENARIO 5: S3 upload fails
Upscaling succeeds but upload fails
       │
       └─▶ Miner returns None
           └─▶ Validator marks as failure


MITIGATION STRATEGIES:

1. Monitor services with PM2
   pm2 logs video-upscaler  # Check for errors

2. Health checks
   curl http://localhost:29115/health  # Add health endpoint

3. Redis monitoring
   redis-cli ping  # Ensure queue is working

4. S3 connection test
   Test bucket access on startup

5. Automatic restart
   pm2 start --max-restarts 10  # Auto-restart on crash
```

---

## 9. Performance Optimization Tips

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION STRATEGIES                               │
└─────────────────────────────────────────────────────────────────────────┘

1. GPU UTILIZATION
   ┌──────────────────────────────────────────────────────┐
   │ • Use GPU for Video2X RealESRGAN                     │
   │ • Monitor: watch -n 1 nvidia-smi                     │
   │ • Ensure CUDA drivers installed                      │
   │ • Avoid running multiple GPU tasks simultaneously    │
   └──────────────────────────────────────────────────────┘

2. NETWORK BANDWIDTH
   ┌──────────────────────────────────────────────────────┐
   │ • Use fast S3 endpoint (low latency)                 │
   │ • Consider CDN if available                          │
   │ • Test bandwidth: speedtest-cli                      │
   │ • Typical needs:                                     │
   │   - Download: 50-500 MB/video                        │
   │   - Upload: 100 MB-1 GB/video                        │
   └──────────────────────────────────────────────────────┘

3. STORAGE MANAGEMENT
   ┌──────────────────────────────────────────────────────┐
   │ • Ensure Redis deletion service running              │
   │ • Monitor disk usage: df -h                          │
   │ • Clean temp files regularly                         │
   │ • Use SSD for video processing                       │
   └──────────────────────────────────────────────────────┘

4. CONCURRENT PROCESSING
   ┌──────────────────────────────────────────────────────┐
   │ • Miner can handle multiple requests                 │
   │ • Limited by GPU memory                              │
   │ • FastAPI handles async I/O                          │
   │ • Adjust worker count if needed                      │
   └──────────────────────────────────────────────────────┘

5. RESPONSE TIME TARGET
   ┌──────────────────────────────────────────────────────┐
   │ UPSCALING:                                           │
   │   • 30-second video: ~30-60 seconds total            │
   │   • Timeout: 40 seconds                              │
   │                                                      │
   │ COMPRESSION:                                         │
   │   • 30-second video: ~60-90 seconds total            │
   │   • Timeout: 120 seconds                             │
   └──────────────────────────────────────────────────────┘
```

---

## Summary

**Key Takeaways:**

1. **Task Warrant System**: Miners declare capabilities; validators route accordingly
2. **Two Task Types**: Upscaling (Video2X) and Compression (AI encoder)
3. **Priority by Stake**: Higher stake → more tasks (but not higher rewards)
4. **Rewards by Performance**: Quality scores and consistency determine earnings
5. **Service Architecture**: Main miner + 3 services (upscaler, compressor, deleter)
6. **PM2 Management**: All services monitored and auto-restarted
7. **Timeout Management**: 40s for upscaling, 120s for compression
8. **Storage Cleanup**: Redis-based deletion after 10 minutes

**Next Steps:**
- Ensure all services running: `pm2 list`
- Monitor logs: `pm2 logs video-miner`
- Check GPU: `nvidia-smi`
- Verify S3 access: Test upload/download


