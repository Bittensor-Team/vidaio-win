# Vidaio VSR Pipeline Architecture

## Complete Pipeline Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validator     │───▶│     Miner       │───▶│ video_upscaler  │───▶│  VSR Server     │───▶│  Cloudflare R2  │
│                 │    │                 │    │                 │    │                 │    │                 │
│ - Sends request │    │ - Receives      │    │ - HTTP POST     │    │ - Processes     │    │ - Stores video  │
│ - elk.mp4 URL   │    │ - Calls VSR     │    │ - /upscale-video│    │ - Uploads to R2 │    │ - Returns URL   │
│ - Task type     │    │ - Returns URL   │    │ - Gets response │    │ - Presigned URL │    │ - 7-day expiry  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        │                        │                        │                        │
         │                        │                        │                        │                        │
         │                        │                        │                        │                        │
         │                        ▼                        ▼                        ▼                        ▼
         │                ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │                │   Response      │    │   JSON Response │    │   Video File    │    │   Presigned URL │
         │                │                 │    │                 │    │                 │    │                 │
         │                │ - optimized_    │    │ - uploaded_     │    │ - HEVC/MP4      │    │ - 7-day access  │
         │                │   video_url     │    │   video_url     │    │ - Vidaio format │    │ - Public access │
         │                │ - processing_   │    │ - task_type     │    │ - Resolution    │    │ - Auto cleanup  │
         │                │   time          │    │ - metadata      │    │   validated     │    │   (10 min)      │
         │                └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
         └─────────────────────────────────────────────────────────────────────────────────────────────────────
                                        Validator can access video via presigned URL
                                        and score it using Vidaio scoring system
```

## Components

### 1. Validator (Mock)
- **File**: `mock_validator.py`
- **Purpose**: Simulates validator behavior
- **Actions**:
  - Sends upscaling requests to miner
  - Downloads processed video from R2 URL
  - Scores video using Vidaio metrics
  - Validates resolution and quality

### 2. Miner
- **File**: `neurons/miner_vsr.py`
- **Purpose**: Processes validator requests
- **Actions**:
  - Receives VideoUpscalingProtocol
  - Calls video_upscaler() utility
  - Returns processed video URL

### 3. Video Upscaler Utility
- **File**: `services/miner_utilities/miner_utils.py`
- **Purpose**: HTTP client for VSR server
- **Actions**:
  - Sends POST to `/upscale-video`
  - Handles response parsing
  - Returns video URL

### 4. VSR Server (Enhanced)
- **File**: `vsr_worker_server_enhanced.py`
- **Purpose**: Video processing and upload
- **Actions**:
  - Processes video frames with VSR
  - Uploads to Cloudflare R2
  - Returns presigned URL
  - Schedules cleanup

### 5. Cloudflare R2 Storage
- **Endpoint**: `https://ed23ab68357ef85e85a67a8fa27fab47.r2.cloudflarestorage.com`
- **Bucket**: `vidaio`
- **Purpose**: Store processed videos
- **Features**:
  - Presigned URLs (7-day expiry)
  - Auto cleanup (10 minutes)
  - Public access

## Data Flow

1. **Request**: Validator → Miner (VideoUpscalingProtocol)
2. **Processing**: Miner → video_upscaler() → VSR Server
3. **Upload**: VSR Server → Cloudflare R2
4. **Response**: R2 → VSR Server → video_upscaler() → Miner → Validator
5. **Access**: Validator downloads from R2 presigned URL
6. **Scoring**: Validator scores video using Vidaio metrics

## Configuration

### Environment Variables
```bash
# Cloudflare R2
export R2_ACCESS_KEY="your_access_key"
export R2_SECRET_KEY="your_secret_key"

# Redis (optional)
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"

# VSR Server
export VSR_HOST="localhost"
export VSR_PORT="29115"
```

### Server Endpoints
- **VSR Server**: `http://localhost:29115/upscale-video`
- **Health Check**: `http://localhost:29115/health`
- **Task Types**: `http://localhost:29115/task_types`

## Task Types Supported
- **SD2HD**: 480p → 1080p (2x scale)
- **HD24K**: 1080p → 4K (2x scale)  
- **SD24K**: 480p → 4K (4x scale)
- **4K28K**: 4K → 8K (2x scale)

## Scoring System
- **VMAF**: Video quality assessment
- **PIE-APP**: Perceptual image error
- **Length Score**: Content duration factor
- **Final Score**: Combined metric for rewards



