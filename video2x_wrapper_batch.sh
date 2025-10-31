#!/bin/bash
# Video2X Batch Processing Wrapper
# Uses shared storage and manifest files for efficient batch processing

set -e

INPUT="$1"
OUTPUT="$2"
SCALE="$3"
CODEC="${4:-libx264}"
PRESET="${5:-slow}"
CRF="${6:-28}"

# Validate inputs
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ] || [ -z "$SCALE" ]; then
    echo "Usage: $0 <input_video> <output_video> <scale_factor> [codec] [preset] [crf]"
    echo "Example: $0 input.mp4 output.mp4 2 libx264 slow 28"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found"
    exit 1
fi

echo "🎬 Video2X Batch Processing Wrapper"
echo "==================================="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Scale: ${SCALE}x"
echo "Codec: $CODEC"
echo "Preset: $PRESET"
echo "CRF: $CRF"
echo ""

# Create temporary directory
TEMP_DIR="/tmp/video2x_$$"
mkdir -p "$TEMP_DIR"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Get video properties
echo "📊 Analyzing video properties..."
FPS_RAW=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")
WIDTH=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT")
HEIGHT=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT")
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$INPUT")

# Convert FPS to decimal
FPS=$(echo "scale=2; $FPS_RAW" | bc -l)

echo "  📐 Resolution: ${WIDTH}x${HEIGHT}"
echo "  🎞️  FPS: $FPS"
echo "  ⏱️  Duration: ${DURATION}s"

# Calculate new dimensions
NEW_WIDTH=$((WIDTH * SCALE))
NEW_HEIGHT=$((HEIGHT * SCALE))
echo "  📐 Target: ${NEW_WIDTH}x${NEW_HEIGHT}"

# Step 1: Duplicate last 2 frames (fix Video2X bug)
echo ""
echo "🔧 Step 1: Duplicating last 2 frames (Video2X bug fix)..."
TEMP_INPUT="$TEMP_DIR/padded_input.mp4"

/usr/bin/ffmpeg -i "$INPUT" \
    -vf "tpad=stop_mode=clone:stop_duration=0.067" \
    -c:v libx264 -b:v 2M \
    "$TEMP_INPUT" -y -loglevel error

echo "  ✅ Frame duplication completed"

# Step 2: Extract frames
echo ""
echo "📸 Step 2: Extracting frames..."
FRAMES_DIR="$TEMP_DIR/input_frames"
mkdir -p "$FRAMES_DIR"

/usr/bin/ffmpeg -i "$TEMP_INPUT" -vf "fps=$FPS" "$FRAMES_DIR/frame_%06d.png" -y -loglevel error

FRAME_COUNT=$(ls "$FRAMES_DIR"/*.png | wc -l)
echo "  ✅ Extracted $FRAME_COUNT frames"

# Step 3: Create batch manifests
echo ""
echo "📋 Step 3: Creating batch manifests..."
OUTPUT_DIR="$TEMP_DIR/output_frames"
MANIFEST_DIR="$TEMP_DIR/manifests"
STATUS_DIR="$TEMP_DIR/status"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MANIFEST_DIR"
mkdir -p "$STATUS_DIR"

# Generate manifests using Python script
python3 /workspace/vidaio-win/batch_manifest_generator.py \
    "$FRAMES_DIR" \
    "$FRAME_COUNT" \
    6 \
    8

echo "  ✅ Created batch manifests"

# Step 4: Setup workers and start batch processing
echo ""
echo "⚡ Step 4: Starting batch processing with 6 workers..."

# Setup 6 workers
WORKER_URLS=("http://127.0.0.1:8090" "http://127.0.0.1:8091" "http://127.0.0.1:8092" "http://127.0.0.1:8093" "http://127.0.0.1:8094" "http://127.0.0.1:8095")
NUM_WORKERS=6

# Check worker health
echo "🔍 Checking worker health..."
HEALTHY_WORKERS=()
for i in "${!WORKER_URLS[@]}"; do
    URL="${WORKER_URLS[$i]}"
    if curl -s --connect-timeout 5 "$URL/health" > /dev/null 2>&1; then
        HEALTHY_WORKERS+=("$URL")
        echo "  ✅ Worker $((i+1)) ($URL) is healthy"
    else
        echo "  ❌ Worker $((i+1)) ($URL) is not responding"
    fi
done

if [ ${#HEALTHY_WORKERS[@]} -eq 0 ]; then
    echo "❌ No healthy workers found! Please start workers with: bash start_workers_batch.sh"
    exit 1
fi

echo "✅ ${#HEALTHY_WORKERS[@]} workers ready"

# Start processing on each worker
echo "🚀 Starting batch processing..."
for i in "${!HEALTHY_WORKERS[@]}"; do
    worker_url="${WORKER_URLS[$i]}"
    manifest_path="$MANIFEST_DIR/worker_${i}_manifest.txt"
    
    if [ -f "$manifest_path" ]; then
        echo "  📤 Sending manifest to Worker $((i+1))..."
        curl -s -X POST "$worker_url/process_manifest" \
            -H "Content-Type: application/json" \
            -d "{\"manifest_path\": \"$manifest_path\"}" > /dev/null
    else
        echo "  ⚠️  Manifest not found for Worker $((i+1)): $manifest_path"
    fi
done

# Step 5: Monitor progress
echo ""
echo "⏳ Monitoring batch processing progress..."

start_time=$(date +%s)
all_completed=false

while [ "$all_completed" = false ]; do
    all_completed=true
    total_progress=0
    
    for i in "${!HEALTHY_WORKERS[@]}"; do
        worker_url="${WORKER_URLS[$i]}"
        
        # Get worker status
        status=$(curl -s "$worker_url/status" 2>/dev/null || echo '{"status":"error"}')
        worker_status=$(echo "$status" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "error")
        progress=$(echo "$status" | python3 -c "import sys, json; print(json.load(sys.stdin)['progress_percent'])" 2>/dev/null || echo "0")
        
        if [ "$worker_status" != "completed" ] && [ "$worker_status" != "error" ]; then
            all_completed=false
        fi
        
        total_progress=$((total_progress + $(echo "$progress" | cut -d. -f1)))
    done
    
    if [ "$all_completed" = false ]; then
        avg_progress=$((total_progress / ${#HEALTHY_WORKERS[@]}))
        echo "  📊 Progress: ${avg_progress}% (checking every 5s...)"
        sleep 5
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "  ✅ Batch processing completed in ${elapsed}s"

# Step 6: Verify all frames are processed
echo ""
echo "🔍 Step 6: Verifying processed frames..."
PROCESSED_COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
echo "  📊 Processed frames: $PROCESSED_COUNT / $FRAME_COUNT"

if [ "$PROCESSED_COUNT" -ne "$FRAME_COUNT" ]; then
    echo "  ⚠️  Warning: Frame count mismatch! Some frames may be missing."
    echo "  📋 Missing frames:"
    for frame_num in $(seq 1 $FRAME_COUNT); do
        frame_name=$(printf "frame_%06d.png" $frame_num)
        if [ ! -f "$OUTPUT_DIR/$frame_name" ]; then
            echo "    - $frame_name"
        fi
    done
fi

# Step 7: Reconstruct video
echo ""
echo "🎬 Step 7: Reconstructing video..."
/usr/bin/ffmpeg -framerate "$FPS" -i "$OUTPUT_DIR/frame_%06d.png" \
    -c:v libx264 -b:v 8M \
    -pix_fmt yuv420p "$OUTPUT" -y -loglevel error

echo "  ✅ Video reconstruction completed"

# Verify output
if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    OUTPUT_SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "🎉 Video upscaling completed successfully!"
    echo "📁 Output: $OUTPUT ($OUTPUT_SIZE)"
    echo "📐 Resolution: ${NEW_WIDTH}x${NEW_HEIGHT}"
    echo "🎞️  FPS: $FPS"
    echo "⚡ Processing Method: Batch Processing (6 workers)"
    echo "📊 Total Time: ${elapsed}s"
else
    echo "❌ Error: Output file was not created or is empty"
    exit 1
fi
