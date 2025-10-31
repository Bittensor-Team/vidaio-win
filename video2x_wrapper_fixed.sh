#!/bin/bash
# Video2X wrapper using FFmpeg + Real-ESRGAN
# This provides Video2X-compatible interface for Vidaio subnet

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

echo "🎬 Video2X Wrapper - FFmpeg + Real-ESRGAN"
echo "=========================================="
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
STOP_DURATION=$(echo "scale=3; 2 / $FPS" | bc -l)
TEMP_INPUT="$TEMP_DIR/padded_input.mp4"

/usr/bin/ffmpeg -i "$INPUT" \
    -vf "tpad=stop_mode=clone:stop_duration=0.067" \
    -c:v libx264 -b:v 2M \
    "$TEMP_INPUT" -y -loglevel error

echo "  ✅ Frame duplication completed"

# Step 2: Extract frames
echo ""
echo "📸 Step 2: Extracting frames..."
FRAMES_DIR="$TEMP_DIR/frames"
mkdir -p "$FRAMES_DIR"

/usr/bin/ffmpeg -i "$TEMP_INPUT" -vf "fps=$FPS" "$FRAMES_DIR/frame_%06d.png" -y -loglevel error

FRAME_COUNT=$(ls "$FRAMES_DIR"/*.png | wc -l)
echo "  ✅ Extracted $FRAME_COUNT frames"

# Step 3: Upscale frames with Real-ESRGAN using 6 workers
echo ""
echo "⚡ Step 3: Upscaling frames with Real-ESRGAN (6 workers)..."
UPSCALED_DIR="$TEMP_DIR/upscaled_frames"
mkdir -p "$UPSCALED_DIR"

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
    echo "❌ No healthy workers found! Please start workers with: bash start_workers.sh"
    exit 1
fi

echo "✅ ${#HEALTHY_WORKERS[@]} workers ready"

# Get frame files
FRAME_FILES=($(ls "$FRAMES_DIR"/*.png | sort))
TOTAL_FRAMES=${#FRAME_FILES[@]}
echo "📊 Processing $TOTAL_FRAMES frames with ${#HEALTHY_WORKERS[@]} workers..."

# Distribute frames across workers
FRAMES_PER_WORKER=$((TOTAL_FRAMES / ${#HEALTHY_WORKERS[@]}))
REMAINING_FRAMES=$((TOTAL_FRAMES % ${#HEALTHY_WORKERS[@]}))

echo "  📋 Frames per worker: $FRAMES_PER_WORKER (remaining: $REMAINING_FRAMES)"

# Process frames in parallel using workers
process_frames_parallel() {
    local worker_id=$1
    local worker_url=$2
    local start_frame=$3
    local end_frame=$4
    
    echo "  🔧 Worker $((worker_id+1)) processing frames $start_frame-$end_frame..."
    local processed=0
    local failed=0
    
    for ((i=start_frame; i<end_frame; i++)); do
        FRAME_FILE="${FRAME_FILES[$i]}"
        FRAME_NAME=$(basename "$FRAME_FILE")
        OUTPUT_FRAME="$UPSCALED_DIR/$FRAME_NAME"
        
        # Upload frame to worker
        if curl -s -X POST -F "file=@$FRAME_FILE" "$worker_url/upscale" -o "$OUTPUT_FRAME" > /dev/null 2>&1; then
            processed=$((processed + 1))
        else
            failed=$((failed + 1))
            echo "    ⚠️  Frame $i failed"
        fi
    done
    
    echo "  ✅ Worker $((worker_id+1)) completed: $processed processed, $failed failed"
}

# Start parallel processing
start_time=$(date +%s)
pids=()

current_frame=0
for i in "${!HEALTHY_WORKERS[@]}"; do
    worker_url="${HEALTHY_WORKERS[$i]}"
    
    # Calculate frame range for this worker
    frames_for_worker=$FRAMES_PER_WORKER
    if [ $i -lt $REMAINING_FRAMES ]; then
        frames_for_worker=$((frames_for_worker + 1))
    fi
    
    end_frame=$((current_frame + frames_for_worker))
    
    # Start worker in background
    process_frames_parallel $i "$worker_url" $current_frame $end_frame &
    pids+=($!)
    
    current_frame=$end_frame
done

# Wait for all workers to complete
echo "⏳ Waiting for all workers to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "  ✅ Frame upscaling completed in ${elapsed}s"

# Step 4: Reconstruct video
echo ""
echo "🎬 Step 4: Reconstructing video..."
/usr/bin/ffmpeg -framerate "$FPS" -i "$UPSCALED_DIR/frame_%06d.png" \
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
else
    echo "❌ Error: Output file was not created or is empty"
    exit 1
fi
