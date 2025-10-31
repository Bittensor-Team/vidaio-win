cd /workspace/vidaio-subnet && source tier2_env/bin/activate && python3 local_validation.py --reference test_input.mp4 --processed test_compressed.mp4 --task compression --vmaf-threshold 80 --verbose


cd /workspace/vidaio-subnet && ffmpeg -i test_upscaled.mp4 -vf "scale=640:360" test_upscaled_scaled.mp4 -y -loglevel error && source tier2_env/bin/activate && python3 local_validation.py --reference test_input.mp4 --processed test_upscaled_scaled.mp4 --task upscaling --verbose

cd /workspace/vidaio-subnet && source tier2_env/bin/activate && python3 << 'EOF'
import subprocess
import os

print("="*80)
print("TIER 2 IMPROVED COMPRESSION WITH AV1 CODEC")
print("="*80)

input_file = "test_input.mp4"
output_file = "test_compressed_av1.mp4"

print(f"\n🎬 Compressing with AV1 codec (CRF 32)...")
cmd = [
    'ffmpeg', '-i', input_file,
    '-c:v', 'libaom-av1',
    '-crf', '32',
    '-cpu-used', '4',
    '-pix_fmt', 'yuv420p',
    '-b:v', '0',
    output_file, '-y'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode == 0:
    print("✅ AV1 compression completed")
else:
    print(f"Note: {result.stderr[:200]}")

# Get file sizes
if os.path.exists(output_file):
    ref_size = os.path.getsize(input_file)
    comp_size = os.path.getsize(output_file)
    comp_ratio = ref_size / comp_size
    
    print(f"\n📊 AV1 COMPRESSION METRICS:")
    print(f"  Reference: {ref_size:,} bytes ({ref_size/1024/1024:.3f} MB)")
    print(f"  Compressed: {comp_size:,} bytes ({comp_size/1024/1024:.3f} MB)")
    print(f"  Compression Ratio: {comp_ratio:.2f}x (was 1.81x with H.264)")
    print(f"  Size Reduction: {(1 - comp_size/ref_size)*100:.1f}%")
    
    # Calculate improved score
    compression_component = min(comp_ratio / 5.0, 1.0)
    quality_factor = 0.9
    improved_score = 0.7 * compression_component + 0.3 * quality_factor
    
    print(f"\n  Estimated New Score: ~{improved_score:.4f} (was 0.3163)")
else:
    print("⚠️ AV1 encoding not available, trying H.264 with different settings...")
    
    cmd = [
        'ffmpeg', '-i', input_file,
        '-c:v', 'libx264',
        '-crf', '32',  # Higher CRF for smaller file
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_file, '-y'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        ref_size = os.path.getsize(input_file)
        comp_size = os.path.getsize(output_file)
        comp_ratio = ref_size / comp_size
        
        print(f"\n📊 H.264 COMPRESSION (CRF 32) METRICS:")
        print(f"  Reference: {ref_size:,} bytes")
        print(f"  Compressed: {comp_size:,} bytes")
        print(f"  Compression Ratio: {comp_ratio:.2f}x")
        print(f"  Size Reduction: {(1 - comp_size/ref_size)*100:.1f}%")

EOF


cd /workspace/vidaio-subnet && source tier2_env/bin/activate && python3 local_validation.py --reference test_input.mp4 --processed test_compressed_av1.mp4 --task compression --vmaf-threshold 75 --verbose 2>&1 | tail -20


export PATH="/workspace/miniconda/envs/vidaio/bin:$PATH" && which python && which pip && python --version


/workspace/miniconda/envs/vidaio/bin/python test/test_tsdsr.py --pretrained_model_name_or_path checkpoint/sd3 -i imgs/test/elk_frame.png -o outputs/new --lora_dir checkpoint/tsdsr --embedding_dir dataset/default --upscale 4 --process_size 256 --device cuda

/workspace/miniconda/envs/vidaio/bin/python test/test_tsdsr.py --pretrained_model_name_or_path checkpoint/sd3 -i imgs/test/elk_frame.png -o outputs/new --lora_dir checkpoint/tsdsr --embedding_dir dataset/default --upscale 2 --process_size 128 --device cuda


@vidaio_pipeline_test.py this is the way to use parallel workers

simplified_vidaio_evaluation.py -> this is with scoring, 
lets score the