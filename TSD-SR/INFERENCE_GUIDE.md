# TSD-SR Inference Guide

This guide will help you run TSD-SR inference on a single image frame extracted from `elk.mp4`.

## Quick Start

### 1. Extract Frame from Video
A frame has already been extracted from `elk.mp4` and saved as `imgs/test/elk_frame.png`.

### 2. Set Up Models
Run the setup script to download required models:

```bash
cd /workspace/vidaio-win/TSD-SR
python setup_models.py
```

This will:
- Download the SD3 model from HuggingFace
- Create placeholder prompt embeddings (you'll need to replace these with real ones)

### 3. Download TSD-SR LoRA Weights
Download the TSD-SR LoRA weights from:
- [Google Drive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI?usp=drive_link)
- [OneDrive](https://1drv.ms/f/c/d75249b59f444489/EsQQ2LLXp7pHsYMBVubgcsYBvEQXMmcNXGnz695odCGByQ?e=cRTmOX)

Place the downloaded files in `checkpoint/tsdsr/`:
- `transformer.safetensors`
- `vae.safetensors`

### 4. Download Real Prompt Embeddings
Download the prompt embeddings from the same Google Drive link and place them in `dataset/default/`:
- `prompt_embeds.pt`
- `pool_embeds.pt`

### 5. Run Inference
```bash
python run_inference.py
```

Or run the original test script directly:
```bash
python test/test_tsdsr.py \
--pretrained_model_name_or_path checkpoint/sd3 \
-i imgs/test/elk_frame.png \
-o outputs/test \
--lora_dir checkpoint/tsdsr \
--embedding_dir dataset/default
```

## Directory Structure

After setup, your directory should look like:
```
TSD-SR/
├── checkpoint/
│   ├── sd3/                    # SD3 model from HuggingFace
│   └── tsdsr/                  # TSD-SR LoRA weights
│       ├── transformer.safetensors
│       └── vae.safetensors
├── dataset/
│   └── default/                # Prompt embeddings
│       ├── prompt_embeds.pt
│       └── pool_embeds.pt
├── imgs/
│   └── test/
│       └── elk_frame.png       # Extracted frame from elk.mp4
├── outputs/
│   └── test/                   # Output super-resolved images
└── test/
    └── test_tsdsr.py           # Main inference script
```

## Parameters

The inference script supports several parameters:

- `--upscale`: Upscaling factor (default: 4)
- `--process_size`: Processing size for images (default: 512)
- `--align_method`: Color alignment method (`wavelet`, `adain`, `nofix`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--mixed_precision`: Precision (`fp16` or `fp32`)

## Expected Output

The super-resolved image will be saved as `outputs/test/elk_frame.png` with 4x upscaling applied to the original frame.

## Troubleshooting

1. **CUDA out of memory**: Try reducing `--process_size` or using `--mixed_precision fp16`
2. **Missing models**: Ensure all required models are downloaded and placed in correct directories
3. **Import errors**: Install requirements with `pip install -r requirements.txt`

## Notes

- The extracted frame is 1920x1080 pixels
- With 4x upscaling, the output will be 7680x4320 pixels
- Processing time depends on your GPU and image size
- For best results, use the real prompt embeddings instead of placeholders


