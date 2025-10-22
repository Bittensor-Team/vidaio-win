# Clarity Upscaler Demo Scripts

This directory contains several demo scripts to upscale images locally without the web UI.

## Available Scripts

### 1. `final_demo.py` (Recommended)
**Comprehensive demo with multiple upscaling methods**

```bash
# Create test image and upscale
python final_demo.py --test --scale 2 --method enhanced

# Use your own image
python final_demo.py your_image.jpg --scale 2 --method enhanced

# Compare all methods
python final_demo.py your_image.jpg --scale 2 --compare

# Available methods:
# - basic: Basic LANCZOS upscaling
# - enhanced: Enhanced with sharpening (recommended)
# - super_res: Super resolution (may have issues)
# - nearest, bilinear, bicubic, lanczos, box, hamming: Different interpolation methods
```

### 2. `simple_upscale.py`
**Lightweight Real-ESRGAN upscaling with PIL fallback**

```bash
# Try Real-ESRGAN first, fallback to PIL
python simple_upscale.py your_image.jpg --scale 2

# Force PIL fallback
python simple_upscale.py your_image.jpg --scale 2 --fallback

# Use specific Real-ESRGAN model
python simple_upscale.py your_image.jpg --scale 2 --model RealESRGAN_x2plus
```

### 3. `demo_upscale.py`
**Full Clarity Upscaler with Stable Diffusion (Advanced)**
*Note: Requires full web UI setup and may have compatibility issues*

```bash
python demo_upscale.py your_image.jpg --scale 2 --creativity 0.35
```

## Quick Start

1. **Test with a sample image:**
   ```bash
   python final_demo.py --test --scale 2 --method enhanced
   ```

2. **Upscale your own image:**
   ```bash
   python final_demo.py your_image.jpg --scale 2 --method enhanced
   ```

3. **Compare different methods:**
   ```bash
   python final_demo.py your_image.jpg --scale 2 --compare
   ```

## Features

### `final_demo.py` Features:
- ✅ **Multiple upscaling methods** (8 different algorithms)
- ✅ **Automatic test image generation** with `--test`
- ✅ **Method comparison** with `--compare`
- ✅ **Enhanced upscaling** with sharpening and contrast
- ✅ **Super resolution** techniques
- ✅ **Fast processing** (0.001s - 0.023s)
- ✅ **No external dependencies** (uses only PIL)

### Supported Methods:
1. **Basic LANCZOS** - High quality, fast
2. **Enhanced** - With sharpening and contrast (recommended)
3. **Super Resolution** - Multiple enhancement steps
4. **Nearest Neighbor** - Pixelated, very fast
5. **Bilinear** - Smooth, fast
6. **Bicubic** - Good quality, fast
7. **Box** - Simple averaging
8. **Hamming** - Smooth with slight blur

## Examples

### Basic Usage
```bash
# Create test image and upscale 2x
python final_demo.py --test --scale 2

# Upscale existing image 4x
python final_demo.py photo.jpg --scale 4 --method enhanced
```

### Comparison
```bash
# Compare all methods on your image
python final_demo.py photo.jpg --scale 2 --compare
```

### Different Methods
```bash
# Use different upscaling methods
python final_demo.py photo.jpg --method basic
python final_demo.py photo.jpg --method super_res
python final_demo.py photo.jpg --method bicubic
```

## Output

The scripts will create upscaled images with descriptive filenames:
- `test_image_upscaled_enhanced_2x.png`
- `photo_upscaled_basic_4x.jpg`
- `output_basic_2x.png` (when comparing)

## Requirements

- Python 3.7+
- PIL/Pillow
- NumPy (for test image generation)

## Troubleshooting

1. **Import errors**: Make sure you're in the correct directory
2. **File not found**: Check the input image path
3. **Permission errors**: Make sure the script is executable (`chmod +x final_demo.py`)

## Performance

Typical processing times on a modern system:
- **Nearest neighbor**: ~0.001s
- **Bilinear/Bicubic**: ~0.003-0.004s  
- **Basic LANCZOS**: ~0.006s
- **Enhanced**: ~0.021s
- **Super resolution**: ~0.050s

## Notes

- The **enhanced** method provides the best balance of quality and speed
- **Super resolution** may have compatibility issues with some PIL versions
- All methods use CPU processing (no GPU required)
- Output images are automatically optimized for file size
