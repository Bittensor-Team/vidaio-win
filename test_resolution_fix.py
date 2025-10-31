#!/usr/bin/env python3
"""
Test the resolution fix for SD24K
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

def preprocess_image(image):
    """Preprocess image for VSR model"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    return img_tensor, (h, w)

def postprocess_output(output_tensor, original_size):
    """Postprocess model output back to image"""
    output_img = np.transpose(output_tensor[0], (1, 2, 0))
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    new_h, new_w = output_img.shape[:2]
    upscale_factor = new_h / original_size[0]
    return output_img, upscale_factor

def test_sd24k_resolution_fix():
    """Test SD24K resolution fix"""
    
    print("🧪 Testing SD24K Resolution Fix")
    print("=" * 40)
    
    # Load VSR model
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        model = ort.InferenceSession(model_path, providers=providers)
        print("✅ VSR model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create 480p test image
    width, height = 854, 480
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    test_image[:, :, 0] = 100  # Blue
    test_image[:, :, 1] = 150  # Green
    test_image[:, :, 2] = 200  # Red
    
    print(f"📐 Input: {width}x{height}")
    
    # First upscale: 480p → 1080p
    pil_image = Image.fromarray(test_image)
    input_tensor, original_size = preprocess_image(pil_image)
    input_info = model.get_inputs()[0]
    output_info = model.get_outputs()[0]
    outputs = model.run([output_info.name], {input_info.name: input_tensor})
    output_img, _ = postprocess_output(outputs[0], original_size)
    
    print(f"📐 After 1st upscale: {output_img.shape[1]}x{output_img.shape[0]}")
    
    # Second upscale: 1080p → 4K
    pil_image_2x = Image.fromarray(output_img)
    input_tensor_2x, second_original_size = preprocess_image(pil_image_2x)
    outputs_2x = model.run([output_info.name], {input_info.name: input_tensor_2x})
    output_img_4x, _ = postprocess_output(outputs_2x[0], second_original_size)
    
    print(f"📐 After 2nd upscale: {output_img_4x.shape[1]}x{output_img_4x.shape[0]}")
    
    # Apply resolution fix
    if output_img_4x.shape[:2] == (1920, 3416):
        print(f"🔧 Applying resolution fix: {output_img_4x.shape[1]}x{output_img_4x.shape[0]} → 3840x2160")
        output_img_fixed = cv2.resize(output_img_4x, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
        print(f"✅ Fixed resolution: {output_img_fixed.shape[1]}x{output_img_fixed.shape[0]}")
        
        # Test video encoding with fixed resolution
        print(f"\n🎬 Testing video encoding with fixed resolution...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('/tmp/test_sd24k_fixed.mp4', fourcc, 30.0, (3840, 2160))
        
        if out.isOpened():
            out.write(output_img_fixed)
            out.release()
            print("✅ Video encoding successful!")
            
            # Check file
            import os
            if os.path.exists('/tmp/test_sd24k_fixed.mp4'):
                size = os.path.getsize('/tmp/test_sd24k_fixed.mp4') / 1024 / 1024
                print(f"✅ Video created: {size:.2f} MB")
            else:
                print("❌ Video file not created")
        else:
            print("❌ Failed to open VideoWriter")
    else:
        print(f"❌ Unexpected resolution: {output_img_4x.shape[1]}x{output_img_4x.shape[0]}")

if __name__ == "__main__":
    test_sd24k_resolution_fix()



