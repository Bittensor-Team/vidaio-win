#!/usr/bin/env python3
"""
Debug VSR model output resolution to understand SD24K encoding issue
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

def test_vsr_resolution():
    """Test what resolution the VSR model actually outputs"""
    
    print("🔍 DEBUGGING VSR MODEL OUTPUT RESOLUTION")
    print("=" * 50)
    
    # Load VSR model
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        model = ort.InferenceSession(model_path, providers=providers)
        print("✅ VSR model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test different input resolutions
    test_cases = [
        ("480p", 854, 480, "SD24K input"),
        ("1080p", 1920, 1080, "HD24K input"),
        ("4K", 3840, 2160, "4K28K input")
    ]
    
    for name, width, height, description in test_cases:
        print(f"\n🧪 Testing {description} ({name}): {width}x{height}")
        
        # Create test image
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        test_image[:, :, 0] = 100  # Blue
        test_image[:, :, 1] = 150  # Green
        test_image[:, :, 2] = 200  # Red
        
        # Convert to PIL
        pil_image = Image.fromarray(test_image)
        
        # Preprocess
        input_tensor, original_size = preprocess_image(pil_image)
        print(f"   Input tensor shape: {input_tensor.shape}")
        print(f"   Original size: {original_size}")
        
        # Run VSR inference
        input_info = model.get_inputs()[0]
        output_info = model.get_outputs()[0]
        outputs = model.run([output_info.name], {input_info.name: input_tensor})
        
        # Postprocess
        output_img, upscale_factor = postprocess_output(outputs[0], original_size)
        output_h, output_w = output_img.shape[:2]
        
        print(f"   Output size: {output_w}x{output_h}")
        print(f"   Upscale factor: {upscale_factor:.2f}x")
        print(f"   Expected 2x upscale: {width*2}x{height*2}")
        
        # Check if it matches expected 2x upscale
        expected_w, expected_h = width * 2, height * 2
        if output_w == expected_w and output_h == expected_h:
            print(f"   ✅ Resolution matches expected 2x upscale")
        else:
            print(f"   ❌ Resolution mismatch! Expected {expected_w}x{expected_h}, got {output_w}x{output_h}")
        
        # Test 4x upscaling (SD24K case)
        if name == "480p":
            print(f"\n   🔄 Testing 4x upscaling (SD24K):")
            
            # First upscale: 480p → 1080p
            pil_image_2x = Image.fromarray(output_img)
            input_tensor_2x, second_original_size = preprocess_image(pil_image_2x)
            outputs_2x = model.run([output_info.name], {input_info.name: input_tensor_2x})
            output_img_4x, _ = postprocess_output(outputs_2x[0], second_original_size)
            
            final_h, final_w = output_img_4x.shape[:2]
            print(f"   Final output size: {final_w}x{final_h}")
            print(f"   Expected 4K: 3840x2160")
            
            if final_w == 3840 and final_h == 2160:
                print(f"   ✅ 4x upscaling produces correct 4K resolution")
            else:
                print(f"   ❌ 4x upscaling produces wrong resolution!")
                print(f"   📊 Resolution difference: {3840-final_w}x{2160-final_h}")

if __name__ == "__main__":
    test_vsr_resolution()