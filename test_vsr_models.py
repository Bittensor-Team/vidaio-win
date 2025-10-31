#!/usr/bin/env python3
"""
Test script for Microsoft Edge VideoSuperResolution ONNX models
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

def load_image(image_path):
    """Load and preprocess image for VSR model"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = img_rgb.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Convert to float32 and normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension: [1, C, H, W]
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    
    print(f"Input tensor shape: {img_tensor.shape}")
    return img_tensor, (h, w)

def postprocess_output(output_tensor, original_size):
    """Postprocess model output back to image"""
    # Remove batch dimension and transpose back to HWC
    output_img = np.transpose(output_tensor[0], (1, 2, 0))
    
    # Denormalize from [0, 1] to [0, 255]
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    
    # Get new dimensions
    new_h, new_w = output_img.shape[:2]
    print(f"Output image size: {new_w}x{new_h}")
    
    # Calculate upscaling factor
    upscale_factor = new_h / original_size[0]
    print(f"Upscaling factor: {upscale_factor:.2f}x")
    
    return output_img

def test_vsr_model(model_path, input_tensor, output_dir):
    """Test a single VSR model"""
    print(f"\n{'='*60}")
    print(f"Testing model: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    try:
        # Load ONNX model
        print("Loading ONNX model...")
        session = ort.InferenceSession(model_path)
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"Input: {input_info.name} - {input_info.shape} ({input_info.type})")
        print(f"Output: {output_info.name} - {output_info.shape} ({output_info.type})")
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        
        outputs = session.run([output_info.name], {input_info.name: input_tensor})
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Postprocess output
        output_img = postprocess_output(outputs[0], input_tensor.shape[2:])
        
        # Save output image
        output_filename = f"vsr_output_{os.path.splitext(os.path.basename(model_path))[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert RGB to BGR for OpenCV
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        
        print(f"Output saved to: {output_path}")
        
        return True, inference_time
        
    except Exception as e:
        print(f"Error testing model {model_path}: {e}")
        return False, 0

def main():
    # Paths
    input_image = "/workspace/vidaio-win/optimization_tests/test_frame.png"
    vsr_dir = "/workspace/vidaio-win/VideoSuperResolution"
    output_dir = "/workspace/vidaio-win/optimization_tests/vsr_outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input image
    print("Loading input image...")
    input_tensor, original_size = load_image(input_image)
    
    # Find all ONNX models
    onnx_models = [f for f in os.listdir(vsr_dir) if f.endswith('.onnx')]
    onnx_models.sort()
    
    print(f"\nFound {len(onnx_models)} ONNX models:")
    for model in onnx_models:
        print(f"  - {model}")
    
    # Test each model
    results = []
    for model_name in onnx_models:
        model_path = os.path.join(vsr_dir, model_name)
        success, inference_time = test_vsr_model(model_path, input_tensor, output_dir)
        results.append((model_name, success, inference_time))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for model_name, success, inference_time in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{model_name:30} | {status:7} | {inference_time:.3f}s")
    
    print(f"\nOutput images saved to: {output_dir}")

if __name__ == "__main__":
    main()





