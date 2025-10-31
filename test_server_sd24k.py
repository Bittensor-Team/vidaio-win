#!/usr/bin/env python3
"""
Test SD24K processing directly in the server code
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import tempfile
import os

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

def encode_vidaio_video_from_memory(frames, output_path, fps=30.0, width=1920, height=1080):
    """Encode video from frames in memory using OpenCV VideoWriter (FAST!)"""
    try:
        print(f"🎬 Encoding video from memory: {len(frames)} frames...")
        print(f"📐 Resolution: {width}x{height}, FPS: {fps}")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to open VideoWriter")
            return False
        
        # Process frames
        for i, frame in enumerate(frames):
            # Ensure frame is correct size
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Ensure frame is BGR (OpenCV expects BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already BGR from OpenCV
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            out.write(frame)
            
            if i % 30 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames")
        
        # Release VideoWriter
        out.release()
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video encoded: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error encoding video: {e}")
        return False

def test_sd24k_processing():
    """Test SD24K processing with resolution fix"""
    
    print("🧪 Testing SD24K Processing with Resolution Fix")
    print("=" * 50)
    
    # Load VSR model
    model_path = "/workspace/vidaio-win/VideoSuperResolution/VideoSuperResolution-DXM-FP32.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        model = ort.InferenceSession(model_path, providers=providers)
        print("✅ VSR model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create test frames (simulate 5 seconds at 30 FPS)
    num_frames = 150
    input_width, input_height = 854, 480
    target_width, target_height = 3840, 2160
    
    print(f"📊 Creating {num_frames} test frames ({input_width}x{input_height})")
    
    processed_frames = []
    
    for i in range(num_frames):
        # Create test frame
        test_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        test_image[:, :, 0] = (i * 2) % 255  # Blue
        test_image[:, :, 1] = (i * 3) % 255  # Green
        test_image[:, :, 2] = (i * 5) % 255  # Red
        
        # Add text
        cv2.putText(test_image, f"Frame {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process with VSR (2x upscale)
        pil_image = Image.fromarray(test_image)
        input_tensor, original_size = preprocess_image(pil_image)
        input_info = model.get_inputs()[0]
        output_info = model.get_outputs()[0]
        outputs = model.run([output_info.name], {input_info.name: input_tensor})
        output_img, _ = postprocess_output(outputs[0], original_size)
        
        # Second upscale for 4x total
        pil_image_2x = Image.fromarray(output_img)
        input_tensor_2x, second_original_size = preprocess_image(pil_image_2x)
        outputs_2x = model.run([output_info.name], {input_info.name: input_tensor_2x})
        output_img_4x, _ = postprocess_output(outputs_2x[0], second_original_size)
        
        # Apply resolution fix
        if output_img_4x.shape[:2] == (1920, 3416):
            print(f"🔧 Fixing resolution: {output_img_4x.shape[1]}x{output_img_4x.shape[0]} → {target_width}x{target_height}")
            output_img_fixed = cv2.resize(output_img_4x, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            output_img_fixed = output_img_4x
            print(f"⚠️  Unexpected resolution: {output_img_4x.shape[1]}x{output_img_4x.shape[0]}")
        
        processed_frames.append(output_img_fixed)
        
        if i % 30 == 0:
            print(f"  Processed {i+1}/{num_frames} frames")
    
    print(f"✅ Processed {len(processed_frames)} frames")
    
    # Test video encoding
    output_path = "/tmp/test_sd24k_processing.mp4"
    success = encode_vidaio_video_from_memory(
        processed_frames, output_path, fps=30.0, width=target_width, height=target_height
    )
    
    if success:
        print(f"🎉 SUCCESS! SD24K processing and encoding works!")
        print(f"📁 Output: {output_path}")
    else:
        print(f"❌ FAILED! Video encoding failed")

if __name__ == "__main__":
    test_sd24k_processing()



