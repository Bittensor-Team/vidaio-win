#!/usr/bin/env python3
"""
Debug Server
Run server in foreground and test to see detailed errors
"""

import subprocess
import time
import requests
import json

def test_server():
    """Test the server with detailed error reporting"""
    
    # Start server in background
    print("🚀 Starting VSR Server...")
    process = subprocess.Popen([
        "python", "vsr_worker_server_enhanced.py", 
        "--port", "29115", 
        "--model", "DXM-FP32",
        "--output-folder", "/workspace/vidaio-win/test_output"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for server to start
    time.sleep(15)
    
    # Test health endpoint
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:29115/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test upscale endpoint
    print("🔍 Testing upscale endpoint...")
    payload = {
        "payload_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4",
        "task_type": "SD2HD"
    }
    
    try:
        response = requests.post(
            "http://localhost:29115/upscale-video",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"📊 Response Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request error: {e}")
    
    # Stop server
    process.terminate()
    process.wait()
    
    # Print server output
    print("\n📋 Server Output:")
    print(process.stdout.read())

if __name__ == "__main__":
    test_server()



