#!/usr/bin/env python3
"""
Debug server error by running it in foreground and making a request
"""

import subprocess
import time
import requests
import json
import signal
import os
import sys

def test_server_debug():
    """Run server in foreground and test SD24K"""
    
    print("🔍 DEBUGGING SERVER ERROR")
    print("=" * 30)
    
    # Start server in background
    env = os.environ.copy()
    env['R2_ACCESS_KEY'] = "eacd80b3fcf1a6d0c98572134a05680c"
    env['R2_SECRET_KEY'] = "b5a4e900e924f029dd804f096a765c9cc590d997bb25aa7dd0e0fa78ca36ed5f"
    
    cmd = [
        'python', 'vsr_worker_server_enhanced.py',
        '--port', '29115',
        '--model', 'DXM-FP32',
        '--output-folder', '/workspace/vidaio-win/test_output'
    ]
    
    print("🚀 Starting server...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:29115/health', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        process.terminate()
        return
    
    # Test SD24K
    print("\n🧪 Testing SD24K...")
    payload = {
        'payload_url': 'http://localhost:8080/elk_1080p.mp4',
        'task_type': 'SD24K'
    }
    
    try:
        response = requests.post(
            'http://localhost:29115/upscale-video',
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS: {result}")
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Stop server
    print("\n🛑 Stopping server...")
    process.terminate()
    process.wait()
    
    # Show server output
    print("\n📋 SERVER OUTPUT:")
    print("-" * 40)
    output = process.stdout.read()
    if output:
        print(output)
    else:
        print("No output captured")

if __name__ == "__main__":
    test_server_debug()



