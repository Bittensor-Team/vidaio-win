#!/usr/bin/env python3
"""
Test script to debug parallel processing performance
"""

import aiohttp
import asyncio
import time
import json

async def test_parallel_performance():
    """Test parallel processing performance with detailed logging"""
    
    url = 'http://localhost:29115/upscale-video'
    payload = {
        'payload_url': 'http://localhost:8080/elk_1080p.mp4',
        'task_type': 'SD2HD'
    }
    
    print("🚀 Testing Parallel Processing Performance")
    print("=" * 50)
    print(f"📹 Video: {payload['payload_url']}")
    print(f"🎯 Task: {payload['task_type']}")
    print(f"⏰ Start time: {time.strftime('%H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ SUCCESS: {processing_time:.1f}s")
                    print(f"📊 File size: {result.get('file_size_mb', 0):.2f} MB")
                    print(f"🔗 Video URL: {result.get('uploaded_video_url', 'N/A')[:60]}...")
                    print(f"⏰ End time: {time.strftime('%H:%M:%S')}")
                    
                    # Performance analysis
                    if processing_time > 120:
                        print(f"⚠️  SLOW: {processing_time:.1f}s (expected ~40s with true parallelization)")
                    elif processing_time < 60:
                        print(f"🚀 FAST: {processing_time:.1f}s (excellent parallelization!)")
                    else:
                        print(f"📈 GOOD: {processing_time:.1f}s (reasonable parallelization)")
                        
                else:
                    error = await response.text()
                    print(f"❌ FAILED: {response.status}")
                    print(f"Error: {error}")
                    
    except asyncio.TimeoutError:
        print(f"⏰ TIMEOUT: Request took longer than 10 minutes")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_parallel_performance())



