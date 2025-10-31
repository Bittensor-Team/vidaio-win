#!/usr/bin/env python3
"""
Test Local Video
Test VSR server with local video file
"""

import asyncio
import aiohttp
import time
import os
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

async def test_local_video():
    """Test with local video file"""
    logger.info("🔍 Testing VSR server with local video...")
    
    # Use local elk video
    local_video_path = "/workspace/vidaio-win/real_vidaio_tests/elk_1080p.mp4"
    
    if not os.path.exists(local_video_path):
        logger.error(f"❌ Local video not found: {local_video_path}")
        return
    
    # Create a simple HTTP server to serve the local file
    import http.server
    import socketserver
    import threading
    
    # Start simple HTTP server
    port = 8080
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        # Change to the directory containing the video
        os.chdir("/workspace/vidaio-win/real_vidaio_tests")
        
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test video URL
        video_url = f"http://localhost:{port}/elk_1080p.mp4"
        logger.info(f"📹 Video URL: {video_url}")
        
        # Test VSR server
        payload = {
            "payload_url": video_url,
            "task_type": "SD2HD"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.info("🎬 Sending upscale request...")
                async with session.post(
                    "http://localhost:29115/upscale-video",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    logger.info(f"📊 Response Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ Request successful!")
                        logger.info(f"📹 Video URL: {data.get('uploaded_video_url', 'N/A')}")
                        logger.info(f"📊 File size: {data.get('file_size_mb', 0):.2f} MB")
                        logger.info(f"🎯 Task type: {data.get('task_type', 'N/A')}")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Request failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        
        # Stop HTTP server
        httpd.shutdown()

if __name__ == "__main__":
    asyncio.run(test_local_video())



