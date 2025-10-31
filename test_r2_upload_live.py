#!/usr/bin/env python3
"""
Test R2 Upload Live
Test the VSR server with R2 upload enabled
"""

import asyncio
import aiohttp
import time
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

async def test_r2_upload():
    """Test VSR server with R2 upload"""
    logger.info("🚀 Testing VSR Server with R2 Upload")
    logger.info("=" * 50)
    
    # Test video URL (local HTTP server)
    video_url = "http://localhost:8080/elk_1080p.mp4"
    task_type = "SD2HD"
    
    payload = {
        "payload_url": video_url,
        "task_type": task_type
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"🎬 Sending request: {task_type}")
            logger.info(f"📹 Video: {video_url}")
            
            start_time = time.time()
            
            async with session.post(
                "http://localhost:29115/upscale-video",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                
                processing_time = time.time() - start_time
                logger.info(f"⏱️  Processing time: {processing_time:.1f}s")
                logger.info(f"📊 Response Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Request successful!")
                    logger.info(f"📹 Video URL: {data.get('uploaded_video_url', 'N/A')}")
                    logger.info(f"📊 File size: {data.get('file_size_mb', 0):.2f} MB")
                    logger.info(f"🎯 Task type: {data.get('task_type', 'N/A')}")
                    
                    # Check if it's an R2 URL
                    video_url = data.get('uploaded_video_url', '')
                    if 'r2.cloudflarestorage.com' in video_url:
                        logger.info("🎉 SUCCESS: Video uploaded to R2!")
                    elif video_url.startswith('file://'):
                        logger.warning("⚠️  Video saved locally only (R2 upload failed)")
                    else:
                        logger.info(f"📋 Video URL type: {video_url[:50]}...")
                        
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request failed: {error_text}")
                    
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_r2_upload())



