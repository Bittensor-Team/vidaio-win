#!/usr/bin/env python3
"""
Minimal VSR Test
Test the VSR server with minimal processing
"""

import asyncio
import aiohttp
import time
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

async def test_minimal_request():
    """Test with minimal request"""
    logger.info("🔍 Testing minimal VSR request...")
    
    # Test with a very small video
    payload = {
        "payload_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4",
        "task_type": "SD2HD"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # First test health
            async with session.get("http://localhost:29115/health") as response:
                if response.status != 200:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return
                logger.info("✅ Health check passed")
            
            # Test upscale endpoint
            logger.info("🎬 Sending upscale request...")
            async with session.post(
                "http://localhost:29115/upscale-video",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                logger.info(f"📊 Response Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Request successful!")
                    logger.info(f"📹 Video URL: {data.get('uploaded_video_url', 'N/A')}")
                    logger.info(f"📊 File size: {data.get('file_size_mb', 0):.2f} MB")
                    logger.info(f"🎯 Task type: {data.get('task_type', 'N/A')}")
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request failed: {error_text}")
                    
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_minimal_request())



