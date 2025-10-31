#!/usr/bin/env python3
"""
Simple VSR Test
Tests the VSR server with a local file
"""

import asyncio
import aiohttp
import time
import os
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

async def test_simple_vsr():
    """Test VSR server with a simple request"""
    server_url = "http://localhost:29115"
    
    # Test with a simple video URL (not file://)
    test_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4"
    
    payload = {
        "payload_url": test_url,
        "task_type": "SD2HD"
    }
    
    logger.info("🎬 Testing VSR Server with simple request...")
    logger.info(f"📹 Video URL: {test_url}")
    logger.info(f"🎯 Task: SD2HD")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/upscale-video",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                logger.info(f"📊 Response Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Request successful!")
                    logger.info(f"📹 Response: {data}")
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request failed: {error_text}")
                    
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_vsr())



