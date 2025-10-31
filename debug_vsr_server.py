#!/usr/bin/env python3
"""
Debug VSR Server
Simple test to debug the 500 error
"""

import asyncio
import aiohttp
import time
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

async def test_health():
    """Test health endpoint"""
    logger.info("🔍 Testing health endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:29115/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Health check passed: {data}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

async def test_simple_upscale():
    """Test simple upscale request"""
    logger.info("🔍 Testing simple upscale request...")
    
    # Use a very simple test video
    payload = {
        "payload_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4",
        "task_type": "SD2HD"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:29115/upscale-video",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                logger.info(f"📊 Response Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Request successful: {data}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request failed: {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Request error: {e}")
        return False

async def main():
    """Main debug function"""
    logger.info("🚀 Debugging VSR Server")
    logger.info("=" * 40)
    
    # Test 1: Health check
    if not await test_health():
        logger.error("❌ Health check failed. Server not running?")
        return
    
    # Test 2: Simple upscale
    await test_simple_upscale()

if __name__ == "__main__":
    asyncio.run(main())



