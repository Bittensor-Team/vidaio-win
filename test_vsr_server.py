#!/usr/bin/env python3
"""
Test client for Vidaio VSR Server
"""

import asyncio
import aiohttp
import time
import json
from loguru import logger

SERVER_URL = "http://localhost:29115"

async def test_health():
    """Test health endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{SERVER_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"✅ Health check passed: {data}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status}")
                return False

async def test_stats():
    """Test stats endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{SERVER_URL}/stats") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"📊 Server stats: {json.dumps(data, indent=2)}")
                return True
            else:
                logger.error(f"❌ Stats check failed: {response.status}")
                return False

async def test_upscale(task_type: str, video_url: str):
    """Test upscaling endpoint"""
    payload = {
        "payload_url": video_url,
        "task_type": task_type,
        "maximum_optimized_size_mb": 100
    }
    
    logger.info(f"🎬 Testing {task_type} upscaling...")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SERVER_URL}/upscale-video",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                elapsed = time.time() - start_time
                logger.info(f"✅ {task_type} upscaling completed in {elapsed:.2f}s")
                logger.info(f"📹 Output URL: {data['optimized_video_url']}")
                logger.info(f"📊 File size: {data['file_size_mb']:.2f} MB")
                return True
            else:
                error_text = await response.text()
                logger.error(f"❌ {task_type} upscaling failed: {response.status} - {error_text}")
                return False

async def main():
    """Main test function"""
    logger.info("🧪 Starting VSR Server Tests...")
    
    # Test health
    if not await test_health():
        logger.error("❌ Server not healthy, exiting")
        return
    
    # Test stats
    await test_stats()
    
    # Test upscaling with different task types
    test_video_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
    
    test_tasks = ["SD2HD", "HD24K", "4K28K"]
    
    for task_type in test_tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {task_type}")
        logger.info(f"{'='*50}")
        
        success = await test_upscale(task_type, test_video_url)
        if not success:
            logger.error(f"❌ {task_type} test failed")
        else:
            logger.info(f"✅ {task_type} test passed")
        
        # Wait between tests
        await asyncio.sleep(2)
    
    logger.info("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())





