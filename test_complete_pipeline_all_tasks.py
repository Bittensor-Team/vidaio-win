#!/usr/bin/env python3
"""
Complete Pipeline Test - All Tasks
Test the entire Vidaio pipeline for all 4 task types
"""

import asyncio
import aiohttp
import time
import json
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Test configuration
VIDEO_URL = "http://localhost:8080/elk_1080p.mp4"
VSR_SERVER_URL = "http://localhost:29115"
TASK_TYPES = ["SD2HD", "HD24K", "SD24K", "4K28K"]

async def test_single_task(task_type: str) -> dict:
    """Test a single task type through the complete pipeline"""
    
    logger.info(f"🎯 Testing {task_type} task...")
    logger.info(f"📹 Video: {VIDEO_URL}")
    
    start_time = time.time()
    
    try:
        # Step 1: Send request to VSR Server (simulating Miner → video_upscaler() → VSR Server)
        payload = {
            "payload_url": VIDEO_URL,
            "task_type": task_type
        }
        
        logger.info(f"📤 Sending request to VSR Server...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{VSR_SERVER_URL}/upscale-video",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
            ) as response:
                
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if video was uploaded to R2
                    video_url = data.get('uploaded_video_url', '')
                    is_r2_upload = 'r2.cloudflarestorage.com' in video_url
                    
                    result = {
                        "task_type": task_type,
                        "status": "success",
                        "processing_time": processing_time,
                        "video_url": video_url,
                        "file_size_mb": data.get('file_size_mb', 0),
                        "r2_uploaded": is_r2_upload,
                        "response_data": data
                    }
                    
                    logger.info(f"✅ {task_type} completed successfully!")
                    logger.info(f"⏱️  Processing time: {processing_time:.1f}s")
                    logger.info(f"📹 Video URL: {video_url[:80]}...")
                    logger.info(f"📊 File size: {data.get('file_size_mb', 0):.2f} MB")
                    logger.info(f"☁️  R2 Upload: {'✅ Yes' if is_r2_upload else '❌ No'}")
                    
                    return result
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ {task_type} failed: {response.status}")
                    logger.error(f"Error: {error_text}")
                    
                    return {
                        "task_type": task_type,
                        "status": "failed",
                        "processing_time": processing_time,
                        "error": f"HTTP {response.status}: {error_text}",
                        "r2_uploaded": False
                    }
                    
    except asyncio.TimeoutError:
        processing_time = time.time() - start_time
        logger.error(f"⏰ {task_type} timed out after {processing_time:.1f}s")
        return {
            "task_type": task_type,
            "status": "timeout",
            "processing_time": processing_time,
            "error": "Request timed out",
            "r2_uploaded": False
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ {task_type} error: {e}")
        return {
            "task_type": task_type,
            "status": "error",
            "processing_time": processing_time,
            "error": str(e),
            "r2_uploaded": False
        }

async def test_complete_pipeline():
    """Test the complete pipeline for all task types"""
    
    logger.info("🚀 COMPLETE VIDAIO PIPELINE TEST")
    logger.info("=" * 60)
    logger.info("📋 Testing: Validator → Miner → video_upscaler() → VSR Server → R2 Storage → Response")
    logger.info("=" * 60)
    
    # Check VSR Server health first
    logger.info("🔍 Step 1: Checking VSR Server Health...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VSR_SERVER_URL}/health", timeout=10) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ VSR Server healthy: {health_data.get('model_name', 'Unknown')}")
                    logger.info(f"📊 Supported tasks: {health_data.get('supported_tasks', [])}")
                else:
                    logger.error(f"❌ VSR Server unhealthy: {response.status}")
                    return
    except Exception as e:
        logger.error(f"❌ Cannot connect to VSR Server: {e}")
        return
    
    # Test all task types
    logger.info(f"\n🎬 Step 2: Testing All Task Types...")
    logger.info(f"📹 Test video: {VIDEO_URL}")
    logger.info(f"🎯 Tasks: {', '.join(TASK_TYPES)}")
    logger.info("-" * 60)
    
    results = []
    total_start_time = time.time()
    
    for i, task_type in enumerate(TASK_TYPES, 1):
        logger.info(f"\n📋 Task {i}/{len(TASK_TYPES)}: {task_type}")
        logger.info("-" * 40)
        
        result = await test_single_task(task_type)
        results.append(result)
        
        # Small delay between tasks to avoid overwhelming the server
        if i < len(TASK_TYPES):
            logger.info("⏳ Waiting 5 seconds before next task...")
            await asyncio.sleep(5)
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info(f"\n📊 PIPELINE TEST SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    r2_uploads = [r for r in results if r.get('r2_uploaded', False)]
    
    logger.info(f"✅ Successful: {len(successful)}/{len(results)}")
    logger.info(f"❌ Failed: {len(failed)}/{len(results)}")
    logger.info(f"☁️  R2 Uploads: {len(r2_uploads)}/{len(results)}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    
    # Detailed results
    logger.info(f"\n📋 DETAILED RESULTS:")
    logger.info("-" * 60)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        r2_icon = "☁️" if result.get('r2_uploaded', False) else "📁"
        
        logger.info(f"{status_icon} {result['task_type']}: {result['status']} ({result['processing_time']:.1f}s) {r2_icon}")
        
        if result['status'] == 'success':
            logger.info(f"   📹 URL: {result['video_url'][:60]}...")
            logger.info(f"   📊 Size: {result['file_size_mb']:.2f} MB")
        elif 'error' in result:
            logger.info(f"   ❌ Error: {result['error']}")
    
    # Overall assessment
    logger.info(f"\n🎯 OVERALL ASSESSMENT:")
    logger.info("-" * 60)
    
    if len(successful) == len(results):
        logger.info("🎉 ALL TASKS PASSED! Pipeline is fully operational!")
        logger.info("✅ Validator → Miner → video_upscaler() → VSR Server → R2 Storage → Response")
    elif len(successful) > 0:
        logger.info(f"⚠️  PARTIAL SUCCESS: {len(successful)}/{len(results)} tasks passed")
        logger.info("🔧 Some tasks failed - check logs above for details")
    else:
        logger.info("❌ ALL TASKS FAILED! Pipeline needs debugging")
        logger.info("🔧 Check VSR Server logs and configuration")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())
