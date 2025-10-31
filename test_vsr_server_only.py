#!/usr/bin/env python3
"""
Test VSR Server Only (without R2)
Tests the VSR server functionality without requiring R2 credentials
"""

import asyncio
import aiohttp
import time
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class VSRServerTester:
    def __init__(self):
        self.server_url = "http://localhost:29115"
        self.test_video_path = "/workspace/vidaio-win/real_vidaio_tests/elk_1080p.mp4"
        self.task_types = ["SD2HD", "HD24K", "SD24K", "4K28K"]
        self.results = {}
        
    async def test_server_health(self) -> bool:
        """Test if VSR server is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ VSR Server healthy: {data}")
                        return True
                    else:
                        logger.error(f"❌ VSR Server unhealthy: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to VSR Server: {e}")
            return False
    
    async def test_task_types(self) -> bool:
        """Test available task types"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/task_types") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Available task types: {data}")
                        return True
                    else:
                        logger.error(f"❌ Failed to get task types: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error getting task types: {e}")
            return False
    
    async def upload_test_video(self) -> Optional[str]:
        """Upload elk video to a temporary location for testing"""
        if not os.path.exists(self.test_video_path):
            logger.error(f"❌ Test video not found: {self.test_video_path}")
            return None
            
        # For testing, we'll use the local file path
        logger.info(f"📹 Using test video: {self.test_video_path}")
        return self.test_video_path
    
    async def send_upscaling_request(self, video_url: str, task_type: str) -> Optional[Dict]:
        """Send upscaling request to VSR server"""
        payload = {
            "payload_url": video_url,
            "task_type": task_type
        }
        
        logger.info(f"🎬 Sending {task_type} upscaling request...")
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/upscale-video",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ {task_type} completed in {elapsed:.2f}s")
                        logger.info(f"📹 Response: {data}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {task_type} failed: {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"⏰ {task_type} request timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"❌ {task_type} request failed: {e}")
            return None
    
    async def test_complete_pipeline(self) -> Dict:
        """Test the complete pipeline for all task types"""
        logger.info("🚀 Starting VSR Server Test (without R2)")
        logger.info("=" * 60)
        
        # Test 1: Server Health
        logger.info("🔍 Step 1: Testing VSR Server Health")
        if not await self.test_server_health():
            return {"error": "VSR Server not available"}
        
        # Test 2: Task Types
        logger.info("🔍 Step 2: Testing Available Task Types")
        if not await self.test_task_types():
            return {"error": "Cannot get task types"}
        
        # Test 3: Upload Test Video
        logger.info("🔍 Step 3: Preparing Test Video")
        video_url = await self.upload_test_video()
        if not video_url:
            return {"error": "No test video available"}
        
        # Test 4: Process Each Task Type
        results = {}
        
        for task_type in self.task_types:
            logger.info(f"🔍 Step 4: Testing {task_type} Pipeline")
            logger.info("-" * 40)
            
            # Send upscaling request
            response = await self.send_upscaling_request(video_url, task_type)
            if not response:
                results[task_type] = {"error": "Upscaling request failed"}
                continue
            
            # Check response format
            if "uploaded_video_url" in response:
                logger.info(f"✅ {task_type} - R2 upload successful")
                logger.info(f"   📹 R2 URL: {response['uploaded_video_url']}")
            elif "optimized_video_url" in response:
                logger.info(f"✅ {task_type} - Local file response")
                logger.info(f"   📹 Local URL: {response['optimized_video_url']}")
            else:
                logger.warning(f"⚠️  {task_type} - Unknown response format")
            
            results[task_type] = {
                "success": True,
                "response": response,
                "has_r2_url": "uploaded_video_url" in response,
                "has_local_url": "optimized_video_url" in response
            }
            
            logger.info(f"✅ {task_type} Pipeline Complete")
            logger.info("")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print test summary"""
        logger.info("📊 VSR SERVER TEST SUMMARY")
        logger.info("=" * 60)
        
        if "error" in results:
            logger.error(f"❌ Test failed: {results['error']}")
            return
        
        success_count = 0
        r2_count = 0
        local_count = 0
        total_count = len(self.task_types)
        
        for task_type in self.task_types:
            if task_type in results:
                result = results[task_type]
                if result.get("success"):
                    success_count += 1
                    if result.get("has_r2_url"):
                        r2_count += 1
                        logger.info(f"✅ {task_type}: R2 Upload Success")
                    elif result.get("has_local_url"):
                        local_count += 1
                        logger.info(f"✅ {task_type}: Local File Response")
                    else:
                        logger.warning(f"⚠️  {task_type}: Unknown response format")
                else:
                    logger.error(f"❌ {task_type}: {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"❌ {task_type}: No result")
        
        logger.info("=" * 60)
        logger.info(f"📈 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        logger.info(f"☁️  R2 Uploads: {r2_count}/{total_count}")
        logger.info(f"📁 Local Files: {local_count}/{total_count}")
        
        if success_count == total_count:
            if r2_count > 0:
                logger.info("🎉 ALL TESTS PASSED! R2 integration working!")
            else:
                logger.info("🎉 ALL TESTS PASSED! Server working (R2 credentials needed)")
        else:
            logger.warning(f"⚠️  {total_count - success_count} tests failed. Check logs above.")

async def main():
    """Main test function"""
    tester = VSRServerTester()
    
    # Run complete pipeline test
    results = await tester.test_complete_pipeline()
    
    # Print summary
    tester.print_summary(results)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())



