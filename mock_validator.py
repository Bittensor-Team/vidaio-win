#!/usr/bin/env python3
"""
Mock Validator for Testing Vidaio VSR Pipeline
Simulates validator behavior to test the complete pipeline:
Validator → Miner → video_upscaler() → VSR Server → Cloudflare R2
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

class MockValidator:
    def __init__(self):
        self.miner_url = "http://localhost:29115"
        self.test_video_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4"
        self.elk_video_path = "/workspace/vidaio-win/real_vidaio_tests/elk_1080p.mp4"
        self.task_types = ["SD2HD", "HD24K", "SD24K", "4K28K"]
        self.results = {}
        
    async def test_vsr_server_health(self) -> bool:
        """Test if VSR server is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.miner_url}/health") as response:
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
                async with session.get(f"{self.miner_url}/task_types") as response:
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
        if not os.path.exists(self.elk_video_path):
            logger.error(f"❌ Test video not found: {self.elk_video_path}")
            return None
            
        # For testing, we'll use the local file path
        # In real scenario, this would be uploaded to a public URL
        logger.info(f"📹 Using test video: {self.elk_video_path}")
        return self.elk_video_path
    
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
                    f"{self.miner_url}/upscale-video",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ {task_type} completed in {elapsed:.2f}s")
                        logger.info(f"📹 Output URL: {data.get('uploaded_video_url', 'N/A')}")
                        logger.info(f"📊 Processing time: {data.get('processing_time', 0):.2f}s")
                        logger.info(f"📏 File size: {data.get('file_size_mb', 0):.2f} MB")
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
    
    async def download_and_verify_video(self, video_url: str, task_type: str) -> Optional[Dict]:
        """Download video from R2 and verify it"""
        if not video_url:
            logger.error(f"❌ No video URL provided for {task_type}")
            return None
            
        logger.info(f"📥 Downloading {task_type} video from R2...")
        
        try:
            # Create temporary file for download
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{task_type}.mp4", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Download video
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status == 200:
                        with open(temp_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        file_size = os.path.getsize(temp_path)
                        logger.info(f"✅ Downloaded {task_type} video: {file_size / (1024*1024):.2f} MB")
                        
                        # Get video info using ffprobe
                        video_info = await self.get_video_info(temp_path)
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        return {
                            "file_size_mb": file_size / (1024*1024),
                            "video_info": video_info,
                            "download_success": True
                        }
                    else:
                        logger.error(f"❌ Failed to download {task_type} video: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error downloading {task_type} video: {e}")
            return None
    
    async def get_video_info(self, video_path: str) -> Dict:
        """Get video information using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = None
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        "width": video_stream.get("width"),
                        "height": video_stream.get("height"),
                        "duration": float(data.get("format", {}).get("duration", 0)),
                        "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                        "codec": video_stream.get("codec_name"),
                        "bitrate": int(data.get("format", {}).get("bit_rate", 0))
                    }
                else:
                    return {"error": "No video stream found"}
            else:
                return {"error": f"ffprobe failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Error getting video info: {e}"}
    
    def calculate_vidaio_score(self, video_info: Dict, task_type: str) -> Dict:
        """Calculate Vidaio-style scoring metrics"""
        if "error" in video_info:
            return {"error": video_info["error"]}
        
        # Expected resolutions for each task type
        expected_resolutions = {
            "SD2HD": (1920, 1080),
            "HD24K": (3840, 2160),
            "SD24K": (3840, 2160),
            "4K28K": (7680, 4320)
        }
        
        width = video_info.get("width", 0)
        height = video_info.get("height", 0)
        duration = video_info.get("duration", 0)
        fps = video_info.get("fps", 0)
        
        expected_width, expected_height = expected_resolutions.get(task_type, (0, 0))
        
        # Resolution validation
        resolution_correct = (width == expected_width and height == expected_height)
        
        # Basic quality metrics (simplified)
        duration_score = min(duration / 10.0, 1.0)  # Normalize to 10 seconds
        resolution_score = 1.0 if resolution_correct else 0.0
        fps_score = min(fps / 30.0, 1.0)  # Normalize to 30 fps
        
        # Combined score (simplified Vidaio scoring)
        combined_score = (duration_score * 0.3 + resolution_score * 0.5 + fps_score * 0.2)
        
        return {
            "resolution_correct": resolution_correct,
            "expected_resolution": f"{expected_width}x{expected_height}",
            "actual_resolution": f"{width}x{height}",
            "duration_seconds": duration,
            "fps": fps,
            "duration_score": duration_score,
            "resolution_score": resolution_score,
            "fps_score": fps_score,
            "combined_score": combined_score,
            "vidaio_ready": resolution_correct and duration > 0
        }
    
    async def test_complete_pipeline(self) -> Dict:
        """Test the complete pipeline for all task types"""
        logger.info("🚀 Starting Complete Vidaio VSR Pipeline Test")
        logger.info("=" * 60)
        
        # Test 1: Server Health
        logger.info("🔍 Step 1: Testing VSR Server Health")
        if not await self.test_vsr_server_health():
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
            
            # Download and verify video
            video_url_result = response.get("uploaded_video_url")
            if not video_url_result:
                results[task_type] = {"error": "No video URL in response"}
                continue
            
            # Download video from R2
            download_result = await self.download_and_verify_video(video_url_result, task_type)
            if not download_result:
                results[task_type] = {"error": "Video download failed"}
                continue
            
            # Calculate Vidaio score
            video_info = download_result.get("video_info", {})
            score = self.calculate_vidaio_score(video_info, task_type)
            
            results[task_type] = {
                "success": True,
                "response": response,
                "download": download_result,
                "video_info": video_info,
                "score": score,
                "r2_url": video_url_result
            }
            
            logger.info(f"✅ {task_type} Pipeline Complete")
            logger.info(f"   📊 Score: {score.get('combined_score', 0):.3f}")
            logger.info(f"   📏 Resolution: {score.get('actual_resolution', 'N/A')}")
            logger.info(f"   ⏱️  Duration: {score.get('duration_seconds', 0):.1f}s")
            logger.info(f"   🎯 Vidaio Ready: {score.get('vidaio_ready', False)}")
            logger.info("")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print test summary"""
        logger.info("📊 PIPELINE TEST SUMMARY")
        logger.info("=" * 60)
        
        if "error" in results:
            logger.error(f"❌ Pipeline failed: {results['error']}")
            return
        
        success_count = 0
        total_count = len(self.task_types)
        
        for task_type in self.task_types:
            if task_type in results:
                result = results[task_type]
                if result.get("success"):
                    success_count += 1
                    score = result.get("score", {})
                    logger.info(f"✅ {task_type}: Score {score.get('combined_score', 0):.3f} | {score.get('actual_resolution', 'N/A')}")
                else:
                    logger.error(f"❌ {task_type}: {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"❌ {task_type}: No result")
        
        logger.info("=" * 60)
        logger.info(f"📈 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count == total_count:
            logger.info("🎉 ALL TESTS PASSED! Pipeline is ready for production!")
        else:
            logger.warning(f"⚠️  {total_count - success_count} tests failed. Check logs above.")

async def main():
    """Main test function"""
    validator = MockValidator()
    
    # Run complete pipeline test
    results = await validator.test_complete_pipeline()
    
    # Print summary
    validator.print_summary(results)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())



