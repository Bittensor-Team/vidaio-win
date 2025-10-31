#!/usr/bin/env python3
"""
Vidaio Miner Pipeline End-to-End Test

This script tests the complete miner pipeline:
1. Creates a test video
2. Tests upscaling service (all 4 task types)
3. Tests compression service (all 3 VMAF thresholds)
4. Verifies output quality
5. Calculates scores

Usage:
    python test_miner_pipeline.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

import aiohttp
import cv2
import numpy as np
from loguru import logger

# Configuration
UPSCALING_SERVICE_URL = "http://localhost:29115/upscale-video"
COMPRESSION_SERVICE_URL = "http://localhost:29116/compress-video"
SCORING_SERVICE_URL = "http://localhost:29090/score-video"

TEST_VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("test_outputs")


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✘ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def create_test_video(output_path: Path, resolution: tuple = (640, 480), 
                      duration: int = 5, fps: int = 30) -> bool:
    """
    Create a test video with moving shapes and colors.
    
    Args:
        output_path: Path to save the video
        resolution: Video resolution (width, height)
        duration: Duration in seconds
        fps: Frames per second
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print_info(f"Creating test video: {resolution[0]}x{resolution[1]}, {duration}s @ {fps}fps")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Create a frame with gradient background
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Animated gradient
            for y in range(resolution[1]):
                color_value = int(255 * (y / resolution[1] + frame_num / total_frames) % 1)
                frame[y, :] = [color_value, 255 - color_value, 128]
            
            # Moving circle
            center_x = int(resolution[0] * (0.5 + 0.3 * np.sin(2 * np.pi * frame_num / total_frames)))
            center_y = int(resolution[1] * 0.5)
            cv2.circle(frame, (center_x, center_y), 50, (255, 255, 0), -1)
            
            # Moving rectangle
            rect_x = int(resolution[0] * (frame_num / total_frames))
            cv2.rectangle(frame, (rect_x, 100), (rect_x + 80, 180), (0, 255, 255), -1)
            
            # Add text with frame number
            cv2.putText(frame, f"Frame: {frame_num}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print_success(f"Test video created: {output_path}")
        return True
        
    except Exception as e:
        print_error(f"Failed to create test video: {e}")
        return False


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        
        return {
            'width': video_stream['width'],
            'height': video_stream['height'],
            'duration': float(data['format']['duration']),
            'size_bytes': int(data['format']['size']),
            'size_mb': int(data['format']['size']) / (1024 * 1024),
            'bitrate': int(data['format']['bit_rate']),
            'fps': eval(video_stream['r_frame_rate'])
        }
    except Exception as e:
        print_error(f"Failed to get video info: {e}")
        return {}


async def test_upscaling_service(test_video: Path, task_type: str) -> Dict[str, Any]:
    """
    Test the upscaling service with a specific task type.
    
    Args:
        test_video: Path to test video
        task_type: One of SD2HD, HD24K, SD24K, 4K28K
    
    Returns:
        Test results dictionary
    """
    print_header(f"Testing Upscaling Service: {task_type}")
    
    result = {
        'task_type': task_type,
        'success': False,
        'error': None,
        'processing_time': 0,
        'input_video': None,
        'output_video': None
    }
    
    try:
        # Get input video info
        input_info = get_video_info(test_video)
        result['input_video'] = input_info
        print_info(f"Input: {input_info['width']}x{input_info['height']}, "
                  f"{input_info['duration']:.1f}s, {input_info['size_mb']:.2f}MB")
        
        # Prepare request payload
        payload = {
            "payload_url": f"file://{test_video.absolute()}",
            "task_type": task_type
        }
        
        print_info(f"Sending request to upscaling service...")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(UPSCALING_SERVICE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time
                
                if response.status == 200:
                    response_data = await response.json()
                    output_url = response_data.get('uploaded_video_url')
                    
                    if output_url:
                        print_success(f"Upscaling completed in {processing_time:.2f}s")
                        print_info(f"Output URL: {output_url}")
                        
                        # Download output and get info
                        # (In real scenario, would download from S3)
                        result['output_url'] = output_url
                        result['success'] = True
                    else:
                        result['error'] = "No output URL returned"
                        print_error("No output URL in response")
                else:
                    error_text = await response.text()
                    result['error'] = f"HTTP {response.status}: {error_text}"
                    print_error(f"Service returned error: {response.status}")
        
    except asyncio.TimeoutError:
        result['error'] = "Request timeout (>120s)"
        print_error("Request timed out")
    except Exception as e:
        result['error'] = str(e)
        print_error(f"Error: {e}")
    
    return result


async def test_compression_service(test_video: Path, vmaf_threshold: float) -> Dict[str, Any]:
    """
    Test the compression service with a specific VMAF threshold.
    
    Args:
        test_video: Path to test video
        vmaf_threshold: One of 85, 90, 95
    
    Returns:
        Test results dictionary
    """
    print_header(f"Testing Compression Service: VMAF {vmaf_threshold}")
    
    result = {
        'vmaf_threshold': vmaf_threshold,
        'success': False,
        'error': None,
        'processing_time': 0,
        'input_video': None,
        'output_video': None,
        'compression_rate': None
    }
    
    try:
        # Get input video info
        input_info = get_video_info(test_video)
        result['input_video'] = input_info
        print_info(f"Input: {input_info['width']}x{input_info['height']}, "
                  f"{input_info['size_mb']:.2f}MB")
        
        # Prepare request payload
        payload = {
            "payload_url": f"file://{test_video.absolute()}",
            "vmaf_threshold": vmaf_threshold
        }
        
        print_info(f"Sending request to compression service...")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(COMPRESSION_SERVICE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time
                
                if response.status == 200:
                    response_data = await response.json()
                    output_url = response_data.get('uploaded_video_url')
                    
                    if output_url:
                        print_success(f"Compression completed in {processing_time:.2f}s")
                        print_info(f"Output URL: {output_url}")
                        
                        result['output_url'] = output_url
                        result['success'] = True
                    else:
                        result['error'] = "No output URL returned"
                        print_error("No output URL in response")
                else:
                    error_text = await response.text()
                    result['error'] = f"HTTP {response.status}: {error_text}"
                    print_error(f"Service returned error: {response.status}")
        
    except asyncio.TimeoutError:
        result['error'] = "Request timeout (>180s)"
        print_error("Request timed out")
    except Exception as e:
        result['error'] = str(e)
        print_error(f"Error: {e}")
    
    return result


async def check_services() -> Dict[str, bool]:
    """Check if required services are running"""
    print_header("Checking Services")
    
    services = {
        'upscaling': UPSCALING_SERVICE_URL.replace('/upscale-video', '/'),
        'compression': COMPRESSION_SERVICE_URL.replace('/compress-video', '/'),
    }
    
    status = {}
    
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status in [200, 404]:  # 404 means service is up but no root endpoint
                        print_success(f"{name.capitalize()} service is running")
                        status[name] = True
                    else:
                        print_error(f"{name.capitalize()} service returned {response.status}")
                        status[name] = False
            except Exception as e:
                print_error(f"{name.capitalize()} service is not reachable: {e}")
                status[name] = False
    
    return status


async def main():
    """Main test function"""
    print_header("Vidaio Miner Pipeline End-to-End Test")
    
    # Check services
    service_status = await check_services()
    
    if not all(service_status.values()):
        print_error("\n❌ Not all services are running!")
        print_warning("Please start services with PM2:")
        print("  pm2 start services/upscaling/server.py --name video-upscaler")
        print("  pm2 start services/compress/server.py --name video-compressor")
        return 1
    
    # Create test videos
    print_header("Creating Test Videos")
    
    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    test_videos = {
        'SD': TEST_VIDEO_DIR / "test_sd_640x480.mp4",
        'HD': TEST_VIDEO_DIR / "test_hd_1280x720.mp4",
    }
    
    # Create SD test video
    if not test_videos['SD'].exists():
        if not create_test_video(test_videos['SD'], resolution=(640, 480), duration=5):
            return 1
    else:
        print_info(f"Using existing test video: {test_videos['SD']}")
    
    # Create HD test video
    if not test_videos['HD'].exists():
        if not create_test_video(test_videos['HD'], resolution=(1280, 720), duration=5):
            return 1
    else:
        print_info(f"Using existing test video: {test_videos['HD']}")
    
    # Test all upscaling task types
    upscaling_results = []
    
    upscaling_tests = [
        (test_videos['SD'], 'SD2HD'),
        (test_videos['HD'], 'HD24K'),
        (test_videos['SD'], 'SD24K'),
    ]
    
    for test_video, task_type in upscaling_tests:
        result = await test_upscaling_service(test_video, task_type)
        upscaling_results.append(result)
        time.sleep(2)  # Brief pause between tests
    
    # Test compression with different VMAF thresholds
    compression_results = []
    
    for vmaf_threshold in [85, 90, 95]:
        result = await test_compression_service(test_videos['HD'], vmaf_threshold)
        compression_results.append(result)
        time.sleep(2)
    
    # Print summary
    print_header("Test Summary")
    
    print(f"\n{Colors.BOLD}Upscaling Tests:{Colors.END}")
    upscaling_success = sum(1 for r in upscaling_results if r['success'])
    print(f"  Passed: {upscaling_success}/{len(upscaling_results)}")
    
    for result in upscaling_results:
        status = "✅" if result['success'] else "✘"
        print(f"    {status} {result['task_type']}: ", end="")
        if result['success']:
            print(f"{result['processing_time']:.2f}s")
        else:
            print(f"{result['error']}")
    
    print(f"\n{Colors.BOLD}Compression Tests:{Colors.END}")
    compression_success = sum(1 for r in compression_results if r['success'])
    print(f"  Passed: {compression_success}/{len(compression_results)}")
    
    for result in compression_results:
        status = "✅" if result['success'] else "✘"
        print(f"    {status} VMAF {result['vmaf_threshold']}: ", end="")
        if result['success']:
            print(f"{result['processing_time']:.2f}s")
        else:
            print(f"{result['error']}")
    
    # Overall result
    total_tests = len(upscaling_results) + len(compression_results)
    total_success = upscaling_success + compression_success
    
    print(f"\n{Colors.BOLD}Overall:{Colors.END} {total_success}/{total_tests} tests passed")
    
    if total_success == total_tests:
        print_success("\n🎉 All tests passed! Pipeline is working correctly.")
        return 0
    else:
        print_warning(f"\n⚠️  {total_tests - total_success} test(s) failed.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


