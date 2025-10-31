#!/usr/bin/env python3
"""
Complete Vidaio VSR Pipeline Test
Tests the entire pipeline: Validator → Miner → VSR Server → R2
"""

import asyncio
import subprocess
import time
import os
import signal
import sys
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class PipelineTester:
    def __init__(self):
        self.processes = []
        self.vsr_server_port = 29115
        
    def start_vsr_server(self):
        """Start the enhanced VSR server"""
        logger.info("🚀 Starting VSR Server...")
        
        cmd = [
            "python", "vsr_worker_server_enhanced.py",
            "--port", str(self.vsr_server_port),
            "--model", "DXM-FP32",
            "--output-folder", "/workspace/vidaio-win/test_output"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("VSR Server", process))
            
            # Wait a bit for server to start
            time.sleep(5)
            
            # Check if server is running
            if process.poll() is None:
                logger.info("✅ VSR Server started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ VSR Server failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start VSR Server: {e}")
            return False
    
    def test_r2_access(self):
        """Test R2 access before running pipeline"""
        logger.info("🔍 Testing R2 Access...")
        
        try:
            result = subprocess.run(
                ["python", "test_r2_access.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ R2 access test passed")
                return True
            else:
                logger.error(f"❌ R2 access test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ R2 test error: {e}")
            return False
    
    def run_pipeline_test(self):
        """Run the complete pipeline test"""
        logger.info("🧪 Running Complete Pipeline Test...")
        
        try:
            result = subprocess.run(
                ["python", "mock_validator.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Pipeline test completed successfully")
                logger.info("📊 Test Output:")
                print(result.stdout)
                return True
            else:
                logger.error(f"❌ Pipeline test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Pipeline test timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline test error: {e}")
            return False
    
    def cleanup(self):
        """Clean up running processes"""
        logger.info("🧹 Cleaning up processes...")
        
        for name, process in self.processes:
            if process.poll() is None:
                logger.info(f"🛑 Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logger.info(f"✅ {name} stopped")
    
    def run_complete_test(self):
        """Run the complete test suite"""
        logger.info("🚀 Starting Complete Vidaio VSR Pipeline Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Test R2 Access
            logger.info("📋 Step 1: Testing R2 Access")
            if not self.test_r2_access():
                logger.error("❌ R2 access test failed. Cannot proceed.")
                return False
            
            # Step 2: Start VSR Server
            logger.info("📋 Step 2: Starting VSR Server")
            if not self.start_vsr_server():
                logger.error("❌ Failed to start VSR Server. Cannot proceed.")
                return False
            
            # Step 3: Run Pipeline Test
            logger.info("📋 Step 3: Running Pipeline Test")
            if not self.run_pipeline_test():
                logger.error("❌ Pipeline test failed.")
                return False
            
            logger.info("🎉 ALL TESTS PASSED! Pipeline is ready for production!")
            return True
            
        except KeyboardInterrupt:
            logger.info("⏹️  Test interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            return False
        finally:
            self.cleanup()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n⏹️  Received interrupt signal. Cleaning up...")
    sys.exit(0)

def main():
    """Main function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if we're in the right directory
    if not os.path.exists("vsr_worker_server_enhanced.py"):
        logger.error("❌ Please run this script from the /workspace/vidaio-win directory")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        "vsr_worker_server_enhanced.py",
        "mock_validator.py", 
        "test_r2_access.py",
        "neurons/miner_vsr.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    # Check environment variables
    if not os.getenv("R2_ACCESS_KEY") or not os.getenv("R2_SECRET_KEY"):
        logger.warning("⚠️  R2_ACCESS_KEY and R2_SECRET_KEY not set. R2 tests will fail.")
        logger.info("   Set them with: export R2_ACCESS_KEY='your_key' && export R2_SECRET_KEY='your_secret'")
    
    # Run the test
    tester = PipelineTester()
    success = tester.run_complete_test()
    
    if success:
        logger.info("🎉 Pipeline test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()



