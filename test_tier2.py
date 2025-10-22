#!/usr/bin/env python3
"""
Simple test script for Tier 2 algorithms
"""

import os
import sys
import subprocess
from pathlib import Path

def test_upscaling():
    """Test Tier 2 upscaling"""
    print("🎬 Testing Tier 2 Upscaling...")
    
    # Check if we have the test video
    if not os.path.exists("sample_input.mp4"):
        print("❌ Test video not found")
        return False
    
    # Run upscaling
    try:
        result = subprocess.run([
            sys.executable, "tier2_upscaling.py", 
            "sample_input.mp4", "output_upscaled.mp4"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        if result.returncode == 0 and os.path.exists("output_upscaled.mp4"):
            print("✅ Upscaling successful!")
            return True
        else:
            print("❌ Upscaling failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Upscaling timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_compression():
    """Test Tier 2 compression"""
    print("\n🎬 Testing Tier 2 Compression...")
    
    # Check if we have the test video
    if not os.path.exists("sample_input.mp4"):
        print("❌ Test video not found")
        return False
    
    # Run compression
    try:
        result = subprocess.run([
            sys.executable, "tier2_compression.py", 
            "sample_input.mp4", "output_compressed.mp4"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        if result.returncode == 0 and os.path.exists("output_compressed.mp4"):
            print("✅ Compression successful!")
            return True
        else:
            print("❌ Compression failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Compression timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_validation():
    """Test validation with local_validation.py"""
    print("\n🔍 Testing Validation...")
    
    if not os.path.exists("output_upscaled.mp4") or not os.path.exists("sample_input.mp4"):
        print("❌ Output files not found")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "local_validation.py",
            "--reference", "sample_input.mp4",
            "--processed", "output_upscaled.mp4",
            "--task", "upscaling",
            "--verbose"
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        if result.returncode == 0:
            print("✅ Validation successful!")
            return True
        else:
            print("❌ Validation failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Validation timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("======================================================================")
    print("  TIER 2 ALGORITHM TESTING")
    print("======================================================================")
    
    # Test upscaling
    upscaling_success = test_upscaling()
    
    # Test compression
    compression_success = test_compression()
    
    # Test validation
    validation_success = test_validation()
    
    print("\n======================================================================")
    print("  RESULTS SUMMARY")
    print("======================================================================")
    print(f"Upscaling: {'✅ PASS' if upscaling_success else '❌ FAIL'}")
    print(f"Compression: {'✅ PASS' if compression_success else '❌ FAIL'}")
    print(f"Validation: {'✅ PASS' if validation_success else '❌ FAIL'}")
    
    if upscaling_success and compression_success and validation_success:
        print("\n🎉 ALL TESTS PASSED! Tier 2 algorithms are working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
