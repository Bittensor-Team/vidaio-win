#!/usr/bin/env python3
"""
Test the scoring pipeline using FFmpeg upscaled videos vs original input videos
This validates the PIE-APP calculation fix before running expensive VSR processing
"""

import sys
import os
import json
sys.path.append('/workspace/vidaio-win')

from simplified_vidaio_evaluation import (
    calculate_pie_app_improved, 
    calculate_vidaio_score,
    create_ground_truth_videos
)

def test_scoring_pipeline():
    """Test scoring pipeline with FFmpeg vs original videos"""
    print("🧪 TESTING SCORING PIPELINE")
    print("="*60)
    print("Using FFmpeg upscaled videos vs original input videos")
    print("This validates PIE-APP calculation before VSR processing")
    print("="*60)
    
    # Step 1: Create ground truth videos if they don't exist
    print("\n🎬 STEP 1: Creating Ground Truth Videos")
    print("-" * 40)
    
    input_videos, ground_truth_videos = create_ground_truth_videos()
    
    # Step 2: Test scoring pipeline
    print("\n📊 STEP 2: Testing Scoring Pipeline")
    print("-" * 40)
    
    # Test cases: FFmpeg upscaled vs original input
    test_cases = [
        {
            'name': 'SD2HD (FFmpeg vs Original)',
            'reference': input_videos['480p'],  # Original 480p input
            'processed': ground_truth_videos['sd2hd'],  # FFmpeg 480p→1080p
            'expected_resolution': '1920x1080',
            'content_length': 10.0  # 10 seconds
        },
        {
            'name': 'HD24K (FFmpeg vs Original)', 
            'reference': input_videos['1080p'],  # Original 1080p input
            'processed': ground_truth_videos['hd24k'],  # FFmpeg 1080p→4K
            'expected_resolution': '3840x2160',
            'content_length': 10.0
        },
        {
            'name': '4K28K (FFmpeg vs Original)',
            'reference': input_videos['4k'],  # Original 4K input
            'processed': ground_truth_videos['4k28k'],  # FFmpeg 4K→8K
            'expected_resolution': '7680x4320',
            'content_length': 10.0
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"   Reference: {test_case['reference']}")
        print(f"   Processed: {test_case['processed']}")
        
        # Check if files exist
        if not os.path.exists(test_case['reference']):
            print(f"   ❌ Reference video not found: {test_case['reference']}")
            continue
            
        if not os.path.exists(test_case['processed']):
            print(f"   ❌ Processed video not found: {test_case['processed']}")
            continue
        
        try:
            # Test PIE-APP calculation
            print(f"   🧮 Calculating PIE-APP...")
            pie_app_score = calculate_pie_app_improved(
                test_case['reference'], 
                test_case['processed']
            )
            
            print(f"   📊 PIE-APP Score: {pie_app_score:.4f}")
            
            # Test full Vidaio scoring
            print(f"   🧮 Calculating Vidaio score...")
            vidaio_score = calculate_vidaio_score(
                test_case['reference'],
                test_case['processed'], 
                test_case['content_length']
            )
            
            print(f"   📊 Vidaio Final Score: {vidaio_score['final_score']:.4f}")
            print(f"   📊 VMAF Score: {vidaio_score['vmaf_score']:.1f}")
            print(f"   📊 Quality Score: {vidaio_score['quality_score']:.4f}")
            print(f"   📊 Length Score: {vidaio_score['length_score']:.4f}")
            
            # Validate PIE-APP fix
            if pie_app_score == 1.0:
                print(f"   ⚠️  PIE-APP still hardcoded to 1.0 - fix not working")
            elif pie_app_score == 0.0:
                print(f"   ❌ PIE-APP calculation failed - returned 0.0")
            else:
                print(f"   ✅ PIE-APP calculation working! Score: {pie_app_score:.4f}")
            
            results[test_case['name']] = {
                'pie_app_score': pie_app_score,
                'vidaio_score': vidaio_score,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[test_case['name']] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # Summary
    print(f"\n📋 SCORING PIPELINE TEST SUMMARY")
    print("="*60)
    
    successful_tests = sum(1 for r in results.values() if r['status'] == 'success')
    total_tests = len(results)
    
    print(f"Successful Tests: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        print(f"\n📊 PIE-APP Scores (should be different values, not 1.0):")
        for name, result in results.items():
            if result['status'] == 'success':
                pie_app = result['pie_app_score']
                vidaio = result['vidaio_score']['final_score']
                print(f"   {name:>25}: PIE-APP={pie_app:.4f}, Vidaio={vidaio:.4f}")
    
    # Save results
    with open('/workspace/vidaio-win/scoring_pipeline_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: /workspace/vidaio-win/scoring_pipeline_test_results.json")
    
    return results

if __name__ == "__main__":
    test_scoring_pipeline()





