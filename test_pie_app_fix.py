#!/usr/bin/env python3
"""
Test the fixed PIE-APP calculation
"""

import sys
import os
sys.path.append('/workspace/vidaio-win')

from simplified_vidaio_evaluation import calculate_pie_app_improved

def test_pie_app_calculation():
    """Test PIE-APP calculation with sample videos"""
    print("🧪 Testing PIE-APP Calculation Fix")
    print("="*50)
    
    # Test with the existing ground truth videos if they exist
    test_cases = [
        {
            'name': 'SD2HD vs Ground Truth',
            'reference': '/tmp/ground_truth_sd2hd.mp4',
            'distorted': '/tmp/vsr_SD2HD_10w_fixed.mp4'
        },
        {
            'name': 'HD24K vs Ground Truth', 
            'reference': '/tmp/ground_truth_hd24k.mp4',
            'distorted': '/tmp/vsr_HD24K_12w.mp4'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        
        if not os.path.exists(test_case['reference']):
            print(f"   ❌ Reference video not found: {test_case['reference']}")
            continue
            
        if not os.path.exists(test_case['distorted']):
            print(f"   ❌ Distorted video not found: {test_case['distorted']}")
            continue
        
        try:
            pie_app_score = calculate_pie_app_improved(
                test_case['reference'], 
                test_case['distorted']
            )
            
            print(f"   ✅ PIE-APP Score: {pie_app_score:.4f}")
            
            if pie_app_score == 1.0:
                print(f"   ⚠️  Still hardcoded to 1.0 - fix not working")
            elif pie_app_score == 0.0:
                print(f"   ❌ Calculation failed - returned 0.0")
            else:
                print(f"   ✅ PIE-APP calculation working! Score: {pie_app_score:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Expected: PIE-APP scores should be different values (not 1.0 or 0.0)")
    print(f"   Good range: 0.3-0.8 (depending on quality)")

if __name__ == "__main__":
    test_pie_app_calculation()





