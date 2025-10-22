#!/usr/bin/env python3
"""
Compare different Tier 2 pipeline methods
"""

import subprocess
import time
import json
import os
from datetime import datetime

def run_pipeline(method, use_previous=True):
    """Run a specific pipeline method and measure time"""
    print(f"\n{'='*80}")
    print(f"TESTING METHOD: {method.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    if method == "simple":
        cmd = ["python", "tier2_pipeline_simple.py", "--use-previous"] if use_previous else ["python", "tier2_pipeline_simple.py"]
    elif method == "chunked":
        cmd = ["python", "tier2_pipeline_chunked.py", "--use-previous"] if use_previous else ["python", "tier2_pipeline_chunked.py"]
    else:
        print(f"Unknown method: {method}")
        return None
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {method} completed in {elapsed:.1f}s")
            return {"method": method, "success": True, "time": elapsed, "output": result.stdout}
        else:
            print(f"❌ {method} failed after {elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            return {"method": method, "success": False, "time": elapsed, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ {method} timed out after {elapsed:.1f}s")
        return {"method": method, "success": False, "time": elapsed, "error": "Timeout"}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {method} error after {elapsed:.1f}s: {e}")
        return {"method": method, "success": False, "time": elapsed, "error": str(e)}

def main():
    print("🚀 TIER 2 PIPELINE COMPARISON")
    print("="*80)
    
    methods = ["simple", "chunked"]
    results = []
    
    for method in methods:
        result = run_pipeline(method, use_previous=True)
        if result:
            results.append(result)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<15} {'Success':<8} {'Time (s)':<10} {'Status'}")
    print("-" * 50)
    
    for result in results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{result['method']:<15} {'Yes' if result['success'] else 'No':<8} {result['time']:<10.1f} {status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Find fastest successful method
    successful = [r for r in results if r["success"]]
    if successful:
        fastest = min(successful, key=lambda x: x["time"])
        print(f"\n🏆 Fastest method: {fastest['method']} ({fastest['time']:.1f}s)")
    else:
        print("\n❌ No methods completed successfully")

if __name__ == "__main__":
    main()
