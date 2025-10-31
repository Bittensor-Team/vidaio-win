#!/usr/bin/env python3
"""
Test VSR workers health and functionality
"""

import requests
import time

def test_worker(port):
    """Test a single VSR worker"""
    url = f"http://127.0.0.1:{port}"
    
    try:
        # Health check
        resp = requests.get(f"{url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Worker {port}: {data.get('model_name', 'unknown')} - {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Worker {port}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Worker {port}: {e}")
        return False

def test_all_workers(base_port=8100, num_workers=6):
    """Test all VSR workers"""
    print(f"🔍 Testing {num_workers} VSR workers on ports {base_port}-{base_port + num_workers - 1}")
    print("="*60)
    
    healthy_workers = []
    
    for i in range(num_workers):
        port = base_port + i
        if test_worker(port):
            healthy_workers.append(port)
    
    print(f"\n📊 Results: {len(healthy_workers)}/{num_workers} workers healthy")
    
    if healthy_workers:
        print(f"✅ Healthy workers: {healthy_workers}")
    else:
        print("❌ No healthy workers found")
    
    return healthy_workers

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VSR workers")
    parser.add_argument("--workers", type=int, default=6, help="Number of workers to test")
    parser.add_argument("--base-port", type=int, default=8100, help="Base port number")
    
    args = parser.parse_args()
    
    test_all_workers(args.base_port, args.workers)





