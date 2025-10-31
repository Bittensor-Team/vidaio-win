#!/usr/bin/env python3
"""
Start multiple VSR workers on different ports
"""

import subprocess
import time
import sys
import signal
import os

def start_vsr_workers(num_workers=6, base_port=8100, model="DXM-FP32"):
    """Start multiple VSR workers"""
    processes = []
    
    print(f"🚀 Starting {num_workers} VSR workers on ports {base_port}-{base_port + num_workers - 1}")
    print(f"🤖 Model: {model}")
    
    try:
        for i in range(num_workers):
            port = base_port + i
            print(f"   Starting worker {i+1} on port {port}...")
            
            # Start worker process
            proc = subprocess.Popen([
                'python', '/workspace/vidaio-win/vsr_worker_server.py',
                '--port', str(port),
                '--model', model
            ])
            
            processes.append((i+1, port, proc))
            time.sleep(2)  # Give each worker time to start
        
        print(f"\n✅ All {num_workers} workers started!")
        print("Press Ctrl+C to stop all workers")
        
        # Wait for interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all workers...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up all processes
        for worker_id, port, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"   ✅ Worker {worker_id} (port {port}) stopped")
            except:
                try:
                    proc.kill()
                    print(f"   🔥 Worker {worker_id} (port {port}) force killed")
                except:
                    print(f"   ❌ Failed to stop worker {worker_id} (port {port})")
        
        print("🏁 All workers stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start multiple VSR workers")
    parser.add_argument("--workers", type=int, default=6, help="Number of workers to start")
    parser.add_argument("--base-port", type=int, default=8100, help="Base port number")
    parser.add_argument("--model", type=str, default="DXM-FP32", 
                       choices=["DXM-FP32", "DXM-FP16", "E-FP32", "E-FP16", "FP32", "FP16"],
                       help="VSR model to use")
    
    args = parser.parse_args()
    
    start_vsr_workers(args.workers, args.base_port, args.model)





