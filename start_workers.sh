#!/bin/bash

echo "🚀 Starting 6 Real-ESRGAN Worker Servers..."
echo ""

for i in {0..5}; do
    port=$((8090 + i))
    echo "[Worker $((i+1))/6] Starting on port $port..."
    /workspace/miniconda/envs/vidaio/bin/python /workspace/vidaio-win/upscaler_worker_server.py --port $port > /tmp/worker_$port.log 2>&1 &
    sleep 2
done

echo ""
echo "✅ All 6 workers started!"
echo "   Ports: 8090, 8091, 8092, 8093, 8094, 8095"
echo ""
echo "Monitor logs with:"
echo "   tail -f /tmp/worker_*.log"
echo ""
echo "Kill all workers with:"
echo "   pkill -f upscaler_worker_server"
