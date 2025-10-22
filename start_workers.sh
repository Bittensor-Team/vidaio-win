#!/bin/bash

echo "🚀 Starting 5 Real-ESRGAN Worker Servers..."
echo ""

for i in {0..4}; do
    port=$((8090 + i))
    echo "[Worker $((i+1))/5] Starting on port $port..."
    python /workspace/vidaio-subnet/upscaler_worker_server.py --port $port > /tmp/worker_$port.log 2>&1 &
    sleep 2
done

echo ""
echo "✅ All 5 workers started!"
echo "   Ports: 8090, 8091, 8092, 8093, 8094"
echo ""
echo "Monitor logs with:"
echo "   tail -f /tmp/worker_*.log"
echo ""
echo "Kill all workers with:"
echo "   pkill -f upscaler_worker_server"
