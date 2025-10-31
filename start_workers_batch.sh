#!/bin/bash

echo "🚀 Starting 6 Real-ESRGAN Batch Processing Workers..."
echo ""

for i in {0..5}; do
    port=$((8090 + i))
    echo "[Worker $((i+1))/6] Starting batch worker on port $port..."
    WORKER_ID=$i /workspace/miniconda/envs/vidaio/bin/python3 /workspace/vidaio-win/upscaler_worker_batch.py --port $port --worker-id $i > /tmp/batch_worker_$port.log 2>&1 &
    sleep 2
done

echo ""
echo "✅ All 6 batch workers started!"
echo "   Ports: 8090, 8091, 8092, 8093, 8094, 8095"
echo "   Logs: /tmp/batch_worker_*.log"
echo ""
echo "🔍 Check worker health:"
echo "   curl http://127.0.0.1:8090/health"
echo ""
echo "📊 Check worker status:"
echo "   curl http://127.0.0.1:8090/status"
echo ""
echo "🛑 Stop workers:"
echo "   pkill -f upscaler_worker_batch.py"
