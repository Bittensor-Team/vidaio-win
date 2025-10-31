#!/bin/bash
# Vidaio Miner - Quick Start Script
# This script activates the conda environment and provides the next steps

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Vidaio Miner - Quick Start                          ║"
echo "║        Wallet: lucky_key | Hotkey: lucky_hotkey1           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vidaio 2>/dev/null || true

echo "✓ Conda environment activated: vidaio"
python --version
echo ""

cd /workspace/vidaio-subnet

echo "📋 Setup Checklist:"
echo ""
echo "1. Configure S3 Bucket Credentials:"
echo "   nano .env"
echo ""
echo "2. Install System Dependencies (one-time):"
echo "   sudo apt update"
echo "   sudo apt install -y npm redis-server ffmpeg jq"
echo "   sudo npm install -g pm2"
echo "   sudo systemctl start redis-server"
echo ""
echo "3. Install Video2X:"
echo "   mkdir -p services/upscaling/models"
echo "   wget -P services/upscaling/models https://github.com/k4yt3x/video2x/releases/download/6.3.1/video2x-linux-ubuntu2404-amd64.deb"
echo "   sudo dpkg -i services/upscaling/models/video2x-linux-ubuntu2404-amd64.deb"
echo ""
echo "4. Start Miner (all services):"
echo ""
echo "   pm2 start \"python services/upscaling/server.py\" --name video-upscaler"
echo "   pm2 start \"python services/compress/server.py\" --name video-compressor"
echo "   pm2 start \"python services/miner_utilities/file_deletion_server.py\" --name video-deleter"
echo "   pm2 start \"python3 neurons/miner.py --wallet.name lucky_key --wallet.hotkey lucky_hotkey1 --subtensor.network finney --netuid 85 --axon.port 8091 --logging.debug\" --name video-miner"
echo ""
echo "5. Monitor:"
echo "   pm2 logs video-miner"
echo ""
echo "📖 Full documentation available in:"
echo "   - SETUP_SUMMARY.md (overview)"
echo "   - MINER_SETUP.md (detailed guide)"
echo ""
