#!/bin/bash
# watch_training.sh
# Watch training progress in real-time

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Watching training progress (Ctrl+C to stop)..."
echo ""

while true; do
    clear
    echo "=================================================================================="
    echo "TRAINING PROGRESS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================================="
    echo ""
    
    python scripts/check_progress.py
    
    echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
