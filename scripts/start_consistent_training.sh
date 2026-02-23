#!/bin/bash
# Start training all models with consistent architecture

set -e

echo "=================================================================================="
echo "STARTING CONSISTENT ARCHITECTURE TRAINING"
echo "=================================================================================="
echo "Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32"
echo "Dataset sizes: 160K, 240K, 480K, 640K"
echo ""

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Train each model in background
for size in 160 240 480 640; do
    echo "Starting training for x${size}K..."
    python src/train_autoencoder.py \
        --config "configs/x${size}_static_32_consistent.json" \
        --loss-type log_ratio \
        --run-dir "results/runs/runs_autoencoder_x${size}_static_32_consistent" \
        > "training_x${size}K.log" 2>&1 &
    
    echo "  → Training started in background (PID: $!)"
    echo "  → Log: training_x${size}K.log"
    echo ""
done

echo "=================================================================================="
echo "All training jobs started!"
echo "Monitor progress with: tail -f training_x*K.log"
echo "Check status with: ps aux | grep train_autoencoder"
echo "=================================================================================="
