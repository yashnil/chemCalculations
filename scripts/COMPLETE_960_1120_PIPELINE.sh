#!/bin/bash
# COMPLETE_960_1120_PIPELINE.sh
# Complete pipeline: Train models, update metrics, regenerate plots

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETE PIPELINE: 960K AND 1120K MODELS"
echo "=================================================================================="
echo ""

# Step 1: Train models
echo "=================================================================================="
echo "STEP 1: TRAINING MODELS"
echo "=================================================================================="
bash scripts/TRAIN_960_1120.sh

# Step 2: Update metrics and regenerate plots
echo ""
echo "=================================================================================="
echo "STEP 2: UPDATING METRICS AND REGENERATING PLOTS"
echo "=================================================================================="
python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "=================================================================================="
echo ""
echo "All models trained and plots updated for:"
echo "  - 160K, 320K, 480K, 640K, 800K, 960K, 1120K"
echo ""
echo "Check plots/performance_vs_size.png to see if performance has asymptoted."
