#!/bin/bash
# TRAIN_3360_3680_4000.sh
# Train models for 3360K, 3680K, and 4000K datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TRAINING MODELS FOR 3360K, 3680K, AND 4000K"
echo "=================================================================================="
echo ""

# Check if datasets exist
for size in 3360 3680 4000; do
    if [ ! -f "data/datasets/all_gas_fastchem_x${size}.csv" ]; then
        echo "❌ Error: Dataset x${size} not found!"
        exit 1
    fi
done

echo "✅ All datasets found"
echo ""

# Check if configs exist, create if needed
if [ ! -f "configs/x3360_optimal_retrained.json" ] || \
   [ ! -f "configs/x3680_optimal_retrained.json" ] || \
   [ ! -f "configs/x4000_optimal_retrained.json" ]; then
    echo "Creating config files..."
    python scripts/train_3360_3680_4000_models.py <<< "n"  # Create configs but don't train
    echo ""
fi

# Train each model
for size in 3360 3680 4000; do
    echo "=================================================================================="
    echo "Training x${size}K model..."
    echo "=================================================================================="
    python src/train_autoencoder.py \
        --config configs/x${size}_optimal_retrained.json \
        --loss-type log_ratio \
        --run-dir results/runs/runs_autoencoder_x${size}_optimal_retrained

    if [ $? -eq 0 ]; then
        echo "✅ x${size}K training complete"
    else
        echo "❌ x${size}K training failed"
        exit 1
    fi
    
    echo ""
done

echo "=================================================================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=================================================================================="
echo ""
echo "Next steps:"
echo "  1. Update metrics: python scripts/update_plots_for_optimal_retrained.py"
echo "  2. Regenerate plots: python scripts/regenerate_all_plots_consistent.py"
echo "  3. Check plots/performance_vs_size.png to see if performance has asymptoted"
