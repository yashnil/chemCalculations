#!/bin/bash
# TRAIN_2400_2720_3040.sh
# Train models for 2400K, 2720K, and 3040K datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TRAINING MODELS FOR 2400K, 2720K, AND 3040K"
echo "=================================================================================="
echo ""

# Check if datasets exist
for size in 2400 2720 3040; do
    if [ ! -f "data/datasets/all_gas_fastchem_x${size}.csv" ]; then
        echo "❌ Error: Dataset x${size} not found!"
        exit 1
    fi
done

echo "✅ All datasets found"
echo ""

# Check if configs exist, create if needed
if [ ! -f "configs/x2400_optimal_retrained.json" ] || \
   [ ! -f "configs/x2720_optimal_retrained.json" ] || \
   [ ! -f "configs/x3040_optimal_retrained.json" ]; then
    echo "Creating config files..."
    python scripts/train_2400_2720_3040_models.py <<< "n"  # Create configs but don't train
    echo ""
fi

# Train each model
for size in 2400 2720 3040; do
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
