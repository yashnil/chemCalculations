#!/bin/bash
# TRAIN_1760_2080.sh
# Train models for 1760K and 2080K datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TRAINING MODELS FOR 1760K AND 2080K"
echo "=================================================================================="
echo ""

# Check if datasets exist
if [ ! -f "data/datasets/all_gas_fastchem_x1760.csv" ]; then
    echo "❌ Error: Dataset x1760 not found!"
    exit 1
fi

if [ ! -f "data/datasets/all_gas_fastchem_x2080.csv" ]; then
    echo "❌ Error: Dataset x2080 not found!"
    exit 1
fi

echo "✅ Datasets found"
echo ""

# Check if configs exist, create if needed
if [ ! -f "configs/x1760_optimal_retrained.json" ] || [ ! -f "configs/x2080_optimal_retrained.json" ]; then
    echo "Creating config files..."
    python scripts/train_1760_2080_models.py <<< "n"  # Create configs but don't train
    echo ""
fi

# Train x1760
echo "=================================================================================="
echo "Training x1760K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x1760_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x1760_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x1760K training complete"
else
    echo "❌ x1760K training failed"
    exit 1
fi

echo ""

# Train x2080
echo "=================================================================================="
echo "Training x2080K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x2080_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x2080_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x2080K training complete"
else
    echo "❌ x2080K training failed"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=================================================================================="
echo ""
echo "Next steps:"
echo "  1. Update metrics: python scripts/update_plots_for_optimal_retrained.py"
echo "  2. Regenerate plots: python scripts/regenerate_all_plots_consistent.py"
echo "  3. Check plots/performance_vs_size.png to see if performance has asymptoted"
