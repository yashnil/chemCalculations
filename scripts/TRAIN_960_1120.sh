#!/bin/bash
# TRAIN_960_1120.sh
# Train models for 960K and 1120K datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TRAINING MODELS FOR 960K AND 1120K"
echo "=================================================================================="
echo ""

# Check if datasets exist
if [ ! -f "data/datasets/all_gas_fastchem_x960.csv" ]; then
    echo "❌ Error: Dataset x960 not found!"
    exit 1
fi

if [ ! -f "data/datasets/all_gas_fastchem_x1120.csv" ]; then
    echo "❌ Error: Dataset x1120 not found!"
    exit 1
fi

echo "✅ Datasets found"
echo ""

# Check if configs exist, create if needed
if [ ! -f "configs/x960_optimal_retrained.json" ] || [ ! -f "configs/x1120_optimal_retrained.json" ]; then
    echo "Creating config files..."
    python scripts/train_960_1120_models.py <<< "n"  # Create configs but don't train
    echo ""
fi

# Train x960
echo "=================================================================================="
echo "Training x960K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x960_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x960_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x960K training complete"
else
    echo "❌ x960K training failed"
    exit 1
fi

echo ""

# Train x1120
echo "=================================================================================="
echo "Training x1120K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x1120_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x1120_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x1120K training complete"
else
    echo "❌ x1120K training failed"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=================================================================================="
echo ""
echo "Next steps:"
echo "  1. Update metrics: python scripts/update_and_regenerate_all.py"
echo "  2. Regenerate plots: python scripts/regenerate_all_plots_consistent.py"
echo "  3. Check plots/performance_vs_size.png to see if performance has asymptoted"
