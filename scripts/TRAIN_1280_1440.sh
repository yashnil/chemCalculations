#!/bin/bash
# TRAIN_1280_1440.sh
# Train models for 1280K and 1440K datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TRAINING MODELS FOR 1280K AND 1440K"
echo "=================================================================================="
echo ""

# Check if datasets exist
if [ ! -f "data/datasets/all_gas_fastchem_x1280.csv" ]; then
    echo "❌ Error: Dataset x1280 not found!"
    exit 1
fi

if [ ! -f "data/datasets/all_gas_fastchem_x1440.csv" ]; then
    echo "❌ Error: Dataset x1440 not found!"
    exit 1
fi

echo "✅ Datasets found"
echo ""

# Check if configs exist, create if needed
if [ ! -f "configs/x1280_optimal_retrained.json" ] || [ ! -f "configs/x1440_optimal_retrained.json" ]; then
    echo "Creating config files..."
    python scripts/train_1280_1440_models.py <<< "n"  # Create configs but don't train
    echo ""
fi

# Train x1280
echo "=================================================================================="
echo "Training x1280K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x1280_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x1280_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x1280K training complete"
else
    echo "❌ x1280K training failed"
    exit 1
fi

echo ""

# Train x1440
echo "=================================================================================="
echo "Training x1440K model..."
echo "=================================================================================="
python src/train_autoencoder.py \
    --config configs/x1440_optimal_retrained.json \
    --loss-type log_ratio \
    --run-dir results/runs/runs_autoencoder_x1440_optimal_retrained

if [ $? -eq 0 ]; then
    echo "✅ x1440K training complete"
else
    echo "❌ x1440K training failed"
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
