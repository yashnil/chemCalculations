#!/bin/bash
# regenerate_pred_vs_true.sh
# Regenerate pred_vs_true_test.png using the best model (x800_optimal_retrained)

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "REGENERATING pred_vs_true_test.png WITH BEST MODEL"
echo "=================================================================================="
echo ""

# Set paths for best model
BEST_MODEL="results/runs/runs_autoencoder_x800_optimal_retrained/best_model.py"
DATASET="data/datasets/all_gas_fastchem_x800.csv"
OUTPUT="plots/pred_vs_true_test_filtered.png"

# Check if best model exists
if [ ! -f "$BEST_MODEL" ]; then
    echo "⚠️  Error: Best model not found at: $BEST_MODEL"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "⚠️  Error: Dataset not found at: $DATASET"
    exit 1
fi

echo "Using:"
echo "  Model: $BEST_MODEL"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT"
echo ""

# Set environment variables and run plot script
export BEST_MODULE="$BASE_DIR/$BEST_MODEL"
export CSV_PATH="$BASE_DIR/$DATASET"
export OUT_PNG="$BASE_DIR/$OUTPUT"

echo "Generating plot..."
python src/plot.py

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "✅ Successfully generated: $OUTPUT"
    ls -lh "$OUTPUT"
else
    echo ""
    echo "⚠️  Warning: Plot may not have been generated. Check for errors above."
fi

echo ""
echo "=================================================================================="
echo "COMPLETE"
echo "=================================================================================="
