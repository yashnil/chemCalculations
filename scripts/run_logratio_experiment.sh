#!/bin/bash
# Run experiment with log-ratio loss on x160 dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================================="
echo "LOG-RATIO LOSS EXPERIMENT"
echo "==================================================================="
echo ""
echo "This will train a model using the fractional/log-ratio loss:"
echo "  L_log-ratio = |log_10(ŷ/y)|"
echo ""
echo "Using best model architecture (x160_new):"
echo "  - Latent dim: 192"
echo "  - Hidden layers: [512, 512, 512]"
echo "  - Activation: SiLU"
echo "  - Dataset: x160 (160,000 samples)"
echo ""
echo "==================================================================="

# Set paths
CONFIG_PATH="$PROJECT_ROOT/configs/x160_logratio_config.json"
RUN_DIR="$PROJECT_ROOT/src/runs_autoencoder_x160_logratio"
CSV_PATH="$PROJECT_ROOT/data/datasets/all_gas_fastchem_x160.csv"

# Check if dataset exists
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ Error: Dataset not found at $CSV_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config not found at $CONFIG_PATH"
    exit 1
fi

echo "📊 Dataset: $CSV_PATH"
echo "⚙️  Config: $CONFIG_PATH"
echo "📁 Run directory: $RUN_DIR"
echo ""

# Run training
echo "🚀 Starting training..."
python src/train_autoencoder.py \
    --config "$CONFIG_PATH" \
    --loss-type log_ratio \
    --run-dir "$RUN_DIR"

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  1. Run diagnostics: python src/diagnostics.py"
echo "  2. Compare results in plots/comparison_metrics.csv"
echo ""

