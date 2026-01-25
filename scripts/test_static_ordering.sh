#!/bin/bash
# Test static species ordering with different species counts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================================="
echo "TESTING STATIC SPECIES ORDERING"
echo "==================================================================="
echo ""
echo "This will train models with static species lists (24, 32, 36 species)"
echo "and compare performance against the dynamic top-20 baseline."
echo ""
echo "Using best model architecture (x160_logratio):"
echo "  - Latent dim: 192"
echo "  - Hidden layers: [512, 512, 512]"
echo "  - Activation: SiLU"
echo "  - Loss: Log-ratio"
echo "  - Dataset: x160 (160,000 samples)"
echo ""
echo "==================================================================="

# Test configurations
CONFIGS=(
    "configs/x160_static_24_config.json:results/runs/runs_autoencoder_x160_static_24"
    "configs/x160_static_32_config.json:results/runs/runs_autoencoder_x160_static_32"
    "configs/x160_static_36_config.json:results/runs/runs_autoencoder_x160_static_36"
)

CSV_PATH="$PROJECT_ROOT/data/datasets/all_gas_fastchem_x160.csv"

# Check if dataset exists
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ Error: Dataset not found at $CSV_PATH"
    exit 1
fi

echo "📊 Dataset: $CSV_PATH"
echo ""

# Run training for each config
for config_pair in "${CONFIGS[@]}"; do
    IFS=':' read -r config_path run_dir <<< "$config_pair"
    
    if [ ! -f "$config_path" ]; then
        echo "⚠️  Warning: Config not found at $config_path, skipping..."
        continue
    fi
    
    echo ""
    echo "==================================================================="
    echo "Training with: $(basename $config_path)"
    echo "Run directory: $run_dir"
    echo "==================================================================="
    
    python src/train_autoencoder.py \
        --config "$config_path" \
        --loss-type log_ratio \
        --run-dir "$run_dir"
    
    echo ""
    echo "✅ Training complete for $(basename $config_path)"
done

echo ""
echo "==================================================================="
echo "✅ All training runs complete!"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "  1. Run diagnostics for each model"
echo "  2. Compare results in plots/comparison_metrics.csv"
echo ""

