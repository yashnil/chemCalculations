#!/bin/bash
# Master script to retrain models and regenerate all plots for 160, 320, 480, 640, 800

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "MASTER SCRIPT: RETRAIN MODELS AND REGENERATE ALL PLOTS"
echo "=================================================================================="
echo "Dataset sizes: 160, 320, 480, 640, 800K"
echo "Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32"
echo ""

# Step 1: Check datasets
echo "Step 1: Checking datasets..."
MISSING=""
for size in 160 320 480 640 800; do
    if [ -f "data/datasets/all_gas_fastchem_x${size}.csv" ]; then
        echo "  ✓ x${size}K: EXISTS"
    else
        echo "  ✗ x${size}K: MISSING"
        MISSING="$MISSING $size"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "⚠️  Missing datasets:$MISSING"
    echo "   These need to be generated first using:"
    echo "   python scripts/generate_large_datasets.py"
    echo "   Then run FastChem and merge results."
    echo ""
    read -p "Continue with available datasets only? (y/n): " response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Step 2: Train models for available datasets
echo ""
echo "Step 2: Training models..."
for size in 160 320 480 640 800; do
    if [ ! -f "data/datasets/all_gas_fastchem_x${size}.csv" ]; then
        echo "  ⏭️  Skipping x${size}K - dataset not found"
        continue
    fi
    
    if [ -d "results/runs/runs_autoencoder_x${size}_optimal_retrained" ]; then
        echo "  ⏭️  Skipping x${size}K - already trained"
        continue
    fi
    
    echo "  Training x${size}K..."
    python src/train_autoencoder.py \
        --config "configs/x${size}_optimal_retrained.json" \
        --loss-type log_ratio \
        --run-dir "results/runs/runs_autoencoder_x${size}_optimal_retrained" \
        > "training_x${size}K.log" 2>&1 &
    
    echo "    → Started in background (PID: $!)"
done

echo ""
echo "Waiting for training to complete..."
wait

# Step 3: Update comparison metrics and regenerate plots
echo ""
echo "Step 3: Updating comparison metrics and regenerating plots..."
python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ COMPLETE!"
echo "=================================================================================="
echo ""
echo "All plots regenerated with optimal_retrained runs:"
echo "  - plots/comparison_metrics.csv"
echo "  - plots/performance_vs_size.png"
echo "  - plots/loss_curves.png"
echo "  - plots/asymptote_analysis.png"
echo "  - plots/model_comparison.png"
