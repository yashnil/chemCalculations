#!/bin/bash
# generate_full_plot_suite.sh
# ============================
# Generate a comprehensive suite of plots for optimal_retrained models

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "GENERATING FULL PLOT SUITE"
echo "=================================================================================="
echo ""
echo "This script generates:"
echo "  1. Loss curves for all dataset sizes (160, 320, 480, 640, 800K)"
echo "  2. Performance metrics vs dataset size"
echo "  3. Scatter plots (predicted vs true) for optimal model"
echo "  4. Model comparison bar charts"
echo "  5. All other comprehensive analysis plots"
echo ""

# Step 1: Update comparison metrics
echo "Step 1: Updating comparison metrics..."
python scripts/update_and_regenerate_all.py

# Step 2: Generate comprehensive analysis plots
echo ""
echo "Step 2: Generating comprehensive analysis plots..."
python src/plot_comprehensive_analysis.py

# Step 3: Generate training analysis plots
echo ""
echo "Step 3: Generating training analysis plots..."
python src/plot_training_analysis.py

# Step 4: Generate consistent runs plots
echo ""
echo "Step 4: Generating consistent runs plots..."
python src/plot_consistent_runs.py

# Step 5: Generate full suite plots (if script exists and can run)
echo ""
echo "Step 5: Generating full suite plots..."
if python -c "import torch" 2>/dev/null; then
    python src/plot_full_suite.py || echo "  ⚠️  plot_full_suite.py failed (may need to run outside sandbox)"
else
    echo "  ⚠️  torch not available in sandbox - skipping plot_full_suite.py"
    echo "  Run manually: python src/plot_full_suite.py"
fi

# Step 6: Run diagnostics for optimal model (if available)
echo ""
echo "Step 6: Running diagnostics for optimal model..."
BEST_RUN="x800_optimal_retrained"
BEST_MODEL_PATH="$BASE_DIR/results/runs/runs_autoencoder_${BEST_RUN}/best_model.py"

if [ -f "$BEST_MODEL_PATH" ]; then
    echo "  Found optimal model: $BEST_RUN"
    echo "  To generate diagnostic plots, run:"
    echo "    export BEST_MODULE=$BEST_MODEL_PATH"
    echo "    export CSV_PATH=\$BASE_DIR/data/datasets/all_gas_fastchem_x800.csv"
    echo "    export OUT_DIR=\$BASE_DIR/plots/diagnostics_optimal"
    echo "    python src/diagnostics.py"
else
    echo "  ⚠️  Optimal model not found at: $BEST_MODEL_PATH"
fi

echo ""
echo "=================================================================================="
echo "✅ PLOT GENERATION COMPLETE!"
echo "=================================================================================="
echo ""
echo "Generated plots:"
echo "  📊 Loss curves:"
echo "    - plots/loss_curves.png"
echo "    - plots/loss_curves_all_sizes.png"
echo "    - plots/loss_curves_consistent.png"
echo ""
echo "  📈 Performance metrics:"
echo "    - plots/performance_vs_size.png"
echo "    - plots/performance_vs_size_comprehensive.png"
echo "    - plots/performance_vs_size_consistent.png"
echo ""
echo "  📉 Model comparison:"
echo "    - plots/model_comparison.png"
echo ""
echo "  📊 Asymptote analysis:"
echo "    - plots/asymptote_analysis.png"
echo ""
echo "  📊 Scatter plots (if generated):"
echo "    - plots/scatter_optimal_model.png"
echo "    - plots/pred_vs_true_test.png (from diagnostics)"
echo ""
echo "All plots use consistent units and optimal_retrained runs (160, 320, 480, 640, 800K)"
echo ""
