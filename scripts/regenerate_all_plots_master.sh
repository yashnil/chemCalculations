#!/bin/bash
# Master script to regenerate ALL plots with consistent architecture

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "REGENERATING ALL PLOTS WITH CONSISTENT ARCHITECTURE"
echo "=================================================================================="
echo ""

# Step 1: Update comparison metrics
echo "Step 1: Updating comparison metrics..."
python scripts/update_and_regenerate_all.py

# Step 2: Generate all plot types
echo ""
echo "Step 2: Generating comprehensive plots..."
python src/plot_comprehensive_analysis.py

echo ""
echo "Step 3: Generating training analysis plots..."
python src/plot_training_analysis.py

echo ""
echo "Step 4: Generating consistent runs plots..."
python src/plot_consistent_runs.py

echo ""
echo "=================================================================================="
echo "✅ ALL PLOTS REGENERATED!"
echo "=================================================================================="
echo ""
echo "Key plots generated:"
echo "  - plots/loss_curves.png (default - uses consistent runs)"
echo "  - plots/loss_curves_consistent.png"
echo "  - plots/loss_curves_all_sizes.png"
echo "  - plots/performance_vs_size.png (uses consistent runs)"
echo "  - plots/performance_vs_size_consistent.png"
echo "  - plots/performance_vs_size_comprehensive.png"
echo "  - plots/asymptote_analysis.png"
echo "  - plots/model_comparison.png"
echo ""
echo "Best model: x640_consistent (640K samples)"
echo "  Test Loss: 0.030339"
echo "  Val Loss: 0.028549"
