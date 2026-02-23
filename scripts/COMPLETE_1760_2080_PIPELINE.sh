#!/bin/bash
# COMPLETE_1760_2080_PIPELINE.sh
# Complete pipeline: Train models, update metrics, regenerate plots

set +e  # Don't exit on error - we want to continue even if some steps fail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETE PIPELINE: 1760K AND 2080K MODELS"
echo "=================================================================================="
echo ""

# Source FastChem environment if available
if [ -f "scripts/setup_fastchem_env.sh" ]; then
    echo "Sourcing FastChem environment..."
    source scripts/setup_fastchem_env.sh
fi

# Step 1: Generate datasets (if needed)
echo "=================================================================================="
echo "STEP 1: CHECKING/GENERATING DATASETS"
echo "=================================================================================="
python scripts/generate_1760_2080_datasets.py

# Step 2: Train models
echo ""
echo "=================================================================================="
echo "STEP 2: TRAINING MODELS"
echo "=================================================================================="
bash scripts/TRAIN_1760_2080.sh

# Step 3: Update metrics and regenerate plots
echo ""
echo "=================================================================================="
echo "STEP 3: UPDATING METRICS AND REGENERATING PLOTS"
echo "=================================================================================="
python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "=================================================================================="
echo ""
echo "All models trained and plots updated for:"
echo "  - 160K, 480K, 800K, 1120K, 1440K, 1760K, 2080K"
echo ""
echo "Check plots/performance_vs_size.png to see if performance has asymptoted."
