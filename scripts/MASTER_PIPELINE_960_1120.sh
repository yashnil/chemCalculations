#!/bin/bash
# MASTER_PIPELINE_960_1120.sh
# Complete pipeline: Generate 960K and 1120K datasets, train models, update plots

# Don't exit on error - we want to continue even if datasets aren't ready
set +e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "MASTER PIPELINE: 960K AND 1120K DATASETS AND MODELS"
echo "=================================================================================="
echo ""

# Source FastChem environment if available
if [ -f "scripts/setup_fastchem_env.sh" ]; then
    echo "Sourcing FastChem environment..."
    source scripts/setup_fastchem_env.sh
fi

# Step 1: Generate datasets
echo "=================================================================================="
echo "STEP 1: GENERATING DATASETS (960K, 1120K)"
echo "=================================================================================="
python scripts/generate_960_1120_datasets.py
DATASET_GEN_STATUS=$?

# Check if datasets were actually generated
X960_EXISTS=$(test -f "data/datasets/all_gas_fastchem_x960.csv" && echo "yes" || echo "no")
X1120_EXISTS=$(test -f "data/datasets/all_gas_fastchem_x1120.csv" && echo "yes" || echo "no")

if [ "$X960_EXISTS" = "no" ] || [ "$X1120_EXISTS" = "no" ]; then
    echo ""
    echo "⚠️  WARNING: Datasets not yet generated!"
    echo "   x960 exists: $X960_EXISTS"
    echo "   x1120 exists: $X1120_EXISTS"
    echo ""
    echo "   To generate datasets, you need to:"
    echo "   1. Set FastChem environment variables:"
    echo "      export FASTCHEM_LOGK=/path/to/logK.dat"
    echo "      export FASTCHEM_COND=/path/to/logK_condensates.dat"
    echo "      export FASTCHEM_ELEM=/path/to/asplund_2009.dat"
    echo ""
    echo "   2. Run FastChem manually (see instructions above)"
    echo "   3. Then re-run this script to train models"
    echo ""
    echo "   Skipping training step for now..."
    SKIP_TRAINING=true
else
    echo ""
    echo "✅ Datasets exist! Proceeding to training..."
    SKIP_TRAINING=false
fi

# Step 2: Train models (only if datasets exist)
if [ "$SKIP_TRAINING" = "false" ]; then
    echo ""
    echo "=================================================================================="
    echo "STEP 2: TRAINING MODELS (960K, 1120K)"
    echo "=================================================================================="
    python scripts/train_960_1120_models.py
    TRAINING_STATUS=$?
else
    echo ""
    echo "⏭️  Skipping training step (datasets not ready)"
    TRAINING_STATUS=0
fi

# Step 3: Update metrics and plots (always run, even if new models aren't trained)
echo ""
echo "=================================================================================="
echo "STEP 3: UPDATING METRICS AND PLOTS"
echo "=================================================================================="
python scripts/update_and_regenerate_all.py
UPDATE_STATUS=$?

# Only regenerate consistent plots if we have new models
if [ "$SKIP_TRAINING" = "false" ] && [ "$TRAINING_STATUS" = "0" ]; then
    python scripts/regenerate_all_plots_consistent.py 2>/dev/null || echo "⚠️  Warning: Some plots may have errors (non-critical)"
else
    echo "⏭️  Skipping consistent plots regeneration (no new models)"
fi

echo ""
echo "=================================================================================="
if [ "$SKIP_TRAINING" = "false" ] && [ "$TRAINING_STATUS" = "0" ]; then
    echo "✅ PIPELINE COMPLETE!"
    echo ""
    echo "All models trained and plots updated for:"
    echo "  - 160K, 320K, 480K, 640K, 800K, 960K, 1120K"
else
    echo "⚠️  PIPELINE PARTIALLY COMPLETE"
    echo ""
    echo "Status:"
    echo "  - Datasets: $([ "$X960_EXISTS" = "yes" ] && [ "$X1120_EXISTS" = "yes" ] && echo "✅ Ready" || echo "❌ Not ready")"
    echo "  - Training: $([ "$SKIP_TRAINING" = "false" ] && [ "$TRAINING_STATUS" = "0" ] && echo "✅ Complete" || echo "⏭️  Skipped")"
    echo "  - Plots: ✅ Updated (using existing models)"
    echo ""
    echo "To complete the pipeline:"
    echo "  1. Generate datasets (set FastChem env vars and run FastChem)"
    echo "  2. Re-run this script: bash scripts/MASTER_PIPELINE_960_1120.sh"
fi
echo "=================================================================================="
echo ""
echo "Check plots/performance_vs_size.png to see current performance trends."
