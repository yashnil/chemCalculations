#!/bin/bash
# Cleanup and consolidate plots directory

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=================================================================================="
echo "CLEANING UP AND CONSOLIDATING PLOTS DIRECTORY"
echo "=================================================================================="
echo ""

# Step 1: Move plots from diagnostics_optimal/ to plots/ (if not duplicates)
echo "Step 1: Moving plots from diagnostics_optimal/ to plots/..."
if [ -d "plots/diagnostics_optimal" ]; then
    for file in plots/diagnostics_optimal/*.png; do
        if [ -f "$file" ]; then
            basename=$(basename "$file")
            # Only move if it doesn't exist in main plots/ or if diagnostics version is newer
            if [ ! -f "plots/$basename" ] || [ "$file" -nt "plots/$basename" ]; then
                echo "  Moving $basename"
                mv "$file" "plots/$basename"
            else
                echo "  Skipping $basename (already exists and is newer)"
            fi
        fi
    done
    
    # Remove empty diagnostics_optimal directory
    if [ -d "plots/diagnostics_optimal" ] && [ -z "$(ls -A plots/diagnostics_optimal)" ]; then
        rmdir plots/diagnostics_optimal
        echo "  Removed empty diagnostics_optimal directory"
    fi
fi

# Step 2: Remove outdated study plots
echo ""
echo "Step 2: Removing outdated study plots..."
OUTDATED_PLOTS=(
    "resolution_study.png"
    "resolution_study_log_mae.png"
    "resolution_study_log_r2.png"
    "resolution_study_test_loss.png"
    "resolution_study_validation_loss.png"
    "latent_dim_study.png"
    "layer_width_study.png"
    "dataset_size_study_optimal.png"
    "large_dataset_comparison.png"
    "overfitting_analysis.png"
)

for plot in "${OUTDATED_PLOTS[@]}"; do
    if [ -f "plots/$plot" ]; then
        echo "  Removing outdated: $plot"
        rm "plots/$plot"
    fi
done

# Step 3: Remove outdated CSV files (keep comparison_metrics.csv)
echo ""
echo "Step 3: Cleaning up CSV files..."
OUTDATED_CSVS=(
    "dataset_size_results_optimal.csv"
    "latent_dim_results.csv"
    "layer_width_results.csv"
    "hyperparameters_table.csv"
)

for csv in "${OUTDATED_CSVS[@]}"; do
    if [ -f "plots/$csv" ]; then
        echo "  Removing outdated CSV: $csv"
        rm "plots/$csv"
    fi
done

# Step 4: Remove duplicate hist_obs files (keep the ones from diagnostics_optimal if they exist)
echo ""
echo "Step 4: Checking for duplicates..."
# The diagnostics_optimal versions are likely newer/better, so we'll keep those

# Step 5: List final plots
echo ""
echo "=================================================================================="
echo "CLEANUP COMPLETE"
echo "=================================================================================="
echo ""
echo "Remaining plots:"
ls -1 plots/*.png 2>/dev/null | wc -l | xargs echo "  Total PNG files:"
echo ""
echo "Key plots:"
ls -1 plots/*.png 2>/dev/null | grep -E "(loss_curves|performance|model_comparison|asymptote|parity|scatter|MAE_per_species|error_distribution|residual)" | sed 's/^/  - /'
