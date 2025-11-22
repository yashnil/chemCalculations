#!/bin/bash
# Retrain all datasets (x32, x48, x64, x80) with locked target species
# This script assumes datasets already exist and just retrains the models

set -e  # Exit on error

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "Retraining all datasets with locked targets"
echo "=========================================="

# Function to train a single dataset
train_dataset() {
    local tag=$1
    local csv_path="$BASE_DIR/datasets/all_gas_fastchem_${tag}.csv"
    
    if [ ! -f "$csv_path" ]; then
        echo "ERROR: Dataset not found: $csv_path"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Training ${tag} dataset"
    echo "=========================================="
    echo "CSV: $csv_path"
    
    # Clean up old run directory
    if [ -d "runs_autoencoder" ]; then
        echo "Removing old runs_autoencoder directory..."
        rm -rf runs_autoencoder
    fi
    
    # Train
    export CSV_PATH="$csv_path"
    python train_autoencoder.py
    
    # Run diagnostics
    export CSV_PATH="$csv_path"
    export BEST_MODULE="$BASE_DIR/runs_autoencoder/best_model.py"
    export OUT_DIR="$BASE_DIR/runs_autoencoder/diagnostics"
    python diagnostics.py
    
    # Generate plot
    export CSV_PATH="$csv_path"
    export BEST_MODULE="$BASE_DIR/runs_autoencoder/best_model.py"
    export OUT_DIR="$BASE_DIR/runs_autoencoder/diagnostics"
    export OUT_PNG="$BASE_DIR/runs_autoencoder/pred_vs_true_test.png"
    python plot.py
    
    # Archive
    archive_dir="$BASE_DIR/runs_autoencoder_${tag}"
    if [ -d "$archive_dir" ]; then
        echo "Removing old archive: $archive_dir"
        rm -rf "$archive_dir"
    fi
    mv runs_autoencoder "$archive_dir"
    echo "Archived to: $archive_dir"
}

# Train all datasets
train_dataset "x32"
train_dataset "x48"
train_dataset "x64"
train_dataset "x80"

echo ""
echo "=========================================="
echo "All datasets retrained successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python make_comparison_metrics.py"
echo "2. Run: python plot_resolution_study.py"
echo ""

