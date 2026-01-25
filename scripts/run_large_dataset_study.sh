#!/bin/bash
# run_large_dataset_study.sh
# ==========================
# Master script to run the large dataset study (x240, x480, x640)
# 
# This script:
# 1. Generates FastChem job shards for x240, x480, x640
# 2. (Optional) Runs FastChem on all shards (takes hours - can run overnight)
# 3. Merges FastChem outputs
# 4. Trains models
# 5. Runs diagnostics and comparison
#
# Usage:
#   ./run_large_dataset_study.sh                    # Prepare jobs only
#   ./run_large_dataset_study.sh --run-fastchem     # Also run FastChem (long!)
#   ./run_large_dataset_study.sh --skip-data-gen     # Skip data generation, train only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

echo "================================================================================"
echo "LARGE DATASET STUDY: x240, x480, x640"
echo "================================================================================"
echo ""

# Parse arguments
RUN_FASTCHEM=false
SKIP_DATA_GEN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-fastchem)
            RUN_FASTCHEM=true
            shift
            ;;
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Generate dataset job shards
if [ "$SKIP_DATA_GEN" = false ]; then
    echo "Step 1: Generating FastChem job shards..."
    echo "----------------------------------------"
    python scripts/generate_large_datasets.py
    
    echo ""
    echo "✅ Job shards prepared!"
    echo ""
    echo "⚠️  IMPORTANT: FastChem jobs are ready but NOT yet executed."
    echo "   To run FastChem (this will take several hours):"
    echo "   1. For each dataset (x240, x480, x640), run:"
    echo "      python scripts/data_generation/run_fastchem_all.py \\"
    echo "          --jobs-root fastchem_jobs_X240 \\"
    echo "          --logk \$FASTCHEM_LOGK \\"
    echo "          --logk-cond \$FASTCHEM_COND \\"
    echo "          --chunksize 128"
    echo ""
    echo "   2. After FastChem completes, merge results:"
    echo "      python scripts/data_generation/merge_fastchem_outputs.py \\"
    echo "          --jobs-root fastchem_jobs_X240 \\"
    echo "          --reference-csv data/datasets/all_gas_fastchem_x160.csv \\"
    echo "          --output-csv data/datasets/all_gas_fastchem_X240.csv"
    echo ""
    
    if [ "$RUN_FASTCHEM" = true ]; then
        echo "Running FastChem now (this will take hours)..."
        echo ""
        
        for tag in x240 x480 x640; do
            echo "Running FastChem for $tag..."
            python scripts/data_generation/run_fastchem_all.py \
                --jobs-root "results/fastchem_jobs/fastchem_jobs_${tag}" \
                --logk "${FASTCHEM_LOGK}" \
                --logk-cond "${FASTCHEM_COND}" \
                --chunksize 128 || {
                    echo "⚠️  FastChem failed for $tag - continuing..."
                }
            
            echo "Merging results for $tag..."
            python scripts/data_generation/merge_fastchem_outputs.py \
                --jobs-root "results/fastchem_jobs/fastchem_jobs_${tag}" \
                --reference-csv data/datasets/all_gas_fastchem_x160.csv \
                --output-csv "data/datasets/all_gas_fastchem_${tag}.csv" || {
                    echo "⚠️  Merge failed for $tag - continuing..."
                }
        done
    else
        echo "⏭️  Skipping FastChem execution (use --run-fastchem to run)"
        echo "   You can run FastChem manually or overnight, then continue with:"
        echo "   ./run_large_dataset_study.sh --skip-data-gen"
        exit 0
    fi
fi

# Step 2: Train models
echo ""
echo "================================================================================"
echo "Step 2: Training models"
echo "================================================================================"
echo ""

python scripts/train_large_datasets.py

echo ""
echo "================================================================================"
echo "STUDY COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Comparison metrics: plots/comparison_metrics.csv"
echo "  - Comparison plot: plots/large_dataset_comparison.png"
echo "  - Individual model diagnostics in runs_autoencoder_*_static_32/"
echo ""
