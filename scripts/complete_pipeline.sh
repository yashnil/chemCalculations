#!/bin/bash
# Complete pipeline: Generate missing datasets, then train all models

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETE PIPELINE: GENERATE DATASETS & TRAIN MODELS"
echo "=================================================================================="
echo ""

# Check FastChem environment variables
if [ -z "$FASTCHEM_LOGK" ] || [ -z "$FASTCHEM_COND" ]; then
    echo "⚠️  FastChem environment variables not set"
    echo ""
    echo "   FastChem requires 3 files:"
    echo "   1. logK.dat (set via FASTCHEM_LOGK)"
    echo "   2. logK_condensates.dat (set via FASTCHEM_COND)"
    echo "   3. element_abundances file (set via FASTCHEM_ELEM, or will be inferred)"
    echo ""
    echo "   Example:"
    echo "   export FASTCHEM_LOGK=/path/to/FastChem/tables/logK.dat"
    echo "   export FASTCHEM_COND=/path/to/FastChem/tables/logK_condensates.dat"
    echo "   export FASTCHEM_ELEM=/path/to/FastChem/element_abundances/asplund_2009.dat"
    echo ""
    echo "   If FASTCHEM_ELEM is not set, the script will try to infer it from the logK path."
    echo ""
    read -p "Continue anyway? (y/n): " response
    if [ "$response" != "y" ]; then
        echo "Aborted. Set environment variables and run again."
        exit 1
    fi
fi

# Step 1: Generate missing datasets
echo "=================================================================================="
echo "STEP 1: GENERATING MISSING DATASETS (320K, 800K)"
echo "=================================================================================="

for tag in x320 x800; do
    size=$(echo $tag | sed 's/x//')
    output_csv="data/datasets/all_gas_fastchem_${tag}.csv"
    jobs_root="results/fastchem_jobs/fastchem_jobs_${tag}"
    
    if [ -f "$output_csv" ]; then
        echo "✓ Dataset $tag already exists"
        continue
    fi
    
    echo ""
    echo "Processing $tag..."
    
    # Check if job shards exist
    if [ ! -d "$jobs_root" ]; then
        echo "  Preparing job shards..."
        python scripts/data_generation/prepare_fastchem_jobs.py \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-root "$jobs_root" \
            --total-samples $((size * 1000)) \
            --shard-size 2000 \
            --strategy empirical \
            --temp-jitter 50.0 \
            --logp-jitter 0.1 \
            --dex-jitter 0.05
    fi
    
    # Run FastChem
    if [ -n "$FASTCHEM_LOGK" ] && [ -n "$FASTCHEM_COND" ]; then
        echo "  Running FastChem (this will take 2-4 hours)..."
        FASTCHEM_CMD="python scripts/data_generation/run_fastchem_all.py \
            --jobs-root \"$jobs_root\" \
            --logk \"$FASTCHEM_LOGK\" \
            --logk-cond \"$FASTCHEM_COND\" \
            --chunksize 128"
        
        # Add element abundances if set
        if [ -n "$FASTCHEM_ELEM" ]; then
            FASTCHEM_CMD="$FASTCHEM_CMD --element-abundances \"$FASTCHEM_ELEM\""
        fi
        
        eval $FASTCHEM_CMD > "fastchem_${tag}.log" 2>&1
        
        # Merge results
        echo "  Merging results..."
        python scripts/data_generation/merge_fastchem_outputs.py \
            --jobs-root "$jobs_root" \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-csv "$output_csv" \
            > "merge_${tag}.log" 2>&1
        
        echo "  ✓ Dataset $tag generated"
    else
        echo "  ⚠️  FastChem environment variables not set"
        echo "     Job shards prepared at: $jobs_root"
        echo "     Run FastChem manually, then merge results"
    fi
done

# Step 2: Check all datasets are ready
echo ""
echo "=================================================================================="
echo "STEP 2: CHECKING DATASETS"
echo "=================================================================================="

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
    echo "   Please generate these datasets first"
    echo "   Then run this script again to train models"
    exit 1
fi

# Step 3: Train all models
echo ""
echo "=================================================================================="
echo "STEP 3: TRAINING ALL MODELS"
echo "=================================================================================="
echo "Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32"
echo ""

for size in 160 320 480 640 800; do
    run_dir="results/runs/runs_autoencoder_x${size}_optimal_retrained"
    
    if [ -d "$run_dir" ] && [ -f "$run_dir/summary.json" ]; then
        echo "  ⏭️  Skipping x${size}K - already trained"
        continue
    fi
    
    echo "  Training x${size}K model..."
    python src/train_autoencoder.py \
        --config "configs/x${size}_optimal_retrained.json" \
        --loss-type log_ratio \
        --run-dir "$run_dir" \
        > "training_x${size}K.log" 2>&1 &
    
    TRAIN_PID=$!
    echo "    → Started (PID: $TRAIN_PID)"
done

echo ""
echo "Waiting for all training jobs to complete..."
wait

echo ""
echo "=================================================================================="
echo "STEP 4: UPDATING METRICS AND REGENERATING PLOTS"
echo "=================================================================================="

python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ COMPLETE!"
echo "=================================================================================="
echo ""
echo "All models trained and plots regenerated:"
echo "  - plots/comparison_metrics.csv"
echo "  - plots/performance_vs_size.png"
echo "  - plots/loss_curves.png"
echo "  - plots/asymptote_analysis.png"
echo "  - plots/model_comparison.png"
