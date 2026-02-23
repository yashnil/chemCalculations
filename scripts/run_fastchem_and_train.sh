#!/bin/bash
# Complete pipeline: Run FastChem for missing datasets, then train all models

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Source FastChem environment
source scripts/setup_fastchem_env.sh

echo "=================================================================================="
echo "COMPLETE PIPELINE: GENERATE DATASETS & TRAIN MODELS"
echo "=================================================================================="
echo ""

# Step 1: Generate missing datasets (320K, 800K)
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
    echo "  Running FastChem (this will take 2-4 hours)..."
    echo "    Log: fastchem_${tag}.log"
    python scripts/data_generation/run_fastchem_all.py \
        --jobs-root "$jobs_root" \
        --logk "$FASTCHEM_LOGK" \
        --logk-cond "$FASTCHEM_COND" \
        --element-abundances "$FASTCHEM_ELEM" \
        --chunksize 128 \
        > "fastchem_${tag}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ FastChem completed for $tag"
        
        # Merge results
        echo "  Merging results..."
        python scripts/data_generation/merge_fastchem_outputs.py \
            --jobs-root "$jobs_root" \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-csv "$output_csv" \
            > "merge_${tag}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Dataset $tag generated successfully"
        else
            echo "  ✗ Failed to merge results for $tag"
            exit 1
        fi
    else
        echo "  ✗ FastChem failed for $tag (check fastchem_${tag}.log)"
        exit 1
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
    echo "   Cannot proceed with training"
    exit 1
fi

# Step 3: Train all models
echo ""
echo "=================================================================================="
echo "STEP 3: TRAINING ALL MODELS"
echo "=================================================================================="
echo "Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32"
echo ""

TRAIN_PIDS=()
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
    TRAIN_PIDS+=($TRAIN_PID)
    echo "    → Started (PID: $TRAIN_PID, log: training_x${size}K.log)"
done

if [ ${#TRAIN_PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Waiting for all training jobs to complete..."
    for pid in "${TRAIN_PIDS[@]}"; do
        wait $pid
        echo "  ✓ Training job $pid completed"
    done
else
    echo "  All models already trained"
fi

# Step 4: Update metrics and regenerate plots
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
echo ""
echo "Best model comparison available in comparison_metrics.csv"
