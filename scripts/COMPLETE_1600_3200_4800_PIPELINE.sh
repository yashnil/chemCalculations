#!/bin/bash
# COMPLETE_1600_3200_4800_PIPELINE.sh
# End-to-end pipeline for 1600K, 3200K, 4800K datasets
# These complete the 800-increment study: 800, 1600, 2400, 3200, 4000, 4800

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETE PIPELINE: x1600, x3200, x4800 (800K increments)"
echo "=================================================================================="
echo ""

# Source FastChem environment
if [ -f "scripts/setup_fastchem_env.sh" ]; then
    source scripts/setup_fastchem_env.sh
else
    echo "ERROR: setup_fastchem_env.sh not found"
    exit 1
fi

SIZES=(1600 3200 4800)

for SIZE in "${SIZES[@]}"; do
    TAG="x${SIZE}"
    DATASET="data/datasets/all_gas_fastchem_${TAG}.csv"
    JOBS_ROOT="results/fastchem_jobs/fastchem_jobs_${TAG}"
    CONFIG="configs/${TAG}_optimal_retrained.json"
    RUN_DIR="results/runs/runs_autoencoder_${TAG}_optimal_retrained"
    TOTAL_SAMPLES=$((SIZE * 1000))

    echo ""
    echo "=================================================================================="
    echo "Processing ${TAG} (${TOTAL_SAMPLES} samples)"
    echo "=================================================================================="

    # --- STEP 1: Dataset ---
    if [ -f "$DATASET" ]; then
        echo "Dataset exists, skipping generation"
    else
        echo "Generating dataset..."

        # Use x800 as reference since x160 was deleted
        REF_CSV="data/datasets/all_gas_fastchem_x800.csv"

        # Prepare job shards
        if [ ! -d "$JOBS_ROOT" ]; then
            echo "  Preparing FastChem job shards..."
            python scripts/data_generation/prepare_fastchem_jobs.py \
                --reference-csv "$REF_CSV" \
                --output-root "$JOBS_ROOT" \
                --total-samples "$TOTAL_SAMPLES" \
                --shard-size 2000 \
                --strategy empirical \
                --temp-jitter 50.0 \
                --logp-jitter 0.1 \
                --dex-jitter 0.05
        fi

        # Run FastChem
        echo "  Running FastChem (this may take several hours)..."
        python scripts/data_generation/run_fastchem_all.py \
            --jobs-root "$JOBS_ROOT" \
            --logk "$FASTCHEM_LOGK" \
            --logk-cond "$FASTCHEM_COND" \
            --element-abundances "$FASTCHEM_ELEM" \
            --chunksize 128

        # Merge results
        echo "  Merging results..."
        python scripts/data_generation/merge_fastchem_outputs.py \
            --jobs-root "$JOBS_ROOT" \
            --reference-csv "$REF_CSV" \
            --output-csv "$DATASET"

        # Clean up job shards to save disk space
        echo "  Cleaning up job shards..."
        rm -rf "$JOBS_ROOT"

        echo "  Dataset ${TAG} generated: $(wc -l < "$DATASET") rows"
    fi

    # --- STEP 2: Train model ---
    if [ -f "${RUN_DIR}/summary.json" ]; then
        echo "Model already trained, skipping"
    else
        echo "Training model..."

        # Create config
        python3 -c "
import json
from pathlib import Path
config = {
    'data': {
        'train_frac': 0.85, 'val_frac': 0.10, 'test_frac': 0.05,
        'target_topk_species': 20, 'include_fz_as_feature': True,
        'use_static_species_list': True, 'static_species_list_path': 'static_species_list_32.json',
        'input_cols_manual': None, 'target_cols_manual': None,
        'csv_path': 'data/datasets/all_gas_fastchem_${TAG}.csv'
    },
    'optimization': {'epochs': 200, 'batch_size': 512, 'learning_rate': 5e-4,
                    'weight_decay': 1e-5, 'grad_clip': 5.0, 'seed': 42},
    'architecture': {'latent_dim': 192, 'encoder_hidden': [512, 512, 512],
                    'dynamics_hidden': [512, 512, 512], 'decoder_hidden': [512, 512, 512],
                    'activation': 'silu', 'dropout': 0.0},
    'loss': {'type': 'log_ratio', 'use_weighted': True},
    'normalization': {'temp_divisor': 4000.0, 'input_log_scale': 10.0,
                     'abund_epsilon_offset': 12.0, 'abund_dex_scale': 10.0,
                     'target_zero_floor': 1e-30, 'target_log_scale': 30.0, 'log_eps': 1e-30},
    'scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5,
                 'patience': 10, 'min_lr': 1e-6}
}
Path('configs').mkdir(exist_ok=True)
Path('configs/${TAG}_optimal_retrained.json').write_text(json.dumps(config, indent=2))
print('Config created: configs/${TAG}_optimal_retrained.json')
"

        python src/train_autoencoder.py \
            --config "$CONFIG" \
            --loss-type log_ratio \
            --run-dir "$RUN_DIR"

        echo "  Model ${TAG} training complete"
    fi

    echo "  ${TAG} DONE"
done

echo ""
echo "=================================================================================="
echo "All models complete! Updating plots..."
echo "=================================================================================="

# Update comparison_metrics.csv and regenerate plots
python scripts/update_plots_800_increment.py

echo ""
echo "=================================================================================="
echo "PIPELINE COMPLETE"
echo "=================================================================================="
echo "Models trained: x1600, x3200, x4800"
echo "Full study: 800, 1600, 2400, 3200, 4000, 4800"
