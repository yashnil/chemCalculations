#!/bin/bash
# complete_all_3360_3680_4000.sh
# Complete end-to-end pipeline for 3360K, 3680K, 4000K datasets

set -e  # Exit on error

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETE END-TO-END PIPELINE: 3360K, 3680K, 4000K"
echo "=================================================================================="
echo ""

# Source FastChem environment if available
if [ -f "scripts/setup_fastchem_env.sh" ]; then
    echo "Sourcing FastChem environment..."
    source scripts/setup_fastchem_env.sh
    echo "✅ FastChem environment loaded"
else
    echo "⚠️  Warning: setup_fastchem_env.sh not found"
    echo "   Make sure FASTCHEM_LOGK, FASTCHEM_COND, and FASTCHEM_ELEM are set"
fi

echo ""
echo "Step 1: Checking current status..."
python scripts/check_progress.py

echo ""
echo "=================================================================================="
echo "Step 2: Completing x3360 dataset and model"
echo "=================================================================================="

# Check if x3360 model is complete
if [ -f "results/runs/runs_autoencoder_x3360_optimal_retrained/summary.json" ]; then
    echo "✅ x3360 model already complete, skipping"
else
    echo "Training x3360 model..."
    # Create config if needed
    if [ ! -f "configs/x3360_optimal_retrained.json" ]; then
        python -c "
import json
from pathlib import Path
config = {
    'data': {
        'train_frac': 0.85, 'val_frac': 0.10, 'test_frac': 0.05,
        'target_topk_species': 20, 'include_fz_as_feature': True,
        'use_static_species_list': True, 'static_species_list_path': 'static_species_list_32.json',
        'input_cols_manual': None, 'target_cols_manual': None,
        'csv_path': 'data/datasets/all_gas_fastchem_x3360.csv'
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
Path('configs/x3360_optimal_retrained.json').write_text(json.dumps(config, indent=2))
"
    fi
    
    python src/train_autoencoder.py \
        --config configs/x3360_optimal_retrained.json \
        --loss-type log_ratio \
        --run-dir results/runs/runs_autoencoder_x3360_optimal_retrained
    
    echo "✅ x3360 model training complete"
fi

echo ""
echo "=================================================================================="
echo "Step 3: Completing x3680 dataset and model"
echo "=================================================================================="

# Check if dataset exists
if [ -f "data/datasets/all_gas_fastchem_x3680.csv" ]; then
    echo "✅ x3680 dataset already exists, skipping generation"
else
    echo "Completing x3680 FastChem jobs..."
    
    # Check if jobs are prepared
    if [ ! -d "results/fastchem_jobs/fastchem_jobs_x3680" ]; then
        echo "Preparing FastChem jobs for x3680..."
        python scripts/data_generation/prepare_fastchem_jobs.py \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-root results/fastchem_jobs/fastchem_jobs_x3680 \
            --total-samples 3680000 \
            --shard-size 2000 \
            --strategy empirical \
            --temp-jitter 50.0 \
            --logp-jitter 0.1 \
            --dex-jitter 0.05
    fi
    
    # Run FastChem (will skip already completed jobs)
    if [ -n "$FASTCHEM_LOGK" ] && [ -n "$FASTCHEM_COND" ]; then
        echo "Running FastChem for x3680 (will resume from checkpoint)..."
        python scripts/data_generation/run_fastchem_all.py \
            --jobs-root results/fastchem_jobs/fastchem_jobs_x3680 \
            --logk "$FASTCHEM_LOGK" \
            --logk-cond "$FASTCHEM_COND" \
            --chunksize 128 \
            ${FASTCHEM_ELEM:+--element-abundances "$FASTCHEM_ELEM"}
        
        echo "Merging FastChem results for x3680..."
        python scripts/data_generation/merge_fastchem_outputs.py \
            --jobs-root results/fastchem_jobs/fastchem_jobs_x3680 \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-csv data/datasets/all_gas_fastchem_x3680.csv
        
        echo "✅ x3680 dataset complete"
    else
        echo "❌ Error: FastChem environment variables not set"
        echo "   Please set FASTCHEM_LOGK and FASTCHEM_COND, then run manually"
        exit 1
    fi
fi

# Train x3680 model
if [ -f "results/runs/runs_autoencoder_x3680_optimal_retrained/summary.json" ]; then
    echo "✅ x3680 model already complete, skipping"
else
    echo "Training x3680 model..."
    # Create config if needed
    if [ ! -f "configs/x3680_optimal_retrained.json" ]; then
        python -c "
import json
from pathlib import Path
config = {
    'data': {
        'train_frac': 0.85, 'val_frac': 0.10, 'test_frac': 0.05,
        'target_topk_species': 20, 'include_fz_as_feature': True,
        'use_static_species_list': True, 'static_species_list_path': 'static_species_list_32.json',
        'input_cols_manual': None, 'target_cols_manual': None,
        'csv_path': 'data/datasets/all_gas_fastchem_x3680.csv'
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
Path('configs/x3680_optimal_retrained.json').write_text(json.dumps(config, indent=2))
"
    fi
    
    python src/train_autoencoder.py \
        --config configs/x3680_optimal_retrained.json \
        --loss-type log_ratio \
        --run-dir results/runs/runs_autoencoder_x3680_optimal_retrained
    
    echo "✅ x3680 model training complete"
fi

echo ""
echo "=================================================================================="
echo "Step 4: Completing x4000 dataset and model"
echo "=================================================================================="

# Check if dataset exists
if [ -f "data/datasets/all_gas_fastchem_x4000.csv" ]; then
    echo "✅ x4000 dataset already exists, skipping generation"
else
    echo "Generating x4000 dataset..."
    
    # Prepare jobs
    if [ ! -d "results/fastchem_jobs/fastchem_jobs_x4000" ]; then
        echo "Preparing FastChem jobs for x4000..."
        python scripts/data_generation/prepare_fastchem_jobs.py \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-root results/fastchem_jobs/fastchem_jobs_x4000 \
            --total-samples 4000000 \
            --shard-size 2000 \
            --strategy empirical \
            --temp-jitter 50.0 \
            --logp-jitter 0.1 \
            --dex-jitter 0.05
    fi
    
    # Run FastChem
    if [ -n "$FASTCHEM_LOGK" ] && [ -n "$FASTCHEM_COND" ]; then
        echo "Running FastChem for x4000..."
        python scripts/data_generation/run_fastchem_all.py \
            --jobs-root results/fastchem_jobs/fastchem_jobs_x4000 \
            --logk "$FASTCHEM_LOGK" \
            --logk-cond "$FASTCHEM_COND" \
            --chunksize 128 \
            ${FASTCHEM_ELEM:+--element-abundances "$FASTCHEM_ELEM"}
        
        echo "Merging FastChem results for x4000..."
        python scripts/data_generation/merge_fastchem_outputs.py \
            --jobs-root results/fastchem_jobs/fastchem_jobs_x4000 \
            --reference-csv data/datasets/all_gas_fastchem_x160.csv \
            --output-csv data/datasets/all_gas_fastchem_x4000.csv
        
        echo "✅ x4000 dataset complete"
    else
        echo "❌ Error: FastChem environment variables not set"
        echo "   Please set FASTCHEM_LOGK and FASTCHEM_COND, then run manually"
        exit 1
    fi
fi

# Train x4000 model
if [ -f "results/runs/runs_autoencoder_x4000_optimal_retrained/summary.json" ]; then
    echo "✅ x4000 model already complete, skipping"
else
    echo "Training x4000 model..."
    # Create config if needed
    if [ ! -f "configs/x4000_optimal_retrained.json" ]; then
        python -c "
import json
from pathlib import Path
config = {
    'data': {
        'train_frac': 0.85, 'val_frac': 0.10, 'test_frac': 0.05,
        'target_topk_species': 20, 'include_fz_as_feature': True,
        'use_static_species_list': True, 'static_species_list_path': 'static_species_list_32.json',
        'input_cols_manual': None, 'target_cols_manual': None,
        'csv_path': 'data/datasets/all_gas_fastchem_x4000.csv'
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
Path('configs/x4000_optimal_retrained.json').write_text(json.dumps(config, indent=2))
"
    fi
    
    python src/train_autoencoder.py \
        --config configs/x4000_optimal_retrained.json \
        --loss-type log_ratio \
        --run-dir results/runs/runs_autoencoder_x4000_optimal_retrained
    
    echo "✅ x4000 model training complete"
fi

echo ""
echo "=================================================================================="
echo "Step 5: Updating metrics and plots"
echo "=================================================================================="

python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ ALL COMPLETE!"
echo "=================================================================================="
echo ""
echo "Final status:"
python scripts/check_progress.py
