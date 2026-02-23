#!/bin/bash
# complete_x4000.sh
# Complete x4000: merge, train, update plots

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "COMPLETING x4000 PIPELINE"
echo "=================================================================================="
echo ""

# Step 1: Merge FastChem results (if dataset doesn't exist)
if [ ! -f "data/datasets/all_gas_fastchem_x4000.csv" ]; then
    echo "Step 1: Merging FastChem results for x4000..."
    python scripts/data_generation/merge_fastchem_outputs.py \
        --jobs-root results/fastchem_jobs/fastchem_jobs_x4000 \
        --reference-csv data/datasets/all_gas_fastchem_x160.csv \
        --output-csv data/datasets/all_gas_fastchem_x4000.csv
    
    echo "✅ x4000 dataset merged"
else
    echo "✅ x4000 dataset already exists, skipping merge"
fi

# Step 2: Train model (if not complete)
if [ ! -f "results/runs/runs_autoencoder_x4000_optimal_retrained/summary.json" ]; then
    echo ""
    echo "Step 2: Training x4000 model..."
    
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
else
    echo "✅ x4000 model already complete, skipping training"
fi

# Step 3: Update plots
echo ""
echo "Step 3: Updating metrics and plots..."
python scripts/update_plots_for_optimal_retrained.py

echo ""
echo "=================================================================================="
echo "✅ x4000 PIPELINE COMPLETE!"
echo "=================================================================================="
echo ""
echo "Final status:"
python scripts/check_progress.py
