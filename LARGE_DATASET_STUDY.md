# Large Dataset Study: x240, x480, x640

## Overview

This study tests whether larger datasets (240K, 480K, 640K samples) improve model performance compared to the current best model (x160_static_32 with 160K samples).

## Why This Study?

1. **Previous x224 test was unfair**: Used old hyperparameters (latent_dim=96, huber loss) instead of optimal ones
2. **Current best uses optimal config**: x160_static_32 uses latent_dim=192, log_ratio loss, static 32 species
3. **More data often helps**: Neural networks typically benefit from more training data
4. **Fair comparison**: All models use identical architecture, loss function, and species ordering

## Current Best Model (Baseline)

- **Model**: x160_static_32
- **Test Loss**: 1.50×10⁻⁴
- **Log MAE**: 0.0224
- **Log R²**: 0.9994
- **Dataset Size**: 160K samples

## Study Setup

### Datasets to Test
- **x240**: 240,000 samples (1.5× baseline)
- **x480**: 480,000 samples (3× baseline)
- **x640**: 640,000 samples (4× baseline)

### Configuration (Same for All)
- **Architecture**: latent_dim=192, width=512, layers=3
- **Loss**: log_ratio loss
- **Species**: Static 32 species (33 including e-)
- **Training**: 200 epochs, batch_size=512, lr=5e-4

## Files Created

### Config Files
- `configs/x240_static_32_config.json`
- `configs/x480_static_32_config.json`
- `configs/x640_static_32_config.json`

### Scripts
- `scripts/generate_large_datasets.py` - Prepare FastChem job shards
- `scripts/train_large_datasets.py` - Train models and run diagnostics
- `scripts/run_large_dataset_study.sh` - Master script (orchestrates everything)

## Step-by-Step Instructions

### Step 1: Generate FastChem Job Shards ✅ DONE

Job shards have been prepared:
- `fastchem_jobs_x240/` - 120 shards (240K samples)
- `fastchem_jobs_x480/` - 240 shards (480K samples)
- `fastchem_jobs_x640/` - 320 shards (640K samples)

### Step 2: Run FastChem (Overnight Recommended)

**Estimated time**: 2-4 hours per dataset (depending on hardware)

For each dataset (x240, x480, x640):

```bash
# Set FastChem environment variables
export FASTCHEM_LOGK=/path/to/logK.dat
export FASTCHEM_COND=/path/to/logK_condensates.dat

# Run FastChem
python scripts/data_generation/run_fastchem_all.py \
    --jobs-root fastchem_jobs_x240 \
    --logk "$FASTCHEM_LOGK" \
    --logk-cond "$FASTCHEM_COND" \
    --chunksize 128

# Merge results
python scripts/data_generation/merge_fastchem_outputs.py \
    --jobs-root fastchem_jobs_x240 \
    --reference-csv data/datasets/all_gas_fastchem_x160.csv \
    --output-csv data/datasets/all_gas_fastchem_x240.csv
```

**Or use the master script** (if you want to run FastChem automatically):
```bash
./scripts/run_large_dataset_study.sh --run-fastchem
```

### Step 3: Train Models

Once datasets are ready, train all models:

```bash
python scripts/train_large_datasets.py
```

This will:
1. Train models for x240, x480, x640
2. Run diagnostics for each
3. Update comparison metrics
4. Create comparison plots

**Estimated training time**: ~30-60 minutes per model (200 epochs)

### Step 4: View Results

Results will be in:
- `plots/comparison_metrics.csv` - All metrics
- `plots/large_dataset_comparison.png` - Comparison plot
- `runs_autoencoder_x240_static_32/` - Individual model diagnostics
- `runs_autoencoder_x480_static_32/` - Individual model diagnostics
- `runs_autoencoder_x640_static_32/` - Individual model diagnostics

## Expected Outcomes

### Best Case
- Lower test_loss and log_mae than x160_static_32
- Confirms that more data improves performance
- May find optimal dataset size (e.g., x480 is best)

### Worst Case
- Similar or slightly worse metrics
- Indicates x160 is near-optimal
- Diminishing returns from more data

### Most Likely
- Small improvements (1-5% better metrics)
- Confirms x160 is close to optimal
- Useful for understanding data scaling behavior

## Quick Start (If Datasets Already Exist)

If you've already generated the datasets:

```bash
# Train all models and run diagnostics
python scripts/train_large_datasets.py

# Or skip training if models already exist
python scripts/train_large_datasets.py --skip-training
```

## Troubleshooting

### Datasets Not Found
If training fails with "Dataset not found":
1. Check that CSV files exist: `data/datasets/all_gas_fastchem_x240.csv`, etc.
2. If missing, run Step 2 (FastChem generation)

### FastChem Fails
- Check environment variables: `echo $FASTCHEM_LOGK $FASTCHEM_COND`
- Verify FastChem Python bindings: `python -c "import pyfastchem"`
- Check individual shard logs in `fastchem_jobs_*/job_*/results/`

### Training Fails
- Check config files exist: `ls configs/x*_static_32_config.json`
- Verify dataset CSV format matches x160
- Check GPU/memory availability

## Comparison Metrics

The study will compare:
- **Test Loss** (normalized space)
- **Log MAE** (orders of magnitude error)
- **Log R²** (variance explained)
- **Linear MAE/MSE** (for reference)

All metrics will be compared against x160_static_32 baseline.
