#!/bin/bash
# TARGETED_OVERSAMPLE_PIPELINE.sh
# ===============================
# Generate targeted samples in low-P (hot Jupiter) and high C/O regions,
# run FastChem, merge, and augment the x4800 dataset for retraining.
#
# Goal: Reduce Log MAE in independent validation (hot Jupiter 0.27 dex, C/O sweep 0.16 dex)
#
# Prerequisites: all_gas_fastchem_x800.csv and all_gas_fastchem_x4800.csv must exist

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================================================="
echo "TARGETED OVERSAMPLE PIPELINE"
echo "  Regions: low-P (10^-6 to 10^-4 bar), high C/O (1.5 to 2.5)"
echo "  Augmenting: x4800 base dataset"
echo "=================================================================================="
echo ""

# Source FastChem environment
if [ -f "scripts/setup_fastchem_env.sh" ]; then
    source scripts/setup_fastchem_env.sh
else
    echo "ERROR: setup_fastchem_env.sh not found"
    exit 1
fi

REF_CSV="data/datasets/all_gas_fastchem_x800.csv"
BASE_DATASET="data/datasets/all_gas_fastchem_x4800.csv"
TARGETED_JOBS="results/fastchem_jobs/fastchem_jobs_targeted"
TARGETED_CSV="data/datasets/all_gas_fastchem_targeted.csv"
AUGMENTED_CSV="data/datasets/all_gas_fastchem_x4800_augmented.csv"

# Check prerequisites
if [ ! -f "$REF_CSV" ]; then
    echo "ERROR: Reference CSV not found: $REF_CSV"
    exit 1
fi
if [ ! -f "$BASE_DATASET" ]; then
    echo "ERROR: Base dataset not found: $BASE_DATASET (train x4800 first)"
    exit 1
fi

# --- STEP 1: Prepare targeted job shards ---
echo "Step 1: Preparing targeted oversample job shards..."
python scripts/data_generation/prepare_targeted_oversample.py \
    --output-root "$TARGETED_JOBS" \
    --reference-csv "$REF_CSV" \
    --n-low-p 50000 \
    --n-high-co 50000 \
    --shard-size 2000 \
    --seed 2026

# --- STEP 2: Run FastChem on targeted conditions ---
echo ""
echo "Step 2: Running FastChem on targeted conditions (this may take 1-2 hours)..."
python scripts/data_generation/run_fastchem_all.py \
    --jobs-root "$TARGETED_JOBS" \
    --logk "$FASTCHEM_LOGK" \
    --logk-cond "$FASTCHEM_COND" \
    --element-abundances "$FASTCHEM_ELEM" \
    --chunksize 128

# --- STEP 3: Merge targeted results ---
echo ""
echo "Step 3: Merging targeted FastChem results..."
python scripts/data_generation/merge_fastchem_outputs.py \
    --jobs-root "$TARGETED_JOBS" \
    --reference-csv "$REF_CSV" \
    --output-csv "$TARGETED_CSV"

# --- STEP 4: Filter T > 750K (consistent with training pipeline) ---
echo ""
echo "Step 4: Filtering T > 750K and merging with base dataset..."
python << PYEOF
import pandas as pd

targeted = pd.read_csv("$TARGETED_CSV")
base = pd.read_csv("$BASE_DATASET")

# Apply same filter as training pipeline
targeted = targeted[targeted["T_K"] > 750].reset_index(drop=True)
print(f"  Targeted after T>750K filter: {len(targeted)} rows")

# Ensure column order matches
assert list(targeted.columns) == list(base.columns), "Column mismatch"
augmented = pd.concat([base, targeted], axis=0, ignore_index=True)
augmented.to_csv("$AUGMENTED_CSV", index=False)
print(f"  Base: {len(base)}, Targeted: {len(targeted)}, Augmented: {len(augmented)}")
PYEOF

# --- STEP 5: Clean up job shards (optional, saves disk space) ---
echo ""
echo "Step 5: Cleaning up job shards..."
rm -rf "$TARGETED_JOBS"

echo ""
echo "=================================================================================="
echo "TARGETED OVERSAMPLE COMPLETE"
echo "=================================================================================="
echo "  Augmented dataset: $AUGMENTED_CSV"
echo "  Base rows: $(wc -l < "$BASE_DATASET" | tr -d ' ') - 1"
echo "  Augmented rows: $(wc -l < "$AUGMENTED_CSV" | tr -d ' ') - 1"
echo ""
echo "Next: Train model on augmented dataset:"
echo "  python src/train_autoencoder.py \\"
echo "    --config configs/x4800_augmented.json \\"
echo "    --loss-type log_ratio \\"
echo "    --run-dir results/runs/runs_autoencoder_x4800_augmented"
echo ""
