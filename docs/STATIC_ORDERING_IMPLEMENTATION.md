# Static Species Ordering Implementation

## Overview

This document describes the implementation of **static species ordering** for the FlowMapAutoencoder model. This addresses the issue where different training runs could select different species based on dataset-specific statistics, making model comparison and deployment difficult.

## Problem Statement

**Previous Approach (Dynamic Top-K):**
- Species selection based on mean abundance in each training dataset
- Different datasets → different species → different model architectures
- Ordering could change between runs
- Difficult to compare models or ensure consistency
- Model outputs unpredictable between runs

**Solution (Static Ordering):**
- Fixed species list determined once from comprehensive analysis
- Same species in same order across all training runs
- Consistent model architecture
- Predictable outputs
- Easier model comparison and deployment

## Implementation

### 1. Species Statistics Analysis

**Script:** `scripts/analyze_species_statistics.py`

Analyzes all species in the dataset and computes:
- Mean, min, max, std, median, p95 abundance
- Non-zero fraction
- Cumulative coverage percentage

**Output:** `plots/species_statistics.csv`

**Key Findings:**
- Top 20 species: **96.76%** coverage
- Top 24 species: **98.77%** coverage  
- Top 32 species: **99.68%** coverage
- Top 36 species: **99.86%** coverage

### 2. Static Species List Generation

**Script:** `scripts/generate_static_species_list.py`

Generates JSON files with static species lists for different counts:
- `configs/static_species_list_24.json` (25 species including e-)
- `configs/static_species_list_32.json` (33 species including e-)
- `configs/static_species_list_36.json` (37 species including e-)

Each list is ordered by mean abundance, with electron (e-) always first if present.

### 3. Code Updates

**File:** `src/train_autoencoder.py`

**New Configuration Variables:**
```python
USE_STATIC_SPECIES_LIST = False
STATIC_SPECIES_LIST_PATH: Optional[str] = None
```

**Updated Function:** `resolve_target_columns()`
- Priority 1: Manual override (`TARGET_COLS_MANUAL`)
- Priority 2: Static species list (if enabled)
- Priority 3: Dynamic selection (fallback)

**Config Support:**
- `use_static_species_list`: Boolean flag
- `static_species_list_path`: Path to JSON file (relative to `configs/`)

### 4. Configuration Files

Created three test configurations:
- `configs/x160_static_24_config.json`
- `configs/x160_static_32_config.json`
- `configs/x160_static_36_config.json`

All use:
- Best architecture: latent_dim=192, hidden=[512,512,512], SiLU activation
- Log-ratio loss
- x160 dataset (160,000 samples)

## Usage

### Analyze Species Statistics

```bash
python scripts/analyze_species_statistics.py data/datasets/all_gas_fastchem_x160.csv
```

### Generate Static Lists

```bash
python scripts/generate_static_species_list.py plots/species_statistics.csv
```

### Train with Static Ordering

```bash
# Using config file
python src/train_autoencoder.py \
    --config configs/x160_static_32_config.json \
    --loss-type log_ratio \
    --run-dir runs_autoencoder_x160_static_32

# Or use the test script
./scripts/test_static_ordering.sh
```

## Benefits

1. **Consistency**: All models use the same output species in the same order
2. **Reproducibility**: Results are directly comparable across runs
3. **Deployment**: Inference always produces the same species
4. **Analysis**: Can see coverage before training

## Testing Plan

1. ✅ Analyze species statistics
2. ✅ Generate static lists (24, 32, 36 species)
3. ✅ Update code to support static ordering
4. 🔄 Train models with static ordering
5. ⏳ Compare performance vs. dynamic top-20 baseline
6. ⏳ Determine optimal species count

## Expected Results

We expect:
- **24 species**: Slightly better than 20 (98.77% vs 96.76% coverage)
- **32 species**: Best balance (99.68% coverage, minimal extra complexity)
- **36 species**: Marginal improvement (99.86% coverage, more parameters)

The static ordering should provide:
- More consistent results across runs
- Better generalization (fixed ordering learned once)
- Easier model comparison and deployment

## Files Created/Modified

**New Files:**
- `scripts/analyze_species_statistics.py`
- `scripts/generate_static_species_list.py`
- `scripts/test_static_ordering.sh`
- `configs/static_species_list_24.json`
- `configs/static_species_list_32.json`
- `configs/static_species_list_36.json`
- `configs/x160_static_24_config.json`
- `configs/x160_static_32_config.json`
- `configs/x160_static_36_config.json`
- `plots/species_statistics.csv`

**Modified Files:**
- `src/train_autoencoder.py` (added static ordering support)

