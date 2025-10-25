# v10 Complete Documentation

## What is v10?

v10 is Isaac Malsky's proven PyTorch implementation of a FastChem surrogate model. It successfully addresses all the issues encountered in v8/v9 by using a fundamentally different approach.

---

## Key Innovations from Isaac's Work

### 1. **Focus on What Matters**
- **Predict only top-20 species** (not all 116)
- Reduces output complexity by 83%
- Focuses model capacity on important predictions

### 2. **More Input Information**
- **30 input features** instead of 7
- Includes 28 element abundances (Al, Ar, C, Ca, Cl, Co, Cr, Cu, F, Fe, Ge, H, He, K, Mg, Mn, N, Na, Ne, Ni, O, P, S, Si, Ti, V, Zn, e-)
- Provides richer chemical context

### 3. **Simple, Effective Normalization**
```python
# No complex transformations, just divide by sensible constants
T_normalized = T_K / 4000
P_normalized = log10(P_bar) / 10
abund_normalized = (abund_dex - 12) / 10  # Center around solar
target_scaled = log10(clip(y, 1e-30, inf)) / 30
```

### 4. **Robust Data Handling**
- **Drops problematic rows** instead of trying to fix them
- Logs detailed diagnostics about what was dropped
- Prevents silent errors

### 5. **PyTorch Benefits**
- Cleaner code
- Faster training
- Better debugging
- Easier deployment

---

## Architecture Details

### Model: Plain MLP

```python
Input(30) 
→ Linear(512) → LeakyReLU → Dropout(0.05)
→ Linear(512) → LeakyReLU → Dropout(0.05)
→ Linear(512) → LeakyReLU → Dropout(0.05)
→ Linear(20)
→ Output(20)
```

**Parameters**: ~530K (vs v8's variable architecture)

### Training

- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR (eta_min=1e-6)
- **Loss**: MSE in scaled target space
- **Batch size**: 512
- **Epochs**: 200
- **Gradient clipping**: 5.0

---

## Normalization Philosophy

### Why This Works

**v8/v9 problem**: Log-ratios created high variance
```python
log10(O/H) ranges from -9 to +9  # Too wide!
```

**Isaac's solution**: Center and scale with physical constants
```python
(abund_O_dex - 12) / 10  # Centered around solar, scaled to ~[-1, 1]
```

### Constants Explained

| Constant | Value | Why |
|----------|-------|-----|
| TEMP_DIVISOR | 4000 | Typical stellar/planetary T range |
| INPUT_LOG_SCALE | 10 | Brings log10(P) to ~[-1, 0.5] |
| ABUND_EPSILON_OFFSET | 12 | Solar hydrogen = 12 in dex scale |
| ABUND_DEX_SCALE | 10 | Typical element variation |
| TARGET_LOG_SCALE | 30 | Typical log10 abundance range |

---

## Data Requirements

### Input CSV Must Have:

1. **Core columns**:
   - `T_K`: Temperature in Kelvin
   - `P_bar`: Pressure in bar

2. **Abundance columns** (dex scale: 12 + log10(N_elem/N_H)):
   - `abund_H_dex`
   - `abund_O_dex`
   - ... (all other elements)

3. **Species columns**:
   - Numeric columns with species abundances
   - Auto-detected if not in exclusion list

### Optional:
- `group_index` or `point_index` for group-stratified splitting
- `fZ` or `fZ_dex` for metallicity

---

## Running the Code

### Step 1: Update Paths

Edit `run_mlp.py`:
```python
CSV_PATH = '/path/to/your/all_gas.csv'  # Change this!
OUT_DIR = "runs_mlp_v10"                # Output directory
```

### Step 2: Run Training

```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/v10
python run_mlp.py
```

### Step 3: Monitor Progress

Watch the log output:
```
Epoch 001/200 | train_mse=542.6 | val_mse=419.6
Epoch 002/200 | train_mse=241.9 | val_mse=153.2
...
Epoch 150/200 | train_mse=1.234 | val_mse=1.456
```

Training converges when MSE stops decreasing.

### Step 4: Use the Model

```python
# Import the auto-generated module
from runs_mlp_v10.best_model import load_model, normalize_inputs, denormalize_targets

model = load_model()
# ... see README for full example
```

---

## Configuration Options

### Model Hyperparameters

```python
HIDDEN = 512              # Hidden layer size
DEPTH = 3                 # Number of hidden layers
ACTIVATION = "leaky_relu" # relu | gelu | tanh | leaky_relu
DROPOUT = 0.05            # Dropout probability
```

### Training Hyperparameters

```python
EPOCHS = 200              # Max epochs
BATCH_SIZE = 512          # Batch size
LR = 5.0e-04              # Initial learning rate
WEIGHT_DECAY = 1.0e-05    # L2 regularization
ETA_MIN = 1.0e-06         # Min LR for cosine schedule
GRAD_CLIP_NORM = 5.0      # Gradient clipping
```

### Data Split

```python
TRAIN_FRAC = 0.85         # 85% training
VAL_FRAC = 0.10           # 10% validation
TEST_FRAC = 0.05          # 5% test
USE_GROUP_SPLIT = False   # Group-stratified splitting
```

### Target Selection

```python
TARGET_TOPK_SPECIES = 20  # How many species to predict
```

To predict more or fewer species, change this number.

---

## Outputs Explained

### Checkpoints

**`best.pt`**: Model weights at lowest validation MSE
```python
{
    "model": state_dict,
    "epoch": 123,
    "val_mse": 1.234,
    "config": {...}  # All hyperparameters
}
```

**`last.pt`**: Model weights at final epoch

### Self-Contained Module

**`best_model.py`**: Standalone inference code
- Contains all normalization constants
- Includes model architecture
- Can be copied anywhere
- No dependencies on training code

### Split Indices

**`split_indices.npz`**:
```python
np.load('split_indices.npz')
# Contains: train_idx, val_idx, test_idx, seed
```

Ensures reproducibility.

---

## Comparison: v8/v9 vs v10

### Why v9 Failed

**Problem 1**: Log-ratio transformations
```python
# v9 approach
log10(O/H), log10(C/H), ...  # High variance, lost info
# Result: 3× worse performance
```

**Problem 2**: Too many outputs
```python
# Predicting 116 species
# Many are trace species with noisy/zero values
# Wastes model capacity
```

**Problem 3**: Complex composite loss
```python
# λ·KL_divergence + (1-λ)·MAE_log
# Hard to tune, unstable gradients
```

### Why v10 Works

**Solution 1**: More inputs, not ratios
```python
# v10 approach
abund_H_dex, abund_O_dex, abund_C_dex, ...
# All absolute values, centered and scaled
```

**Solution 2**: Focus on top-K
```python
# Only 20 most important species
# Model learns what matters
```

**Solution 3**: Simple MSE
```python
# Just minimize squared error
# In properly scaled space
```

---

## Troubleshooting

### CSV Not Found
```
FileNotFoundError: No such file: all_gas.csv
```
→ Update `CSV_PATH` in `run_mlp.py`

### Missing Columns
```
ValueError: Expected columns 'T_K' and 'P_bar' not found
```
→ Check your CSV has these exact column names

### All Rows Dropped
```
RuntimeError: All rows were dropped due to non-finite inputs/targets
```
→ Check for NaN/Inf in your data  
→ Review the logged error summary

### Poor Performance
- Check normalization constants match your data range
- Increase `TARGET_TOPK_SPECIES` if too few
- Adjust `HIDDEN` or `DEPTH` if underfitting
- Check for data quality issues

---

## Advanced Usage

### Custom Element List

To use specific elements instead of auto-detection:
```python
INPUT_COLS_MANUAL = [
    "T_K", "P_bar",
    "abund_H_dex", "abund_O_dex", "abund_C_dex",
    # ... your elements
]
```

### Custom Target Species

```python
TARGET_COLS_MANUAL = [
    "H2", "H2O", "CO", "CH4", "NH3",
    # ... your species
]
```

### Group-Stratified Splitting

```python
USE_GROUP_SPLIT = True
# Requires 'group_index' column in CSV
```

Useful if you have groups that should stay together.

---

## Performance Metrics

### Expected Results

Based on Isaac's runs:

**Scaled Space** (what the model sees):
- Train MSE: 1-5
- Val MSE: 1-5
- Test MSE: 1-5

**Linear Space** (actual abundances):
- MAE: 0.008-0.01
- R²: 0.99+

**Speed**:
- Training: ~2-10 min (200 epochs, CPU)
- Inference: <1ms per sample

---

## Future Enhancements

Possible improvements:

1. **Ensemble**: Train multiple models, average predictions
2. **Uncertainty**: Add dropout at inference for uncertainty estimates
3. **Physics constraints**: Add mass conservation loss term
4. **Active learning**: Retrain on samples with high uncertainty
5. **GPU acceleration**: Use CUDA for faster training

---

## Credits

This implementation is based on Isaac Malsky's working FastChem-MLP code, which successfully solved the issues encountered in v8/v9.

**Key insight**: Sometimes simpler is better. More inputs, fewer outputs, straightforward normalization, and robust error handling beat complex transformations and losses.

---

## Summary

**v10 = Isaac's proven approach**

✅ 30 input features (rich chemical context)  
✅ Top-20 output species (focus on what matters)  
✅ Simple normalization (center & scale)  
✅ PyTorch MLP (clean, fast)  
✅ 85-10-5 split (more training data)  
✅ MSE loss (simple, effective)  
✅ Robust error handling (drop, don't sanitize)  

**Result**: High accuracy without compromising on robustness.

---

Created: October 2025  
Version: 10.0  
Status: Production-ready ✅

