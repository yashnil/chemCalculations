# ✅ V9 Changes Reverted and Simplified

## What Happened

### Initial v9 Attempt (FAILED)
- Tried to use log-ratios: log₁₀(O/H, C/H, N/H, S/H)
- Reduced inputs from 7 to 6
- **Result**: 3× worse performance (MAE_log: 0.142 vs 0.047)

### Root Cause
- Log-ratios created high-variance features (std ≈ 3.66 vs 2.5)
- Lost information about absolute hydrogen abundance
- Model couldn't learn effectively

### Solution Implemented ✅
**Keep it simple**: Only change what matters, keep what works.

## Current v9 Configuration

### Changes from v8:
1. ✅ **70-15-15 split** (was 60-15-25) — more training data
2. ✅ **Temperature normalization**: T/T_max → [0,1] (was StandardScaler)

### Kept from v8:
- ✅ Element encoding: log₁₀(comp_X) + 9 for all 5 elements
- ✅ 7 input features (same count as v8)
- ✅ Model architecture (Optuna-tuned)
- ✅ Loss functions (composite loss)
- ✅ Output head (softplus)

## Input Features (v9 Final)

```python
# 7 features total
1. T / T_max              # [0, 1] — CHANGED from v8
2. log10(P)               # Same as v8
3. log10(comp_H) + 9      # Same as v8
4. log10(comp_O) + 9      # Same as v8
5. log10(comp_C) + 9      # Same as v8
6. log10(comp_N) + 9      # Same as v8
7. log10(comp_S) + 9      # Same as v8
```

## Files Updated

All scripts now use **7 inputs** with v8-style element encoding:

- ✅ `utils.py` — Reverted to 7 features, T/T_max normalization only
- ✅ `baseline_checks.py` — 70-15-15 split, T normalization
- ✅ `train_baseline.py` — 7-input model
- ✅ `tune.py` — 7-input hyperparameter search
- ✅ `finalize.py` — 7-input production model
- ✅ `final_train.py` — 7-input alternative training
- ✅ `documentation.txt` — Updated to reflect changes
- ✅ `README.md` — Updated quick start

## Ready to Run

The pipeline is now correctly configured and ready to test:

```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/v9

# Clean slate - old artefacts removed
python baseline_checks.py      # Creates 70-15-15 split, T normalization
python train_baseline.py       # Should perform similar to v8
python tune.py                 # Tune hyperparameters
python finalize.py             # Production model
```

## Expected Performance

**Should be close to v8 baseline:**
- v8: MAE_log ≈ 0.047, R²_log ≈ 0.954
- v9: Similar or slightly better due to more training data

## Key Learnings

1. **Don't change too much at once** — isolate changes for testing
2. **Log-ratios are problematic** — high variance, loss of information
3. **More training data is good** — 70-15-15 should help
4. **Bounded features may help** — T/T_max in [0,1] is cleaner

---

**Status**: ✅ Ready to run  
**Changes**: Minimal and focused  
**Risk**: Low (only split ratio and T normalization changed)

