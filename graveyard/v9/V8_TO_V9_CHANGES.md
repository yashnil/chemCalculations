# v8 → v9 Migration Guide

## Summary of Changes

| Aspect | v8 | v9 | Impact |
|--------|----|----|--------|
| **Train/Val/Test Split** | 60% / 15% / 25% | 70% / 15% / 15% | +17% more training samples |
| **Input Features** | 7 | 6 | Reduced dimensionality |
| **Temperature** | StandardScaler(T) | T / T_max → [0,1] | Bounded, more stable |
| **Elements** | log₁₀(H,O,C,N,S) + 9 | log₁₀(O/H, C/H, N/H, S/H) | H-relative, astrophysically standard |

## Detailed Input Transformation Changes

### v8 Inputs (7 features)
```python
# 1. Temperature - linearly scaled
X["temperature"] = df["temperature"]  # then StandardScaler

# 2. Pressure
X["pressure"] = np.log10(df["pressure"])

# 3-7. Elements (5 features)
for elem in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    X[elem] = np.log10(df[elem]) + 9.0
```

### v9 Inputs (6 features)
```python
# 1. Temperature - normalized to [0,1]
T_max = df["temperature"].max()
X["temperature_norm"] = df["temperature"] / T_max

# 2. Pressure (same)
X["log_pressure"] = np.log10(df["pressure"])

# 3-6. H-relative element ratios (4 features)
X["log_O_H"] = np.log10(df["comp_O"] / df["comp_H"])
X["log_C_H"] = np.log10(df["comp_C"] / df["comp_H"])
X["log_N_H"] = np.log10(df["comp_N"] / df["comp_H"])
X["log_S_H"] = np.log10(df["comp_S"] / df["comp_H"])
```

## Code Changes by File

### `utils.py`
- ✅ Updated `load_XY()` to return `T_max` as 9th value
- ✅ Changed input transformations to v9 format
- ✅ Updated column names

### `baseline_checks.py`
- ✅ Changed split from 60-15-25 to 70-15-15
- ✅ Applied v9 input transformations
- ✅ Saves `normalization_params.pkl` with T_max

### `train_baseline.py`
- ✅ Changed input shape from `(7,)` to `(6,)`
- ✅ Applied v9 transformations

### `tune.py`
- ✅ Updated `load_XY()` call to handle T_max return value
- ✅ Changed input shape from `(7,)` to `(6,)`

### `finalize.py`
- ✅ Updated `load_XY()` call
- ✅ Changed input shape to `(6,)`
- ✅ Saves T_max and input feature names in `final_report.json`

### `final_train.py`
- ✅ Applied v9 data loading with T-bin filtering
- ✅ Changed input shape to `(6,)`
- ✅ Updated feature transformations
- ✅ Removed redundant softmax layers

### `model_heads.py`, `losses.py`, `metrics.py`
- ✅ No changes needed (architecture-agnostic)

### `diagnostics.py`
- ⚠️ May need updates if it references specific input column names
- Works as-is if it only uses model outputs

## New Artefacts in v9

```
artefacts/
├── normalization_params.pkl    # NEW: {"T_max": value}
├── input_scaler.pkl            # Modified: fits 6 features
├── splits.npz                  # Modified: 70-15-15 split
├── final_report.json           # Enhanced: includes T_max, n_inputs, input_features
└── ... (rest same as v8)
```

## Expected Performance Changes

### Advantages
1. **More training data**: 70% vs 60% → better generalization
2. **Physical meaning**: H-relative ratios match astrophysical notation
3. **Simpler model**: 6 inputs → fewer parameters in first layer
4. **Bounded temperature**: [0,1] normalization may help gradient flow

### Potential Concerns
1. **Test set smaller**: 15% vs 25% → less precise error estimates
2. **H-ratio sensitivity**: If H varies greatly, ratios may have larger dynamic range
3. **Division concerns**: Requires comp_H > 0 (already enforced in data)

## Backward Compatibility

**Not compatible!** v9 models cannot use v8 data/scalers and vice versa:

- Different number of input features (6 vs 7)
- Different scaling (T_max normalization vs StandardScaler)
- Different split indices (70-15-15 vs 60-15-25)

## Testing Checklist

Before running full pipeline:

- [ ] `baseline_checks.py` creates 6-feature scaler
- [ ] `normalization_params.pkl` contains valid T_max
- [ ] Split sizes are approximately 70-15-15
- [ ] `train_baseline.py` builds model with Input((6,))
- [ ] No errors loading data in `utils.py`
- [ ] `tune.py` runs without shape mismatches
- [ ] `finalize.py` saves T_max in final_report.json

## Migration Script (if needed)

If you have v8 artefacts and want to convert data:

```python
import joblib, numpy as np, pandas as pd

# Load v8 artefacts
v8_scaler = joblib.load("v8/artefacts/input_scaler.pkl")
v8_splits = np.load("v8/artefacts/splits.npz")

# Note: Cannot directly migrate!
# You must re-run baseline_checks.py with v9 transformations
print("Run: python v9/baseline_checks.py")
```

## Quick Verification

After setting up v9:

```bash
cd v9
python -c "from utils import load_XY; X_tr, X_v, X_te, Y_tr, Y_v, Y_te, sc, sp, T = load_XY(); print(f'Inputs: {X_tr.shape[1]}, T_max: {T:.1f}')"
```

Expected output: `Inputs: 6, T_max: 3000.0` (or similar)

---

**Ready to run!** Execute the pipeline in order:
1. `baseline_checks.py`
2. `train_baseline.py`
3. `tune.py`
4. `finalize.py`
5. `diagnostics.py`

