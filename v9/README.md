# FastChem-Surrogate v9

## What's New in v9

Two **focused improvements** over v8:

1. **70-15-15 split** (previously 60-15-25) — +17% more training data
2. **Temperature normalized to [0,1]** — T_norm = T / T_max

**Element encoding**: Same as v8 (log₁₀ + 9) to maintain performance

**Result**: 7 input features (same as v8)

## Quick Start

```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/v9

# Step 1: Preprocessing & splits (70-15-15, T normalization)
python baseline_checks.py

# Step 2: Baseline model
python train_baseline.py

# Step 3: Hyperparameter tuning (~20 min)
python tune.py

# Step 4: Final training
python finalize.py

# Step 5: Diagnostics (optional)
python diagnostics.py
```

## Input Features (v9)

| Feature | Description | Range | Change from v8 |
|---------|-------------|-------|----------------|
| temperature_norm | T / T_max | [0, 1] | ✅ New normalization |
| log_pressure | log₁₀(P [bar]) | ~[-10, 5] | Same |
| log_H | log₁₀(comp_H) + 9 | ~[0, 9] | Same |
| log_O | log₁₀(comp_O) + 9 | ~[0, 9] | Same |
| log_C | log₁₀(comp_C) + 9 | ~[0, 9] | Same |
| log_N | log₁₀(comp_N) + 9 | ~[0, 9] | Same |
| log_S | log₁₀(comp_S) + 9 | ~[0, 9] | Same |

## Outputs

116 gas-phase species (normalized mole fractions, Σ = 1.0)

## Key Changes from v8

### What Changed
- ✅ Split ratio: 70-15-15 (was 60-15-25)
- ✅ Temperature: T/T_max normalization (was StandardScaler)
- ✅ Saves T_max for inference

### What Stayed the Same
- ✅ Element encoding: log₁₀ + 9
- ✅ 7 input features
- ✅ Model architecture options
- ✅ Loss functions
- ✅ Output head (softplus)

## Why Not Log-Ratios?

We initially tried H-relative ratios (log₁₀(O/H) etc.) to reduce to 6 inputs, but this caused **3× worse performance**:

- **v8**: MAE_log = 0.047, R²_log = 0.954 ✅
- **v9 with ratios**: MAE_log = 0.142, R²_log = 0.830 ❌

**Issue**: Log-ratios create high-variance features and lose absolute abundance information.

**Solution**: Keep v8's element encoding, only change split and T normalization.

## Artefacts

All saved in `artefacts/`:
- `input_scaler.pkl` — StandardScaler for 7 features
- `normalization_params.pkl` — {"T_max": value} ← NEW
- `splits.npz` — Train/val/test indices (70-15-15)
- `baseline_model.keras` — Baseline model
- `optuna_study.pkl` — Tuning results
- `final_model.keras` — **Production model**
- `final_report.json` — Metrics + metadata

## Inference Example

```python
import joblib, numpy as np
from tensorflow import keras

# Load
model = keras.load_model("artefacts/final_model.keras")
scaler = joblib.load("artefacts/input_scaler.pkl")
T_max = joblib.load("artefacts/normalization_params.pkl")["T_max"]

# Input: T=1500K, P=0.1 bar, composition
T, P = 1500.0, 0.1
comp = {"H": 0.85, "O": 0.10, "C": 0.03, "N": 0.015, "S": 0.005}

# Transform
x = np.array([[
    T / T_max,                    # temperature_norm
    np.log10(P),                  # log_pressure
    np.log10(comp["H"]) + 9.0,   # log_H
    np.log10(comp["O"]) + 9.0,   # log_O
    np.log10(comp["C"]) + 9.0,   # log_C
    np.log10(comp["N"]) + 9.0,   # log_N
    np.log10(comp["S"]) + 9.0    # log_S
]])

# Predict
y = model.predict(scaler.transform(x))
```

## Expected Performance

Similar or slightly better than v8 due to:
- More training data (70% vs 60%)
- Bounded temperature normalization may help gradients

## Notes

- **Data source**: Same as v8 (`all_gas.csv`)
- **Compatibility**: NOT compatible with v8 models/scalers
- **Recommendation**: Compare v9 vs v8 results to validate improvements
- **Documentation**: See `documentation.txt` for full details

---

**Ready to run!** Execute baseline_checks.py to get started.
