# FastChem-Surrogate v9

## What's New in v9

Three key improvements over v8:

1. **70-15-15 split** (previously 60-15-25) — more training data
2. **Temperature normalized to [0,1]** — T_norm = T / T_max
3. **H-relative element ratios** — log₁₀(O/H), log₁₀(C/H), log₁₀(N/H), log₁₀(S/H)

**Result**: 6 input features instead of 7

## Quick Start

```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/v9

# Step 1: Preprocessing & splits (70-15-15)
python baseline_checks.py

# Step 2: Baseline model
python train_baseline.py

# Step 3: Hyperparameter tuning (~20 min)
python tune.py

# Step 4: Final training
python finalize.py

# Step 5: Diagnostics
python diagnostics.py
```

## Input Features (v9)

| Feature | Description | Range |
|---------|-------------|-------|
| temperature_norm | T / T_max | [0, 1] |
| log_pressure | log₁₀(P [bar]) | ~[-10, 5] |
| log_O_H | log₁₀(O/H) | varies |
| log_C_H | log₁₀(C/H) | varies |
| log_N_H | log₁₀(N/H) | varies |
| log_S_H | log₁₀(S/H) | varies |

## Outputs

116 gas-phase species (normalized mole fractions, Σ = 1.0)

## Key Files

- `utils.py` — Data loading with v9 transformations
- `baseline_checks.py` — Creates splits, scaler, saves T_max
- `train_baseline.py` — Reference model (256-256-128 architecture)
- `tune.py` — Optuna hyperparameter search
- `finalize.py` — Production model training
- `losses.py` — Composite loss (balanced KL + log MAE)
- `model_heads.py` — Softplus output head
- `diagnostics.py` — Advanced analysis and plots

## Artefacts

All saved in `artefacts/`:
- `input_scaler.pkl` — StandardScaler for 6 features
- `normalization_params.pkl` — {"T_max": value}
- `splits.npz` — Train/val/test indices
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
    T / T_max,
    np.log10(P),
    np.log10(comp["O"] / comp["H"]),
    np.log10(comp["C"] / comp["H"]),
    np.log10(comp["N"] / comp["H"]),
    np.log10(comp["S"] / comp["H"])
]])

# Predict
y = model.predict(scaler.transform(x))
```

## Notes

- **Data source**: Same as v8 (`all_gas.csv`)
- **Compatibility**: NOT compatible with v8 models/scalers
- **Performance**: Expected similar or better than v8 due to more training data
- **Documentation**: See `documentation.txt` for full details

---

For questions or issues, see `documentation.txt` or compare with v8 implementation.

