# FastChem Neural Network Emulator

A high-performance machine learning surrogate model for chemical equilibrium calculations in planetary and stellar atmospheres.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution: ML Emulator](#solution-ml-emulator)
3. [Performance Comparison](#performance-comparison)
4. [Directory Structure](#directory-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Model Architecture](#model-architecture)
8. [Version History](#version-history)
9. [Advanced Usage](#advanced-usage)
10. [Citation](#citation)
11. [Contact](#contact)

---

## Problem Statement

### The Challenge

Chemical equilibrium calculations are fundamental to modeling planetary and stellar atmospheres. These calculations determine the abundances of hundreds of molecular and atomic species as functions of temperature, pressure, and elemental composition. Traditional approaches use iterative numerical solvers like [FastChem](https://github.com/exoclime/FastChem), which:

- **Are computationally expensive**: Each evaluation takes ~7ms
- **Don't scale well**: Atmospheric models require millions of evaluations
- **Become prohibitive**: 3D simulations and retrieval analyses are extremely slow

### The Bottleneck

A typical exoplanet atmospheric retrieval requires:
- **~10⁶–10⁸ chemistry evaluations** per model fit
- **Hours to days** of computation time
- **Limits scientific exploration** of parameter space

### Example Use Cases

1. **Exoplanet Atmospheric Retrievals**: Inferring atmospheric composition from spectra
2. **General Circulation Models (GCMs)**: 3D climate simulations
3. **Population Studies**: Exploring parameter space across many planets
4. **Real-time Analysis**: Interactive exploration of atmospheric models

**Bottom line**: FastChem is accurate but too slow for modern astrophysical applications.

---

## Solution: ML Emulator

### Our Approach

We replace the iterative FastChem solver with a **neural network emulator** that:

✅ **Maintains high accuracy**: Linear MAE < 0.01, R² > 0.99  
✅ **Achieves massive speed-up**: **140× faster** than FastChem  
✅ **Handles diverse conditions**: 100–3000 K, 10⁻¹⁰–10⁵ bar  
✅ **Works with complex chemistry**: Predicts abundances for key species  
✅ **Is production-ready**: Robust error handling, self-contained inference  

### Key Innovation

Rather than predicting all 116+ species (many at trace levels), we:

1. **Focus on the top-20 most abundant species** by mean contribution
2. **Use rich input features**: 30 inputs including T, P, and 28 elemental abundances
3. **Apply simple, effective normalization**: Physical constants, not complex transformations
4. **Employ robust data handling**: Drop problematic samples, log diagnostics

**Result**: A model that learns what matters, with cleaner signals and better gradients.

---

## Performance Comparison

### Speed Benchmark

| Method | Latency per Evaluation | Relative Speed |
|--------|------------------------|----------------|
| **FastChem (CPU)** | ~7.0 ms | 1× (baseline) |
| **ML Emulator (CPU)** | ~0.05 ms | **140×** faster |
| **ML Emulator (GPU)** | ~0.01 ms | **700×** faster |

### Accuracy Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Linear MAE** | 0.009 | Average error in linear abundance space |
| **R² Score** | 0.99+ | Excellent correlation with ground truth |
| **Training MSE** | 1–5 | Loss in scaled target space |

### Scalability Impact

For a typical atmospheric retrieval with 10⁷ evaluations:

| Method | Time Required | Practical Feasibility |
|--------|---------------|----------------------|
| **FastChem** | ~19.4 hours | Slow, limits exploration |
| **ML Emulator** | **~8 minutes** | Fast, enables population studies |

**Impact**: What took hours now takes minutes. What was impossible is now routine.

---

## Directory Structure

```
v10/
│
├── run_mlp.py                   # Main training script
│   └── Trains the ML emulator from scratch
│
├── plot.py                      # Visualization utilities
│   └── Generate diagnostic plots and parity diagrams
│
├── investigate.py               # Data analysis tools
│   └── Explore datasets and model predictions
│
├── CHECK_SETUP.sh               # Setup verification script
│   └── Verify dependencies and data availability
│
├── README.md                    # This file
├── START_HERE.md                # Quick start guide
├── DOCUMENTATION.md             # Complete technical documentation
├── COMPARISON_WITH_V8_V9.md     # Version comparison and lessons learned
└── SUMMARY.md                   # Executive summary
│
└── runs_mlp_v10/                # Output directory (created during training)
    ├── best.pt                  # Best model checkpoint
    ├── last.pt                  # Final epoch checkpoint
    ├── best_model.py            # Self-contained inference module
    ├── split_indices.npz        # Train/val/test split indices
    ├── train.log                # Training log
    └── [diagnostic plots]       # Optional visualizations
```

### Key Files Explained

- **`run_mlp.py`**: Training pipeline. Configure hyperparameters, run training, generates checkpoints.
- **`best_model.py`** (generated): Standalone inference module. Contains model architecture, weights, and normalization.
- **`plot.py`**: Post-training visualization. Creates parity plots, residual analysis, etc.
- **`investigate.py`**: Data exploration. Inspect training data, analyze predictions.

---

## Installation

### Requirements

- **Python**: 3.9 or higher
- **PyTorch**: 2.0 or higher
- **NumPy**: 1.20+
- **Pandas**: 1.3+
- **Scikit-learn**: 1.0+

### Setup

```bash
# Clone or navigate to the repository
cd /path/to/FastChem-emulator/v10

# Install dependencies
pip install torch numpy pandas scikit-learn

# Verify setup (optional)
./CHECK_SETUP.sh
```

### Data Requirements

Your input CSV must contain:

**Core columns**:
- `T_K`: Temperature in Kelvin
- `P_bar`: Pressure in bar

**Elemental abundances** (dex scale: 12 + log₁₀(N_elem/N_H)):
- `abund_H_dex`, `abund_He_dex`, `abund_O_dex`, `abund_C_dex`, etc.
- At minimum: H, He, C, N, O, S
- Optionally: Al, Ar, Ca, Cl, Co, Cr, Cu, F, Fe, Ge, K, Mg, Mn, Na, Ne, Ni, P, Si, Ti, V, Zn, e⁻

**Species columns**: Numeric abundances for gas-phase species (e.g., H2, H2O, CO, CH4)

---

## Quick Start

### 1. Configure Data Path

Edit `run_mlp.py`:

```python
# Line 33
CSV_PATH = '/path/to/your/fastchem_data.csv'
```

### 2. Train the Model

```bash
python run_mlp.py
```

**Expected output**:
```
2025-10-25 | INFO | Device: cpu
2025-10-25 | INFO | Loaded: 25600 rows × 125 cols
2025-10-25 | INFO | Resolved INPUT columns (30): ['T_K', 'P_bar', 'abund_H_dex', ...]
2025-10-25 | INFO | Resolved TARGET columns (20): [top species]
2025-10-25 | INFO | Split sizes: Train=21760 | Val=2560 | Test=1280
2025-10-25 | INFO | Model params: 530K

Epoch 001/200 | train_mse=542.6 | val_mse=419.6 | best=Yes
Epoch 002/200 | train_mse=241.9 | val_mse=153.2 | best=Yes
...
Epoch 150/200 | train_mse=1.234 | val_mse=1.456 | best=No

Done in 487 s. Best val_mse=1.456 @ epoch 148
TEST MSE: 1.523
```

**Training time**: ~5–15 minutes on modern CPU

### 3. Use the Trained Model

```python
import pandas as pd
import numpy as np
from runs_mlp_v10.best_model import load_model, normalize_inputs, denormalize_targets

# Load model
model = load_model(device='cpu')

# Prepare input data
df_input = pd.DataFrame({
    'T_K': [1500.0, 2000.0],
    'P_bar': [0.1, 1.0],
    'abund_H_dex': [12.0, 12.0],
    'abund_O_dex': [8.69, 8.69],
    'abund_C_dex': [8.43, 8.43],
    # ... include all other required abund_*_dex columns
})

# Normalize inputs
X = normalize_inputs(df_input)

# Predict (returns scaled abundances)
import torch
with torch.no_grad():
    y_scaled = model(X).numpy()

# Denormalize to linear abundances
y_linear = denormalize_targets(y_scaled)

print("Predicted abundances:", y_linear)
```

---

## Model Architecture

### Neural Network Design

**Type**: Feedforward Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input(30) → Linear(512) → LeakyReLU → Dropout(0.05)
          → Linear(512) → LeakyReLU → Dropout(0.05)
          → Linear(512) → LeakyReLU → Dropout(0.05)
          → Linear(20) → Output(20)
```

**Parameters**: ~530,000

### Input Features (30 total)

1. **T_K**: Temperature (normalized: T/4000)
2. **P_bar**: Pressure (log₁₀-scaled: log₁₀(P)/10)
3–30. **Elemental abundances** (centered & scaled: (abund_dex - 12)/10):
   - H, He, C, N, O, S (always)
   - Al, Ar, Ca, Cl, Co, Cr, Cu, F, Fe, Ge, K, Mg, Mn, Na, Ne, Ni, P, Si, Ti, V, Zn, e⁻ (when available)

### Output Targets (20 species)

Top-20 most abundant species by mean linear abundance across training set.  
Examples: H₂, H₂O, CO, CH₄, NH₃, He, CO₂, N₂, O₂, etc.

Auto-detected from data or manually specified.

### Normalization

**Philosophy**: Use physical constants for simple, interpretable scaling.

| Feature Type | Transformation | Range |
|--------------|----------------|-------|
| Temperature | T_K / 4000 | [0.025, 0.75] |
| Pressure | log₁₀(P_bar) / 10 | [-1.0, 0.5] |
| Abundances | (abund_dex - 12) / 10 | ≈ [-1, 1] |
| Targets | log₁₀(clipped) / 30 | Scaled log-space |

### Training Configuration

```python
Optimizer:      AdamW (lr=5e-4, weight_decay=1e-5)
Scheduler:      CosineAnnealingLR (η_min=1e-6)
Loss:           MSE in scaled target space
Batch size:     512
Epochs:         200 (early stopping if val loss plateaus)
Grad clipping:  5.0
Data split:     85% train / 10% val / 5% test
```

---

## Version History

### Evolution of the FastChem Emulator

#### v8: TensorFlow Baseline (2024)
- **Framework**: TensorFlow/Keras
- **Inputs**: 7 features (T, log P, 5 elements with log₁₀+9 encoding)
- **Outputs**: All 116 species
- **Split**: 60-15-25
- **Performance**: MAE_log ≈ 0.047, R² ≈ 0.954
- **Status**: ✅ Good baseline, but room for improvement

#### v9: Failed Log-Ratio Experiment (2025)
- **Key change**: Attempted log-ratio inputs (log₁₀(O/H), log₁₀(C/H), etc.)
- **Motivation**: Reduce from 7 to 6 inputs, match astrophysical conventions
- **Result**: **3× worse performance** (MAE_log ≈ 0.142, R² ≈ 0.830)
- **Root cause**: High variance features, loss of absolute abundance information
- **Lesson**: Simple transformations > clever feature engineering
- **Status**: ❌ Abandoned

#### v10: PyTorch Production Model (2025) — **Current**
- **Framework**: PyTorch
- **Inputs**: **30 features** (T, P, 28 element abundances)
- **Outputs**: **Top-20 species** (focused on most abundant)
- **Split**: 85-10-5 (more training data)
- **Normalization**: Simple physical constants (T/4000, (abund-12)/10)
- **Architecture**: 3-layer MLP, 512 hidden units, LeakyReLU
- **Loss**: Simple MSE (not composite)
- **Performance**: **Linear MAE ≈ 0.009, R² > 0.99**
- **Speed-up**: **140× faster** than FastChem
- **Status**: ✅ **Production-ready**

### v10 Improvements Over v9

| Aspect | v9 | v10 | Improvement |
|--------|-------|-----|-------------|
| **Inputs** | 6 (log-ratios) | 30 (abundances) | 5× more information |
| **Outputs** | 116 (all species) | 20 (top species) | Reduced noise |
| **Normalization** | Complex ratios | Simple constants | Low variance |
| **Framework** | TF/Keras | PyTorch | Cleaner code |
| **Accuracy** | MAE_log: 0.142 | Lin MAE: 0.009 | **16× better** |
| **R² Score** | 0.830 | 0.99+ | **Better fit** |
| **Training** | Unstable | Stable | Faster convergence |

**Key insight**: v10 succeeds by providing **more input information** (30 features vs 6-7) and **simpler transformations** (physical constants vs ratios), allowing the model to learn what matters without hand-crafted feature engineering.

---

## Advanced Usage

### Custom Hyperparameters

Edit `run_mlp.py` configuration section:

```python
# Model architecture
HIDDEN = 512              # Hidden layer size (try 256, 512, 1024)
DEPTH = 3                 # Number of layers (try 2-5)
ACTIVATION = "leaky_relu" # relu | gelu | tanh | leaky_relu
DROPOUT = 0.05            # Dropout rate (0.0-0.2)

# Training
EPOCHS = 200              # Max epochs
BATCH_SIZE = 512          # Batch size (256, 512, 1024)
LR = 5e-4                 # Learning rate
WEIGHT_DECAY = 1e-5       # L2 regularization

# Target selection
TARGET_TOPK_SPECIES = 20  # How many species to predict (10-50)
```

### Custom Species Selection

To predict specific species instead of auto-selecting top-K:

```python
# In run_mlp.py, set:
TARGET_COLS_MANUAL = [
    "H2", "H2O", "CO", "CH4", "NH3",
    "He", "CO2", "N2", "O2", "H",
    # ... your species of interest
]
```

### GPU Acceleration

```python
# In run_mlp.py:
DEVICE_FALLBACK = "cuda"  # Use GPU if available

# Or at inference:
model = load_model(device='cuda')
```

### Batch Inference

```python
from runs_mlp_v10.best_model import load_model, normalize_inputs, denormalize_targets
import pandas as pd
import torch

# Load large dataset
df = pd.read_csv('my_conditions.csv')  # Must have all required columns

# Normalize
X = normalize_inputs(df)

# Predict in batches
model = load_model(device='cuda')
model.eval()

batch_size = 10000
predictions = []

with torch.no_grad():
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size].cuda()
        y_batch = model(X_batch).cpu().numpy()
        predictions.append(y_batch)

y_scaled = np.vstack(predictions)
y_linear = denormalize_targets(y_scaled)
```

### Error Analysis

```python
from runs_mlp_v10.best_model import load_model, normalize_inputs, denormalize_targets, TARGET_COLS
import pandas as pd

# Load test data with ground truth
df_test = pd.read_csv('test_data.csv')

# Predict
X = normalize_inputs(df_test)
y_pred = denormalize_targets(model(X).detach().numpy())

# Compare with ground truth
y_true = df_test[TARGET_COLS].values

# Compute errors
errors = y_pred - y_true
rel_errors = errors / (y_true + 1e-10)

# Analyze
print("Mean Absolute Error:", np.abs(errors).mean())
print("Median Relative Error:", np.median(np.abs(rel_errors)))
```

---

## Citation

If you use this emulator in your research, please cite:

```bibtex
@software{fastchem_ml_emulator,
  author = {Mohanty, Yashnil and Malsky, Isaac},
  title = {FastChem Neural Network Emulator},
  year = {2025},
  url = {https://github.com/yashnil/chemCalculations}
}
```

### Acknowledging FastChem

This emulator is trained on data generated by [FastChem](https://github.com/exoclime/FastChem):

```bibtex
@article{Stock2018,
  author = {Stock, Joachim W. and Kitzmann, Daniel and Patzer, A. Beate C.},
  title = {FastChem: A computer program for efficient complex chemical equilibrium calculations in the neutral/ionized gas phase with applications to stellar and planetary atmospheres},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year = {2018},
  volume = {479},
  pages = {865--874},
  doi = {10.1093/mnras/sty1531}
}
```

---

## Contact

### Project Maintainers

**Yashnil Mohanty**  
Email: [ymohanty@ucsc.edu](mailto:ymohanty@ucsc.edu)  
Affiliation: University of California, Santa Cruz

**Isaac Malsky** (Original PyTorch implementation)  
Affiliation: UC Santa Cruz

### Support

- **Issues**: [GitHub Issues](https://github.com/yashnil/chemCalculations/issues)
- **Documentation**: See `DOCUMENTATION.md` for detailed technical information
- **Questions**: Email ymohanty@ucsc.edu

### Contributing

We welcome contributions! Areas of interest:
- Extended element coverage
- Condensed-phase species
- Uncertainty quantification
- GPU optimization
- Integration with atmospheric modeling codes (petitRADTRANS, BART, etc.)

---

## License

MIT License. See `LICENSE` file for details.

---

## Acknowledgments

- **FastChem** team for the original chemical equilibrium solver
- **UCSC Exoplanet Group** for computational resources
- **PyTorch** team for the deep learning framework
- **Isaac Malsky** for the initial PyTorch implementation and key architectural insights

---

<p align="center">
  <strong>FastChem ML Emulator — Making atmospheric chemistry fast enough for the future</strong>
</p>

<p align="center">
  <em>From hours to minutes. From impossible to routine.</em>
</p>
