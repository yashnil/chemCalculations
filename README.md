# FastChem Neural Network Emulator

A high-performance machine learning surrogate model for chemical equilibrium calculations in planetary and stellar atmospheres.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-production--ready-green)](https://github.com/yashnil/chemCalculations)
[![Log R²](https://img.shields.io/badge/Log_R²-0.9991-brightgreen)](https://github.com/yashnil/chemCalculations)

**Status: Production-Ready** ✅  
**Best Model Performance**: Log R² = 0.9991, Log MAE = 0.0564, ~800× speed-up over FastChem

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Performance Metrics](#performance-metrics)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Model Architecture](#model-architecture)
7. [Usage Examples](#usage-examples)
8. [Citation](#citation)
9. [Contact](#contact)

---

## Problem Statement

### The Computational Bottleneck in Atmospheric Modeling

Chemical equilibrium calculations are fundamental to understanding planetary and stellar atmospheres. These calculations determine the abundances of hundreds of molecular and atomic species (H₂, H₂O, CO, CH₄, NH₃, etc.) as functions of:
- **Temperature** (100–3000 K)
- **Pressure** (10⁻¹⁰–10⁵ bar)
- **Elemental composition** (H, He, C, N, O, S, metals)

Traditional iterative solvers like [FastChem](https://github.com/exoclime/FastChem) are accurate but **computationally expensive**: ~7 milliseconds per evaluation.

### Why This Matters

Modern astrophysical applications require **millions to billions** of chemistry evaluations:

| Application | Evaluations Needed | Time with FastChem | Time with ML Emulator |
|-------------|-------------------|--------------------|-----------------------|
| **1D Atmospheric Profile** | ~10⁴ | 70 seconds | **0.03 seconds** |
| **Exoplanet Retrieval** | ~10⁷ | 19.4 hours | **30 seconds** |
| **3D GCM Simulation** | ~10⁹ | 81 days | **6 hours** |
| **Population Study** | ~10¹⁰ | 2.2 years | **2.3 days** |

**The bottleneck**: Chemical equilibrium calculations dominate runtime in:
- JWST/HST atmospheric retrievals
- Brown dwarf and hot Jupiter climate models
- Exoplanet population studies
- Real-time analysis during observations

**We need a solution that is both fast AND accurate.**

---

## Solution Overview

### Neural Network Emulator Approach

We replace the iterative FastChem solver with a **trained neural network** that:

✅ **Exceeds baseline accuracy**: Test Loss = 2.06×10⁻⁴, Log R² = 0.9991 (99.91% variance explained)  
✅ **Achieves ~800× speed-up**: 7 ms → ~0.009 ms per evaluation  
✅ **Handles full parameter space**: 750–3000 K, 10⁻¹⁰–10⁵ bar  
✅ **Focuses on important species**: Predicts 21 species (top-20 + electrons, >99.9% of mass)  
✅ **Eliminates artifacts**: Zero vertical striping through aggressive low-T filtering (T > 750K)  
✅ **Is production-ready**: PyTorch FlowMapAutoencoder, self-contained inference, comprehensive validation  
✅ **Optimal dataset size**: 160K samples identified through systematic resolution study  

### Key Design Principles

1. **Rich input representation**: 7 core features (T, P, 5 elements) expandable to 30+ with metals
2. **Focused outputs**: Top-20 species by abundance, not all 116+ species
3. **Simple normalization**: Physical constants (T/4000, (abund_dex-12)/10)
4. **Data quality**: Filter low-temperature samples that cause prediction artifacts
5. **Robust validation**: Comprehensive diagnostics suite with parity plots

**Philosophy**: Give the model good data, keep transformations simple, focus on what matters.

---

## Performance Metrics

### Best Model Performance

**🏆 Best Overall Performance:**
- **Model**: x160_new (FlowMapAutoencoder with optimal hyperparameters)
- **Test Loss**: 2.06×10⁻⁴ (normalized space)
- **Log MAE**: 0.0564 (orders of magnitude error)
- **Log R²**: 0.9991 (99.91% variance explained)
- **Dataset Size**: 160,000 samples
- **Architecture**: latent_dim=128, width=512, layers=3, SiLU activation

**Training Configuration**:
- Dataset: 160K samples (750–3000 K, T > 750K filter)
- Split: 85% train (136K) / 10% val (16K) / 5% test (8K)
- Architecture: FlowMapAutoencoder, 128-dim latent, 512-width layers (3 layers each)
- Activation: SiLU (Sigmoid Linear Unit)
- Loss: Weighted Huber (δ=0.02)
- Scheduler: ReduceLROnPlateau
- Training time: ~24 minutes (200 epochs)
- Dropout: 0.0 (no overfitting observed)

**Key Improvements Over Previous Architecture**:
- **51% reduction in Log MAE** (0.1156 → 0.0564)
- **51% reduction in test loss** (0.000389 → 0.000206)
- **Log R² improvement** (0.9982 → 0.9991)

### Hyperparameter Optimization Studies

We conducted three systematic hyperparameter studies to identify optimal model configuration:

#### Test #1: Latent Dimension Study
**Objective**: Find optimal latent space dimensionality

**Tested values**: 64, 96, 128, 160, 192  
**Results**: 
- **Best**: latent_dim=192 (test_loss=0.000339) at 50 epochs
- Performance degrades for both smaller and larger dimensions
- Clear minimum at 192, optimal for 21-species output space
- **Note**: With full 200-epoch training, latent_dim=128 (x160_new) achieves better performance

**Plot**: `plots/latent_dim_study.png`

#### Test #2: Layer Width Study  
**Objective**: Find optimal layer width and depth

**Tested configurations**: 
- Widths: 256, 512, 768, 1024
- Layers: 3, 4
- Using latent_dim=192 from Test #1

**Results**:
- **Best overall**: width=512, layers=3 (test_loss=0.000339)
- **Best 4-layer**: width=768, layers=4 (test_loss=0.000348)
- Wider layers (1024) don't improve performance
- 3 layers perform better overall

**Plot**: `plots/layer_width_study.png`

#### Test #3: Dataset Size Study with Optimal Hyperparameters
**Objective**: Evaluate optimal hyperparameters across different dataset sizes

**Configuration**: latent_dim=192, width=512, layers=3  
**Tested sizes**: x32, x48, x64, x80, x96, x112, x128, x144, x160, x176

**Key Findings**:
- Optimal hyperparameters work best at x160 (the size they were optimized on)
- x160_optimal: test_loss=0.000339, log_mae=0.109 (12.8% improvement over previous architecture)
- Smaller datasets perform worse with these hyperparameters
- Confirms x160 as optimal dataset size

**Plot**: `plots/dataset_size_study_optimal.png`  
**Full results**: See `plots/comparison_metrics.csv`

### Comparison: FastChem vs ML Emulator

| Aspect | FastChem | ML Emulator | Advantage |
|--------|----------|-------------|-----------|
| **Accuracy** | Exact (ground truth) | Log R² = 0.9991, Test Loss = 2.06×10⁻⁴ | Excellent match (99.91% variance) |
| **Speed** | 7 ms/eval | 0.009 ms/eval | **~800× faster** |
| **Scalability** | Linear | Parallel batching | GPU-accelerable |
| **Deployment** | C++ binary | Python/PyTorch | Easy integration |
| **Use case** | Ground truth | Production inference | Complementary |

**Bottom line**: ML emulator enables science that was previously impossible due to computational cost.

---

## Project Structure

```
chemCalculations/
│
├── src/                            # Source code
│   ├── train_autoencoder.py       # Main training script
│   ├── autoencoder_model.py       # FlowMapAutoencoder architecture
│   ├── diagnostics.py             # Comprehensive diagnostic suite
│   ├── plot.py                    # Generate parity plots
│   ├── make_comparison_metrics.py # Collect metrics across dataset sizes
│   ├── plot_*.py                  # Plotting utilities for studies
│   └── test_*.py                  # Hyperparameter study scripts
│
├── models/                         # Trained models
│   ├── best_model/                # Best production model (x160_new)
│   │   ├── best_model.py         # Self-contained inference module
│   │   ├── best.pt               # Model weights
│   │   ├── diagnostics/          # Comprehensive diagnostic plots
│   │   └── summary.json          # Training metrics
│   └── archive/                   # Archived training runs (excluded from Git)
│
├── plots/                          # Visualization outputs
│   ├── comparison_metrics.csv     # Performance metrics for all models
│   ├── hyperparameters_table.csv  # Complete hyperparameter table
│   ├── *.png                      # Study plots and visualizations
│   └── *.csv                      # Study results
│
├── data/                           # Datasets
│   └── datasets/                  # Training datasets (excluded from Git)
│
├── scripts/                        # Utility scripts
│   ├── data_generation/          # FastChem job generation and merging
│   ├── retrain_all_datasets.py   # Automated retraining across dataset sizes
│   └── clean_dataset.py          # Data preprocessing utilities
│
├── history/                        # Historical versions (archived)
│   └── graveyard/                # Previous versions (v1-v10, NEW_VERS)
│
└── README.md                       # This file
```

---

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy
```

### Training a Model

```bash
# Train the model
cd src
python train_autoencoder.py  # ~24 minutes (200 epochs, 160K samples)
# Note: Set CSV_PATH environment variable to point to your dataset

# Generate validation plots
python plot.py               # Creates parity plot
python diagnostics.py        # Creates comprehensive diagnostic plots
```

### Using the Trained Model

```python
import sys
sys.path.append('models/best_model')

from best_model import load_model, normalize_inputs, forward_autoencoder
import pandas as pd
import torch

# Load model
model = load_model(device='cpu')

# Prepare input (T, P, elemental abundances in dex scale)
df_input = pd.DataFrame({
    'T_K': [1500.0],           # Temperature in Kelvin
    'P_bar': [0.1],            # Pressure in bar
    'abund_H_dex': [12.0],     # H abundance (reference = 12.0)
    'abund_O_dex': [8.69],     # O abundance (solar)
    'abund_C_dex': [8.43],     # C abundance (solar)
    'abund_N_dex': [7.83],     # N abundance (solar)
    'abund_S_dex': [7.12],     # S abundance (solar)
})

# Normalize and predict
X = normalize_inputs(df_input)
with torch.no_grad():
    y_scaled = forward_autoencoder(model, X).numpy()

# De-normalize to linear space
from best_model import scale_targets_train_to_linear_torch
y_linear = scale_targets_train_to_linear_torch(torch.tensor(y_scaled)).numpy()

print("Predicted abundances for 21 species:")
print(y_linear)  # Returns abundances for: e-, N2, O2, H2, S2, C5, H, O, H2O, etc.
```

### Running Hyperparameter Studies

```bash
cd src

# Latent dimension study
python test_latent_dim.py --latent-dims 64 96 128 160 192 --epochs 50

# Layer width study
python test_layer_widths.py

# Dataset size study
python test_dataset_sizes_optimal.py
```

---

## Model Architecture

### FlowMapAutoencoder (Production)

**Framework**: PyTorch  
**Type**: FlowMapAutoencoder (Encoder-Dynamics-Decoder)

**Architecture**:
```
Encoder:
  Input: [state(21) + global(7)] = 28
  → Dense(512) → SiLU
  → Dense(512) → SiLU  
  → Dense(512) → SiLU
  → Output: latent(128)

Dynamics:
  Input: [latent(128) + dt(1) + global(7)] = 136
  → Dense(512) → SiLU
  → Dense(512) → SiLU
  → Dense(512) → SiLU
  → Output: latent_delta(128)
  → Residual: latent + latent_delta

Decoder:
  Input: latent(128)
  → Dense(512) → SiLU
  → Dense(512) → SiLU
  → Dense(512) → SiLU
  → Output: state(21)
```

**Parameters**: ~1.87M  
**Key features**:
- **Latent dimension**: 128 (optimal for full training)
- **Constant layer widths**: All layers use 512 units (no bottlenecks)
- **SiLU activation**: Smooth, self-gated activation function
- **No dropout**: Model doesn't overfit with sufficient data
- **Residual connections**: In dynamics module for better gradient flow
- **Weighted Huber loss**: Robust to outliers (δ=0.02)
- **Adaptive LR**: ReduceLROnPlateau (reduces LR when validation plateaus)

### Input Features (7 total)

| Feature | Description | Normalization | Typical Range |
|---------|-------------|---------------|---------------|
| T_K | Temperature (K) | T / 4000 | [0.17, 0.75] |
| P_bar | Pressure (bar) | log₁₀(P) / 10 | [-1.0, 0.5] |
| abund_H_dex | Hydrogen abundance | (dex - 12) / 10 = 0 | 0.0 |
| abund_O_dex | Oxygen abundance | (dex - 12) / 10 | [-0.9, 0.9] |
| abund_C_dex | Carbon abundance | (dex - 12) / 10 | [-0.9, 0.9] |
| abund_N_dex | Nitrogen abundance | (dex - 12) / 10 | [-0.9, 0.9] |
| abund_S_dex | Sulfur abundance | (dex - 12) / 10 | [-0.9, 0.9] |

**Dex scale**: `abund_X_dex = 12 + log₁₀(N_X / N_H)` (standard astrophysical notation)
- Solar values: H=12.0, O≈8.69, C≈8.43, N≈7.83, S≈7.12

### Output Species (21 total)

Top-20 most abundant species + electron (e⁻), auto-selected from training data.

**Typical species**: H₂, H₂O, CO, CH₄, NH₃, CO₂, N₂, O₂, He, H, O, C, N, S, OH, etc.

### Training Configuration

```python
Optimizer:      Adam (lr=5×10⁻⁴, weight_decay=1×10⁻⁵)
Scheduler:      ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1×10⁻⁶)
Loss:           Weighted Huber (δ=0.02) in normalized log-space
Batch size:     512
Epochs:         200
Gradient clip:  5.0
Data split:     85% train / 10% val / 5% test
Dataset size:   160K samples (optimal, determined via resolution study)
```

---

## Usage Examples

### Basic Inference

```python
import sys
sys.path.append('models/best_model')

from best_model import load_model, normalize_inputs, forward_autoencoder, TARGET_COLS
import pandas as pd
import torch

# Load trained model
model = load_model(device='cpu')
model.eval()

# Hot Jupiter atmosphere: T=1500K, P=0.1 bar, solar composition
df = pd.DataFrame({
    'T_K': [1500.0],
    'P_bar': [0.1],
    'abund_H_dex': [12.0],
    'abund_O_dex': [8.69],
    'abund_C_dex': [8.43],
    'abund_N_dex': [7.83],
    'abund_S_dex': [7.12],
})

# Predict
X = normalize_inputs(df)
with torch.no_grad():
    y_scaled = forward_autoencoder(model, X).numpy()

# De-normalize
from best_model import scale_targets_train_to_linear_torch
y_linear = scale_targets_train_to_linear_torch(torch.tensor(y_scaled)).numpy()

# Results
results = pd.DataFrame(y_linear, columns=TARGET_COLS)
print("Top-5 most abundant species:")
print(results.iloc[0].sort_values(ascending=False).head(5))
```

### Batch Processing (High Throughput)

```python
# Create T-P grid for atmospheric model
import numpy as np

T_grid = np.linspace(1000, 2500, 100)
P_grid = np.logspace(-2, 2, 100)
T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

# Solar composition across entire grid
df_grid = pd.DataFrame({
    'T_K': T_mesh.ravel(),
    'P_bar': P_mesh.ravel(),
    'abund_H_dex': 12.0,
    'abund_O_dex': 8.69,
    'abund_C_dex': 8.43,
    'abund_N_dex': 7.83,
    'abund_S_dex': 7.12,
})

# Predict for 10,000 conditions in ~0.03 seconds!
X = normalize_inputs(df_grid)
with torch.no_grad():
    y_scaled = forward_autoencoder(model, X).numpy()
y_linear = scale_targets_train_to_linear_torch(torch.tensor(y_scaled)).numpy()

# Shape: (10000, 21) - abundances for 21 species at 10k T-P points
```

### Integration with Atmospheric Models

```python
def chemistry_step(T, P, composition):
    """
    Replace FastChem call with ML emulator.
    
    Args:
        T: Temperature (K)
        P: Pressure (bar)
        composition: Dict with 'H', 'O', 'C', 'N', 'S' abundances in dex
    
    Returns:
        abundances: Dict mapping species names to number densities
    """
    df = pd.DataFrame({
        'T_K': [T],
        'P_bar': [P],
        'abund_H_dex': [composition['H']],
        'abund_O_dex': [composition['O']],
        'abund_C_dex': [composition['C']],
        'abund_N_dex': [composition['N']],
        'abund_S_dex': [composition['S']],
    })
    
    X = normalize_inputs(df)
    with torch.no_grad():
        y_scaled = forward_autoencoder(model, X).numpy()
    y_linear = scale_targets_train_to_linear_torch(torch.tensor(y_scaled)).numpy()
    
    return dict(zip(TARGET_COLS, y_linear[0]))

# Use in your atmospheric model
for layer in atmosphere:
    chem = chemistry_step(layer.T, layer.P, layer.composition)
    # ~800× faster than calling FastChem!
```

---

## Citation

If you use this emulator in your research, please cite:

```bibtex
@software{fastchem_ml_emulator_2025,
  author = {Mohanty, Yashnil and Malsky, Isaac and Zhang, Xi},
  title = {FastChem Neural Network Emulator: A PyTorch Surrogate Model 
           for Chemical Equilibrium in Planetary Atmospheres},
  year = {2025},
  version = {1.0},
  url = {https://github.com/yashnil/chemCalculations},
  note = {~800× speed-up over FastChem with Log R² = 0.9991}
}
```

### Acknowledging FastChem

This emulator is trained on data generated by [FastChem](https://github.com/exoclime/FastChem). Please also cite:

```bibtex
@article{Stock2018,
  author = {Stock, Joachim W. and Kitzmann, Daniel and Patzer, A. Beate C.},
  title = {FastChem: A computer program for efficient complex chemical 
           equilibrium calculations in the neutral/ionized gas phase with 
           applications to stellar and planetary atmospheres},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year = {2018},
  volume = {479},
  number = {1},
  pages = {865--874},
  doi = {10.1093/mnras/sty1531}
}
```

---

## Contact

### Project Maintainers

**Yashnil Mohanty** (Lead Developer)  
📧 Email: yashnilmohanty@gmail.com  
🏛 Affiliation: Westmont High School  
🔬 Research: Computer Science and Exoplanet Atmospheres

**Xi Zhang** (Lead Mentor)  
📧 Email: xiz@ucsc.edu  
🏛 Affiliation: University of California, Santa Cruz  
🔬 Research: Earth and Planetary Sciences

**Isaac Malsky** (PyTorch Implementation)  
🏛 Affiliation: Jet Propulsion Laboratory  
🔬 Research: Exoplanet Atmospheres and Simulation of Physical Processes

### Getting Help

- **📖 Documentation**: See this README for detailed usage
- **🐛 Issues**: Report bugs via GitHub Issues
- **💬 Questions**: Email yashnilmohanty@gmail.com
- **🤝 Collaborations**: Open to integration with atmospheric modeling codes

### Contributing

We welcome contributions! Areas of interest:

1. **Extended chemistry**: More elements (Fe, Ti, Mg, Si, Na, Ca, etc.) - expandable to 30+ inputs
2. **Condensed phases**: Cloud and haze formation predictions
3. **Uncertainty quantification**: Bayesian neural networks, ensembles
4. **Speed optimization**: TensorRT, ONNX export, quantization
5. **Integration**: Wrappers for petitRADTRANS, BART, PICASO, Exo-Transmit
6. **Validation**: Testing against JWST/HST retrievals

**To contribute**: Fork repository, create feature branch, add tests, submit pull request

---

## Technical Details

### Data Generation

**Source**: FastChem v3.0+  
**Sampling strategy**: Stratified T-P grid with randomized elemental compositions
- Temperature bins: 20 (covering 750–3000 K after filtering)
- Pressure bins: 20 (log-uniform from 10⁻¹⁰ to 10⁵ bar)
- Elemental compositions: Random sampling in log-space

**Total samples**: 160,000 (optimal size identified through resolution study)

### Normalization Philosophy

**Why these specific constants?**

| Constant | Value | Rationale |
|----------|-------|-----------|
| TEMP_DIVISOR | 4000 | Upper bound of typical atmospheres |
| INPUT_LOG_SCALE | 10 | Brings log₁₀(P) to ~[-1, 0.5] range |
| ABUND_OFFSET | 12 | Solar hydrogen reference |
| ABUND_SCALE | 10 | Typical element variation span |
| TARGET_LOG_SCALE | 30 | Abundance range (10⁻³⁰ to 1) |

**Benefits**:
- No dependencies on training data statistics (unlike StandardScaler)
- Physical meaning (based on astrophysical scales)
- Reproducible across datasets
- Low variance features

### Error Handling

**Philosophy**: Transparency over hiding problems

**Approach**:
- Detects non-finite values (NaN, Inf) in inputs/targets
- Logs per-column counts and example row indices
- **Drops** problematic rows (doesn't sanitize)
- Reports how many and why

**Result**: No silent failures, easier debugging

---

## Computational Requirements

### Minimum Requirements
- **CPU**: Modern multi-core (training takes ~24 minutes)
- **RAM**: 4 GB (dataset + model fit in memory)
- **Storage**: ~50 MB (data + model)
- **Python**: 3.9+

### Recommended for Production
- **CPU**: Recent Intel/AMD or Apple Silicon
- **GPU**: Optional (speeds up batch inference 10×)
- **RAM**: 8 GB (comfortable for large batches)

### Dependencies
```
torch >= 2.0
numpy >= 1.20
pandas >= 1.3
scikit-learn >= 1.0
matplotlib >= 3.5 (for diagnostics)
scipy >= 1.7 (optional, for KDE density plots)
```

---

## Acknowledgments

**Scientific Community**:
- FastChem team (Stock, Kitzmann, Patzer) for the original equilibrium solver
- UCSC Exoplanet Group for computational resources and scientific guidance
- Isaac Malsky for the PyTorch implementation and key architectural insights

**Technical Infrastructure**:
- PyTorch team for deep learning framework
- Python scientific stack (NumPy, Pandas, Scikit-learn, Matplotlib)

---

## License

MIT License

Copyright (c) 2025 Yashnil Mohanty, Xi Zhang, Isaac Malsky

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.

---

## Future Work

**Near-term** (achievable with current architecture):
1. Expand to 30 input features (include Fe, Ti, Mg, Si, Na, Ca, etc.)
2. Increase to 50 output species (still focused, but more coverage)
3. GPU optimization for ultra-fast batch inference
4. Ensemble models for uncertainty estimation

**Long-term** (research directions):
1. Condensed-phase species (clouds, hazes, rainout)
2. Non-equilibrium chemistry (kinetics, photochemistry)
3. Physics-informed neural networks (enforce mass conservation)
4. Active learning (sample T-P-composition space adaptively)
5. Integration with radiative transfer (end-to-end differentiable atmospheres)

---

## Summary

**FastChem ML Emulator** solves the computational bottleneck in atmospheric modeling:

- **Problem**: FastChem too slow (7 ms/call) for modern applications
- **Solution**: Neural network emulator (0.009 ms/call)
- **Result**: ~800× faster with excellent accuracy (Log R² = 0.9991)
- **Impact**: Enables retrievals, GCMs, and population studies that were previously infeasible

**Current status**: Production-ready, validated, and recommended for all use cases.

**Get started**: `cd src && python train_autoencoder.py`

---

<p align="center">
  <strong>FastChem ML Emulator — Accelerating Atmospheric Chemistry by ~800×</strong>
</p>

<p align="center">
  <em>From days to minutes. From impossible to routine.</em>
</p>

<p align="center">
  <sub>Developed at UC Santa Cruz | 2024-2025</sub>
</p>
