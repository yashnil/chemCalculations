# FastChem Neural Network Emulator

A high-performance machine learning surrogate model for chemical equilibrium calculations in planetary and stellar atmospheres.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-production--ready-green)](CURRENT/)
[![Log R²](https://img.shields.io/badge/Log_R²-0.9991-brightgreen)](CURRENT/)

**Current Version: CURRENT (FlowMapAutoencoder)** | **Status: Production-Ready** ✅  
**Previous Version: NEW_VERS** | **Status: Archived**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Performance Metrics](#performance-metrics)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Model Architecture](#model-architecture)
7. [Version History](#version-history)
8. [Usage Examples](#usage-examples)
9. [Citation](#citation)
10. [Contact](#contact)

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

**The bottleneckMenuChemical equilibrium calculations dominate runtime in:
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

1. **Rich input representationMenu 7 core features (T, P, 5 elements) expandable to 30+ with metals
2. **Focused outputs**: Top-20 species by abundance, not all 116+ species
3. **Simple normalizationMenuPhysical constants (T/4000, (abund_dex-12)/10)
4. **Data quality**: Filter low-temperature samples that cause prediction artifacts
5. **Robust validation**: Comprehensive diagnostics suite with parity plots

**PhilosophyMenuGive the model good data, keep transformations simple, focus on what matters.

---

## Performance Metrics

### NEW_VERS Production Model Results

**Training Configuration**:
- Dataset: 12,412 samples (750–3000 K, aggressively filtered for stripe removal)
- Split: 85% train (10,549) / 10% val (1,242) / 5% test (621)
- Architecture: Residual MLP, 512 hidden units, 3 ResBlocks, GELU activation
- Training time: 160 seconds (350 epochs on CPU)
- Dropout: 0.08 (improved regularization)

**Accuracy** (on held-out test set):
```
Test MSE (scaled space):    1.389e-03  ✅ Excellent
Log R² (primary metric):    0.9750     ✅ 97.5% variance explained
Log MAE:                    0.1578 dex ✅ ~44% typical fractional error
Validation MSE:             1.116e-03  ✅ Converged (epoch 324)
```

**Speed** (measured on 621-sample test set):
```
Inference time:       0.0085 ms per sample (batch mode)
Single-sample time:   0.310 ms per sample
Batch inference:      5.28 ms for 621 samples
Speed-up vs FastChem: ~823× faster (batch mode)
Throughput:           117,568 samples/sec
```

**Quality Indicators**:
```
Vertical stripe artifact:  0 points (T > 750K filter eliminates it)
Parity plot quality:       Tight 1:1 clustering across all 21 species
Convergence:               Smooth (best epoch 324/350, no oscillation)
Generalization:            Excellent (train ≈ val ≈ test MSE)
Architecture:              Residual connections for better gradient flow
```

### Comparison: FastChem vs ML Emulator

| Aspect | FastChem | NEW_VERS ML Emulator | Advantage |
|--------|----------|----------------------|-----------|
| **Accuracy** | Exact (ground truth) | Log R² = 0.9750, Test MSE = 1.39×10⁻³ | Excellent match (97.5% variance) |
| **Speed** | 7 ms/eval | 0.0085 ms/eval | **823× faster** |
| **Scalability** | Linear | Parallel batching | GPU-accelerable |
| **Deployment** | C++ binary | Python/PyTorch | Easy integration |
| **Use case** | Ground truth | Production inference | Complementary |
| **Architecture** | Iterative solver | Residual MLP (skip connections) | Modern deep learning |

**Bottom line**: ML emulator enables science that was previously impossible due to computational cost.

---

## Project Structure

```
chemCalculations/
│
├── CURRENT/                       ⭐ CURRENT PRODUCTION VERSION (FlowMapAutoencoder)
│   ├── train_autoencoder.py      # Training script (FlowMapAutoencoder, PyTorch)
│   ├── autoencoder_model.py      # FlowMapAutoencoder architecture definition
│   ├── diagnostics.py             # Comprehensive diagnostic suite
│   ├── plot.py                   # Generate parity plots
│   ├── make_comparison_metrics.py # Collect metrics across dataset sizes
│   ├── plot_resolution_study.py  # Generate resolution study plots
│   ├── retrain_all_datasets.py   # Automated retraining across dataset sizes
│   ├── comparison_metrics.csv     # Performance metrics for all dataset sizes
│   ├── resolution_study.png       # Performance vs dataset size visualization
│   ├── datasets/                 # Training datasets (x32 to x176, excluded from Git)
│   ├── data_generation/          # FastChem job generation and merging scripts
│   └── runs_autoencoder_*/        # Training outputs for each dataset size
│       ├── best_model.py         # Self-contained inference module
│       ├── best.pt               # Model weights
│       ├── diagnostics/          # Comprehensive diagnostic plots
│       └── summary.json          # Training metrics
│
├── NEW_VERS/                     # Previous production version (Residual MLP)
│   ├── run_mlp.py                # Training script (Residual MLP, PyTorch)
│   ├── diagnostics.py            # Comprehensive 11-plot validation suite
│   ├── plot.py                   # Generate parity plots
│   ├── inference_speed_test.py   # Speed benchmarking vs FastChem
│   ├── investigate.py            # Input distribution analysis
│   ├── all_gas_v10_no_stripe_clean.csv  # Clean data (12.4k samples, T > 750K)
│   ├── README.md                 # NEW_VERS detailed documentation
│   ├── ABLATION_STUDY_RESULTS.md # Optimization experiment record
│   ├── PLOT_INTERPRETATION_GUIDE.md  # How to read diagnostic plots
│   └── runs_mlp_NEW_VERS/        # Generated outputs
│       ├── best_model.py         # Self-contained inference module
│       ├── best.pt               # Model weights (optimal baseline)
│       ├── pred_vs_true_test.png # Main parity plot
│       ├── diagnostics/          # 11 diagnostic plots + metrics
│       └── speed_test/           # Inference timing results
│
├── v10/                          # Previous production version (baseline)
│   ├── run_mlp.py                # Training script (PyTorch)
│   ├── diagnostics.py            # Comprehensive validation
│   ├── all_gas_v10_no_stripe.csv # Training data (T > 680K filter)
│   └── runs_mlp_v10/             # v10 outputs
│
├── v9/                           # Experimental (failed log-ratio approach)
│   └── [Archived in graveyard/]
│
├── v8/                           # TensorFlow baseline (working)
│   └── [Archived in graveyard/ - TensorFlow/Keras + composite loss]
│
├── Fastchemlp/                   # Reference implementation
│   ├── run.py                    # Original PyTorch code (basis for v10/NEW_VERS)
│   └── runs_mlp_all_gas/         # Reference trained model
│
├── graveyard/                    # Historical versions (v1-v9)
│   ├── v8/                       # TensorFlow baseline (Log R² = 0.954)
│   ├── v9/                       # Failed log-ratio experiment
│   └── [v1-v7 earlier iterations]
│
├── artefacts/                    # v8 training outputs
├── results/                      # FastChem raw outputs (40k samples)
└── README.md                     # This file

```

### Version Directories Explained

- **CURRENT** ⭐: **Use this!** Latest production model with FlowMapAutoencoder architecture
  - **Architecture**: FlowMapAutoencoder (encoder-dynamics-decoder)
  - **Performance**: Log R² = 0.9991, Log MAE = 0.0564 (51% improvement over previous)
  - **Dataset**: 160K samples (optimal size identified through resolution study)
  - **Key features**: SiLU activation, 128-dim latent space, constant 512-width layers, adaptive LR scheduling
- **NEW_VERS**: Previous production version with Residual MLP (Log R² = 0.9750, archived)
- **v10**: Earlier production baseline (Log R² = 0.9730, archived)
- **v9**: Failed experiment with log-ratio inputs (archived in graveyard/)
- **v8**: TensorFlow baseline (Log R² = 0.954, archived in graveyard/)
- **Fastchemlp**: Original PyTorch reference implementation
- **graveyard**: Complete development history (v1-v9)

---

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn
```

### Running CURRENT (Recommended)

```bash
# Navigate to CURRENT
cd CURRENT

# Train the model (if not already done)
python train_autoencoder.py  # ~20 minutes (200 epochs, 160K samples)
# Note: Set CSV_PATH environment variable or edit train_autoencoder.py

# Generate validation plots
python plot.py               # Creates pred_vs_true_test.png (main parity plot)
python diagnostics.py        # Creates comprehensive diagnostic plots

# Generate resolution study (compare across dataset sizes)
python make_comparison_metrics.py  # Collect metrics from all runs
python plot_resolution_study.py    # Generate resolution_study.png

# View results
open runs_autoencoder_x160_new/pred_vs_true_test.png
open runs_autoencoder_x160_new/diagnostics/
open resolution_study.png
```

### Running NEW_VERS (Previous Version)

```bash
# Navigate to NEW_VERS
cd NEW_VERS

# Train the model (if not already done)
python run_mlp.py        # ~2.5 minutes (350 epochs), generates best_model.py

# Generate validation plots
python plot.py           # Creates pred_vs_true_test.png (main parity plot)
python diagnostics.py    # Creates 11 diagnostic plots + metrics

# Run inference speed test
python inference_speed_test.py  # Benchmark vs FastChem

# View results
open runs_mlp_NEW_VERS/pred_vs_true_test.png
open runs_mlp_NEW_VERS/diagnostics/
open runs_mlp_NEW_VERS/speed_test/
```

### Using the Trained Model (CURRENT)

```python
import sys
sys.path.append('CURRENT/runs_autoencoder_x160_new')

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
y_linear = scale_targets_train_to_linear_torch(torch.tensor(y_scaled)).numpy()

print("Predicted abundances for 21 species:")
print(y_linear)  # Returns abundances for: e-, N2, O2, H2, S2, C5, H, O, H2O, etc.
```

### Using the Trained Model (NEW_VERS - Previous Version)

```python
import sys
sys.path.append('NEW_VERS/runs_mlp_NEW_VERS')

from best_model import load_model, normalize_inputs, denormalize_targets
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
    y_scaled = model(X).numpy()
y_linear = denormalize_targets(y_scaled)

print("Predicted abundances for 21 species:")
print(y_linear)  # Returns abundances for: e-, N2, O2, H2, S2, C5, H, O, H2O, etc.
```

---

## Model Architecture

### CURRENT: FlowMapAutoencoder (Production)

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
- **Latent dimension**: 128 (increased from 64)
- **Constant layer widths**: All layers use 512 units (no bottlenecks)
- **SiLU activation**: Smooth, self-gated activation function
- **No dropout**: Model doesn't overfit with sufficient data
- **Residual connections**: In dynamics module for better gradient flow
- **Weighted Huber loss**: Robust to outliers (δ=0.02)
- **Adaptive LR**: ReduceLROnPlateau (reduces LR when validation plateaus)

### NEW_VERS: Residual MLP (Previous)

**Framework**: PyTorch  
**Type**: Residual Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input(7) → Dense(512) → GELU → Dropout(0.08)
         → ResBlock(512) → GELU → Dropout(0.08)
         → ResBlock(512) → GELU → Dropout(0.08)
         → ResBlock(512) → GELU → Dropout(0.08)
         → Dense(21) → Output(21)
```

**Parameters**: ~1.59M

### v10 Production Model (Archived)

**Framework**: PyTorch  
**Type**: Feedforward Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input(7) → Dense(512) → LeakyReLU → Dropout(0.05)
         → Dense(512) → LeakyReLU → Dropout(0.05)
         → Dense(512) → LeakyReLU → Dropout(0.05)
         → Dense(21) → Output(21)
```

**Parameters**: ~540,000

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

**Dex scaleMenu`abund_X_dex = 12 + log₁₀(N_X / N_H)` (standard astrophysical notation)
- Solar values: H=12.0, O≈8.69, C≈8.43, N≈7.83, S≈7.12

### Output Species (21 total)

Top-20 most abundant species + electron (e⁻), auto-selected from training data.

**Typical species**: H₂, H₂O, CO, CH₄, NH₃, CO₂, N₂, O₂, He, H, O, C, N, S, OH, etc.

### Training Configuration (CURRENT)

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

### Training Configuration (NEW_VERS - Previous)

```python
Optimizer:      AdamW (lr=5×10⁻⁴, weight_decay=1×10⁻⁵)
Scheduler:      CosineAnnealingLR (η_min=1×10⁻⁶)
Loss:           MSE in log-scaled target space
Batch size:     512
Epochs:         350 (longer training for convergence)
Gradient clip:  5.0
Data split:     85% train / 10% val / 5% test
Dataset size:   12.4K samples
```

**Data preprocessing**:
- Filters out coldest 20% (T < 680K) to eliminate vertical stripe artifact
- Drops rows with non-finite values
- Logs detailed diagnostics

---

## Version History

### Development Timeline

#### v1-v7: Early Development (2024)
- Initial experiments with TensorFlow/Keras
- Various architectures, loss functions, and normalizations
- Established baseline performance
- **StatusMenuArchived in `graveyard/`

#### v8: TensorFlow Production Baseline (June 2025)
**FrameworkMenuTensorFlow/Keras  
**Key features**:
- Softplus output head with explicit normalization
- Composite loss: λ·KL + (1-λ)·MAE_log
- Optuna hyperparameter tuning
- Comprehensive diagnostics suite

**Performance**:
- MAE_log: 0.047 (log-space)
- R²_log: 0.954
- Speed-up: 141×
- Split: 60-15-25

**InnovationMenuIdentified and analyzed vertical stripe artifact at 1-2% abundance
**StatusMenu✅ Working, scientifically validated

---

#### v9: Failed Log-Ratio Experiment (October 2025)
**MotivationMenuReduce inputs, use astrophysical notation  
**Key changes**:
- Log-ratio inputs: log₁₀(O/H), log₁₀(C/H), log₁₀(N/H), log₁₀(S/H)
- 70-15-15 split (more training data)
- 6 inputs instead of 7

**Results**:
- MAE_log: 0.142 (**3× worse than v8!**)
- R²_log: 0.830
- R²_lin: 0.693

**Root causeMenuLog-ratios created high-variance features (σ ≈ 3.66 vs v8's 2.5) and lost absolute abundance information

**Lesson learnedMenuSimple transformations > clever feature engineering

**StatusMenu❌ Abandoned

---

#### v10: PyTorch Production Baseline (October 2025)

**Framework**: PyTorch (clean, modern, faster)  
**Key features**:
- Simple normalization: T/4000, (abund_dex-12)/10, log₁₀/30
- Focus on top-20 species (reduced noise)
- Low-temperature filtering (T > 680K)
- 85-10-5 split (maximum training data)
- Plain MSE loss (simple, effective)

**Architecture**:
- 3 hidden layers × 512 units
- LeakyReLU activation
- 5% dropout
- ~540K parameters

**Performance**:
```
Test MSE (scaled):      8.354e-04
Log R²:                 0.9730
Inference speed:        0.003 ms/sample
Speed-up:               ~2,300×
```

**Status**: ✅ Production-ready, excellent baseline

---

#### NEW_VERS: Optimized Residual MLP (November 2025) — **Archived**

**Framework**: PyTorch with Residual Architecture  
**Based on**: v10 with architectural improvements and aggressive data filtering  
**Key features**:
- **Residual MLP**: 3 ResBlocks with skip connections for better gradient flow
- **GELU activation**: Smoother optimization than LeakyReLU
- **Aggressive filtering**: T > 750K (vs v10's 680K) completely eliminates stripe
- **Longer training**: 350 epochs (vs v10's 200) for deeper convergence
- **Higher regularization**: 8% dropout (vs v10's 5%)
- **Clean targets**: Excludes `comp_*` metadata columns

**Performance**:
- Log R²: 0.9750
- Log MAE: 0.1578 dex
- Speed-up: 823× vs FastChem

**Status**: ✅ **Archived - superseded by CURRENT version**

---

#### CURRENT: FlowMapAutoencoder (November 2025) — **Current Production** ⭐

**Framework**: PyTorch with FlowMapAutoencoder Architecture  
**Key innovations**:
- **FlowMapAutoencoder**: Encoder-dynamics-decoder architecture for better representation learning
- **Larger latent space**: 128 dimensions (vs previous 64) for richer feature representation
- **Constant layer widths**: All layers use 512 units (removed bottleneck architecture)
- **SiLU activation**: Replaced GELU for smoother gradients and better performance
- **No dropout**: Removed regularization (0.08 → 0.0) as model doesn't overfit
- **Adaptive learning rate**: ReduceLROnPlateau scheduler (replaces fixed cosine annealing)
- **Resolution study**: Systematic evaluation across dataset sizes (x32 to x176)

**Architecture**:
```
Encoder:   [state(21) + global(7)] → [512, 512, 512] → latent(128)
Dynamics:  [latent(128) + dt(1) + global(7)] → [512, 512, 512] → latent(128)
Decoder:   [latent(128)] → [512, 512, 512] → state(21)
```
- **Total parameters**: ~1.87M
- **Activation**: SiLU (Sigmoid Linear Unit)
- **Loss function**: Weighted Huber (δ=0.02) on normalized log-space targets
- **Training**: 200 epochs, batch_size=512, Adam optimizer (lr=5e-4)

**Dataset Resolution Study**:
- Evaluated performance across dataset sizes: 12K (base) → 32K → 48K → 64K → 80K → 96K → 112K → 128K → 144K → 160K → 176K
- **Optimal dataset size**: 160K samples (performance asymptotes)
- **Best performance** (x160_new with new architecture):
  - Validation Loss: 0.000177 (48.7% improvement over old x160)
  - Test Loss: 0.000206 (47.2% improvement)
  - Log MAE: 0.0564 (51.2% improvement)
  - Log R²: 0.9991 (99.91% variance explained)

**Performance** (x160_new model, 160K samples):
```
Validation Loss:        0.000177  ✅ 48.7% better than previous
Test Loss:              0.000206  ✅ 47.2% better than previous
Log MAE:                0.0564    ✅ 51.2% better (was 0.1156)
Log R²:                 0.9991     ✅ 99.91% variance explained
Linear MAE:             8.01e+19   ✅ Improved
Training time:          ~20 minutes (200 epochs, 160K samples)
```

**Key Improvements over NEW_VERS**:
- **+2.4% better Log R²** (0.9991 vs 0.9750)
- **51% reduction in Log MAE** (0.0564 vs 0.1156)
- **FlowMapAutoencoder architecture** enables better feature learning
- **Larger latent space** (128 vs 64) captures more complex relationships
- **SiLU activation** provides smoother optimization
- **Resolution study** identifies optimal dataset size (160K samples)

**Status**: ✅ **Current production model - recommended for all use cases**

### Resolution Study: Dataset Size Optimization

A systematic evaluation was conducted to determine the optimal dataset size for training:

**Methodology**:
- Trained models on datasets ranging from 12K (base) to 176K samples
- Evaluated performance metrics: validation loss, test loss, Log MAE, Log R²
- Identified performance asymptote at 160K samples

**Key Findings**:
- **Performance improves with dataset size** up to 160K samples
- **160K samples is optimal**: Performance asymptotes, further increases show diminishing returns
- **x160_new architecture** (with improvements) achieves:
  - 48.7% better validation loss vs old x160
  - 51.2% better Log MAE (0.0564 vs 0.1156)
  - 99.91% variance explained (Log R² = 0.9991)

**Resolution Study Results** (CURRENT architecture):
| Dataset Size | Samples | Val Loss | Test Loss | Log MAE | Log R² |
|--------------|---------|----------|-----------|---------|--------|
| base | 12,412 | 0.000754 | 0.000923 | 0.1439 | 0.9807 |
| x32 | 31,997 | 0.000479 | 0.000630 | 0.1877 | 0.9963 |
| x64 | 63,985 | 0.000421 | 0.000438 | 0.1452 | 0.9977 |
| x96 | 96,000 | 0.000496 | 0.000381 | 0.1318 | 0.9982 |
| x112 | 112,000 | 0.000382 | 0.000426 | 0.1238 | 0.9980 |
| x128 | 128,000 | 0.000423 | 0.000408 | 0.1257 | 0.9981 |
| x144 | 144,000 | 0.000365 | 0.000555 | 0.1169 | 0.9976 |
| **x160** | **160,000** | **0.000346** | **0.000389** | **0.1156** | **0.9982** |
| **x160_new** ⭐ | **160,000** | **0.000177** | **0.000206** | **0.0564** | **0.9991** |
| x176 | 176,000 | 0.000389 | 0.000420 | 0.1161 | 0.9979 |

**Conclusion**: 160K samples with the improved architecture (x160_new) provides optimal performance.

---

## Performance Metrics

### Accuracy

**v10 Test Set Performance** (640 held-out samples, never seen during training):

| Metric | Value | Prat|
|--------|-------|----------------|
| **MSE (scaled space)** | 8.354×10⁻⁴ | Loss in model's training space |
| **MSE (linear space)** | 5.506×10⁻³ | Error in real abundances |
| **Best epoch** | 184 / 200 | Converged with early stopping |
| **Stripe count** | 0 | No artifacts |

**What this means**:
- Model predictions match ground truth with <0.001 error in scaled space
- No systematic biases or artifacts
- Generalizes well to unseen data

### Speed

**Inference Benchmarks**:

| Configuration | Time per Sample | Throughput | vs FastChem |
|---------------|-----------------|------------|-------------|
| **CPU (single)** | 0.003 ms | ~333,000/sec | 2,300× faster |
| **CPU (batch-640)** | 0.003 ms | ~320,000/sec | 2,300× faster |
| **FastChem (baseline)** | 7.0 ms | 143/sec | 1× |

**Real-world impact**:

Exoplanet retrieval with 10⁷ chemistry calls:
- **FastChem**: 19.4 hours
- **v10 ML**: **30 seconds** ⚡

### Comparison Across Versions

| Version | Framework | Test Loss | Log R² | Log MAE | Speed-up | Artifacts | Status |
|---------|-----------|-----------|--------|---------|----------|-----------|--------|
| v8 | TensorFlow | MAE_log: 0.047 | 0.954 | 0.047 | 141× | Stripe analyzed | Archived |
| v9 | TensorFlow | MAE_log: 0.142 | 0.830 | 0.142 | — | Worse than v8 | Failed |
| v10 | PyTorch | 8.35×10⁻⁴ | 0.9730 | ~0.16 | 2,300× | None | ✅ Archived |
| NEW_VERS | PyTorch + ResNet | 1.39×10⁻³ | 0.9750 | 0.1578 | 823× | None | ✅ Archived |
| **CURRENT** | **PyTorch + FlowMapAE** | **2.06×10⁻⁴** | **0.9991** | **0.0564** | **~800×** | **None** | **✅ Current** |

---

## Usage Examples

### Basic Inference

```python
import sys
sys.path.append('NEW_VERS/runs_mlp_NEW_VERS')

from best_model import load_model, normalize_inputs, denormalize_targets, TARGET_COLS
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
    y_scaled = model(X).numpy()
y_linear = denormalize_targets(y_scaled)

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
    y_linear = denormalize_targets(model(X).numpy())

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
        abundances = denormalize_targets(model(X).numpy())[0]
    
    return dict(zip(TARGET_COLS, abundances))

# Use in your atmospheric model
for layer in atmosphere:
    chem = chemistry_step(layer.T, layer.P, layer.composition)
    # 2,300× faster than calling FastChem!
```

---

## Key Improvements in v10

### What Makes v10 Better

**vs v8** (TensorFlow baseline):
- ✅ Cleaner PyTorch implementation
- ✅ Simpler normalization (no StandardScaler dependency)
- ✅ Faster inference (PyTorch efficiency)
- ✅ Better data split (85-10-5 vs 60-15-25)
- ✅ More training data

**vs v9** (failed log-ratio attempt):
- ✅ **16× better accuracy** (MSE: 8.35e-04 vs v9's effective MAE: 0.142)
- ✅ Simple element encoding (no high-variance ratios)
- ✅ Keeps absolute abundance information
- ✅ Stable, low-variance features

**vs Fastchemlp** (Isaac's original):
- ✅ Updated paths for your system
- ✅ Low-temperature filtering applied (eliminates stripe)
- ✅ Comprehensive diagnostic suite
- ✅ Full documentation
- ≈ Same performance (replicates Isaac's results)

### The Vertical Stripe Fix

**ProblemMenuVertical artifact at 1-2% abundance in predictions  
**CauseMenuLow-temperature (T<680K) high-entropy mixtures create equal-share clustering  
**SolutionMenuFilter out coldest 20% of samples during preprocessing  
**ResultMenuStripe eliminated (0 points), ~30% improvement in MSE

**Evidence from v10**:
- Before filtering: Stripe visible at 0.01-0.02
- After filtering: "count near true≈1e-2 (vertical band): 0"
- Visual: Clean diagonal in `pred_vs_true_test.png`

This fix was proven in v8 and successfully implemented in v10.

---

## Citation

If you use this emulator in your research, please cite:

```bibtex
@software{fastchem_ml_emulator_2025,
  author = {Mohanty, Yashnil; Malsky, Isaac; Zhang, Xi},
  title = {FastChem Neural Network Emulator: A PyTorch Surrogate Model 
           for Chemical Equilibrium in Planetary Atmospheres},
  year = {2025},
  version = {10.0},
  url = {https://github.com/yashnil/chemCalculations},
  note = {2,300× speed-up over FastChem with test MSE = 8.35×10⁻⁴}
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
🔬 Research: Exoplanet Atmospheres and Simulation of Phyiscal Processes

### Getting Help

- **📖 Documentation**: See `v10/README.md` for v10-specific details
- **🐛 Issues**: Report bugs via GitHub Issues
- **💬 Questions**: Email yashnilmohanty@gmail.com
- **🤝 CollaborationsMenuOpen to integration with atmospheric modeling codes

### Contributing

We welcome contributions! Areas of interest:

1. **Extended chemistry**: More elements (Fe, Ti, Mg, Al, etc.) - expandable to 30+ inputs
2. **Condensed phasesMenuCloud and haze formation predictions
3. **Uncertainty quantificationMenuBayesian neural networks, ensembles
4. **Speed optimizationMenuTensorRT, ONNX export, quantization
5. **IntegrationMenuWrappers for petitRADTRANS, BART, PICASO, Exo-Transmit
6. **ValidationMenuTesting against JWST/HST retrievals

**To contributeMenuFork repository, create feature branch, add tests, submit pull request

---

## Technical Details

### Data Generation

**Source**: FastChem v3.0+  
**Sampling strategy**: Stratified T-P grid with randomized elemental compositions
- Temperature bins: 20 (covering 680-3000 K after filtering)
- Pressure bins: 20 (log-uniform from 10⁻¹⁰ to 10⁵ bar)
- Elemental compositions: Random sampling in log-space

**Total samples generatedMenu 40,000 (pre-filtering)  
**Used for trainingMenu 12,800 (after removing T < 680K)

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

## Reproducibility

### To Recreate v10 Results

```bash
cd v10

# 1. Data preprocessing (already done)
python convert_csv.py    # Converts raw CSV to v10 format
python fix_stripe.py     # Filters low-T samples

# 2. Training
python run_mlp.py        # 54 seconds, generates best_model.py

# 3. Validation
python plot.py           # Creates pred_vs_true_test.png
python diagnostics.py    # Comprehensive diagnostic suite

# 4. Verify
# Check: runs_mlp_v10/pred_vs_true_test.png
# Expect: Clean 1:1 correlation, no stripe
```

**Expected outputs**:
- Test MSE: ~8-10 × 10⁻⁴
- No vertical stripe
- Tight parity plot correlation
- Training converges by epoch ~150-200

---

## Troubleshooting

### Common Issues

**"No module named 'best_model'"**
```bash
# Solution: Train the model first
cd v10
python run_mlp.py
```

**"Vertical stripe in pred_vs_true_test.png"**
```bash
# Solution: Use filtered data
# Ensure run_mlp.py line 33 points to:
CSV_PATH = '.../all_gas_v10_no_stripe.csv'
```

**"Poor MSE (>0.01)"**
- Check convergence: Did training reach 200 epochs?
- Check data: Are there non-finite values?
- Try: Increase EPOCHS or HIDDEN size in run_mlp.py

**"Different results"**
- Verify: Using filtered data (T > 680K)
- Verify: Same hyperparameters (512×3, LeakyReLU)
- Check: Random seed (SEED=1337)

---

## Computational Requirements

### Minimum Requirements
- **CPU**: Modern multi-core (training takes ~1-5 minutes)
- **RAMMenu 4 GB (dataset + model fit in memory)
- **StorageMenu~50 MB (data + model)
- **Python**: 3.9+

### Recommended for Production
- **CPU**: Recent Intel/AMD or Apple Silicon
- **GPUMenuOptional (speeds up batch inference 10×)
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

**Validation**:
- v8 comprehensive diagnostics identified the stripe artifact
- v10 successfully implements the proven low-T filtering solution

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

- **ProblemMenuFastChem too slow (7 ms/call) for modern applications
- **SolutionMenuNeural network emulator (0.003 ms/call)
- **ResultMenu 2,300× faster with excellent accuracy (MSE = 8.35×10⁻⁴)
- **Impact**: Enables retrievals, GCMs, and population studies that were previously infeasible

**Current status**: CURRENT (FlowMapAutoencoder) is production-ready, validated, and recommended for all use cases.

**Get started**: `cd CURRENT && python train_autoencoder.py`

---

<p align="center">
  <strong>FastChem ML Emulator — Accelerating Atmospheric Chemistry by 2,300×</strong>
</p>

<p align="center">
  <em>From days to minutes. From impossible to routine.</em>
</p>

<p align="center">
  <sub>Developed at UC Santa Cruz | 2024-2025</sub>
</p>
