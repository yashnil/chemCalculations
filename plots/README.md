# Plots Directory

This directory contains figures and data files produced by the chemCalculations ML emulator pipeline. Below is a description of each plot and data file.

---

## Best model snapshot (x4800_improved)

Values below match `scripts/update_comparison_baseline_vs_improved.py` / `comparison_metrics.csv` (regenerate to refresh).

| Metric | Value |
|--------|--------|
| **Run** | `x4800_improved` |
| **Test loss** | ≈ 7.27×10⁻³ |
| **Log MAE** | ≈ 0.00391 dex |
| **Log R²** | ≈ 0.9999 |
| **MFAE** | ≈ **0.0105** (winsorized mean \|pred−true\|/true over scatter dots; cap 2.0 per pair) |

**Reference (same 4800K-scale test set):** x4800_optimal_retrained MFAE ≈ 0.0179; x4800_mlp MFAE ≈ 0.33.

---

## Model Performance & Diagnostics

### `parity_overall.png`
**Purpose:** Predicted vs. true number densities for all species pooled together. Each point is one (sample, species) pair. Points along the diagonal indicate accurate predictions. Units: cm⁻³.

### `parity_top10.png`
**Purpose:** Parity plots for the 10 most abundant species individually. Each subplot shows predicted vs. true for one species. Useful for spotting species-specific biases.

### `scatter_optimal_model.png`
**Purpose:** Scatter plot of predicted vs. observed abundances for the best model (**x4800_improved**). Title includes Log MAE, Log R², and MFAE (winsorized mean fractional error).

### `MAE_per_species.png`
**Purpose:** Mean Absolute Error (MAE) per species, sorted by error. Red bars indicate species above the global average. Helps identify which species the model struggles with most.

### `AAFE_per_species.png`
**Purpose:** Average Absolute Fractional Error (AAFE) per species. AAFE = mean(|pred − true| / true) for each species, computed only for samples where true ≥ 1e-10 (avoids explosion for rare species). Red bars indicate above-average fractional error. Gray = species with no samples above threshold.

### `residual_vs_observed.png`
**Purpose:** Residuals (predicted − true) vs. observed abundance. Hexbin density plot. Horizontal band around zero indicates good calibration across the abundance range.

### `error_distribution.png`
**Purpose:** Histograms of residuals in linear space (left) and log space (right). Shows whether errors are symmetric and how they are distributed.

### `hist_obs_*.png` (e.g. `hist_obs_H2.png`, `hist_obs_N2.png`)
**Purpose:** Distribution of observed abundances for the top-5 most abundant species. Helps understand the data range and typical values for major species.

---

## FastChem-Style Plots

### `mixing_ratio_vs_T.png`
**Purpose:** Gas-phase number densities vs. temperature at fixed pressure (default 0.5 bar), solar composition. Mirrors Figure 1 of FastChem Cond (Kitzmann et al. 2023). Shows how key species (H2, H2O, N2, CO, CH4, O2, CO2, H2S, NH3) vary with T.

### `mixing_ratio_heatmap_TP.png`
**Purpose:** 2D heatmap of log₁₀(number density) for a key species (e.g. H2O) vs. temperature and pressure. Visualizes how abundance changes across the T–P plane.

---

## Training & Scaling

### `loss_curves_full_suite.png`
**Purpose:** Training and validation loss vs. epoch for multiple dataset sizes (x800 through x4800). Shows convergence behavior.

### `log_mae_curves_full_suite.png`
**Purpose:** Validation Log MAE vs. epoch for multiple dataset sizes. Log MAE (in dex) is the primary metric for chemical abundance accuracy.

### `loss_curves_all_sizes.png`
**Purpose:** Loss curves for all resolution runs in a single view. Used for scaling studies.

### `performance_vs_size_full_suite.png`
**Purpose:** Test metrics (Log MAE, Log R²) vs. dataset size. Shows how model quality improves with more training data.

### `performance_vs_size_comprehensive.png`
**Purpose:** Performance vs. dataset size with asymptote analysis. Includes fitted curves to estimate saturation performance.

### `asymptote_analysis.png`
**Purpose:** Asymptotic behavior of metrics as dataset size increases. Used to estimate performance limits.

---

## Model Comparison

### `baseline_vs_improved_bar.png`
**Purpose:** Bar chart comparing x4800_optimal_retrained (baseline) vs. x4800_improved on test loss, Log MAE, and Log R².

### `baseline_vs_improved_performance.png`
**Purpose:** Performance curves (Log MAE, Log R²) vs. dataset size for baseline and improved models.

### `model_comparison_bar.png`
**Purpose:** Bar chart comparing multiple models (e.g. x800 through x4800) on key metrics.

### `compare_x4800_three_metrics.png`
**Purpose:** Bar chart comparing x4800_optimal_retrained (baseline), x4800_improved (best FlowMap), and x4800_mlp (6×1024 MLP) on test loss, Log MAE, and Log R².

### `compare_x4800_three_parity.png`
**Purpose:** 3-panel parity plot (predicted vs true) for each of the three x4800 models on the same test set. Direct visual comparison of prediction quality.

### `compare_x4800_three_per_species.png`
**Purpose:** Per-species Log MAE comparison for the top 20 species by abundance across the three x4800 models. Grouped bar chart.

---

## Speed Benchmarks

### `bench_throughput.png`
**Purpose:** Throughput (samples/second) vs. batch size for the ML emulator (CPU and GPU). Includes FastChem line-by-line baseline. Shows speedup of the emulator over FastChem.

### `bench_throughput.csv`
**Purpose:** Raw throughput data: batch size, CPU samples/sec, FastChem baseline, speedup factor.

### `speed_benchmark_optimized.csv`
**Purpose:** Inference speed benchmark results (ms/sample, speedup vs. FastChem) from `fast_inference.py`.

### `speed_benchmark_onnx.csv`
**Purpose:** ONNX export benchmark results (if ONNX export was run).

---

## Data Files

### `global_metrics.txt`
**Purpose:** Summary of global metrics for the diagnostic run (copied from **x4800_improved** when using `update_comparison_baseline_vs_improved.py`): Linear MAE, Linear R², Log MAE, Log R², AAFE, **MFAE**, **MFAE_median**, test sample count, species count.

### `comparison_metrics.csv`
**Purpose:** Aggregated metrics for all compared runs (dataset, val_loss, test_loss, log_mae, log_r2, **mfae**, etc.). Used by comparison and scaling plots.

**`mfae`:** Mean fractional absolute error over all parity/scatter points where both true and predicted abundance exceed 1e−10 cm⁻³: mean of \|pred−true\|/true, with each pair capped at 2.0 before averaging (winsorized mean) so a few catastrophic outliers do not dominate the value. Raw arithmetic mean of the same fractions is unstable. See `src/mfae_metrics.py`. Diagnostics also report **MFAE_median** (typical fractional error per point).

### `per_species_errors.csv`
**Purpose:** Per-species metrics: MAE, R², AAFE, max/mean abundance for each predicted species.

### `worst100_samples.csv`
**Purpose:** The 100 test samples with highest per-sample MAE. Includes T_K, P_bar, abundances, and sample_MAE for debugging.

### `diagnostic_summary.txt`
**Purpose:** Text summary of the diagnostic run: global metrics, top species by abundance/error, and list of generated plots.

---

## Subdirectories

### `independent_validation/`
**Purpose:** Outputs from `scripts/independent_validation.py`: parity plots and profiles for Hot Jupiter, Cool Dwarf, T–P grid, C/O sweep, metallicity sweep, etc. Used for out-of-distribution validation.
