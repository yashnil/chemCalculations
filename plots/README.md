# Plots Directory

This directory contains figures and data files produced by the chemCalculations ML emulator pipeline. Below is a description of each plot and data file.

---

## Model Performance & Diagnostics

### `parity_overall.png`
**Purpose:** Predicted vs. true number densities for all species pooled together. Each point is one (sample, species) pair. Points along the diagonal indicate accurate predictions. Units: cm⁻³.

### `parity_top10.png`
**Purpose:** Parity plots for the 10 most abundant species individually. Each subplot shows predicted vs. true for one species. Useful for spotting species-specific biases.

### `scatter_optimal_model.png`
**Purpose:** Scatter plot of predicted vs. observed abundances for the best model (typically x4800_improved). Includes Log MAE and Log R² in the title.

### `MAE_per_species.png`
**Purpose:** Mean Absolute Error (MAE) per species, sorted by error. Red bars indicate species above the global average. Helps identify which species the model struggles with most.

### `AAFE_per_species.png`
**Purpose:** Average Absolute Fractional Error (AAFE) per species. AAFE = mean(|pred − true| / true) for each species. Red bars indicate above-average fractional error. Useful for understanding relative errors across species with very different abundance scales.

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
**Purpose:** Summary of global metrics for the best model: Linear MAE, Linear R², Log MAE, Log R², AAFE, test sample count, species count.

### `comparison_metrics.csv`
**Purpose:** Aggregated metrics for all compared runs (dataset, val_loss, test_loss, log_mae, log_r2, etc.). Used by comparison and scaling plots.

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
