# Baseline vs Improved Model Comparison

Compare the **previous best** (x4800_optimal_retrained) with the **improved** model (x4800_improved) that incorporates four changes:

1. **AdamW** instead of Adam
2. **Train-only normalization** (weights, target_cols from training split only)
3. **Correct loss naming** (loss_linear not mse) and config-driven loss
4. **Optional SimpleMLP** (x4800_improved uses FlowMap; MLP available via config)

## Step 1: Train the improved model

```bash
python -m chemcalculations.train_autoencoder_improved \
  --config configs/x4800_improved.json \
  --run-dir results/runs/runs_autoencoder_x4800_improved
```

This uses the same x4800 dataset and architecture as the baseline, but with AdamW and train-only normalization.

## Step 2: Run diagnostics on both models

For the baseline (if not already done):

```bash
CSV_PATH=data/datasets/all_gas_fastchem_x4800.csv \
BEST_MODULE=results/runs/runs_autoencoder_x4800_optimal_retrained/best_model.py \
OUT_DIR=results/runs/runs_autoencoder_x4800_optimal_retrained/diagnostics \
  python -m chemcalculations.diagnostics
```

For the improved model (run from project root):

```bash
CSV_PATH=data/datasets/all_gas_fastchem_x4800.csv \
BEST_MODULE=results/runs/runs_autoencoder_x4800_improved/best_model.py \
OUT_DIR=results/runs/runs_autoencoder_x4800_improved/diagnostics \
  python -m chemcalculations.diagnostics
```

Or use the comparison script (runs diagnostics on x4800_improved automatically):

```bash
python scripts/update_comparison_baseline_vs_improved.py
```

## Step 3: Update comparison metrics and plots

```bash
python scripts/update_comparison_baseline_vs_improved.py
```

This writes `plots/comparison_metrics.csv` with both x4800_optimal_retrained and x4800_improved, then regenerates the analysis plots.

## Metrics compared

- **test_loss** – validation loss (normalized space)
- **log_mae** – Log MAE (from diagnostics/global_metrics.txt)
- **log_r2** – Log R²
- **linear_mae**, **linear_mse** – linear-space metrics

## Files

| File | Purpose |
|------|---------|
| `chemcalculations.train_autoencoder_improved` | Improved trainer (4 changes) |
| `configs/x4800_improved.json` | Config for improved model (same as x4800_optimal + AdamW) |
| `scripts/update_comparison_baseline_vs_improved.py` | Build comparison CSV and regenerate plots |
