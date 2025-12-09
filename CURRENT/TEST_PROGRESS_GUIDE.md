# Hyperparameter Study Progress Guide

## How to Monitor Test #2 (Layer Width Test)

### Option 1: Quick Status Check
```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/CURRENT
python plot_layer_width_results.py
```

This will show:
- ✅ Completed models with their test_loss values
- ⏳ Models still training
- ⏸️ Models not started yet
- Updated plot: `layer_width_study.png`

### Option 2: Interactive Monitor
```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/CURRENT
./monitor_layer_width_test.sh
```

This provides a live-updating display that refreshes every 30 seconds.

### Option 3: Manual Check
```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/CURRENT
# Check individual models
ls -la runs_autoencoder_width*/summary.json

# Or check all at once
for width in 256 512 768 1024; do
  for layers in 3 4; do
    if [ -f "runs_autoencoder_width${width}_layers${layers}/summary.json" ]; then
      echo "✅ width=${width}, layers=${layers}"
    else
      echo "⏳ width=${width}, layers=${layers}"
    fi
  done
done
```

### Current Progress (as of last check)
- ✅ width=256, layers=3: test_loss=0.000555
- ✅ width=512, layers=3: test_loss=0.000339 (best so far)
- ✅ width=768, layers=3: test_loss=0.000385
- ⏳ Remaining: 5 models

---

## Test #3: Dataset Size Study (Ready to Run)

### Optimal Hyperparameters (from Tests #1 and #2)
- **latent_dim**: 192 (best from latent dim study)
- **layer_width**: 512 (best from layer width study so far)
- **num_layers**: 3 (best from layer width study so far)

*Note: These may be updated once Test #2 fully completes*

### To Run Test #3
```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/CURRENT

# Test all dataset sizes (will take ~23 hours total)
python test_dataset_sizes_optimal.py

# Or test a subset for faster results
python test_dataset_sizes_optimal.py --datasets x64 x96 x128 x160
```

### Expected Outputs
- **Plot**: `dataset_size_study_optimal.png`
- **CSV**: `dataset_size_results_optimal.csv`
- **Run directories**: `runs_autoencoder_optimal_x32`, `runs_autoencoder_optimal_x48`, etc.

### Monitor Test #3 Progress
```bash
# Quick check
python plot_dataset_size_results_optimal.py  # (will create this script)

# Or check manually
ls -la runs_autoencoder_optimal_*/summary.json
```

---

## Summary of All Tests

### Test #1: Latent Dimension Study ✅ COMPLETE
- **Best**: latent_dim=192 (test_loss=0.000339)
- **Plot**: `latent_dim_study.png`
- **Results**: All 5 models (64, 96, 128, 160, 192) completed

### Test #2: Layer Width Study ⏳ IN PROGRESS
- **Current best**: width=512, layers=3 (test_loss=0.000339)
- **Plot**: `layer_width_study.png` (updates as models complete)
- **Progress**: 3/8 models complete
- **Monitor**: Use commands above

### Test #3: Dataset Size Study 📋 READY
- **Status**: Waiting for Test #2 to complete for final optimal params
- **Will test**: x32, x48, x64, x80, x96, x112, x128, x144, x160, x176
- **Script**: `test_dataset_sizes_optimal.py`

---

## Files to Check

### Plots
- `latent_dim_study.png` - Test #1 results
- `layer_width_study.png` - Test #2 results (updates automatically)
- `dataset_size_study_optimal.png` - Test #3 results (after running)

### Results CSVs
- `latent_dim_results.csv` - Test #1 data
- `layer_width_results.csv` - Test #2 data
- `dataset_size_results_optimal.csv` - Test #3 data

### Summary Document
- `hyperparameter_study_summary.md` - Combined summary of all tests

