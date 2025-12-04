# MSE Loss Experiment Results

## Objective
Test whether plain MSE loss in normalized space could replace the weighted Huber loss currently used for training the FlowMapAutoencoder.

## Experimental Setup

### Model Configuration
- **Architecture**: FlowMapAutoencoder (128-dim latent, 512-wide layers, SiLU activation)
- **Dataset**: x160 (136,000 training samples after split)
- **Training**: 200 epochs, Adam optimizer, ReduceLROnPlateau scheduler
- **Loss Functions Compared**:
  - **Huber** (current): Weighted Huber loss (δ=0.02) with species-specific weights
  - **MSE** (experimental): Plain MSE loss in normalized log-space

## Results

### x160_new (Weighted Huber Loss)
```
Loss Type:     huber
Train Samples: 136,000
Test Loss:     0.000206 (normalized)
Test MAE:      8.01e+19 (linear space)
Log MAE:       0.0564
Log R²:        0.9991
```

### x160_mse (Plain MSE Loss)
```
Loss Type:     mse
Train Samples: 136,000
Test Loss:     0.000505 (normalized)
Test MAE:      6.42e+20 (linear space) ⚠️
Log MAE:       NaN ⚠️
Log R²:        NaN ⚠️
```

## Key Findings

### 1. Severe Numerical Instability with MSE
The plain MSE loss exhibits catastrophic numerical instability:
- **MAE is ~8× worse**: 6.42e+20 vs 8.01e+19
- **Infinite MSE values**: Model produces predictions that overflow when converted back to linear space
- **Volatile training**: Validation MAE oscillates wildly between 10²¹ and 10²⁶ throughout training
- **No meaningful convergence**: Despite normalized loss appearing to decrease, predictions in actual space are unusable

### 2. Loss Curve Analysis
From `loss_curves.png`:
- **Normalized loss space**: Both train and val loss appear to converge smoothly to ~0.0005
- **Linear MAE space**: Validation MAE remains at catastrophically high values (10²¹-10²²) with massive spikes
- **Interpretation**: MSE loss optimizes the wrong objective - it minimizes error in normalized log-space but produces numerically unstable predictions in the original abundance space

### 3. Why Weighted Huber Works Better
The weighted Huber loss provides:
- **Robustness to outliers**: Huber loss (δ=0.02) is less sensitive to extreme errors
- **Species-specific weighting**: Accounts for varying abundance ranges and importance
- **Numerical stability**: Prevents gradient explosion from extreme prediction errors
- **Better generalization**: Log MAE of 0.056 vs NaN demonstrates meaningful learning

## Performance vs Dataset Size

From `performance_vs_size.png`, key trends for Huber loss models:
- **Optimal size**: Performance peaks at 160k samples (x160_new)
- **Log MAE**: Drops from 0.188 (x32) to 0.056 (x160_new) - **70% improvement**
- **Log R²**: Increases from 0.996 (x32) to 0.999 (x160_new)
- **Diminishing returns**: x176 shows slight performance degradation, suggesting 160k is optimal

## Model Comparison

From `model_comparison.png` (comparing x160, x176, x160_new):
- **x160_new** (with architectural improvements) achieves:
  - **51% lower test loss** vs x160
  - **51% lower Log MAE** vs x160
  - Consistently best across all metrics

## Conclusions

### ❌ MSE Loss is NOT Suitable
Plain MSE loss in normalized space:
- Produces numerically unstable predictions
- Fails to learn meaningful abundance patterns
- Results in unusable models despite appearing to converge

### ✅ Weighted Huber Loss is Essential
The current weighted Huber approach:
- Provides stable, meaningful predictions
- Achieves excellent performance (Log R² = 0.999)
- Handles the wide dynamic range of chemical abundances (10⁻³⁰ to 1)

### 📊 Dataset Size Recommendations
- **Optimal**: 160k samples provides best performance
- **Minimum**: 96k+ samples for acceptable accuracy
- **Beyond 160k**: Marginal or negative returns observed

## Recommendations

1. **Keep weighted Huber loss** - MSE is fundamentally unsuitable for this problem
2. **Use 160k training samples** - optimal balance of performance and training cost
3. **Current architecture is near-optimal** - x160_new shows excellent results
4. **Future work**: 
   - Investigate why x176 underperforms (possible data quality issues?)
   - Consider ensemble methods for further improvement
   - Test on low-temperature regime specifically

## Files Generated

- `train_autoencoder.py` - Updated with `--loss-type` argument
- `plot_training_analysis.py` - New visualization script
- `loss_curves.png` - Training dynamics comparison
- `performance_vs_size.png` - Dataset scaling analysis
- `model_comparison.png` - Model configuration comparison
- `comparison_metrics.csv` - Updated with x160_mse results

---

**Date**: December 4, 2025  
**Experiment**: MSE vs Weighted Huber Loss Comparison  
**Conclusion**: Weighted Huber loss is essential for numerical stability and accurate chemical abundance prediction.

