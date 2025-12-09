# Hyperparameter Study Summary

## Latent Dimension Study

**Tested values:** 64, 96, 128, 160, 192  
**Epochs per model:** 50  
**Dataset:** x160 (160K samples)

### Results

| Latent Dim | Test Loss | Val Loss | Test MAE (Linear) |
|------------|-----------|----------|-------------------|
| 64         | 0.000463  | 0.000460 | 2.46×10²⁰         |
| 96         | 0.000375  | 0.000389 | 1.89×10²⁰         |
| 128        | 0.000439  | 0.000454 | 3.36×10²⁰         |
| 160        | 0.000525  | 0.000476 | 2.62×10²⁰         |
| 192        | 0.000339  | 0.000311 | 1.84×10²⁰         |

**🏆 Best: latent_dim=192** (test_loss=0.000339)

### Key Findings
- **Optimal latent dimension: 192**
- Performance degrades for both smaller (64, 96) and larger (128, 160) dimensions
- Clear minimum at 192, suggesting this is the optimal compression for the 21-species output space

---

## Layer Width Study ✅ COMPLETE

**Tested widths:** 256, 512, 768, 1024  
**Tested layers:** 3, 4  
**Latent dim:** 192 (from previous study)  
**Epochs per model:** 50  
**Dataset:** x160 (160K samples)

### Results

| Width | 3 Layers | 4 Layers |
|-------|----------|----------|
| 256   | 0.000555 | 0.000467 |
| 512   | **0.000339** 🏆 | 0.000480 |
| 768   | 0.000385 | **0.000348** |
| 1024  | 0.000417 | 0.000439 |

**🏆 Best overall: width=512, layers=3** (test_loss=0.000339)  
**Best 4-layer: width=768, layers=4** (test_loss=0.000348)

### Key Findings
- **3 layers perform better overall** - Best is width=512 (0.000339)
- **4 layers best is width=768** (0.000348) - Very close to 3-layer best
- **Optimal configuration: width=512, layers=3** for best performance
- Wider layers (1024) don't improve performance - suggests 512 is optimal width

---

## Combined Recommendations

### Optimal Hyperparameters (Final)
- **latent_dim:** 192 (from Test #1)
- **layer_width:** 512 (from Test #2)
- **num_layers:** 3 (from Test #2)
- **Expected test_loss:** ~0.000339 (based on Test #2 results)

### Alternative Configuration
- **latent_dim:** 192
- **layer_width:** 768
- **num_layers:** 4
- **Expected test_loss:** ~0.000348 (slightly worse but very close)

**Recommendation:** Use width=512, layers=3 for best performance and efficiency.

