# Why More Data Can Hurt Performance: Analysis of x64 → x80 Degradation

## The Counterintuitive Result

With **locked target species**, x80 (80k samples) performs **worse** than x64 (64k samples):
- Validation Loss: **+27% worse**
- Test Loss: **+47% worse**
- Log-space MAE: Slightly better (-1.4%)
- Log-space R²: Slightly worse (-0.1%)

## Key Findings

### 1. **Catastrophic Species Failures**
- **C1H4**: R² dropped from 0.78 → **-6.77** (catastrophic failure)
- **Sulfur chains** (S3, S4, S8, O2S1): Significant R² drops (0.10-0.55)
- These failures drive the overall loss increase

### 2. **More Extreme Cases**
- x80 has **993 samples** with C1H4 > 1e20 (vs 857 in x64)
- x80 has **28,482 very small** C1H4 values < 1e-10 (vs 22,979 in x64)
- The model struggles with these extreme ranges

### 3. **Training Dynamics**
- x80: 26,560 gradient steps (133 batches/epoch × 200 epochs)
- x64: 21,245 gradient steps (106 batches/epoch × 200 epochs)
- Same number of epochs, but more data per epoch

## Why This Happens

### **1. Model Capacity vs Data Complexity**
With 25% more data, you're exposing the model to:
- More edge cases and rare combinations
- More extreme values (both very large and very small)
- More diverse chemical regimes

**The model architecture might not have enough capacity** to learn all these patterns. It's essentially **underfitting** the increased complexity.

### **2. Hard Cases Dominate Loss**
Even though log-space MAE improved slightly (better average performance), the **worst cases got much worse**:
- A few catastrophic failures (like C1H4) dominate the loss
- The model is making very bad predictions on rare but extreme cases
- These outliers pull up the overall loss metric

### **3. Training Schedule Mismatch**
- **Fixed 200 epochs** for both datasets
- x80 needs more epochs to converge (more data = more to learn)
- Cosine annealing schedule might not be optimal for larger dataset
- Learning rate might decay too quickly before model learns hard cases

### **4. Numerical Instability**
- Extreme values (C1H4 up to 10²³) can cause numerical issues
- Log-space transformations might amplify errors for extreme cases
- Model might collapse on rare but extreme samples

### **5. Distribution Shift (Subtle)**
While overall distributions look similar, x80 has:
- More extreme high-pressure cases (max P: 2.11e5 vs 1.94e5 bar)
- More samples in tail regions of distributions
- More rare chemical combinations

## Solutions

### **Immediate Fixes**
1. **Increase model capacity**: Larger hidden layers, more parameters
2. **Train longer**: More epochs for x80 (e.g., 250-300 epochs)
3. **Adjust learning rate**: Slower decay or higher initial LR for larger dataset
4. **Robust loss**: Use Huber loss or clip extreme errors to prevent catastrophic failures

### **Data-Level Fixes**
1. **Clip extreme values**: Cap C1H4 and other species at reasonable maximums
2. **Better normalization**: Handle extreme values more carefully
3. **Weighted sampling**: Give more weight to hard cases during training
4. **Data cleaning**: Remove or downweight problematic extreme samples

### **Architecture Fixes**
1. **Species-specific heads**: Different architectures for different species
2. **Hierarchical modeling**: Separate models for common vs rare species
3. **Attention mechanisms**: Focus on relevant features for each sample

## The Paradox Explained

**More data doesn't always help when:**
- The model lacks capacity to learn the increased complexity
- The new data includes more hard/rare cases that are difficult to learn
- Training dynamics aren't adjusted for the larger dataset
- Numerical issues arise from extreme values

**The solution isn't less data—it's better training for the data you have.**

