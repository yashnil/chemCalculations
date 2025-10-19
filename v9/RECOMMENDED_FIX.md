# V9 Performance Issue & Recommended Fix

## Problem Identified

The log-ratio transformation creates features with **much higher variance** than v8:

- V8 element features: std ≈ 2.5
- V9 log-ratio features: std ≈ 3.66 (47% larger!)

This is causing poor model performance:
- MAE_log: 0.142 vs 0.047 (v8) — **3× worse**
- R²_log: 0.830 vs 0.954 (v8)

## Root Cause

**V8 transformation** (`log10(comp) + 9`):
- Creates correlated, smooth features
- Values centered in [0, 9]
- Low variance, easy to learn

**V9 transformation** (`log10(comp_X / comp_H)`):
- Ratios swing from -9 to +9 depending on X > H or X < H
- High variance, harder to learn
- Amplifies small composition differences

## Recommended Solutions

### Option 1: Keep Hydrogen as a Feature (RECOMMENDED)

Instead of removing H entirely, keep it and add ratios:

```python
# 7 features total
X["temperature_norm"] = df["temperature"] / T_max
X["log_pressure"] = np.log10(df["pressure"])
X["log_H"] = np.log10(df["comp_H"]) + 9.0          # Keep H!
X["log_O_H"] = np.log10(df["comp_O"] / df["comp_H"])
X["log_C_H"] = np.log10(df["comp_C"] / df["comp_H"])
X["log_N_H"] = np.log10(df["comp_N"] / df["comp_H"])
X["log_S_H"] = np.log10(df["comp_S"] / df["comp_H"])
```

**Why this works:**
- Gives model absolute scale (via log_H) AND relative ratios
- Similar to having both position and velocity in physics
- Only 7 features (same as v8, just different representation)

### Option 2: Use Bounded Ratios

Clip extreme ratios to reduce variance:

```python
def safe_log_ratio(numerator, denominator, clip=(-5, 5)):
    ratio = numerator / denominator
    return np.clip(np.log10(ratio), clip[0], clip[1])

X["log_O_H"] = safe_log_ratio(df["comp_O"], df["comp_H"])
# etc.
```

### Option 3: Revert to V8 Style (SAFEST)

Keep your 70-15-15 split and temperature normalization, but use v8's element encoding:

```python
X["temperature_norm"] = df["temperature"] / T_max  # Keep this
X["log_pressure"] = np.log10(df["pressure"])

# V8 style for elements (7 features total)
X["log_H"] = np.log10(df["comp_H"]) + 9.0
X["log_O"] = np.log10(df["comp_O"]) + 9.0
X["log_C"] = np.log10(df["comp_C"]) + 9.0
X["log_N"] = np.log10(df["comp_N"]) + 9.0
X["log_S"] = np.log10(df["comp_S"]) + 9.0
```

## My Recommendation

**Go with Option 3** for now to validate that the 70-15-15 split and temperature normalization are beneficial. Once that works, you can experiment with Option 1 if you want the astrophysical interpretation.

The pure log-ratio approach (current v9) removes too much information by not including hydrogen's absolute abundance.

## Quick Implementation

I can create a v9b that implements Option 3 if you'd like - it would keep:
- ✅ 70-15-15 split
- ✅ Temperature normalization (T/T_max)
- ✅ V8-style element encoding

This isolates the split/temperature changes from the element representation change.

