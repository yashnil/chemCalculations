#!/usr/bin/env python3
"""
Extract x64's target species list and show how to use it for x80.

This fixes the target species mismatch issue.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load x64's target list
x64_summary = json.load(open(BASE_DIR / "runs_autoencoder_x64" / "summary.json"))
x64_targets = x64_summary["target_cols"]

print("=" * 80)
print("FIXING TARGET SPECIES MISMATCH")
print("=" * 80)

print(f"\n✅ x64 target species list ({len(x64_targets)} species):")
print("   " + ", ".join(x64_targets))

print("\n" + "=" * 80)
print("SOLUTION: Lock target species for all runs")
print("=" * 80)

print("\nTo fix this, you need to set TARGET_COLS_MANUAL in train_autoencoder.py:")
print("\nTARGET_COLS_MANUAL = [")
for i, species in enumerate(x64_targets):
    comma = "," if i < len(x64_targets) - 1 else ""
    print(f'    "{species}"{comma}')
print("]")

print("\n" + "=" * 80)
print("IMPORTANT DISTINCTION")
print("=" * 80)
print("""
What we're fixing:
  ✅ Target species list (what the model predicts) - MUST be identical
  ✅ This is the 21 species columns the model outputs

What stays different:
  ✅ Training/validation/test datasets - these will still differ by size
  ✅ The actual data samples - x80 has more samples than x64

The issue was:
  - x64 auto-selected top 20 species from its dataset → got O3S1
  - x80 auto-selected top 20 species from its dataset → got S6 instead
  - This makes loss values incomparable!

The fix:
  - Lock the target species list to x64's list for ALL runs
  - Now all models predict the same 21 species
  - Loss values become comparable across dataset sizes
""")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. Set TARGET_COLS_MANUAL in train_autoencoder.py to the list above
2. Re-run x80 training (and optionally x48, x32 if you want full consistency)
3. Re-generate comparison_metrics.csv
4. Re-plot resolution_study.png

This will show if the x64→x80 degradation is real or just an artifact
of the target species mismatch.
""")

