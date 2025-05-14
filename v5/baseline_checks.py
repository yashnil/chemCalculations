
#!/usr/bin/env python3
# step 1 -> baseline_checks.py

"""
baseline_checks.py  –  Step‑1 sanity & timing for the FastChem surrogate project
(adapted to column names: comp_H, comp_O, comp_C, comp_N, comp_S)
"""

import os, time, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from pyfastchem import FastChem, FastChemInput, FastChemOutput

CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
LOGK_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"
COND_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK_condensates.dat"

ARTE_DIR   = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# 1‑A  • Load & numeric sanity
# ──────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print("\nLoaded", CSV_PATH, "shape =", df.shape)

assert not df.isna().any().any(), "NaNs present!"
assert np.isfinite(df.values).all(), "Non‑finite values present!"
print("No NaNs / inf values.")

ELEMENT_COLS = [f"comp_{e}" for e in ("H","O","C","N","S")]
META_COLS    = {"temperature", "pressure",
                "group_index", "point_index"} | set(ELEMENT_COLS)

species_cols = [c for c in df.columns if c not in META_COLS]

err = (df[species_cols].sum(axis=1) - 1.0).abs().max()
print(f"Species fractions: max|Σ−1| = {err:.3e}")

# ──────────────────────────────────────────────────────────────────────────
# 1‑B  • Histograms
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
df["temperature"].hist(bins=40, ax=ax[0]);              ax[0].set_title("Temperature")
np.log10(df["pressure"]).hist(bins=40, ax=ax[1]);       ax[1].set_title("log10 Pressure")
plt.tight_layout(); plt.savefig("input_histograms.png", dpi=150)
print("Saved histogram → input_histograms.png")

# ──────────────────────────────────────────────────────────────────────────
# 1‑C  • Feature scaling (pressure→log10, elements→log10+9)
# ──────────────────────────────────────────────────────────────────────────
X = df[["temperature", "pressure", *ELEMENT_COLS]].copy()
X["pressure"] = np.log10(X["pressure"])
for col in ELEMENT_COLS:
    X[col] = np.log10(X[col]) + 9.0          # range ~[0,9]

Y = df[species_cols].values.astype("float32")

# ──────────────────────────────────────────────────────────────────────────
# 1‑D  • Train/val/test split  (60/15/25 from guide)
# ──────────────────────────────────────────────────────────────────────────
X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.40, random_state=42, shuffle=True)
X_val, X_test,  Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.40, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"Split sizes → train:{X_train.shape[0]}, val:{X_val.shape[0]}, test:{X_test.shape[0]}")

joblib.dump(scaler, os.path.join(ARTE_DIR, "input_scaler.pkl"))
np.savez(os.path.join(ARTE_DIR, "splits.npz"),
         train_idx=X_train.shape[0],
         val_idx=X_val.shape[0],
         test_idx=X_test.shape[0])
print("Scaler & split indices saved to ./artefacts")

# ──────────────────────────────────────────────────────────────────────────
# 1‑E  • FastChem latency benchmark
# ──────────────────────────────────────────────────────────────────────────
def fastchem_single_point(row):
    """Run FastChem once; result is discarded – we only measure wall‑time."""
    comp = {e: row[f"comp_{e}"] for e in ("H","O","C","N","S")}

    # ---- build tiny abundance file in a temp dir
    import tempfile, textwrap
    with tempfile.TemporaryDirectory() as tmp:
        abund = os.path.join(tmp, "abund.dat")
        H = comp['H']
        with open(abund, "w") as f:
            f.write("#\ne- 0.0\n")
            for e, v in comp.items():
                f.write(f"{e} {12 + np.log10(v/H):.4f}\n")

        fc  = FastChem(abund, LOGK_PATH, 'none', 0)
        inp = FastChemInput(); out = FastChemOutput()
        inp.temperature = [row["temperature"]]
        inp.pressure    = [row["pressure"]]
        fc.calcDensities(inp, out)

R = 1000
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(df), R, replace=False)

t0 = time.time()
for idx in sample_idx:
    fastchem_single_point(df.iloc[idx])
fastchem_ms = (time.time() - t0) / R * 1e3
print(f"\n⏱️  FastChem benchmark  →  {fastchem_ms:.2f} ms per point ({R} samples)")
