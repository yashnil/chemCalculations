#!/usr/bin/env python3
# step 1 → baseline_checks.py

import os, time, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyfastchem import FastChem, FastChemInput, FastChemOutput

# ──────────────────────────────────────────────────────────
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
LOGK_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"
COND_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK_condensates.dat"
ARTE_DIR   = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 1-A) Load & numeric sanity
# ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\nLoaded {CSV_PATH}  shape={df.shape}")
assert not df.isna().any().any(), "NaNs present!"
assert np.isfinite(df.values).all(), "Non-finite values present!"
print("✔ no NaN/inf")

ELEMENT_COLS = [f"comp_{e}" for e in ("H","O","C","N","S")]
META_COLS    = {"temperature","pressure","group_index","point_index"} | set(ELEMENT_COLS)
species_cols = [c for c in df.columns if c not in META_COLS]
max_dev = (df[species_cols].sum(axis=1) - 1.0).abs().max()
print(f"Species fractions dev from 1: {max_dev:.3e}")

# ──────────────────────────────────────────────────────────
# 1-B) Histograms
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(1,2,figsize=(8,3))
df["temperature"].hist(bins=40, ax=ax[0]); ax[0].set_title("Temperature")
np.log10(df["pressure"]).hist(bins=40, ax=ax[1]); ax[1].set_title("log10 Pressure")
plt.tight_layout()
plt.savefig("input_histograms.png", dpi=150)
print("Saved → input_histograms.png")

# ──────────────────────────────────────────────────────────
# 1-C) Feature scaling prep
# ──────────────────────────────────────────────────────────
X = df[["temperature","pressure",*ELEMENT_COLS]].copy()
X["pressure"] = np.log10(X["pressure"])
for col in ELEMENT_COLS:
    X[col] = np.log10(X[col]) + 9.0
Y = df[species_cols].values.astype("float32")

# ──────────────────────────────────────────────────────────
# 1-D) 60/15/25 random split
# ──────────────────────────────────────────────────────────
all_idx = np.arange(len(df))
train_val_idx, test_idx = train_test_split(all_idx, test_size=0.25, random_state=42, shuffle=True)
train_idx,    val_idx  = train_test_split(train_val_idx, test_size=0.20, random_state=42, shuffle=True)
print(f"Splits → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

# scale *only* on train
scaler    = StandardScaler().fit(X.iloc[train_idx])
X_train   = scaler.transform(X.iloc[train_idx])
X_val     = scaler.transform(X.iloc[val_idx])
X_test    = scaler.transform(X.iloc[test_idx])

joblib.dump(scaler, os.path.join(ARTE_DIR,"input_scaler.pkl"))
np.savez(os.path.join(ARTE_DIR,"splits.npz"),
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
print("Saved scaler + splits.npz → artefacts/")

# ──────────────────────────────────────────────────────────
# 1-E) FastChem latency benchmark
# ──────────────────────────────────────────────────────────
def fastchem_single_point(row):
    comp = {e: row[f"comp_{e}"] for e in ("H","O","C","N","S")}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        abund = os.path.join(tmp, "abund.dat")
        H = comp['H']
        with open(abund,"w") as f:
            f.write("#\ne-  0.0\n")
            for e,v in comp.items():
                f.write(f"{e} {12 + np.log10(v/H):.4f}\n")
        fc = FastChem(abund, LOGK_PATH, 'none', 0)
        inp, out = FastChemInput(), FastChemOutput()
        inp.temperature = [row["temperature"]]
        inp.pressure    = [row["pressure"]]
        fc.calcDensities(inp, out)

R = 1000
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(df), R, replace=False)

t0 = time.time()
for i in sample_idx:
    fastchem_single_point(df.iloc[i])
fastchem_ms = (time.time()-t0)/R*1e3
print(f"\nFastChem benchmark → {fastchem_ms:.2f} ms/pt  ({R} pts)")

# ──────────────────────────────────────────────────────────
# 1-F) Persist FastChem benchmark
# ──────────────────────────────────────────────────────────
with open(os.path.join(ARTE_DIR,"fastchem_benchmark.json"),"w") as fh:
    json.dump({"fastchem_ms": fastchem_ms}, fh, indent=2)
print("Saved → artefacts/fastchem_benchmark.json")
