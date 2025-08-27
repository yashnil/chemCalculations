#!/usr/bin/env python3
# step-1 → baseline_checks.py
# ---------------------------------------------
#  • loads all_gas.csv
#  • drops T-bin 0 samples
#  • does split + scaling
#  • benchmarks FastChem
#  • writes scaler + index arrays to artefacts
# ---------------------------------------------

import os, time, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from pyfastchem import FastChem, FastChemInput, FastChemOutput

CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
LOGK_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"

ARTE_DIR   = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# 1-A  load + basic sanity
# ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print("\nLoaded", CSV_PATH, "shape =", df.shape)

assert not df.isna().any().any(), "NaNs present!"
assert np.isfinite(df.values).all(), "Non-finite values present!"
print("No NaNs / inf values.")

# ───────────────────────────────────────────────────────────────
# 1-B  drop low-T (bin 0) samples  **NEW**
# ───────────────────────────────────────────────────────────────
df["T_bin"] = pd.qcut(df["temperature"], 5, labels=False, duplicates="drop")
df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")
print("After drop →", df.shape[0], "rows")

# ---- define column groups *now* so species_cols is available ----
ELEMENT_COLS = [f"comp_{e}" for e in ("H", "O", "C", "N", "S")]
META_COLS    = {"temperature", "pressure", "group_index", "point_index"} | set(ELEMENT_COLS)
species_cols = [c for c in df.columns if c not in META_COLS]

# ---- renormalise so each row sums to exactly 1 ------------------
row_sum = df[species_cols].sum(axis=1).values          # vector length = n_rows
df[species_cols] = df[species_cols].div(row_sum, axis=0)

assert (df[species_cols].sum(axis=1) - 1.0).abs().max() < 1e-4
print("Species fractions successfully renormalised.")

# ───────────────────────────────────────────────────────────────
# 1-C  column groups
# ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
df["temperature"].hist(bins=40, ax=ax[0]); ax[0].set_title("Temperature")
np.log10(df["pressure"]).hist(bins=40, ax=ax[1]); ax[1].set_title("log10 Pressure")
plt.tight_layout(); plt.savefig("input_histograms.png", dpi=150)
print("Saved histogram → input_histograms.png")

# ───────────────────────────────────────────────────────────────
# 1-D  feature engineering
# ───────────────────────────────────────────────────────────────
X = df[["temperature","pressure",*ELEMENT_COLS]].copy()
X["pressure"] = np.log10(X["pressure"])
for col in ELEMENT_COLS:
    X[col] = np.log10(X[col]) + 9.0          #  ≈ 0 … 9
Y = df[species_cols].values.astype("float32")

# ───────────────────────────────────────────────────────────────
# 1-E  split 60/15/25
# ───────────────────────────────────────────────────────────────
X_train, X_tmp, Y_train, Y_tmp = train_test_split(
    X, Y, test_size=0.40, random_state=42, shuffle=True)
X_val, X_test,  Y_val, Y_test = train_test_split(
    X_tmp, Y_tmp, test_size=0.40, random_state=42)

# grab *row indices* **before** we turn DataFrames into NumPy arrays
train_idx = X_train.index.to_numpy(np.int32)
val_idx   = X_val.index.to_numpy(np.int32)
test_idx  = X_test.index.to_numpy(np.int32)

# scale afterwards
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"Split sizes → train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")

# save artefacts
joblib.dump(scaler, os.path.join(ARTE_DIR,"input_scaler.pkl"))
np.savez(os.path.join(ARTE_DIR,"splits.npz"),
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
print("Scaler & split indices saved →", ARTE_DIR)

# ───────────────────────────────────────────────────────────────
# 1-F  FastChem latency benchmark  (same as before)
# ───────────────────────────────────────────────────────────────
def fastchem_single_point(row):
    comp = {e: row[f"comp_{e}"] for e in ("H","O","C","N","S")}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        abund = os.path.join(tmp,"ab.dat")
        H = comp['H']
        with open(abund,"w") as f:
            f.write("#\ne- 0.0\n")
            for e,v in comp.items():
                f.write(f"{e} {12+np.log10(v/H):.4f}\n")
        fc  = FastChem(abund, LOGK_PATH, 'none', 0)
        inp = FastChemInput(); out = FastChemOutput()
        inp.temperature=[row["temperature"]]; inp.pressure=[row["pressure"]]
        fc.calcDensities(inp,out)

R, rng = 1000, np.random.default_rng(0)
t0 = time.time()
for idx in rng.choice(len(df), R, replace=False):
    fastchem_single_point(df.iloc[idx])
fastchem_ms = (time.time()-t0)/R*1e3
print(f"\nFastChem benchmark  →  {fastchem_ms:.2f} ms / point ({R} samples)")
