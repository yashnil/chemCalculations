#!/usr/bin/env python3
# utils.py  – shared data-loading helpers (v9)
# ---------------------------------------
# v9 changes:
#   - Temperature normalized by global max (0 to 1)
#   - 70-15-15 train/val/test split
#   - Elements: same as v8 (log10 + 9)

import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

ARTE_DIR = "artefacts"   # adjust if you moved the folder
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

# input column names
INPUTS  = ["temperature", "pressure",
           "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
ELEMENT_COLS = INPUTS[2:]                     # the five elemental columns

# build META set once
META = set(INPUTS) | {"group_index", "point_index"}

# ────────────────────────────────────────────────────────────
def _preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Mirrors the preprocessing done in `baseline_checks.py`:
      • drop T-bin 0 rows
      • renormalise gas-phase species so each row sums to 1
      • return (clean_df, species_col_names)
    """
    # drop the low-T bin
    df = df.copy()
    df["T_bin"] = pd.qcut(df["temperature"], 5,
                          labels=False, duplicates="drop")
    df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")

    # identify species columns (everything that isn’t meta / inputs)
    species_cols = [c for c in df.columns if c not in META]

    # renormalise so Σspecies = 1.0
    df[species_cols] = df[species_cols].div(df[species_cols].sum(axis=1),
                                            axis=0)

    return df, species_cols
# ────────────────────────────────────────────────────────────
def load_XY():
    """
    Returns:
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test,
        scaler      (StandardScaler fitted on training inputs),
        species_cols  (list of output column names),
        T_max       (global temperature max for normalization)
    """
    # ─ 1.  read and preprocess dataframe ─────────────────────
    raw_df = pd.read_csv(CSV_PATH)
    df, species_cols = _preprocess_dataframe(raw_df)

    # ─ 2.  build input matrix X (v9: T normalization, v8 element encoding) ──
    X = df[INPUTS].copy()
    
    # Temperature: normalize by global max (v9 change)
    T_max = df["temperature"].max()
    X["temperature"] = X["temperature"] / T_max
    
    # Pressure: log10 (same as v8)
    X["pressure"] = np.log10(X["pressure"])
    
    # Elements: log10 + 9 (same as v8)
    for col in ELEMENT_COLS:
        X[col] = np.log10(X[col]) + 9.0

    # ─ 3.  target matrix Y  ──────────────────────────────────
    Y = df[species_cols].values.astype("float32")

    # ─ 4.  load artefacts created in baseline_checks.py ─────
    split   = np.load(os.path.join(ARTE_DIR, "splits.npz"),
                      allow_pickle=True)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    scaler = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
    X = scaler.transform(X).astype("float32")

    # ─ 5.  slice into the three sets ─────────────────────────
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    return (X_train, X_val, X_test,
            Y_train, Y_val, Y_test,
            scaler, species_cols, T_max)
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # quick sanity check
    Xtr, Xv, Xte, Ytr, Yv, Yte, sc, sp, Tmax = load_XY()
    print("Shapes  –  X:", Xtr.shape, Xv.shape, Xte.shape,
          "|  Y:", Ytr.shape, Yv.shape, Yte.shape)
    print("T_max =", Tmax)
