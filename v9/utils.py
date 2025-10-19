#!/usr/bin/env python3
# utils.py  – shared data-loading helpers (v9)
# ---------------------------------------
# v9 changes:
#   - Temperature normalized by global max (0 to 1)
#   - Elements as H-relative ratios: log10(O/H), log10(C/H), log10(N/H), log10(S/H)
#   - 70-15-15 train/val/test split

import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

ARTE_DIR = "artefacts"   # adjust if you moved the folder
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

# Raw input column names from CSV
RAW_INPUTS  = ["temperature", "pressure",
               "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
ELEMENT_COLS = RAW_INPUTS[2:]                     # the five elemental columns

# build META set once
META = set(RAW_INPUTS) | {"group_index", "point_index"}

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

    # ─ 2.  build input matrix X with v9 transformations ──────
    X = pd.DataFrame()
    
    # Temperature: normalize by global max
    T_max = df["temperature"].max()
    X["temperature_norm"] = df["temperature"] / T_max
    
    # Pressure: log10 as before
    X["log_pressure"] = np.log10(df["pressure"])
    
    # Elements: H-relative ratios (log10 of ratio)
    X["log_O_H"] = np.log10(df["comp_O"] / df["comp_H"])
    X["log_C_H"] = np.log10(df["comp_C"] / df["comp_H"])
    X["log_N_H"] = np.log10(df["comp_N"] / df["comp_H"])
    X["log_S_H"] = np.log10(df["comp_S"] / df["comp_H"])

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
