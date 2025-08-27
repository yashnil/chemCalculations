#!/usr/bin/env python3
# utils.py  –  common I/O helpers for the FastChem-surrogate pipeline
# -------------------------------------------------------------------
import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# CONSTANTS & PATHS
# -------------------------------------------------------------------
ARTE_DIR = "artefacts"      # adjust if your folder is elsewhere
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

INPUTS  = ["temperature", "pressure",
           "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META    = set(INPUTS) | {"group_index", "point_index"}

EPS     = 1e-12             # floor to avoid log10(0)

# -------------------------------------------------------------------
# public helper
# -------------------------------------------------------------------
def load_XY(csv_path: str = CSV_PATH,
            arte_dir: str = ARTE_DIR):
    """
    Load the dataset, apply the input scaler created in step-1, and
    return train / val / test splits where **Y is log10(fraction)**.

    Returns
    -------
    (X_train, X_val, X_test,
     Y_train, Y_val, Y_test,
     scaler, species_cols)
    """
    df = pd.read_csv(csv_path)

    # -------- species columns (116 outputs) ------------------------
    species_cols = [c for c in df.columns if c not in META]

    # -------- X (same numeric treatment as step-1) -----------------
    X = df[INPUTS].copy()
    X["pressure"] = np.log10(X["pressure"])
    for el in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
        X[el] = np.log10(X[el]) + 9.0

    # -------- Y  ->  log10(fraction) -------------------------------
    Y_linear = df[species_cols].values.astype("float32")
    Y_log    = np.log10(Y_linear + EPS).astype("float32")

    # -------- recover split sizes & scaler -------------------------
    split   = np.load(os.path.join(arte_dir, "splits.npz"))
    n_train = int(split["train_idx"])
    n_val   = int(split["val_idx"])

    scaler  = joblib.load(os.path.join(arte_dir, "input_scaler.pkl"))
    X       = scaler.transform(X).astype("float32")

    # -------- slice into the three splits --------------------------
    X_train, Y_train = X[:n_train],              Y_log[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y_log[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],        Y_log[n_train+n_val:]

    return (X_train, X_val, X_test,
            Y_train, Y_val, Y_test,
            scaler, species_cols)
