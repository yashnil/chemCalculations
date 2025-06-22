# utils.py

import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

ARTE_DIR = "artefacts"
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

INPUTS = ["temperature", "pressure",
          "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META   = set(INPUTS) | {"group_index", "point_index"}

def load_XY():
    # — load the full DataFrame —
    df = pd.read_csv(CSV_PATH)

    # — targets —
    species_cols = [c for c in df.columns if c not in META]
    Y = df[species_cols].values.astype("float32")

    # — build input features —
    X = df[INPUTS].copy()
    X["pressure"] = np.log10(X["pressure"])
    for el in INPUTS[2:]:
        X[el] = np.log10(X[el]) + 9.0
    X = X.values  # 2D array

    # — load splits & scaler —
    split = np.load(os.path.join(ARTE_DIR, "splits.npz"), allow_pickle=True)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    scaler = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
    X_scaled = scaler.transform(X).astype("float32")

    # — slice —
    X_train,  Y_train  = X_scaled[train_idx], Y[train_idx]
    X_val,    Y_val    = X_scaled[val_idx],   Y[val_idx]
    X_test,   Y_test   = X_scaled[test_idx],  Y[test_idx]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols
