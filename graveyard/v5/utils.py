
# utils.py  – helper functions reused by several scripts
import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

ARTE_DIR = "artefacts"      # adjust if your folder is elsewhere
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

INPUTS  = ["temperature", "pressure",
           "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META    = set(INPUTS) | {"group_index", "point_index"}

def load_XY():
    """
    Returns
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test,
        scaler (fitted StandardScaler),
        species_cols (list of 116 output column names)
    """
    df = pd.read_csv(CSV_PATH)

    # -------- targets (116 species) ----------
    species_cols = [c for c in df.columns if c not in META]
    Y = df[species_cols].values.astype("float32")

    # -------- inputs  ----------
    X = df[INPUTS].copy()
    X["pressure"] = np.log10(X["pressure"])
    for el in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
        X[el] = np.log10(X[el]) + 9.0

    # load the train/val/test indices & scaler created in step 1
    split   = np.load(os.path.join(ARTE_DIR, "splits.npz"))
    n_train = split["train_idx"].item()
    n_val   = split["val_idx"].item()

    scaler  = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
    X = scaler.transform(X).astype("float32")

    X_train, Y_train = X[:n_train],            Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],      Y[n_train+n_val:]

    return (X_train, X_val, X_test,
            Y_train, Y_val, Y_test,
            scaler, species_cols)
