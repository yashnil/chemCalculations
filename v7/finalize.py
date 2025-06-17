#!/usr/bin/env python3
# step 4a → finalize.py

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
from losses import composite_loss, _mae_log

# ──────────────────────────────────────────────────────────
# 0) Paths & load splits/scaler
# ──────────────────────────────────────────────────────────
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

spl = np.load(os.path.join(ARTE_DIR, "splits.npz"), allow_pickle=True)
train_idx = spl["train_idx"]
val_idx   = spl["val_idx"]
test_idx  = spl["test_idx"]

scaler = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))

# ──────────────────────────────────────────────────────────
# 1) Load full DF & build sample-weights
# ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["T_bin"] = pd.qcut(df["temperature"],   5, labels=False, duplicates="drop")
df["P_bin"] = pd.qcut(np.log10(df["pressure"]), 5, labels=False, duplicates="drop")

inv_t = 1.0 / df["T_bin"].value_counts().sort_index().astype(float)
inv_p = 1.0 / df["P_bin"].value_counts().sort_index().astype(float)
df["sw"] = df["T_bin"].map(lambda b: inv_t[b]) * df["P_bin"].map(lambda b: inv_p[b])

# ──────────────────────────────────────────────────────────
# 2) Build X/Y splits
# ──────────────────────────────────────────────────────────
INPUTS  = ["temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S"]
META    = set(INPUTS) | {"group_index","point_index"}
SPECIES = [c for c in df.columns if c not in META]

X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for col in INPUTS[2:]:
    X[col] = np.log10(X[col]) + 9.0
X = scaler.transform(X).astype("float32")
Y = df[SPECIES].values.astype("float32")

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

# merge train+val for final fitting
X_final    = np.vstack([X_train, X_val])
Y_final    = np.vstack([Y_train, Y_val])
sw_final   = df.loc[np.concatenate([train_idx, val_idx]), "sw"].values
sw_val_only = df.loc[val_idx, "sw"].values

print(f"Final fit: {X_final.shape[0]} samples → test: {X_test.shape[0]} samples")

# ──────────────────────────────────────────────────────────
# 3) Load best hyper-params
# ──────────────────────────────────────────────────────────
study = joblib.load(os.path.join(ARTE_DIR, "optuna_study.pkl"))
best  = study.best_params
print("Best hyper-params:", best)

# ──────────────────────────────────────────────────────────
# 4) Build & compile (softmax head)
# ──────────────────────────────────────────────────────────
keras.backend.clear_session()
model = keras.Sequential(name="surrogate_final")
model.add(keras.layers.Input(shape=(X_final.shape[1],)))
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))

# single softmax output
model.add(keras.layers.Dense(len(SPECIES), activation="softmax"))

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=best["lam"], beta=best.get("beta",1e-3)),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae_lin"), _mae_log]
)
model.summary()

# ──────────────────────────────────────────────────────────
# 5) Train on train+val  (monitor val split, with weights)
# ──────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    ),
    keras.callbacks.TensorBoard(log_dir=os.path.join(ARTE_DIR,"tb_logs")),
]

t0 = time.time()
hist = model.fit(
    X_final,
    Y_final,
    sample_weight=sw_final,
    validation_data=(X_val, Y_val, sw_val_only),
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=callbacks,
)
print(f"Final fit time: {(time.time()-t0)/60:.1f} min")

# ──────────────────────────────────────────────────────────
# 6) Evaluate on untouched test
# ──────────────────────────────────────────────────────────
Y_pred   = model.predict(X_test, batch_size=256, verbose=0)
mae_test = mean_absolute_error(Y_test, Y_pred)
r2_test  = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nFINAL TEST → MAE={mae_test:.4e}  weighted R²={r2_test:.3f}")

# ──────────────────────────────────────────────────────────
# 7) Inference benchmark
# ──────────────────────────────────────────────────────────
FAST_MS = 6.29
N_BENCH = min(1000, len(X_test))
idx     = np.random.choice(len(X_test), N_BENCH, replace=True)
x_bench = X_test[idx]
_ = model.predict(x_bench[:16], batch_size=16)  # warm-up
t0 = time.time()
_  = model.predict(x_bench, batch_size=256, verbose=0)
nn_ms = (time.time() - t0)/N_BENCH * 1e3
print(f"Inference: {nn_ms:.3f} ms/pt  (×{FAST_MS/nn_ms:.1f} vs FastChem)")

# ──────────────────────────────────────────────────────────
# 8) Persist model + report
# ──────────────────────────────────────────────────────────
model.save(os.path.join(ARTE_DIR, "final_model.keras"))
report = {
    "mae_test":   float(mae_test),
    "r2_test":    float(r2_test),
    "latency_ms": nn_ms,
    "speedup":    float(FAST_MS/nn_ms),
    "hyperparams": best,
    "species":     SPECIES
}
with open(os.path.join(ARTE_DIR, "final_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n✔️  Saved final_model.keras + final_report.json")
