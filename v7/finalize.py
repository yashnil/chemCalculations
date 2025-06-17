#!/usr/bin/env python3
# step 4a → finalize.py
# ————————————————————————————————————————————————
# Fit the Optuna-selected architecture on (train+val),
# evaluate on test, benchmark latency, and save artefacts.
# ————————————————————————————————————————————————

import os, time, json, joblib
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from losses import composite_loss, _mae_log      # <— unchanged

# ───────────────────────────────────────────────
# 0) Paths & load full CSV
# ───────────────────────────────────────────────
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ───────────────────────────────────────────────
# 1) Column bookkeeping
# ───────────────────────────────────────────────
INPUTS = ["temperature", "pressure",
          "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META    = set(INPUTS) | {"group_index", "point_index"}
SPECIES = [c for c in df.columns if c not in META]
N_OUT   = len(SPECIES)

# ───────────────────────────────────────────────
# 2) Load splits & scaler
# ───────────────────────────────────────────────
spl     = np.load(os.path.join(ARTE_DIR, "splits.npz"), allow_pickle=True)
train_i = spl["train_idx"];  val_i = spl["val_idx"];  test_i = spl["test_idx"]
scaler  = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))

# ───────────────────────────────────────────────
# 3) Build scaled X / Y arrays
# ───────────────────────────────────────────────
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for c in INPUTS[2:]:
    X[c] = np.log10(X[c]) + 9.0
X = scaler.transform(X).astype("float32")
Y = df[SPECIES].values.astype("float32")

X_train, Y_train = X[train_i], Y[train_i]
X_val,   Y_val   = X[val_i],   Y[val_i]
X_test,  Y_test  = X[test_i],  Y[test_i]

X_final = np.vstack([X_train, X_val])
Y_final = np.vstack([Y_train, Y_val])

print(f"🔹 Final fit on {X_final.shape[0]} samples – test on {X_test.shape[0]} samples")

# ───────────────────────────────────────────────
# 4) Reload Optuna best hyper-parameters
# ───────────────────────────────────────────────
study = joblib.load(os.path.join(ARTE_DIR, "optuna_study.pkl"))
best  = study.best_params
print("Best hyper-parameters:", best)

# ───────────────────────────────────────────────
# 5) Rebuild the model (identical to tune.py)
# ───────────────────────────────────────────────
keras.backend.clear_session()
model = keras.Sequential(name="surrogate_final")
model.add(keras.layers.Input(shape=(X.shape[1],)))
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))
model.add(keras.layers.Dense(N_OUT, activation="softmax"))

# ── Loss: keep β = 0 so NO mass-balance term ──
model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=best["lam"], beta=0.0),
    metrics=[_mae_log],           # MAE (log-space) only
)

# ───────────────────────────────────────────────
# 6) Train on train+val  (monitor val MAE_log)
# ───────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val__mae_log",   # <— keep using the metric you care about
        mode="min",               # <— tell Keras “smaller is better”
        patience=20,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val__mae_log",
        mode="min",               # <— same here
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
]

t0 = time.time()
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=callbacks,
)
print(f"⏱  Final fit: {(time.time()-t0)/60:.1f} min")

# ───────────────────────────────────────────────
# 7) Test-set metrics
# ───────────────────────────────────────────────
pred   = model.predict(X_test, batch_size=256, verbose=0)
mae_lg = _mae_log(tf.constant(Y_test), tf.constant(pred)).numpy()
r2     = r2_score(Y_test, pred, multioutput="variance_weighted")

# ───────────────────────────────────────────────
# 8) Latency benchmark
# ───────────────────────────────────────────────
FASTCHEM_MS = 6.13
idx   = np.random.choice(len(X_test), 1000, replace=True)
x_b   = X_test[idx]
_     = model.predict(x_b[:16], batch_size=16, verbose=0)   # warm-up
t0    = time.time(); model.predict(x_b, batch_size=256, verbose=0)
nn_ms = (time.time()-t0)/1000 * 1e3
speed = FASTCHEM_MS / nn_ms

# ───────────────────────────────────────────────
# 9) Save artefacts
# ───────────────────────────────────────────────
model.save(os.path.join(ARTE_DIR, "final_model.keras"))
with open(os.path.join(ARTE_DIR, "final_report.json"), "w") as f:
    json.dump({
        "mae_log_test":  float(mae_lg),
        "r2_test":       float(r2),
        "latency_ms":    nn_ms,
        "speedup":       speed,
        "hyperparams":   best,
        "species":       SPECIES,
    }, f, indent=2)

print("\n   FINAL TEST")
print(f"   MAE_log       : {mae_lg:10.4e}")
print(f"   weighted R²   : {r2:10.3f}")
print(f"   latency       : {nn_ms:10.3f} ms / pt   (×{speed:.1f} vs FastChem)")
print("✔️  Artefacts saved →", ARTE_DIR)
