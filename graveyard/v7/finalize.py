#!/usr/bin/env python3
# step-4a → finalize.py
# Train Optuna-best model on (train+val), evaluate on test,
# benchmark latency, and save artefacts.
# =========================================================

import os, time, json, joblib, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from utils           import load_XY
from losses          import composite_loss, _mae_log
from model_heads     import softmax_T             # stripe-free head

ARTE_DIR = "artefacts"
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
FASTCHEM_MS = 6.95                               # baseline scalar; update if needed
os.makedirs(ARTE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 0)  Reload split + scaler via helper
# ──────────────────────────────────────────────
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
INPUT_DIM  = X_train.shape[1]
OUTPUT_DIM = len(species_cols)

print(f"🔹 Final fit on {X_train.shape[0]+X_val.shape[0]} samples "
      f"– test on {X_test.shape[0]} samples")

# ──────────────────────────────────────────────
# 1)  Get best Optuna params
# ──────────────────────────────────────────────
study = joblib.load(os.path.join(ARTE_DIR, "optuna_study.pkl"))
best  = study.best_params
print("Best hyper-parameters:", best)

# ──────────────────────────────────────────────
# 2)  Re-build the model
# ──────────────────────────────────────────────
keras.backend.clear_session()
model = keras.Sequential(name="surrogate_final")
model.add(keras.layers.Input(shape=(INPUT_DIM,)))
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))
model.add(keras.layers.Dense(OUTPUT_DIM))            # logits
model.add(softmax_T(OUTPUT_DIM, T=0.5))              # stripe-free softmax

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=best["lam"], beta=0.0),   # β=0: no mass-balance term
    metrics=[_mae_log],
)

# ──────────────────────────────────────────────
# 3)  Train (monitor val _mae_log)
# ──────────────────────────────────────────────
cb = [
    keras.callbacks.EarlyStopping(
        monitor="val__mae_log", mode="min",
        patience=20, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val__mae_log", mode="min",
        factor=0.5, patience=5, min_lr=1e-6
    )
]

t0 = time.time()
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=cb,
)
print(f"⏱  Final fit: {(time.time()-t0)/60:.1f} min")

# ──────────────────────────────────────────────
# 4)  Test metrics
# ──────────────────────────────────────────────
pred      = model.predict(X_test, batch_size=256, verbose=0)
mae_log   = _mae_log(tf.constant(Y_test), tf.constant(pred)).numpy()
r2_weight = r2_score(Y_test, pred, multioutput="variance_weighted")

# ──────────────────────────────────────────────
# 5)  Latency benchmark
# ──────────────────────────────────────────────
bench_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=True)
x_bench   = X_test[bench_idx]
_ = model.predict(x_bench[:16], batch_size=16, verbose=0)   # warm-up
t0 = time.time(); model.predict(x_bench, batch_size=256, verbose=0)
nn_ms   = (time.time()-t0) / len(x_bench) * 1e3
speedup = FASTCHEM_MS / nn_ms

# ──────────────────────────────────────────────
# 6)  Persist artefacts
# ──────────────────────────────────────────────
model.save(os.path.join(ARTE_DIR, "final_model.keras"))
with open(os.path.join(ARTE_DIR, "final_report.json"), "w") as f:
    json.dump({
        "mae_log_test":  float(mae_log),
        "r2_test":       float(r2_weight),
        "latency_ms":    nn_ms,
        "speedup":       speedup,
        "hyperparams":   best,
        "species":       species_cols,
    }, f, indent=2)

print("\n   FINAL TEST")
print(f"   MAE_log       : {mae_log:10.4e}")
print(f"   weighted R²   : {r2_weight:10.3f}")
print(f"   latency       : {nn_ms:10.3f} ms / pt   (×{speedup:.1f} vs FastChem)")
print("✔️  Artefacts saved →", ARTE_DIR)
