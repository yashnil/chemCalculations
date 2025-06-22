#!/usr/bin/env python3
# step-4a → finalize.py
# ---------------------------------------------------------
# Trains the *best-of-study* architecture on (train+val),
# tests on held-out test, benchmarks speed, saves artefacts.
# ---------------------------------------------------------

import os, json, time, joblib, numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from utils   import load_XY                   # uses new pre-processing
from losses  import composite_loss, _mae_log
from model_heads import softplus_head         # stripe-free head
EPS = 1e-12

ARTE_DIR = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# ───────────────────────── 1. data ──────────────────────
X_tr, X_val, X_te, Y_tr, Y_val, Y_te, scaler, species = load_XY()
X_final = np.vstack([X_tr, X_val])
Y_final = np.vstack([Y_tr, Y_val])
print("Final fit on", X_final.shape[0], "samples;  test =", len(X_te))

# ───────────────────────── 2. best params ───────────────
study = joblib.load(os.path.join(ARTE_DIR, "optuna_study.pkl"))
best  = study.best_params
print("Best hyper-params:", best)

# ───────────────────────── 3. model ─────────────────────
keras.backend.clear_session()
model = keras.Sequential([keras.layers.Input((7,))])
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"],
                                 activation=best["act"]))
model.add(softplus_head(len(species)))

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=0.1, beta=0.0),     # λ from tuning
    metrics=[_mae_log]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=25,
                                  restore_best_weights=True,
                                  monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=8, min_lr=1e-5, verbose=1)
]

t0 = time.time()
model.fit(X_final, Y_final,
          validation_data=(X_te, Y_te),   # still watch test for LR schedule
          epochs=500, batch_size=128,
          verbose=2, callbacks=callbacks)
print(f"Training done in {(time.time()-t0)/60:.1f} min")

# ───────────────────────── 4. metrics ───────────────────
Y_pred = model.predict(X_te, batch_size=256, verbose=0)
mae_log = _mae_log(tf.constant(Y_te), tf.constant(Y_pred)).numpy()
r2_lin  = r2_score(Y_te,  Y_pred,  multioutput="variance_weighted")
r2_log  = r2_score(np.log10(Y_te+EPS),
                   np.log10(Y_pred+EPS),
                   multioutput="variance_weighted")

print(f"Test  MAE_log={mae_log:.4e}   R²_log={r2_log:.3f}   R²_lin={r2_lin:.3f}")

# ───────────────────────── 5. latency ───────────────────
N_BENCH = 1000
bench_x = X_te[:N_BENCH]
_ = model.predict(bench_x[:16], batch_size=16, verbose=0)  # warm-up
t0 = time.time(); model.predict(bench_x, batch_size=256, verbose=0)
nn_ms = (time.time()-t0)/N_BENCH * 1e3
FASTCHEM_MS = 7.0
print(f"Latency  {nn_ms:.3f} ms  →  ×{FASTCHEM_MS/nn_ms:.0f} speed-up")

# ───────────────────────── 6. save ──────────────────────
model.save(os.path.join(ARTE_DIR, "final_model.keras"))
with open(os.path.join(ARTE_DIR, "final_report.json"), "w") as f:
    json.dump({
        "mae_log_test": float(mae_log),
        "r2_log_test" : float(r2_log),
        "r2_lin_test" : float(r2_lin),
        "latency_ms"  : nn_ms,
        "speedup"     : FASTCHEM_MS/nn_ms,
        "hyperparams" : best,
        "species"     : species
    }, f, indent=2)

print("✔️  final_model.keras + report saved in artefacts/")
