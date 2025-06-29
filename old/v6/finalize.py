#!/usr/bin/env python3
# step 4a -> finalize.py

import os, json, time, joblib, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Softmax, Lambda
from sklearn.metrics import mean_absolute_error, r2_score
from utils import load_XY
from losses import composite_loss

# 1. Load data & artefacts
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
N_OUT = len(species_cols)

study = joblib.load("artefacts/optuna_study.pkl")
best  = study.best_params
print("Best hyper-parameters from Optuna:", best)

# merge train+val for the final fit
X_final = np.vstack([X_train, X_val])
Y_final = np.vstack([Y_train, Y_val])

# 2. Rebuild the best model
keras.backend.clear_session()

model = keras.Sequential([ keras.layers.Input((7,)) ])
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))
# final dense produces raw logits
model.add(keras.layers.Dense(N_OUT))

# optional “temperature” scaling: divide logits by T before softmax
T = 0.5
model.add(Lambda(lambda x: x / T, name="temp_scale"))

# then a standard Softmax
model.add(Softmax(axis=-1, name="probabilities"))

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=0.6),      
    metrics=[keras.metrics.MeanAbsoluteError(name="mae_lin")]
)

# 3. Train on the full training data
model.fit(
    X_final, Y_final,
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor="loss"
        )
    ]
)

# 4. Evaluate on the untouched test split
Y_pred = model.predict(X_test, batch_size=256, verbose=0)
mae_test = mean_absolute_error(Y_test, Y_pred)
r2_test  = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nFINAL TEST  –  MAE = {mae_test:.4e}   weighted R² = {r2_test:.3f}")

# 5. Latency benchmark vs FastChem
N_BENCH = 1000
rng     = np.random.default_rng(0)
idx     = rng.choice(len(X_test), N_BENCH, replace=True)
x_bench = X_test[idx]

_ = model.predict(x_bench[:16], batch_size=16, verbose=0)  # warm-up
t0 = time.time()
_  = model.predict(x_bench,  batch_size=256, verbose=0)
nn_ms = (time.time() - t0)/N_BENCH * 1e3

FASTCHEM_MS = 6.29
speedup     = FASTCHEM_MS / nn_ms
print(f"Inference latency : {nn_ms:.3f} ms / sample")
print(f"Speed-up vs FastChem: ×{speedup:.1f}")

# 6. Persist artefacts
os.makedirs("artefacts", exist_ok=True)
model.save("artefacts/final_model.keras")

report = {
    "mae_test":   float(mae_test),
    "r2_test":    float(r2_test),
    "latency_ms": nn_ms,
    "speedup":    float(speedup),
    "hyper_params": best,
    "species":    species_cols
}
with open("artefacts/final_report.json", "w") as fh:
    json.dump(report, fh, indent=2)

print("\n✔️  Saved   artefacts/final_model.keras"
      "\n           artefacts/final_report.json")
