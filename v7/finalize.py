# finalize.py  (aka step 4a)
#!/usr/bin/env python3
# step 4a -> finalize.py

import os
import json
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda, Activation
from sklearn.metrics import mean_absolute_error, r2_score
from utils import load_XY
from losses import composite_loss

# 1. Load data & artefacts
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
SPECIES = species_cols

# 2. Retrieve best hyper-parameters
study = joblib.load("artefacts/optuna_study.pkl")
best  = study.best_params
print("Best hyper-parameters:", best)

# merge train+val for the final fit
X_final = np.vstack([X_train, X_val])
Y_final = np.vstack([Y_train, Y_val])

# 3. Rebuild the best model
keras.backend.clear_session()
model = keras.Sequential([ keras.layers.Input((7,)) ])
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))

# plug in our raw_logits → temp_scale → relu_nonneg → renormalize head
model.add(keras.layers.Dense(len(SPECIES), name="raw_logits"))
model.add(Lambda(lambda x: x / 0.5, name="temp_scale"))
model.add(Activation("relu", name="relu_nonneg"))
model.add(Lambda(lambda x: x / (tf.reduce_sum(x, axis=1, keepdims=True) + 1e-12),
                 name="renormalize"))

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(lam=best["lam"], beta=best["beta"]),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae_lin")]
)

# 4. Train on the full training data
model.fit(
    X_final, Y_final,
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=[keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, monitor="loss"
    )]
)

# 5. Evaluate on the untouched test split
Y_pred  = model.predict(X_test, batch_size=256, verbose=0)
mae_test = mean_absolute_error(Y_test, Y_pred)
r2_test  = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nFINAL TEST – MAE = {mae_test:.4e}   weighted R² = {r2_test:.3f}")

# 6. Latency benchmark
FASTCHEM_MS = 6.29
N_BENCH     = min(1000, len(X_test))
idx         = np.random.choice(len(X_test), N_BENCH, replace=True)
x_bench     = X_test[idx]

_ = model.predict(x_bench[:16], batch_size=16)  # warm-up
t0 = time.time()
_  = model.predict(x_bench, batch_size=256)
nn_ms = (time.time() - t0)/N_BENCH * 1e3
speedup = FASTCHEM_MS / nn_ms
print(f"Inference latency : {nn_ms:.3f} ms/sample   (×{speedup:.1f})")

# 7. Persist artefacts
os.makedirs("artefacts", exist_ok=True)
model.save("artefacts/final_model.keras")

report = {
    "mae_test":   float(mae_test),
    "r2_test":    float(r2_test),
    "latency_ms": nn_ms,
    "speedup":    float(speedup),
    "hyper_params": best,
    "species":    SPECIES
}
with open("artefacts/final_report.json", "w") as fh:
    json.dump(report, fh, indent=2)

print("\n✔️  Saved artefacts/final_model.keras and final_report.json")
