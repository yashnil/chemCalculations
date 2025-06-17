#!/usr/bin/env python3
# step 2 -> train_baseline.py (UN-WEIGHTED BASELINE)

import os, json, time, joblib
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
from losses import composite_loss, _mae_log

CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR  = "artefacts"
MODEL_OUT = os.path.join(ARTE_DIR, "baseline_model.keras")
HIST_OUT  = os.path.join(ARTE_DIR, "history.json")

# 1) Load data & artifacts
df     = pd.read_csv(CSV_PATH)
splits = np.load(os.path.join(ARTE_DIR, "splits.npz"), allow_pickle=True)
train_idx = splits["train_idx"]   # <-- now an array of indices
val_idx   = splits["val_idx"]
test_idx  = splits["test_idx"]

scaler = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))

INPUTS  = ["temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S"]
META    = set(INPUTS) | {"group_index","point_index"}
SPECIES = [c for c in df.columns if c not in META]

# build & scale X
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in INPUTS[2:]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")

# build Y
Y = df[SPECIES].values.astype("float32")

# slice by index arrays
X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

print(f"Loaded → train:{X_train.shape[0]}, val:{X_val.shape[0]}, test:{X_test.shape[0]}")

# 2) Build & compile baseline model
model = keras.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(128, activation="gelu"),
    keras.layers.Dense(len(SPECIES), activation="softmax"),
], name="FCM_emulator_baseline")

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=composite_loss(lam=0.6, beta=1e-3),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae_lin"), _mae_log]
)
model.summary()

# 3) Train
t0 = time.time()
hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500,
    batch_size=128,
    verbose=2,
    callbacks=[
      keras.callbacks.EarlyStopping(patience=20,
                                    restore_best_weights=True,
                                    monitor="val_loss"),
      keras.callbacks.TensorBoard(log_dir=os.path.join(ARTE_DIR,"tb_logs"))
    ]
)
print(f"Training time: {(time.time()-t0)/60:.1f} min")

# 4) Evaluate
Y_pred = model.predict(X_test, batch_size=256, verbose=0)
mae    = mean_absolute_error(Y_test, Y_pred)
r2     = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nTest-set MAE={mae:.4e}   weighted R²={r2:.3f}")

# ──────────────────────────────────────────────────────────
# 5) Speed benchmark
N_BENCH = min(1000, len(X_test))
idx     = np.random.choice(len(X_test), N_BENCH, replace=False)
x_bench = X_test[idx]
_       = model.predict(x_bench[:16], batch_size=16)  # warm-up
t0      = time.time()
_       = model.predict(x_bench, batch_size=256)
nn_ms   = (time.time() - t0)/N_BENCH*1e3
print(f"Inference latency : {nn_ms:.3f} ms/pt (×{6.37/nn_ms:.1f} vs FastChem)")

# ──────────────────────────────────────────────────────────
# 6) Persist
os.makedirs(ARTE_DIR, exist_ok=True)
model.save(MODEL_OUT)
with open(HIST_OUT, "w") as fh:
    json.dump(hist.history, fh, indent=2)
print("Saved model →", MODEL_OUT)
print("Saved history →", HIST_OUT)
