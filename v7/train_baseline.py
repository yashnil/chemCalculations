#!/usr/bin/env python3
# step 2 → train_baseline.py

import os, time, json, joblib
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
from losses import composite_loss, _mae_log

# ──────────────────────────────────────────────────────────
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR   = "artefacts"
MODEL_OUT  = os.path.join(ARTE_DIR,"baseline_model.keras")
HIST_OUT   = os.path.join(ARTE_DIR,"history.json")

# ──────────────────────────────────────────────────────────
# 1) Load splits & scaler
splits     = np.load(os.path.join(ARTE_DIR,"splits.npz"), allow_pickle=True)
train_idx  = splits["train_idx"]
val_idx    = splits["val_idx"]
test_idx   = splits["test_idx"]
scaler     = joblib.load(os.path.join(ARTE_DIR,"input_scaler.pkl"))
df         = pd.read_csv(CSV_PATH)

# input/output columns
INPUTS     = ["temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S"]
META       = set(INPUTS)|{"group_index","point_index"}
SPECIES    = [c for c in df.columns if c not in META]

# ──────────────────────────────────────────────────────────
# 2) Build & scale X, Y
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in INPUTS[2:]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")
Y = df[SPECIES].values.astype("float32")

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

print(f"Loaded → train:{len(train_idx)}, val:{len(val_idx)}, test:{len(test_idx)}")

# ──────────────────────────────────────────────────────────
# 3) Build & compile baseline model (λ=0, β=0)
model = keras.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(128, activation="gelu"),
    keras.layers.Dense(len(SPECIES), activation="softmax"),
], name="baseline")
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=composite_loss(lam=0.0, beta=0.0),
    metrics=[ _mae_log ],
)
model.summary()

# ──────────────────────────────────────────────────────────
# 4) Train
t0 = time.time()
hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500, batch_size=128, verbose=2,
    callbacks=[
      keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
      keras.callbacks.TensorBoard(log_dir=os.path.join(ARTE_DIR,"tb_logs"))
    ]
)
train_time = time.time()-t0
print(f"Training took {train_time/60:.1f} min")

with open(HIST_OUT,"w") as fh:
    json.dump(hist.history, fh, indent=2)

# ──────────────────────────────────────────────────────────
# 5) Evaluate
Y_pred = model.predict(X_test, batch_size=256, verbose=0)
mae_log = _mae_log(tf.constant(Y_test), tf.constant(Y_pred)).numpy()
r2      = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nTest MAE_log = {mae_log:.4e}   weighted R² = {r2:.3f}")

# ──────────────────────────────────────────────────────────
# 6) Inference benchmark vs FastChem
# ──────────────────────────────────────────────────────────
# 6-A) NN speed
N_BENCH = min(1000, X_test.shape[0])
idx     = np.random.choice(len(X_test), N_BENCH, replace=False)
x_bench = X_test[idx]

_ = model.predict(x_bench[:16], batch_size=16)  # warm-up
t0 = time.time()
model.predict(x_bench, batch_size=256, verbose=0)
nn_ms = (time.time()-t0)/N_BENCH*1e3

# 6-B) FastChem speed (loaded from baseline_checks)
with open(os.path.join(ARTE_DIR,"fastchem_benchmark.json")) as fh:
    fastchem_ms = json.load(fh)["fastchem_ms"]

speedup = fastchem_ms / nn_ms
print(f"Inference → {nn_ms:.3f} ms/pt  (×{speedup:.1f} vs FastChem)")

# ──────────────────────────────────────────────────────────
# 7) Persist model
# ──────────────────────────────────────────────────────────
os.makedirs(ARTE_DIR, exist_ok=True)
model.save(MODEL_OUT)
print(f"Saved model → {MODEL_OUT}")
