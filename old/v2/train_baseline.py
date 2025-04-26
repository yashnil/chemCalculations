
#!/usr/bin/env python3

# step 2 -> train_baseline.py
"""
step2_train_baseline.py  –  baseline surrogate for FastChem

Inputs  (from step 1)
  • all_gas.csv   (or all_gas1.csv – just keep the path below consistent)
  • artefacts/input_scaler.pkl
  • artefacts/splits.npz

Outputs
  • artefacts/baseline_model.keras
  • artefacts/history.json
  • metrics printed to console
"""

import os, json, time, joblib, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import KLDivergence
from sklearn.metrics import mean_absolute_error, r2_score

# ───────────────────────────────────────────────────────────────────────────
# paths – change if your directories differ
# ───────────────────────────────────────────────────────────────────────────
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR = "artefacts"                            # already created by step 1
MODEL_OUT = os.path.join(ARTE_DIR, "baseline_model.keras")
HIST_OUT  = os.path.join(ARTE_DIR, "history.json")

# ───────────────────────────────────────────────────────────────────────────
# 1) load data, scaler, and split indices
# ───────────────────────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH)
scaler = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
split  = np.load(os.path.join(ARTE_DIR, "splits.npz"))

INPUTS  = ["temperature", "pressure",
           "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META    = set(INPUTS) | {"group_index", "point_index"}
SPECIES = [c for c in df.columns if c not in META]             # 116 outputs

# --- construct X ----------------------------------------------------------
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    X[el] = np.log10(X[el]) + 9.0          # undo earlier log‑shift
X = scaler.transform(X).astype("float32")

# --- construct Y ----------------------------------------------------------
EPS     = 1e-12                     # put with other constants near top
Y_lin   = df[SPECIES].values.astype("float32")
Y       = np.log10(Y_lin + EPS).astype("float32")     # <- NEW: log10

# --- recover the *absolute* sizes of each split --------------------------
n_train = split["train_idx"].item()
n_val   = split["val_idx"].item()

X_train, Y_train = X[:n_train],            Y[:n_train]
X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test,  Y_test  = X[n_train+n_val:],      Y[n_train+n_val:]

print(f"Loaded → train:{X_train.shape[0]}, val:{X_val.shape[0]}, test:{X_test.shape[0]}")

# ───────────────────────────────────────────────────────────────────────────
# 2) model architecture
# ───────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(128, activation="gelu"),
    keras.layers.Dense(len(SPECIES), activation="linear")     # 116 outputs
], name="FCM_emulator_baseline")

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)
model.summary(print_fn=lambda s: print("   " + s))

# ───────────────────────────────────────────────────────────────────────────
# 3) train
# ───────────────────────────────────────────────────────────────────────────
t0 = time.time()
callbacks = [
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,
                                  monitor="val_loss"),
    keras.callbacks.TensorBoard(log_dir=os.path.join(ARTE_DIR, "tb_logs"))
]

hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500,
    batch_size=128,
    verbose=2,
    callbacks=callbacks
)
print(f"Training time: {(time.time() - t0)/60:.1f} min")

# ───────────────────────────────────────────────────────────────────────────
# 4) evaluate on the held‑out test set
# ───────────────────────────────────────────────────────────────────────────
Y_pred_log = model.predict(X_test, batch_size=256, verbose=0)
Y_pred_lin = np.power(10.0, Y_pred_log) - EPS
Y_test_lin = np.power(10.0, Y_test)      - EPS
# (optional) re-normalise rows to sum-to-1
Y_pred_lin /= Y_pred_lin.sum(axis=1, keepdims=True)
Y_test_lin /= Y_test_lin.sum(axis=1, keepdims=True)

mae = mean_absolute_error(Y_test_lin, Y_pred_lin)
r2  = r2_score(Y_test_lin, Y_pred_lin, multioutput="variance_weighted")
print(f"\nTest‑set  MAE={mae:.4e}   weighted R²={r2:.3f}")

# ───────────────────────────────────────────────────────────────────────────
# 4 b)  runtime benchmark – compare with FastChem (5.40 ms / point)
# ───────────────────────────────────────────────────────────────────────────
# ── runtime benchmark – compare with FastChem (5.80 ms / point) ──────────
MAX_BENCH = 1_000
N_BENCH   = min(MAX_BENCH, len(X_test))      # never exceed test‑set size

idx      = np.random.choice(len(X_test), N_BENCH, replace=False)
x_bench  = X_test[idx]

_ = model.predict(x_bench[:10], batch_size=10)        # warm‑up
t0   = time.time()
_    = model.predict(x_bench, batch_size=256)
nn_ms = (time.time() - t0) / N_BENCH * 1e3            # ms per point

FASTCHEM_MS = 5.80                                    # your benchmark
speedup     = FASTCHEM_MS / nn_ms

print(f"\nInference latency  : {nn_ms:.3f} ms / point "
      f"(averaged over {N_BENCH} samples)")
print(f"Speed‑up vs FastChem: ×{speedup:,.1f}")


# ───────────────────────────────────────────────────────────────────────────
# 5) persist artefacts
# ───────────────────────────────────────────────────────────────────────────
os.makedirs(ARTE_DIR, exist_ok=True)
model.save(MODEL_OUT)
with open(HIST_OUT, "w") as fh:
    json.dump(hist.history, fh, indent=2)

print("\nSaved:")
print(" • model   →", MODEL_OUT)
print(" • history →", HIST_OUT)
print("TensorBoard logdir:", os.path.join(ARTE_DIR, "tb_logs"))


'''
Model prelim. results:

Test‑set  MAE=2.1850e-03   weighted R²=0.924
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Inference latency  : 0.062 ms / point (averaged over 640 samples)
Speed‑up vs FastChem: ×93.2

Saved:
 • model   → artefacts/baseline_model.keras
 • history → artefacts/history.json
TensorBoard logdir: artefacts/tb_logs

'''