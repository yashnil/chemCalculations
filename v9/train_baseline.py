#!/usr/bin/env python3
# step-2 → train_baseline.py (v9)  ─ uses ReLU head, no low-T rows
# -----------------------------------------------------------

import os, json, time, joblib, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
from losses      import composite_loss, _mae_log
from model_heads import softplus_head                     # stripe-free head 🎉

CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR  = "artefacts"
MODEL_OUT = os.path.join(ARTE_DIR, "baseline_model.keras")
HIST_OUT  = os.path.join(ARTE_DIR, "history.json")
EPS = 1e-12

# ╭──────────────────── 1. LOAD DATA ───────────────────╮
df = pd.read_csv(CSV_PATH)

# -- drop T-bin 0 samples  (must mirror baseline_checks) --
df["T_bin"] = pd.qcut(df["temperature"], 5,
                      labels=False, duplicates="drop")
df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")

# -- renormalise gas-phase fractions so every row sums to 1.0 --
ELEMENT_COLS = [f"comp_{e}" for e in ("H","O","C","N","S")]
META_COLS    = {"temperature","pressure","group_index","point_index"} | set(ELEMENT_COLS)
SPECIES      = [c for c in df.columns if c not in META_COLS]
df[SPECIES]  = df[SPECIES].div(df[SPECIES].sum(axis=1), axis=0)

# --- scaler + split indices written by baseline_checks ---
scaler   = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
split_np = np.load(os.path.join(ARTE_DIR, "splits.npz"))
train_idx, val_idx, test_idx = (split_np[k] for k in ("train_idx","val_idx","test_idx"))

# ╭──────────────────── 2. FEATURE MATRIX (v9) ──────────╮
X = df[["temperature","pressure", *ELEMENT_COLS]].copy()

# Temperature: normalize by global max (v9 change)
T_max = df["temperature"].max()
X["temperature"] = X["temperature"] / T_max

# Pressure: log10 (same as v8)
X["pressure"] = np.log10(X["pressure"])

# Elements: log10 + 9 (same as v8)
for col in ELEMENT_COLS:
    X[col] = np.log10(X[col]) + 9.0

X = scaler.transform(X).astype("float32")
Y = df[SPECIES].values.astype("float32")

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

print(f"Loaded → train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
print("Lowest T in training set =", df.loc[train_idx, "temperature"].min())

# ╭──────────────────── 3. MODEL (v9: 7 inputs) ─────────╮
model = keras.Sequential([
    keras.layers.Input(shape=(7,)),            # 7 inputs (same as v8)
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(128, activation="gelu"),
    softplus_head(len(SPECIES))                # ← use the new head
])

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=composite_loss(lam=0.1, beta=0.0),    # ← λ down to 0.1
    metrics=[_mae_log]
)

# simple LR decay: halve LR if val_loss hasn’t improved for 6 epochs
callbacks=[keras.callbacks.EarlyStopping(
              monitor="val_loss", patience=20, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(
              monitor="val_loss", factor=0.5, patience=6, verbose=1,
              min_lr=1e-5)]

# ╭──────────────────── 4. TRAIN ────────────────────────╮
hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500,
    batch_size=128,
    verbose=2,
    callbacks=callbacks
)

# ╭──────────────────── 5. TEST + LATENCY ───────────────╮
Y_pred = model.predict(X_test, batch_size=256, verbose=0)

# ── log-space metrics ───────────────────────────────────────────
log_true = np.log10(Y_test + EPS)
log_pred = np.log10(Y_pred + EPS)

mae_log = _mae_log(tf.constant(Y_test), tf.constant(Y_pred)).numpy()   # already in code
r2_log  = r2_score(log_true, log_pred, multioutput="variance_weighted")

# (optional) linear-space R² for completeness
r2_lin  = r2_score(Y_test, Y_pred, multioutput="variance_weighted")

print("\nTest-set metrics")
print(f"   MAE_log (dex)   : {mae_log:.4e}")
print(f"   R²   (log space): {r2_log:7.3f}")
print(f"   R²   (linear)   : {r2_lin:7.3f}   # may be negative, that’s fine")

# quick NN latency check (optional)
_ = model.predict(X_test[:1000], batch_size=256, verbose=0)

# ╭──────────────────── 6. SAVE ─────────────────────────╮
os.makedirs(ARTE_DIR, exist_ok=True)
model.save(MODEL_OUT)
with open(HIST_OUT, "w") as fh:
    json.dump(hist.history, fh, indent=2)
print("Artefacts saved →", ARTE_DIR)
