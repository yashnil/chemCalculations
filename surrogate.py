#!/usr/bin/env python3
"""
Train a 7‑input → 116‑output surrogate neural network for FastChem.

• CSV expected:  all_gas1.csv   (4000 × 125)
• Framework   :  TensorFlow / Keras  (works on CPU; drop‑in for GPU if available)
"""

import os, time, json, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------------------
# 1.  Load data ------------------------------------------------------------
# -------------------------------------------------------------------------
CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas1.csv"
df = pd.read_csv(CSV_PATH)

# ----- define columns -----------------------------------------------------
INPUTS  = ["temperature", "pressure", "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
KEEP    = INPUTS + [c for c in df.columns                      # 116 species
                    if c not in INPUTS
                       and not c.endswith("_index")]

df = df[KEEP]               # drop bookkeeping indices

# ----- numpy arrays -------------------------------------------------------
X = df[INPUTS].values.astype("float32")            # (4000, 7)
Y = df.drop(columns=INPUTS).values.astype("float32")  # (4000, 116)

# ----- splits -------------------------------------------------------------
X_train, X_tmp, Y_train, Y_tmp = train_test_split(
    X, Y, test_size=0.25, random_state=1)
X_val,  X_test, Y_val,  Y_test = train_test_split(
    X_tmp, Y_tmp, test_size=0.4, random_state=1)   # 60/15/25 %

print(f"train  : {X_train.shape}")
print(f"val    : {X_val.shape}")
print(f"test   : {X_test.shape}")

# -------------------------------------------------------------------------
# 2. Build model -----------------------------------------------------------
# -------------------------------------------------------------------------
from tensorflow import keras
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Normalization(),              # will adapt on X_train
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(256, activation="gelu"),
    keras.layers.Dense(128, activation="gelu"),
    keras.layers.Dense(116, activation="softmax")   # outputs sum to 1
])

# adapt normalisation layer
model.layers[0].adapt(X_train)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="kullback_leibler_divergence",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

# -------------------------------------------------------------------------
# 3.  Train ---------------------------------------------------------------
# -------------------------------------------------------------------------
cb = keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, monitor="val_loss")

hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500, batch_size=128, callbacks=[cb], verbose=2)

# -------------------------------------------------------------------------
# 4.  Evaluate -------------------------------------------------------------
# -------------------------------------------------------------------------
# ---- full‑state error ----------------------------------------------------
Y_pred = model.predict(X_test, batch_size=256)
mae_all = mean_absolute_error(Y_test, Y_pred)
r2_all  = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"\nFull 116‑species  |  MAE={mae_all:.4e}   R²={r2_all:.3f}")

# ---- Top‑10 species error -----------------------------------------------
top10_mask = np.zeros_like(Y_test, dtype=bool)
# per‑row Top‑10 indices
top10_idx_each = np.argsort(-Y_test, axis=1)[:, :10]
for i,row in enumerate(top10_idx_each):
    top10_mask[i, row] = True

mae_top10 = np.abs(Y_test[top10_mask] - Y_pred[top10_mask]).mean()
print(f"Top‑10 species     |  MAE={mae_top10:.4e}")

# -------------------------------------------------------------------------
# 5.  Speed test -----------------------------------------------------------
# -------------------------------------------------------------------------
N_BENCH = 1000
rnd_idx = np.random.choice(len(X_test), N_BENCH, replace=False)
x_bench = X_test[rnd_idx]

# warm‑up
_ = model.predict(x_bench[:10], batch_size=10)

t0 = time.time()
_  = model.predict(x_bench, batch_size=256)
t_nn = (time.time()-t0)/N_BENCH

FASTCHEM_PER_CALL = 0.17   # seconds for one point (update if you measured)
speedup = FASTCHEM_PER_CALL / t_nn

print(f"\nAverage inference time per sample : {t_nn*1e3:.2f} ms")
print(f"Estimated speed‑up over FastChem   : ×{speedup:,.0f}")

# -------------------------------------------------------------------------
# 6.  Save artefacts --------------------------------------------------------
# -------------------------------------------------------------------------
OUT_DIR = "/Users/yashnilmohanty/Desktop/FastChem-Materials/model"
os.makedirs(OUT_DIR, exist_ok=True)
model.save(os.path.join(OUT_DIR, "fastchem_emulator.keras"))

with open(os.path.join(OUT_DIR, "normalisation.json"), "w") as f:
    json.dump({"inputs": INPUTS}, f, indent=2)

print(f"\nModel & metadata saved to  {OUT_DIR}")
