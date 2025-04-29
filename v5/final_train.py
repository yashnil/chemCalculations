
#!/usr/bin/env python3
# step 4b -> final_train.py

# ---------------------------------------------------------------

import os, json, time, joblib, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
import optuna                                     # to read the study object
from losses import composite_loss

# ---------------------------------------------------------------
# paths
# ---------------------------------------------------------------
ARTE_DIR  = "artefacts"
CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
MODEL_OUT = os.path.join(ARTE_DIR, "surrogate_final.keras")
CARD_OUT  = os.path.join(ARTE_DIR, "model_card.json")

# ---------------------------------------------------------------
# 1.  Load data, scaler & split
# ---------------------------------------------------------------
df      = pd.read_csv(CSV_PATH)
scaler  = joblib.load(os.path.join(ARTE_DIR, "input_scaler.pkl"))
split   = np.load(os.path.join(ARTE_DIR, "splits.npz"))

INPUTS  = ["temperature", "pressure",
           "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]
META    = set(INPUTS) | {"group_index", "point_index"}
SPECIES = [c for c in df.columns if c not in META]

X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    X[el] = np.log10(X[el]) + 9.0

X = scaler.transform(X).astype("float32")
Y = df[SPECIES].values.astype("float32")

n_train = split["train_idx"].item()
n_val   = split["val_idx"].item()
X_train, Y_train = X[:n_train], Y[:n_train]
X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]

print(f"Train+val : {X_train.shape[0]}   Test : {X_test.shape[0]}")

# ---------------------------------------------------------------
# 2.  Retrieve the best hyper‑parameters (step 3)
# ---------------------------------------------------------------
study   = joblib.load(os.path.join(ARTE_DIR, "optuna_study.pkl"))
best    = study.best_params
print("Best hyper‑params:", best)

# ---------------------------------------------------------------
# 3.  Build & compile the final model
# ---------------------------------------------------------------
model = keras.Sequential([keras.layers.Input((7,))])
for _ in range(best["n_layers"]):
    model.add(keras.layers.Dense(best["units"], activation=best["act"]))
model.add(keras.layers.Dense(len(SPECIES), activation="softmax"))

model.compile(
    optimizer=keras.optimizers.Adam(best["lr"]),
    loss=composite_loss(best["lam"]),                    
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

model.summary()

# ---------------------------------------------------------------
# 4.  Train
# ---------------------------------------------------------------
t0 = time.time()
hist = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=500,
    batch_size=128,
    verbose=2,
    callbacks=[keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, monitor="val_loss")]
)
train_minutes = (time.time()-t0)/60
print(f"Training time: {train_minutes:.1f} min")

# ---------------------------------------------------------------
# 5.  Test‑set evaluation
# ---------------------------------------------------------------
Y_pred = model.predict(X_test, batch_size=256, verbose=0)
mae    = mean_absolute_error(Y_test, Y_pred)
r2     = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
print(f"Test‑set  MAE={mae:.4e}   weighted R²={r2:.3f}")

# ---------------------------------------------------------------
# 6.  Inference‑time benchmark  (compare with 6.42 ms FastChem)
# ---------------------------------------------------------------
FASTCHEM_MS = 6.42
N_BENCH     = min(1000, len(X_test))
bench_idx   = np.random.choice(len(X_test), N_BENCH, replace=False)
x_bench     = X_test[bench_idx]

_ = model.predict(x_bench[:16], batch_size=16, verbose=0)     # warm‑up
t0 = time.time()
_  = model.predict(x_bench, batch_size=256, verbose=0)
nn_ms = (time.time()-t0)/N_BENCH * 1e3

speedup = FASTCHEM_MS / nn_ms
print(f"Inference  : {nn_ms:.3f} ms / point   (×{speedup:,.0f} faster than FastChem)")

# ---------------------------------------------------------------
# 7.  Persist artefacts
# ---------------------------------------------------------------
os.makedirs(ARTE_DIR, exist_ok=True)
model.save(MODEL_OUT)

card = {
    "inputs"       : INPUTS,
    "outputs"      : SPECIES,
    "hyperparams"  : best,
    "test_MAE"     : float(mae),
    "test_R2"      : float(r2),
    "fastchem_ms"  : FASTCHEM_MS,
    "nn_ms"        : float(nn_ms),
    "speedup"      : float(speedup),
    "train_minutes": train_minutes
}
with open(CARD_OUT, "w") as fh:
    json.dump(card, fh, indent=2)

print("\nSaved:")
print(" • final model      →", MODEL_OUT)
print(" • model card JSON  →", CARD_OUT)
