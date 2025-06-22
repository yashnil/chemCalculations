#!/usr/bin/env python3
# step-3 → tune.py
# Optuna H-param search for the FastChem surrogate
# ===============================================

import os, warnings, joblib, optuna, numpy as np, tensorflow as tf
from tensorflow import keras
from utils           import load_XY          # ← your cached split helper
from losses          import composite_loss, _mae_log
from model_heads     import softmax_T        # ← stripe-free head

# ──────────────────────────────────────────────
# 0)  Load the cached train/val/test split
# ──────────────────────────────────────────────
X_train, X_val, _, Y_train, Y_val, _, _, species_cols = load_XY()
INPUT_DIM  = X_train.shape[1]
OUTPUT_DIM = len(species_cols)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# ──────────────────────────────────────────────
# 1)  Optuna objective
# ──────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    keras.backend.clear_session()

    # — search space —
    n_layers = trial.suggest_int("n_layers", 2, 6)
    units    = trial.suggest_categorical("units",  [128, 256, 512])
    act      = trial.suggest_categorical("act",    ["gelu", "swish"])
    lr       = trial.suggest_float      ("lr", 1e-4, 3e-3, log=True)
    lam      = trial.suggest_float      ("lam", 0.0, 1.0)   # blend of KL & log-MAE
    beta     = 0.0                                           # keep mass-balance OFF

    # — model —
    model = keras.Sequential(name="surrogate")
    model.add(keras.layers.Input(shape=(INPUT_DIM,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation=act))
    model.add(keras.layers.Dense(OUTPUT_DIM))        # raw logits
    model.add(softmax_T(OUTPUT_DIM, T=0.5))          # stripe-free softmax

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=composite_loss(lam=lam, beta=beta),
        metrics=[_mae_log],        # our target metric (log-space MAE)
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=128,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val__mae_log", mode="min",
            patience=15, restore_best_weights=True
        )],
    )

    # minimise val log-MAE
    y_val_pred = model.predict(X_val, batch_size=256, verbose=0)
    return float(_mae_log(tf.constant(Y_val), tf.constant(y_val_pred)).numpy())

# ──────────────────────────────────────────────
# 2)  Run / save the Optuna study
# ──────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=60,                    # increase if you have more time
        catch=(tf.errors.InvalidArgumentError,),
    )

    print("\nBest hyper-parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:10s}: {v}")
    print(f"Validation MAE_log : {study.best_value:.4e}")

    os.makedirs("artefacts", exist_ok=True)
    joblib.dump(study, "artefacts/optuna_study.pkl")
    print("Optuna study saved → artefacts/optuna_study.pkl")
