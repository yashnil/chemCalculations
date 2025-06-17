#!/usr/bin/env python3
# step 3 -> tune.py

import os, joblib, warnings
import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error

from utils import load_XY
from losses import composite_loss, _mae_log

# ─────────────────────────────────────────────
# 0) Load data once via updated utils.py
# ─────────────────────────────────────────────
X_train, X_val, X_test, Y_train, Y_val, Y_test, _, species_cols = load_XY()

input_dim  = X_train.shape[1]
output_dim = len(species_cols)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# ─────────────────────────────────────────────
# 1) Optuna objective
# ─────────────────────────────────────────────
def objective(trial: optuna.trial.Trial) -> float:
    keras.backend.clear_session()

    # 1) hyper-params
    n_layers = trial.suggest_int("n_layers", 2, 6)
    units    = trial.suggest_categorical("units", [128, 256, 512])
    act      = trial.suggest_categorical("act",   ["gelu", "swish"])
    lr       = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    lam      = trial.suggest_float("lam", 0.0, 1.0)   # you can tune λ too
    beta     = 0.0                                    # mass‐balance off for now

    # 2) build model
    model = keras.Sequential(name="surrogate")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation=act))
    model.add(keras.layers.Dense(output_dim, activation="softmax"))

    # 3) compile with composite_loss
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=composite_loss(lam=lam, beta=beta),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae_lin"), _mae_log],
    )

    # 4) fit w/ early stopping on val _mae_log
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200, batch_size=128, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
    )

    # 5) return validation MAEₗₒg
    Y_val_pred = model.predict(X_val, batch_size=256, verbose=0)
    return float(_mae_log(tf.constant(Y_val), tf.constant(Y_val_pred)).numpy())


# ─────────────────────────────────────────────
# 2) Run study
# ─────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=60,
        timeout=None,
        catch=(tf.errors.InvalidArgumentError,),
    )

    print("\nBest hyper-parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:10s}: {v}")
    print(f"Validation MAE_log : {study.best_value:.4e}")

    joblib.dump(study, os.path.join("artefacts","optuna_study.pkl"))
    print("Optuna study saved → artefacts/optuna_study.pkl")
