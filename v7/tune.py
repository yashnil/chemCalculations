#!/usr/bin/env python3
# step 3 -> tune.py

import os
import warnings
import joblib
import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from utils import load_XY
from losses import composite_loss

# ──────────────────────────────────────────────────────────────────────────
# 0) Load data & extract dimensions
# ──────────────────────────────────────────────────────────────────────────
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
input_dim = X_train.shape[1]           # number of features
output_dim = len(species_cols)         # number of species to predict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# ──────────────────────────────────────────────────────────────────────────
# 1) Objective for Optuna
# ──────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.trial.Trial) -> float:
    keras.backend.clear_session()

    # hyperparameters
    n_layers = trial.suggest_int("n_layers", 2, 5)
    units    = trial.suggest_categorical("units", [128, 256, 512])
    act      = trial.suggest_categorical("act",   ["gelu", "swish"])
    lr       = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    lam      = trial.suggest_float("lam", 0.3, 0.9)
    beta     = trial.suggest_float("beta", 1e-5, 1e-1, log=True)

    # build model
    model = keras.Sequential(name="surrogate")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation=act))
    model.add(keras.layers.Dense(output_dim, activation="softmax"))

    # compile with composite loss
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=composite_loss(lam=lam, beta=beta),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    # fit
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=128,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=15, monitor="val_loss", restore_best_weights=True
            )
        ],
    )

    # evaluate on validation set
    y_pred = model.predict(X_val, batch_size=256, verbose=0)
    return mean_absolute_error(Y_val, y_pred)


# ──────────────────────────────────────────────────────────────────────────
# 2) Run Optuna study
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=40,
        timeout=2*3600,    # or remove to just limit by n_trials
        catch=(tf.errors.InvalidArgumentError,),
    )

    print("\nBest hyper-parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:10s}: {v}")
    print(f"Validation MAE : {study.best_value:.4e}")

    os.makedirs("artefacts", exist_ok=True)
    joblib.dump(study, "artefacts/optuna_study.pkl")
    print("Optuna study saved → artefacts/optuna_study.pkl")
