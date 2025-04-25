
#!/usr/bin/env python3

# step 3 -> tune.py

import os, warnings, joblib, optuna, numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from utils import load_XY, EPS                        # step‑1 helper

# ────────────────────────────────────────────────────────────────────────
# load the data once
# ────────────────────────────────────────────────────────────────────────
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, species_cols = load_XY()
N_OUT = len(species_cols)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # quiet TensorFlow
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# ────────────────────────────────────────────────────────────────────────
# objective
# ────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.trial.Trial) -> float:
    #   ⚠️  clear graph to avoid variable‑name clashes
    keras.backend.clear_session()

    n_layers = trial.suggest_int("n_layers", 2, 5)
    units    = trial.suggest_categorical("units", [128, 256, 512])
    act      = trial.suggest_categorical("act",  ["gelu", "swish"])
    lr       = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    model = keras.Sequential(name="surrogate")
    model.add(keras.layers.Input((7,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation=act))
    model.add(keras.layers.Dense(N_OUT, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=128,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True, monitor="val_loss")]
    )

    y_pred_log = model.predict(X_val, batch_size=256, verbose=0)
    y_pred_lin = np.power(10.0, y_pred_log)
    y_pred_lin /= y_pred_lin.sum(axis=1, keepdims=True)

    y_true_lin = np.power(10.0, Y_val)
    y_true_lin /= y_true_lin.sum(axis=1, keepdims=True)

    mae = mean_absolute_error(y_true_lin, y_pred_lin)
    return mae

# ────────────────────────────────────────────────────────────────────────
# run Optuna – catch TF errors so the study keeps going
# ────────────────────────────────────────────────────────────────────────
study = optuna.create_study(direction="minimize")
study.optimize(
    objective,
    n_trials=40,
    catch=(tf.errors.InvalidArgumentError,)       # ← extra safety net
)

print("\nBest hyper‑parameters:")
for k, v in study.best_params.items():
    print(f"  {k:10s}: {v}")
print(f"Validation MAE : {study.best_value:.4e}")

# save the study for later inspection
os.makedirs("artefacts", exist_ok=True)
joblib.dump(study, "artefacts/optuna_study.pkl")
print("Optuna study saved → artefacts/optuna_study.pkl")
