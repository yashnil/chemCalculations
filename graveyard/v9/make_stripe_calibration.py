#!/usr/bin/env python3
#make_stripe_calibration.py
import os, json, joblib, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import HuberRegressor

ARTE_DIR   = "artefacts"
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
MODEL_PATH = os.path.join(ARTE_DIR, "final_model.keras")
CARD_JSON  = os.path.join(ARTE_DIR, "final_report.json")
SCALER_PKL = os.path.join(ARTE_DIR, "input_scaler.pkl")
EPS = 1e-12
LO, HI = 0.013, 0.015   # the stripe

# ── load data exactly like diagnostics ───────────────────────────
card   = json.load(open(CARD_JSON))
inputs = card.get("inputs", ["temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S"])

df = pd.read_csv(CSV_PATH)
df["T_bin"] = pd.qcut(df["temperature"], 5, labels=False, duplicates="drop")
df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")

elem_cols = [f"comp_{e}" for e in ("H","O","C","N","S")]
meta = {"temperature","pressure","group_index","point_index", *elem_cols}
species = [c for c in df.columns if c not in meta]

df[species] = df[species].div(df[species].sum(axis=1), axis=0)
scaler = joblib.load(SCALER_PKL)

X = df[inputs].copy()
X["pressure"] = np.log10(X["pressure"])
for el in inputs[2:]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")
Y_true = df[species].values.astype("float32")

keras.config.enable_unsafe_deserialization()
model = keras.models.load_model(
    MODEL_PATH, compile=False, safe_mode=False, custom_objects={"tf": tf}
)

# --- make sure every Lambda sees `tf` ---
def _inject_tf(layer):
    # If this is a Lambda, add tf into its function globals
    if isinstance(layer, keras.layers.Lambda):
        fn = getattr(layer, "function", None) or getattr(layer, "_function", None)
        if fn is not None:
            fn.__globals__.setdefault("tf", tf)
    # Recurse into containers (Sequential/Functional/nested)
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            _inject_tf(sub)

_inject_tf(model)
# (optional belt-and-suspenders)
for lyr in model.layers:
    if isinstance(lyr, keras.layers.Lambda):
        fn = getattr(lyr, "function", None) or getattr(lyr, "_function", None)
        if fn is not None:
            fn.__globals__.setdefault("tf", tf)

Y_pred = model.predict(X, batch_size=256, verbose=0)
Y_pred = np.maximum(Y_pred, 0.0)
Y_pred /= (Y_pred.sum(axis=1, keepdims=True) + EPS)


# ── robust per-species log offset inside the band ─────────────────
N_MIN   = 30          # need enough samples
TRIM_Q  = 10          # use 10–90% trimmed median
CAP_DEX = 0.25        # never correct by more than ±0.25 dex (~×1.78)
FLOOR_P = 1e-10       # ignore vanishing predictions in residuals

offset = {}
debug  = {}           # optional: for inspection

for j, sp in enumerate(species):
    # collect rows where either observed OR predicted sits in the stripe
    band_true = (Y_true[:, j] >= LO) & (Y_true[:, j] <= HI)
    band_pred = (Y_pred[:, j] >= LO) & (Y_pred[:, j] <= HI)
    band = band_true | band_pred
    n_band = int(band.sum())
    if n_band < N_MIN:
        continue

    # residuals only where both sides are numerically meaningful
    yt = Y_true[band, j]
    yp = Y_pred[band, j]
    ok = (yt > EPS) & (yp > FLOOR_P)
    if ok.sum() < N_MIN:
        continue
    res = np.log10((yp[ok] + EPS) / (yt[ok] + EPS))

    # trimmed median to kill outliers
    lo, hi = np.percentile(res, [TRIM_Q, 100-TRIM_Q])
    core = res[(res >= lo) & (res <= hi)]
    if core.size < N_MIN:
        continue

    raw_med = float(np.median(core))
    off = float(np.clip(-raw_med, -CAP_DEX, CAP_DEX))  # what we'll apply

    # keep some debug
    iqr = float(np.percentile(core, 75) - np.percentile(core, 25))
    debug[sp] = {"n_band": n_band, "n_used": int(core.size),
                 "raw_med": raw_med, "offset": off, "iqr": iqr}
    offset[sp] = off

# write patch
json.dump(
    {"band": [LO, HI], "offset": offset, "debug": debug,
     "note": "Inside band, do log10(yhat)+=offset[sp] (trimmed, capped at ±0.25 dex); then renormalize."},
    open("stripe_patch.json", "w"),
    indent=2
)

# pretty print a few sane offsets
for sp, off in list(offset.items())[:5]:
    print(f"  {sp}: offset={off:+.4f} dex")
print(f"Wrote stripe_patch.json with {len(offset)} species offsets.")
