
#!/usr/bin/env python3

# step 5 -> diagnostics.py
# ---------------------------------------------------------------------------

import os, json, joblib, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras
import seaborn as sns
from utils import EPS


# ──────────────────────────────────────────────────────────────────────────
# 0)  paths  (edit if needed)
# ──────────────────────────────────────────────────────────────────────────
ARTE_DIR  = "artefacts"
CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

MODEL_PATH = os.path.join(ARTE_DIR, "surrogate_final.keras")
SCALER_PKL = os.path.join(ARTE_DIR, "input_scaler.pkl")
CARD_JSON  = os.path.join(ARTE_DIR, "model_card.json")

OUT_DIR = os.path.join(ARTE_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# 1)  load everything
# ──────────────────────────────────────────────────────────────────────────
df       = pd.read_csv(CSV_PATH)
scaler   = joblib.load(SCALER_PKL)
model    = keras.models.load_model(MODEL_PATH)

with open(CARD_JSON) as fh:
    card = json.load(fh)
INPUTS   = card["inputs"]          # 7 input names
SPECIES  = card["outputs"]         # 116 outputs (order matches training)

# prepare inputs exactly as during training
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in ["comp_H","comp_O","comp_C","comp_N","comp_S"]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")

Y_true = df[SPECIES].values.astype("float32")
Y_pred = model.predict(X, batch_size=256, verbose=0)

# model output is log10 -> back to linear
Y_pred = np.power(10.0, Y_pred)
Y_pred[Y_pred < EPS] = EPS          # floor for safety
Y_pred /= Y_pred.sum(axis=1, keepdims=True)

# ──────────────────────────────────────────────────────────────────────────
# 2)  global metrics
# ──────────────────────────────────────────────────────────────────────────
global_mae = mean_absolute_error(Y_true, Y_pred)
global_r2  = r2_score(Y_true, Y_pred, multioutput="variance_weighted")
print(f"GLOBAL metrics  –  MAE = {global_mae:.4e}   weighted R² = {global_r2:.3f}")

# save to disk for the report
with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
    fh.write(f"MAE  {global_mae:.6e}\nR2   {global_r2:.6f}\n")

# ──────────────────────────────────────────────────────────────────────────
# 3)  per‑species error table
# ──────────────────────────────────────────────────────────────────────────
mae_sp = np.mean(np.abs(Y_true - Y_pred), axis=0)
r2_sp  = [r2_score(Y_true[:,i], Y_pred[:,i]) for i in range(len(SPECIES))]
tbl    = pd.DataFrame({"species":SPECIES, "MAE":mae_sp, "R2":r2_sp})
tbl.sort_values("MAE").to_csv(os.path.join(OUT_DIR, "per_species_errors.csv"), index=False)

# ──────────────────────────────────────────────────────────────────────────
# 4)  parity plot for the TOP-10 most abundant species (overall)
#     • axes on log–log scale, 1 × 10⁻¹⁰ … 1
#     • values <1e-10 are clipped for display (they’re still in the metrics)
# ──────────────────────────────────────────────────────────────────────────
CLIP = 1e-10                     # anything below this is shown at the floor
top10 = (Y_true.max(axis=0)).argsort()[-10:][::-1]

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for ax, idx in zip(axs, top10):
    true = np.clip(Y_true[:, idx], CLIP, None)
    pred = np.clip(Y_pred[:, idx], CLIP, None)

    ax.scatter(true, pred, s=4, alpha=0.4, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(CLIP, 1.0); ax.set_ylim(CLIP, 1.0)
    ax.plot([CLIP, 1.0], [CLIP, 1.0], 'r--', lw=1)
    ax.set_title(SPECIES[idx], fontsize=9)
    ax.set_xlabel("True"); ax.set_ylabel("Pred")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parity_top10.png"), dpi=180)
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 5)  residual heat-map in T–P space for the single worst-MAE species
#     (unchanged apart from section number)
# ──────────────────────────────────────────────────────────────────────────
worst_idx = mae_sp.argmax()
sp        = SPECIES[worst_idx]
residual  = Y_pred[:, worst_idx] - Y_true[:, worst_idx]

plt.figure(figsize=(6, 4.5))
sc = plt.scatter(df["temperature"], np.log10(df["pressure"]),
                 c=residual, cmap="coolwarm", s=8)
plt.colorbar(sc, label="Pred − True")
plt.xlabel("Temperature [K]")
plt.ylabel("log10 Pressure [bar]")
plt.title(f"Residuals | worst-MAE species: {sp}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"residual_TP_{sp}.png"), dpi=180)
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 6)  MAE vs. species (bar chart – optional, unchanged)
# ──────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
rank = np.argsort(mae_sp)
plt.bar(np.arange(len(SPECIES)), mae_sp[rank])
plt.xticks(np.arange(len(SPECIES)), np.array(SPECIES)[rank],
           rotation=90, fontsize=6)
plt.ylabel("MAE")
plt.title("Per-species MAE (sorted)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "MAE_per_species.png"), dpi=180)
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 7)  NEW  –  sample-wise deviation plot
#     • deviation = mean(|Pred − True|) across the 116 species
#     • plotted for every sample (x-axis = dataset index)
# ──────────────────────────────────────────────────────────────────────────
percent_dev = (np.abs(Y_pred - Y_true) / (Y_true + EPS)) * 100.0
sample_dev  = percent_dev.mean(axis=1)          # mean % per sample

plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(sample_dev)), sample_dev, '.',
         markersize=3, alpha=0.6)
plt.xlabel("Sample index (case #)")
plt.ylabel("Mean % error")
plt.title("Per-sample deviation (%):  mean(|Pred-True|/True) × 100")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_deviation.png"), dpi=180)
plt.close()


print("\nDiagnostics written to:", OUT_DIR)
