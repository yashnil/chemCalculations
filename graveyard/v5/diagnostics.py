#!/usr/bin/env python3
# step-5 → diagnostics.py
# ----------------------------------------------------------------------
# Rich post-training diagnostics for the FastChem-surrogate project
# ----------------------------------------------------------------------

import os, json, joblib, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

ARTE_DIR  = "artefacts"
CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

MODEL_PATH = os.path.join(ARTE_DIR, "surrogate_final.keras")
SCALER_PKL = os.path.join(ARTE_DIR, "input_scaler.pkl")
CARD_JSON  = os.path.join(ARTE_DIR, "model_card.json")

OUT_DIR = os.path.join(ARTE_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

EPS   = 1e-12
CLIP  = 1e-10
LOG10 = np.log(10.0)

# ----------------------------------------------------------------------
# 1.  Load data
# ----------------------------------------------------------------------
df       = pd.read_csv(CSV_PATH)
scaler   = joblib.load(SCALER_PKL)
model    = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=True)

with open(CARD_JSON) as fh:
    card = json.load(fh)

INPUTS   = card["inputs"]        # 7 inputs
SPECIES  = card["outputs"]       # 116 outputs

X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")

Y_true = df[SPECIES].values.astype("float32")

t0 = time.time()
Y_pred = model.predict(X, batch_size=256, verbose=0)
print(f"Predicted {len(X):,} samples in {time.time()-t0:.1f} s")

Y_pred = np.maximum(Y_pred, 0.0)
Y_pred /= Y_pred.sum(axis=1, keepdims=True) + EPS

# ----------------------------------------------------------------------
# 2.  Global metrics
# ----------------------------------------------------------------------
global_mae  = mean_absolute_error(Y_true, Y_pred)
global_r2   = r2_score(Y_true, Y_pred, multioutput="variance_weighted")
balanced_kl = np.mean(
    np.sum(Y_true * (np.log(Y_true + EPS) - np.log(Y_pred + EPS)), axis=1)
)

print(f"GLOBAL  MAE={global_mae:.4e}   R²={global_r2:.3f}   B-KL={balanced_kl:.4e}")

with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
    fh.write(f"MAE        {global_mae:.6e}\n")
    fh.write(f"R2         {global_r2:.6f}\n")
    fh.write(f"BAL_KL     {balanced_kl:.6e}\n")

# ----------------------------------------------------------------------
# 3.  Per-species table
# ----------------------------------------------------------------------
mae_sp = np.mean(np.abs(Y_true - Y_pred), axis=0)
r2_sp  = [r2_score(Y_true[:, i], Y_pred[:, i]) for i in range(len(SPECIES))]
tbl = pd.DataFrame({"species": SPECIES, "MAE": mae_sp, "R2": r2_sp,
                    "max_abun": Y_true.max(axis=0)})
tbl.sort_values("MAE").to_csv(os.path.join(OUT_DIR, "per_species_errors.csv"),
                              index=False)

# ----------------------------------------------------------------------
# 4.  Parity plot (top-10 most abundant species)
# ----------------------------------------------------------------------
top10 = tbl.sort_values("max_abun", ascending=False).head(10)["species"].values
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for ax, sp in zip(axs, top10):
    idx = SPECIES.index(sp)
    true = np.clip(Y_true[:, idx], CLIP, None)
    pred = np.clip(Y_pred[:, idx], CLIP, None)

    ax.scatter(true, pred, s=4, alpha=0.35, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(CLIP, 1.0); ax.set_ylim(CLIP, 1.0)
    ax.plot([CLIP, 1.0], [CLIP, 1.0], 'r--', lw=1)
    # 1-dex band
    ax.fill_between([CLIP, 1], [CLIP*10, 10], [CLIP/10, 0.1],
                    color="grey", alpha=0.10)
    ax.set_title(sp, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parity_top10.png"), dpi=180)
plt.close()

# ----------------------------------------------------------------------
# 5.  Residual T–P hex-bin for the worst species
# ----------------------------------------------------------------------
worst_idx = int(tbl["MAE"].idxmax())
sp        = tbl.loc[worst_idx, "species"]
residual  = Y_pred[:, worst_idx] - Y_true[:, worst_idx]

plt.figure(figsize=(6.2, 4.8))
hb = plt.hexbin(df["temperature"], np.log10(df["pressure"]),
                C=residual, gridsize=70, cmap="coolwarm",
                mincnt=3, linewidths=0.0)
plt.colorbar(hb, label="Pred − True")
plt.xlabel("Temperature [K]")
plt.ylabel("log10 Pressure [bar]")
plt.title(f"Residual hex-bin – worst MAE species: {sp}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"residual_TP_{sp}.png"), dpi=180)
plt.close()

# ----------------------------------------------------------------------
# 6.  MAE vs species – colour-mapped by log max(abundance)
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 4))
rank   = tbl["MAE"].argsort().values
colors = sns.color_palette("viridis", as_cmap=True)(
    (np.log10(tbl["max_abun"].values[rank]) - np.log10(CLIP)) /
    (np.log10(tbl["max_abun"].values[rank]).max() - np.log10(CLIP))
)
plt.bar(np.arange(len(SPECIES)), tbl["MAE"].values[rank], color=colors)
plt.xticks(np.arange(len(SPECIES)), tbl["species"].values[rank],
           rotation=90, fontsize=6)
plt.ylabel("MAE")
plt.title("Per-species MAE (colour = log10 max abundance)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "MAE_per_species.png"), dpi=180)
plt.close()

# ----------------------------------------------------------------------
# 7.  Sample-wise %-error timeline
# ----------------------------------------------------------------------
percent_dev = (np.abs(Y_pred - Y_true) / (Y_true + EPS)) * 100.0
sample_dev  = percent_dev.mean(axis=1)

plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(sample_dev)), sample_dev, '.',
         markersize=2.5, alpha=0.55)
plt.xlabel("Sample index")
plt.ylabel("Mean % error")
plt.title("Per-sample deviation  ⟨|Pred-True|/True⟩ × 100")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_deviation.png"), dpi=180)
plt.close()

# ----------------------------------------------------------------------
# 8.  Cumulative-error curve  &  abundance-vs-MAE scatter
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# (a) cumulative share of total MAE
cum = np.cumsum(tbl["MAE"].values[rank]) / tbl["MAE"].sum()
ax[0].plot(np.arange(1, len(SPECIES)+1), cum, lw=2)
ax[0].axhline(0.95, ls='--', color='k')
ax[0].set_xlabel("# species"); ax[0].set_ylabel("Cumulative MAE share")
ax[0].set_ylim(0, 1); ax[0].set_title("Cumulative MAE curve")

# (b) abundance vs error
ax[1].scatter(np.log10(tbl["max_abun"]), np.log10(tbl["MAE"]), s=30, alpha=0.7)
ax[1].set_xlabel("log10 max(abundance)")
ax[1].set_ylabel("log10 MAE")
ax[1].set_title("Abundance vs MAE")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "extra_diagnostics.png"), dpi=180)
plt.close()

# ----------------------------------------------------------------------
# 9.  Dump worst-100 samples for inspection
# ----------------------------------------------------------------------
idx_worst = (-sample_dev).argsort()[:100]
df.iloc[idx_worst].assign(mean_pct_err=sample_dev[idx_worst]) \
  .to_csv(os.path.join(OUT_DIR, "worst100_samples.csv"), index=False)

print(f"\nDiagnostics written to  {OUT_DIR}")
