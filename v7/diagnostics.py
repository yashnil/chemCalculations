#!/usr/bin/env python3
# step-5 → diagnostics.py  (v2)
# ------------------------------------------------------------
# Post-training diagnostics for the FastChem-surrogate project
# ------------------------------------------------------------

import os, time, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

# ╭──────────────────────── paths ────────────────────────╮
ARTE_DIR   = "artefacts"
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
MODEL_PATH = os.path.join(ARTE_DIR, "final_model.keras")   # ← new name!
SCALER_PKL = os.path.join(ARTE_DIR, "input_scaler.pkl")
OUT_DIR    = os.path.join(ARTE_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
# ╰────────────────────────────────────────────────────────╯

EPS   = 1e-12
CLIP  = 1e-10               # floor for log-scale plots
USE_HEX = True              # set False if you prefer scatter

# ─────────────────────────────────────────────────────────
# 1. Load the data and model
# ─────────────────────────────────────────────────────────
df      = pd.read_csv(CSV_PATH)
scaler  = joblib.load(SCALER_PKL)
model   = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

INPUTS  = ["temperature","pressure",
           "comp_H","comp_O","comp_C","comp_N","comp_S"]
META    = set(INPUTS)|{"group_index","point_index"}
SPECIES = [c for c in df.columns if c not in META]
n_out   = len(SPECIES)

# build scaled X
X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for col in INPUTS[2:]:
    X[col] = np.log10(X[col]) + 9.0
X = scaler.transform(X).astype("float32")
Y_true = df[SPECIES].values.astype("float32")

t0 = time.time()
Y_pred = model.predict(X, batch_size=256, verbose=0)
print(f"Predicted {len(X):,} samples in {time.time()-t0:.1f} s")

# ensure proper normalisation / positivity
Y_pred = np.maximum(Y_pred, 0.0)
Y_pred /= Y_pred.sum(axis=1, keepdims=True) + EPS

# ─────────────────────────────────────────────────────────
# 2. Global metrics
# ─────────────────────────────────────────────────────────
mae_glob = mean_absolute_error(Y_true, Y_pred)
r2_glob  = r2_score(Y_true, Y_pred, multioutput="variance_weighted")
bkl_glob = np.mean(
    np.sum(Y_true * (np.log(Y_true + EPS) - np.log(Y_pred + EPS)), axis=1)
)
print(f"GLOBAL →  MAE={mae_glob:.4e}   R²={r2_glob:.3f}   B-KL={bkl_glob:.4e}")

with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
    fh.write(f"MAE     {mae_glob:.6e}\n")
    fh.write(f"R2      {r2_glob:.6f}\n")
    fh.write(f"BAL_KL  {bkl_glob:.6e}\n")

# ─────────────────────────────────────────────────────────
# 3. Per-species table (MAE & R²)
# ─────────────────────────────────────────────────────────
mae_sp = np.mean(np.abs(Y_true - Y_pred), axis=0)
r2_sp  = [r2_score(Y_true[:,i], Y_pred[:,i]) for i in range(n_out)]
tbl    = pd.DataFrame({"species":SPECIES,
                       "MAE": mae_sp,
                       "R2":  r2_sp,
                       "max_abun": Y_true.max(axis=0)})
tbl.sort_values("MAE").to_csv(
    os.path.join(OUT_DIR, "per_species_errors.csv"), index=False
)

# ─────────────────────────────────────────────────────────
# 4. Parity plots for the 10 most abundant species
# ─────────────────────────────────────────────────────────

JITTER_DEX = 0.03          # 0.03 dex ≈ 7 %; set 0 to disable
GREY       = "#D3D3D3"     # fill colour for ±10 % band

def parity_ax(ax, x_lin, y_lin, title="", clip=CLIP):
    """Density-coloured parity plot with shaded ±10 % band."""
    # jitter only for plotting ------------------------------------------
    if JITTER_DEX > 0:
        noise = 10**np.random.uniform(-JITTER_DEX, JITTER_DEX, size=len(x_lin))
        x_plot = x_lin * noise
        y_plot = y_lin * noise
    else:
        x_plot, y_plot = x_lin, y_lin

    # KDE in log-space ---------------------------------------------------
    lx, ly = np.log10(x_plot), np.log10(y_plot)
    dens   = gaussian_kde(np.vstack([lx, ly]))(np.vstack([lx, ly]))
    order  = dens.argsort()
    lx, ly, dens = lx[order], ly[order], dens[order]

    # convert back to lin for scatter
    ax.scatter(10**lx, 10**ly, c=dens, cmap="viridis",
               s=8, alpha=0.8, linewidths=0)

    # 1-to-1 line
    xx = np.geomspace(clip, 1.0, 300)
    ax.plot(xx, xx, "k--", lw=1)

    # ±10 % shaded band --------------------------------------------------
    ax.fill_between(xx, 0.9*xx, 1.1*xx,
                    color=GREY, alpha=0.25, zorder=0)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(clip, 1.0);   ax.set_ylim(clip, 1.0)
    ax.set_title(title, fontsize=9)

# ---------- make figure for the 10 most abundant species ---------------
top10 = tbl.sort_values("max_abun", ascending=False) \
           .head(10)["species"].values

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for ax, sp in zip(axs, top10):
    idx  = SPECIES.index(sp)
    mask = (Y_true[:, idx] > CLIP) & (Y_pred[:, idx] > CLIP)
    parity_ax(ax,
              Y_true[mask, idx],
              Y_pred[mask, idx],
              title=sp,
              clip=CLIP)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parity_top10.png"), dpi=180)
plt.close()

# ─────────────────────────────────────────────────────────
# 5. Residual T–P hex-bin for the worst-MAE species
# ─────────────────────────────────────────────────────────
worst_idx = tbl["MAE"].idxmax()
sp        = tbl.loc[worst_idx,"species"]
residual  = np.log10(Y_pred[:,worst_idx]+EPS) - \
            np.log10(Y_true[:,worst_idx]+EPS)

plt.figure(figsize=(6.4,4.8))
hb = plt.hexbin(df["temperature"], np.log10(df["pressure"]),
                C=residual, gridsize=70, cmap="coolwarm",
                vmin=-1, vmax=1, mincnt=3, linewidths=0)
plt.colorbar(hb, label=r"$\Delta \log_{10}$ (dex)")
plt.xlabel("Temperature [K]"); plt.ylabel("log₁₀ Pressure [bar]")
plt.title(f"Residual map – worst species: {sp}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,f"residual_TP_{sp}.png"), dpi=180)
plt.close()

# ─────────────────────────────────────────────────────────
# 6. MAE-vs-species bar (colour = log₁₀ max abundance)
# ─────────────────────────────────────────────────────────
plt.figure(figsize=(12,4))
order  = tbl["MAE"].argsort().values
normed = (np.log10(tbl["max_abun"].values[order]) - np.log10(CLIP))
normed = normed / normed.max()
colors = sns.color_palette("viridis", as_cmap=True)(normed)
plt.bar(np.arange(n_out), tbl["MAE"].values[order], color=colors)
plt.xticks(np.arange(n_out), tbl["species"].values[order],
           rotation=90, fontsize=6)
plt.ylabel("MAE"); plt.title("Per-species MAE (colour = log max abundance)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"MAE_per_species.png"), dpi=180)
plt.close()

print(f"\nDiagnostics written to →  {OUT_DIR}")
