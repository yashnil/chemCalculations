#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

h = json.load(open("artefacts/history.json", "r"))
epochs = range(1, len(h["loss"]) + 1)

# composite‐loss
plt.figure(figsize=(6,4))
plt.plot(epochs, h["loss"],    label="train loss")
plt.plot(epochs, h["val_loss"],label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Composite loss")
plt.legend()
plt.tight_layout()
plt.savefig("artefacts/loss_curve.png", dpi=150)
plt.close()

# log‐MAE (if present)
if "mae_log" in h:
    plt.figure(figsize=(6,4))
    plt.plot(epochs, h["mae_log"],     label="train log-MAE")
    plt.plot(epochs, h["val_mae_log"], label="val   log-MAE")
    plt.xlabel("Epoch")
    plt.ylabel("log₁₀ MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artefacts/logmae_curve.png", dpi=150)
    plt.close()
