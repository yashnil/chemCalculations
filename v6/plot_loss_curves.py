
# plot_loss_curves.py
import json
import matplotlib.pyplot as plt

with open("artefacts/history.json") as f:
    h = json.load(f)

epochs = range(1, len(h["loss"])+1)

plt.figure(figsize=(6,4))
plt.plot(epochs, h["loss"],    label="train loss")
plt.plot(epochs, h["val_loss"],label="val loss")
plt.xlabel("Epoch"); plt.ylabel("Composite loss")
plt.legend()
plt.tight_layout()
plt.savefig("artefacts/loss_curve.png", dpi=150)
plt.close()

# if you logged the mae_log metric under name 'mae_log':
if "mae_log" in h:
    plt.figure(figsize=(6,4))
    plt.plot(epochs, h["mae_log"],     label="train log-MAE")
    plt.plot(epochs, h["val_mae_log"], label="val log-MAE")
    plt.xlabel("Epoch"); plt.ylabel("log₁₀ MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artefacts/logmae_curve.png", dpi=150)
    plt.close()
