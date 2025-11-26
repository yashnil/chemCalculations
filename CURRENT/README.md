## UPDATED_VERS – FlowMap Autoencoder Prototype

This directory packages a lightweight training loop that swaps the earlier
residual MLP for the FlowMap Autoencoder architecture you shared (`model.py`).
Use it when you want to experiment with the autoencoder while keeping the same
FastChem dataset and normalisation as the previous grid runs.

### Files

| File | Description |
| ---- | ----------- |
| `autoencoder_model.py` | Exact FlowMapAutoencoder implementation (verbatim from the provided `model.py`). |
| `train_autoencoder.py` | Training script that loads the FastChem CSV, resolves inputs/targets, normalises, and fits the autoencoder. |

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn
```

The script assumes the clean FastChem CSV is available at:

```
/Users/yashnilmohanty/Desktop/chemCalculations/NEW_VERS/all_gas_v10_no_stripe_clean.csv
```

If your path differs, set the `CSV_PATH` environment variable before running.

### Training

```bash
cd /Users/yashnilmohanty/Desktop/chemCalculations/UPDATED_VERS
python train_autoencoder.py
```

Outputs (checkpoints, metrics) are written to `runs_autoencoder/`. The script
automatically splits the data into 85/10/5 train/val/test partitions, logs
progress to stdout, and saves a `summary.json` with the final metrics.

### Notes

* The autoencoder expects `(y_i, g, dt)` inputs. Because the FastChem CSV is a
  static snapshot, the training harness feeds a zero state (`y_i = 0`) and
  constant `dt = 1`, so the network effectively learns a residual mapping from
  the global features `g` to the target species. This keeps the architecture
  intact while still fitting the available data.
* If you later add sequential data (multiple `dt` steps per sample), only the
  mini-batch assembly needs to change; the rest of the pipeline remains valid.

