# Configuration Files

This directory contains JSON configuration files for training the FlowMapAutoencoder model.

## Files

- **`default_config.json`**: Default hyperparameters matching the current best model architecture
- **`x160_logratio_config.json`**: Configuration for log-ratio loss experiment on x160 dataset

## Usage

### Using a config file:

```bash
python -m chemcalculations.train_autoencoder \
    --config configs/default_config.json \
    --loss-type huber \
    --run-dir results/runs/runs_autoencoder_test
```

### Without config file (uses module-level constants):

```bash
python -m chemcalculations.train_autoencoder \
    --loss-type huber \
    --run-dir results/runs/runs_autoencoder_test
```

## Config Structure

```json
{
  "data": {
    "csv_path": "data/datasets/all_gas_fastchem_x160.csv",
    "train_frac": 0.85,
    "val_frac": 0.10,
    "test_frac": 0.05,
    "target_topk_species": 20,
    "include_fz_as_feature": true
  },
  "optimization": {
    "epochs": 200,
    "batch_size": 512,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
    "seed": 42
  },
  "architecture": {
    "latent_dim": 192,
    "encoder_hidden": [512, 512, 512],
    "dynamics_hidden": [512, 512, 512],
    "decoder_hidden": [512, 512, 512],
    "activation": "silu",
    "dropout": 0.0
  },
  "loss": {
    "type": "huber",
    "huber_delta": 0.02,
    "use_weighted": true
  },
  "normalization": {
    "temp_divisor": 4000.0,
    "input_log_scale": 10.0,
    "abund_epsilon_offset": 12.0,
    "abund_dex_scale": 10.0,
    "target_zero_floor": 1e-30,
    "target_log_scale": 30.0,
    "log_eps": 1e-30
  },
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.5,
    "patience": 10,
    "min_lr": 1e-6
  }
}
```

## Loss Types

- **`huber`**: Weighted Huber loss in normalized space (default)
- **`mse`**: Mean Squared Error in normalized space
- **`log_ratio`**: Log-ratio loss in linear space: `L = |log_10(ŷ/y)|`

## Notes

- Config values override module-level constants in `train_autoencoder.py`
- If a config key is missing, the module-level default is used
- The `csv_path` in config can be overridden by the `CSV_PATH` environment variable

