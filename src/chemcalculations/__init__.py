"""
chemcalculations — FastChem neural network emulator
===================================================

Train and run a PyTorch surrogate for gas-phase chemical equilibrium (number
densities in cm\\ :sup:`-3`\\ ) given temperature, pressure, and elemental abundances.

Install::

    pip install -e .

Train (example)::

    python -m chemcalculations.train_autoencoder_improved --config configs/x4800_improved.json \\
        --run-dir results/runs/runs_autoencoder_x4800_improved

Public API (typical imports)::

    from chemcalculations.autoencoder_model import FlowMapAutoencoder, SimpleMLP
    from chemcalculations.mfae_metrics import compute_mfae_for_run
"""

from __future__ import annotations

from ._paths import project_root

__version__ = "0.1.0"

__all__ = ["__version__", "project_root"]
