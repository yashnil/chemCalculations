"""Smoke tests: package imports and version."""

from __future__ import annotations


def test_version():
    import chemcalculations

    assert hasattr(chemcalculations, "__version__")
    assert chemcalculations.__version__ == "0.1.1"


def test_project_root():
    from chemcalculations import project_root

    p = project_root()
    assert (p / "README.md").exists()
    assert (p / "pyproject.toml").exists()


def test_autoencoder_import():
    from chemcalculations.autoencoder_model import FlowMapAutoencoder, SimpleMLP

    assert FlowMapAutoencoder is not None
    assert SimpleMLP is not None


def test_mfae_metrics_import():
    from chemcalculations.mfae_metrics import WINSOR_CAP, compute_mfae_from_arrays

    assert WINSOR_CAP > 0
