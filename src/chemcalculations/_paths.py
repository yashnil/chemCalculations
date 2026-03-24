"""Repository root resolution for scripts living in ``src/chemcalculations/``."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent
