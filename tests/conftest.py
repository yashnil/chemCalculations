"""
Pytest configuration.

On macOS (especially Apple Silicon), importing PyTorch after NumPy/OpenBLAS can
trigger a duplicate OpenMP runtime and abort the interpreter with::

    Fatal Python error: Aborted

Set these *before* torch/numpy are imported by tests.
"""

from __future__ import annotations

import os

# See: https://github.com/pytorch/pytorch/issues/37377 (OpenMP / MKL)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Helps some Accelerate/vecLib setups on macOS
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
