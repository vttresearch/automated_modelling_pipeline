"""
Docstring for amp.torch.utils

This file contains utility functions for PyTorch models
"""

from typing import Sequence

import numpy as np
import torch


def _tensor_to_np(t):
    """Convert a torch.Tensor (or None) to a numpy array."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _fmt(arr, indent=4):
    """Pretty-print a numpy array with consistent indentation."""
    pad = ' ' * indent
    lines = np.array2string(
        arr,
        precision=4,
        suppress_small=True,
        separator=',  ',
        floatmode='fixed',
    ).splitlines()
    return ('\n' + pad).join(lines)


def print_model_attributes(model, model_name: str, attrs: Sequence[str] | None = None):
    """Print requested attributes of a trained AMP model.

    Parameters
    ----------
    model:
        A trained AMP model instance.
    model_name:
        Human-readable name shown in the header.
    attrs:
        Attribute names to print.  Each entry may be a plain name
        (e.g. ``'A'``) or a dotted path to a nested attribute
        (e.g. ``'kf.Q'``).  Tensors and numpy arrays are pretty-printed;
        all other values are printed with ``repr()``.
        When *None* the function falls back to a default set of
        state-space matrices and Kalman-filter covariances:
        ``A, B, C, E, D, b_x, b_y, kf.Q, kf.R, kf.P0``.
    """
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"  type : {type(model).__name__}")

    # ── Context fields ────────────────────────────────────────────────────────
    for ctx_attr in ('latent_dim', 'control_features', 'disturbance_features',
                     'observation_features'):
        val = getattr(model, ctx_attr, None)
        if val is not None:
            print(f"  {ctx_attr} : {val}")

    # ── Default attribute set ─────────────────────────────────────────────────
    if attrs is None:
        attrs = ['A', 'B', 'C', 'E', 'D', 'b_x', 'b_y',
                 'kf.Q', 'kf.R', 'kf.P0']

    # ── Print each requested attribute ───────────────────────────────────────
    for attr in attrs:
        # Support dotted paths like 'kf.Q'
        parts = attr.split('.')
        obj = model
        try:
            for part in parts:
                obj = getattr(obj, part)
        except AttributeError:
            continue

        arr = _tensor_to_np(obj)
        if arr is not None:
            print(f"\n  {attr}  shape={arr.shape}")
            print(f"    {_fmt(arr)}")
        else:
            print(f"\n  {attr} : {obj!r}")

    print(f"{'='*70}")