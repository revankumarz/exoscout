"""Inference - score a light curve with the trained AstroNet.

Exposes ``cnn_score(lc)``, the callable the vetting tool's ``cnn_score`` hook
expects: it takes a light-curve tool result (time/flux + BLS ephemeris), builds
the two views, and returns P(planet) in [0, 1]. Returns None if no model has
been trained yet, so the pipeline works with or without the CNN.
"""

from __future__ import annotations

import os
import functools

import numpy as np

from exoscout.paths import MODEL_PATH


@functools.lru_cache(maxsize=1)
def _get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from .astronet import load_model
        return load_model(MODEL_PATH)
    except Exception:
        return None


def model_available() -> bool:
    return _get_model() is not None


def cnn_score(lc: dict) -> float | None:
    """P(planet) for a light-curve tool result, or None if unavailable."""
    model = _get_model()
    if model is None or not lc or not lc.get("ok"):
        return None
    try:
        import torch
        from .preprocess import make_views
        g, l = make_views(lc["time"], lc["flux"], lc["period"], lc["t0"], lc["duration"])
        with torch.no_grad():
            logit = model(torch.tensor(g).unsqueeze(0), torch.tensor(l).unsqueeze(0))
            return float(torch.sigmoid(logit).item())
    except Exception:
        return None
