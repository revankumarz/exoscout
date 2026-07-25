"""Light curve -> AstroNet input views.

Given a light curve and a transit ephemeris (period, epoch, duration), produce
the two phase-folded, binned, normalised views the CNN expects. Normalisation
follows Shallue & Vanderburg: subtract the median, then divide by the absolute
value of the minimum, so the out-of-transit level is ~0 and the transit depth
is ~ -1 (scale-invariant to the star's brightness).
"""

from __future__ import annotations

import numpy as np

from .astronet import GLOBAL_BINS, LOCAL_BINS


def _fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    return ((time - t0 + 0.5 * period) % period) / period - 0.5  # phase in [-0.5, 0.5)


def _bin_median(phase: np.ndarray, flux: np.ndarray, n_bins: int,
                lo: float, hi: float) -> np.ndarray:
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    view = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = flux[idx == b]
        if sel.size:
            view[b] = np.median(sel)
    # Fill empty bins by linear interpolation over the available ones.
    good = np.isfinite(view)
    if good.sum() < 2:
        return np.zeros(n_bins)
    view = np.interp(np.arange(n_bins), np.flatnonzero(good), view[good])
    return view


def _normalise(view: np.ndarray) -> np.ndarray:
    view = view - np.median(view)
    lo = np.min(view)
    if lo < 0:
        view = view / np.abs(lo)
    return view.astype(np.float32)


def make_views(time, flux, period: float, t0: float, duration: float,
               global_bins: int = GLOBAL_BINS, local_bins: int = LOCAL_BINS,
               n_durations: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (global_view, local_view), each 1D float32 and normalised."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]

    phase = _fold(time, period, t0)

    # Global: the whole folded curve.
    g = _normalise(_bin_median(phase, flux, global_bins, -0.5, 0.5))

    # Local: a window of a few transit durations around the transit centre.
    half = min(0.5, max(1e-3, n_durations * (duration / period)))
    m = np.abs(phase) < half
    if m.sum() >= local_bins // 4:
        l = _normalise(_bin_median(phase[m], flux[m], local_bins, -half, half))
    else:
        # Not enough in-transit coverage: fall back to a central slice of global.
        c = global_bins // 2
        w = local_bins // 2
        l = _normalise(np.interp(np.linspace(c - w, c + w, local_bins),
                                 np.arange(global_bins), g))
    return g, l
