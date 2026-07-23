"""Vetting tool - classic transit false-positive diagnostics.

Given a folded light curve and BLS parameters, compute the standard checks a
human vetter (and the TFOPWG) use to separate real planets from eclipsing
binaries and noise:

  * odd vs even transit depth   -> eclipsing-binary tell (unequal depths)
  * secondary eclipse at phase 0.5 -> stellar companion tell
  * transit depth SNR            -> is the signal real?

The student's CNN slots in as an extra classifier via ``cnn_score`` without
changing this function's contract - the agent just gets one more field.
"""

from __future__ import annotations

import numpy as np


def _in_transit_mask(phase: np.ndarray, half_window: float) -> np.ndarray:
    return np.abs(phase) < half_window


def run_vetting(lc: dict, cnn_score=None) -> dict:
    """Compute vetting diagnostics from a light-curve tool result.

    ``lc`` is the dict returned by ``lightcurve.fetch_and_search`` (needs
    period, t0, duration, and the raw time/flux arrays). ``cnn_score`` is an
    optional callable(lc)->float in [0,1]; the student's CNN plugs in here.
    """
    source = "ExoScout vetting (odd/even, secondary, SNR)"
    if not lc or not lc.get("ok"):
        return {"ok": False, "error": "No valid light curve to vet.", "source": source}

    try:
        time = np.asarray(lc["time"], dtype=float)
        flux = np.asarray(lc["flux"], dtype=float)
        period = float(lc["period"])
        t0 = float(lc["t0"])
        duration = float(lc["duration"])  # days

        # Phase in [-0.5, 0.5]; half-transit window as a fraction of the period.
        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        half_win = max(1e-4, (duration / period) / 2.0)

        # Transit epoch number -> odd/even split.
        epoch = np.round((time - t0) / period).astype(int)
        in_tr = _in_transit_mask(phase, half_win)
        out_tr = np.abs(phase) > 0.5 * 0.5  # far from transit, for the baseline

        baseline = np.nanmedian(flux[out_tr]) if out_tr.any() else np.nanmedian(flux)
        noise = np.nanstd(flux[out_tr]) if out_tr.sum() > 5 else np.nanstd(flux)
        noise = float(noise) if noise > 0 else 1e-6

        def depth_of(mask):
            sel = in_tr & mask
            if sel.sum() < 3:
                return np.nan, 0
            return float(baseline - np.nanmedian(flux[sel])), int(sel.sum())

        odd_depth, n_odd = depth_of(epoch % 2 == 1)
        even_depth, n_even = depth_of(epoch % 2 == 0)
        full_depth, n_in = depth_of(np.ones_like(epoch, dtype=bool))

        # Odd/even difference in units of noise (per-point) -> EB indicator.
        if np.isfinite(odd_depth) and np.isfinite(even_depth):
            oddeven_sigma = abs(odd_depth - even_depth) / (noise + 1e-9)
        else:
            oddeven_sigma = np.nan

        # Secondary eclipse near phase 0.5.
        sec_mask = np.abs(np.abs(phase) - 0.5) < half_win
        if sec_mask.sum() >= 3:
            secondary_depth = float(baseline - np.nanmedian(flux[sec_mask]))
            secondary_sigma = secondary_depth / (noise + 1e-9)
        else:
            secondary_depth, secondary_sigma = np.nan, np.nan

        depth_snr = (full_depth / (noise + 1e-9)) if np.isfinite(full_depth) else np.nan

        # Simple rule-based flags (thresholds are deliberately conservative).
        likely_eb = bool(np.isfinite(oddeven_sigma) and oddeven_sigma > 3.0)
        has_secondary = bool(np.isfinite(secondary_sigma) and secondary_sigma > 3.0)
        weak_signal = bool(np.isfinite(depth_snr) and depth_snr < 5.0)

        flags = []
        if likely_eb:
            flags.append("odd/even depth mismatch (eclipsing-binary risk)")
        if has_secondary:
            flags.append("secondary eclipse detected (stellar companion risk)")
        if weak_signal:
            flags.append("low transit-depth SNR (signal may be spurious)")

        if likely_eb or has_secondary:
            headline = "LIKELY FALSE POSITIVE (astrophysical)"
        elif weak_signal:
            headline = "WEAK / INCONCLUSIVE signal"
        else:
            headline = "PASSES basic vetting (planet-consistent)"

        result = {
            "ok": True,
            "error": None,
            "source": source,
            "odd_depth_ppm": odd_depth * 1e6 if np.isfinite(odd_depth) else None,
            "even_depth_ppm": even_depth * 1e6 if np.isfinite(even_depth) else None,
            "oddeven_sigma": None if np.isnan(oddeven_sigma) else round(oddeven_sigma, 2),
            "secondary_depth_ppm": secondary_depth * 1e6 if np.isfinite(secondary_depth) else None,
            "secondary_sigma": None if np.isnan(secondary_sigma) else round(secondary_sigma, 2),
            "depth_snr": None if np.isnan(depth_snr) else round(depth_snr, 2),
            "n_odd": n_odd,
            "n_even": n_even,
            "flags": flags,
            "summary": headline,
        }

        # Optional CNN classifier hook (the student's model).
        if cnn_score is not None:
            try:
                result["cnn_score"] = float(cnn_score(lc))
            except Exception as e:
                result["cnn_score"] = None
                result["cnn_error"] = f"{type(e).__name__}: {e}"

        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "source": source}
