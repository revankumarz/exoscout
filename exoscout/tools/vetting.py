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


# Detection / false-positive thresholds, in sigma.
ODDEVEN_SIGMA_CUT = 3.0
SECONDARY_SIGMA_CUT = 3.0
DEPTH_SNR_CUT = 7.1  # TESS SPOC transit-detection threshold


def _in_transit_mask(phase: np.ndarray, half_window: float) -> np.ndarray:
    return np.abs(phase) < half_window


def _robust_sigma(x: np.ndarray) -> float:
    """MAD-based per-point scatter, immune to the outliers std would chase."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1e-6
    mad = float(np.nanmedian(np.abs(x - np.nanmedian(x))))
    sigma = 1.4826 * mad
    if sigma <= 0:  # degenerate (e.g. flat synthetic flux) - fall back to std
        sigma = float(np.nanstd(x))
    return sigma if sigma > 0 else 1e-6


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
        sec_mask = np.abs(np.abs(phase) - 0.5) < half_win

        # Baseline/noise from flux that is in neither the primary nor the
        # secondary window - otherwise a real secondary biases both.
        out_tr = ~in_tr & ~sec_mask
        if out_tr.sum() < 20:
            out_tr = ~in_tr

        base_flux = flux[out_tr] if out_tr.any() else flux
        baseline = float(np.nanmedian(base_flux))
        # Robust per-point scatter (MAD-based); std is inflated by residual
        # systematics and outliers that survived cleaning.
        noise = _robust_sigma(base_flux)

        def depth_of(mask):
            """Mean depth in a window and its standard error."""
            sel = mask & np.isfinite(flux)
            n = int(sel.sum())
            if n < 3:
                return np.nan, np.nan, 0
            depth = float(baseline - np.nanmean(flux[sel]))
            return depth, noise / np.sqrt(n), n

        odd_depth, odd_err, n_odd = depth_of(in_tr & (epoch % 2 == 1))
        even_depth, even_err, n_even = depth_of(in_tr & (epoch % 2 == 0))
        full_depth, full_err, n_in = depth_of(in_tr)

        # Odd/even difference in units of its own uncertainty -> EB indicator.
        # The error on the difference propagates from both windows; using the
        # per-point scatter here would understate it by ~sqrt(N) and hide EBs.
        if np.isfinite(odd_depth) and np.isfinite(even_depth):
            diff_err = np.hypot(odd_err, even_err)
            oddeven_sigma = abs(odd_depth - even_depth) / (diff_err + 1e-12)
        else:
            oddeven_sigma = np.nan

        # Secondary eclipse near phase 0.5.
        secondary_depth, secondary_err, n_sec = depth_of(sec_mask)
        if np.isfinite(secondary_depth):
            secondary_sigma = secondary_depth / (secondary_err + 1e-12)
        else:
            secondary_sigma = np.nan

        # Depth SNR is depth over the *error on the mean depth*, which is the
        # quantity thresholds like the TESS SPOC's 7.1-sigma cut refer to.
        depth_snr = (full_depth / (full_err + 1e-12)) if np.isfinite(full_depth) else np.nan

        # Rule-based flags. The significances above are now expressed in units
        # of their own uncertainty, so these are the conventional cuts:
        # 3-sigma for the EB tells, and the SPOC 7.1-sigma detection threshold.
        likely_eb = bool(np.isfinite(oddeven_sigma) and oddeven_sigma > ODDEVEN_SIGMA_CUT)
        has_secondary = bool(np.isfinite(secondary_sigma) and secondary_sigma > SECONDARY_SIGMA_CUT)
        weak_signal = bool(not np.isfinite(depth_snr) or depth_snr < DEPTH_SNR_CUT)

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
            "depth_ppm": full_depth * 1e6 if np.isfinite(full_depth) else None,
            "depth_err_ppm": full_err * 1e6 if np.isfinite(full_err) else None,
            "noise_ppm": noise * 1e6,
            "odd_depth_ppm": odd_depth * 1e6 if np.isfinite(odd_depth) else None,
            "even_depth_ppm": even_depth * 1e6 if np.isfinite(even_depth) else None,
            "oddeven_sigma": None if np.isnan(oddeven_sigma) else round(oddeven_sigma, 2),
            "secondary_depth_ppm": secondary_depth * 1e6 if np.isfinite(secondary_depth) else None,
            "secondary_sigma": None if np.isnan(secondary_sigma) else round(secondary_sigma, 2),
            "depth_snr": None if np.isnan(depth_snr) else round(depth_snr, 2),
            "n_in_transit": n_in,
            "n_secondary": n_sec,
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
