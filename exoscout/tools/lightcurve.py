"""Light-curve tool - fetch a TESS light curve and run a basic transit search.

Uses Lightkurve (public MAST data, no token). Pipeline:
  1. search + download a TESS light curve for the target
  2. clean / flatten / normalise the flux
  3. Box Least Squares (BLS) period search for the strongest transit signal
  4. phase-fold at the best period for plotting

Returns structured, plot-ready arrays plus the fitted transit parameters. The
eventual 'Vetting agent' will call this and layer the student's CNN on top.
"""

from __future__ import annotations

import numpy as np


def fetch_and_search(search_string: str, max_period: float = 15.0) -> dict:
    """Fetch a TESS light curve and run a BLS transit search.

    Returns keys: ok, error, source, sector_count, n_points, period, t0,
    depth, duration, depth_ppm, snr, phase, phase_flux, time, flux.
    """
    source = "MAST / Lightkurve (TESS)"
    try:
        import lightkurve as lk
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"lightkurve import failed: {e}", "source": source}

    try:
        search = lk.search_lightcurve(search_string, mission="TESS")
        if len(search) == 0:
            return {
                "ok": False,
                "error": f"No TESS light curve found for '{search_string}'.",
                "source": source,
            }

        # Prefer short-cadence SPOC/QLP products; just take what is available.
        collection = search.download_all()
        if collection is None or len(collection) == 0:
            return {"ok": False, "error": "Download returned no data.", "source": source}

        # Stitch sectors, remove NaNs/outliers, flatten stellar variability.
        lc = collection.stitch().remove_nans().remove_outliers(sigma=5)
        flat = lc.flatten(window_length=401)

        time = np.asarray(flat.time.value, dtype=float)
        flux = np.asarray(flat.flux.value, dtype=float)
        good = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[good], flux[good]
        if time.size < 200:
            return {"ok": False, "error": "Too few valid points after cleaning.", "source": source}

        # BLS period search.
        baseline = float(time.max() - time.min())
        pmax = float(min(max_period, max(2.0, baseline / 2.0)))
        periods = np.linspace(0.5, pmax, 8000)
        bls = flat.to_periodogram(method="bls", period=periods)

        period = float(bls.period_at_max_power.value)
        t0 = float(bls.transit_time_at_max_power.value)
        duration = float(bls.duration_at_max_power.value)
        depth = float(bls.depth_at_max_power)
        # Rough SNR proxy: peak power over median power.
        power = np.asarray(bls.power.value, dtype=float)
        snr = float(np.nanmax(power) / (np.nanmedian(power) + 1e-9))

        # Phase fold for plotting.
        phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
        order = np.argsort(phase)

        return {
            "ok": True,
            "error": None,
            "source": source,
            "sector_count": len(collection),
            "n_points": int(time.size),
            "period": period,
            "t0": t0,
            "duration": duration,
            "depth": depth,
            "depth_ppm": depth * 1e6,
            "snr": snr,
            "phase": phase[order].tolist(),
            "phase_flux": flux[order].tolist(),
            "time": time.tolist(),
            "flux": flux.tolist(),
        }
    except Exception as e:  # keep the tool total: never raise to the agent
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "source": source}
