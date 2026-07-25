"""Follow-up planning tool - 'where and when can I observe this?'

Given a target's sky position, work out - for a set of ground-based
observatories over the next N nights - whether it is observable under sensible
altitude / airmass / darkness / moon-separation constraints, and on which night
it is best placed. This is the piece that turns a triage verdict into an
actionable observing plan.

Uses astroplan. Observatory coordinates are hard-coded (explicit EarthLocation)
so no site registry download is needed.
"""

from __future__ import annotations

import warnings

import numpy as np

# Curated observatories (name -> lat_deg, lon_deg, height_m). Includes two
# Indian sites alongside the major photometric-follow-up facilities.
OBSERVATORIES = {
    "Vainu Bappu (Kavalur, IN)": (12.5765, 78.8253, 725),
    "IAO Hanle (Ladakh, IN)": (32.7797, 78.9642, 4500),
    "Keck (Mauna Kea, US)": (19.8283, -155.4783, 4160),
    "VLT (Cerro Paranal, CL)": (-24.6275, -70.4044, 2635),
    "Roque (La Palma, ES)": (28.7606, -17.8814, 2396),
}


def plan_followup(ra_deg, dec_deg, target_name: str = "target", nights: int = 14,
                  min_altitude: float = 30.0, max_airmass: float = 2.0,
                  min_moon_sep: float = 30.0) -> dict:
    """Return per-observatory observability over the next ``nights`` nights."""
    source = "astroplan (observability)"
    if ra_deg is None or dec_deg is None:
        return {"ok": False, "error": "No sky coordinates available for this target.",
                "source": source, "observatories": []}

    try:
        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        from astroplan import (Observer, FixedTarget, AltitudeConstraint,
                               AirmassConstraint, AtNightConstraint,
                               MoonSeparationConstraint, observability_table)

        target = FixedTarget(coord=SkyCoord(ra=float(ra_deg) * u.deg,
                                            dec=float(dec_deg) * u.deg),
                             name=target_name)
        constraints = [
            AltitudeConstraint(min=min_altitude * u.deg),
            AirmassConstraint(max=max_airmass),
            AtNightConstraint.twilight_astronomical(),
            MoonSeparationConstraint(min=min_moon_sep * u.deg),
        ]
        start = Time.now()
        end = start + nights * u.day

        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, (lat, lon, height) in OBSERVATORIES.items():
                observer = Observer(location=EarthLocation.from_geodetic(
                    lon * u.deg, lat * u.deg, height * u.m), name=name)
                tbl = observability_table(constraints, observer, [target],
                                          time_range=Time([start, end]),
                                          time_grid_resolution=1 * u.hour)
                ever = bool(tbl["ever observable"][0])
                frac = float(tbl["fraction of time observable"][0])
                # Total hours meeting all constraints across the whole window.
                hours = round(frac * nights * 24, 1)
                results.append({
                    "observatory": name,
                    "observable": ever,
                    "fraction_time": round(frac, 3),
                    "obs_hours_in_window": hours,
                })

        results.sort(key=lambda r: (r["observable"], r["fraction_time"]), reverse=True)
        n_obs = sum(1 for r in results if r["observable"])
        best = results[0]["observatory"] if results and results[0]["observable"] else None
        summary = (f"Observable from {n_obs}/{len(results)} sites over {nights} nights"
                   + (f"; best: {best}" if best else "; not well placed anywhere"))

        return {"ok": True, "error": None, "source": source, "nights": nights,
                "constraints": {"min_altitude": min_altitude, "max_airmass": max_airmass,
                                "min_moon_sep": min_moon_sep},
                "observatories": results, "best": best, "summary": summary}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "source": source,
                "observatories": []}
