"""Stellar-context tool - what kind of star / object is this?

Best-effort enrichment via SIMBAD (astroquery). The most useful field is the
object type (``otype``): if SIMBAD already classifies the source as an eclipsing
binary or variable star, that is a strong false-positive signal the transit
search alone cannot give you.

Network- and version-fragile by nature, so this degrades to a clean 'skipped'
result rather than ever breaking the pipeline.
"""

from __future__ import annotations

import warnings


def stellar_context(ra_deg, dec_deg, radius_arcsec: float = 10.0) -> dict:
    """Look up the nearest SIMBAD object to the target coordinates."""
    source = "SIMBAD (astroquery)"
    if ra_deg is None or dec_deg is None:
        return {"ok": True, "skipped": True, "reason": "no coordinates", "source": source}

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad

        sim = Simbad()
        # Field names differ across astroquery versions; add defensively.
        for field in ("otype", "sp_type", "V", "flux(V)", "otypes"):
            try:
                sim.add_votable_fields(field)
            except Exception:
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg)
            res = sim.query_region(coord, radius=radius_arcsec * u.arcsec)

        if res is None or len(res) == 0:
            return {"ok": True, "found": False, "reason": "no SIMBAD match",
                    "source": source}

        row = res[0]

        def _get(*names):
            for n in names:
                for col in res.colnames:
                    if col.lower() == n.lower():
                        val = row[col]
                        try:
                            val = val.decode() if isinstance(val, bytes) else val
                        except Exception:
                            pass
                        if val is not None and str(val).strip() not in ("", "--"):
                            return str(val)
            return None

        otype = _get("otype", "OTYPE", "main_type")
        eb_flag = bool(otype and any(k in otype.upper() for k in ("EB", "ECLIPS", "SB")))

        return {
            "ok": True, "found": True, "source": source,
            "main_id": _get("main_id", "MAIN_ID"),
            "otype": otype,
            "sp_type": _get("sp_type", "SP_TYPE"),
            "vmag": _get("V", "FLUX_V", "flux_V"),
            "eclipsing_binary_flag": eb_flag,
            "summary": (f"SIMBAD: {otype or 'unknown type'}"
                        + (" - flagged as binary/variable (FP risk)" if eb_flag else "")),
        }
    except Exception as e:
        return {"ok": True, "skipped": True, "reason": f"{type(e).__name__}: {e}",
                "source": source}
