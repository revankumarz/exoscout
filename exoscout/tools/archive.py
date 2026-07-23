"""Archive tool - 'is this candidate already known?'

Queries the NASA Exoplanet Archive TAP service (no token required):
  base: https://exoplanetarchive.ipac.caltech.edu/TAP/sync

Two relevant tables:
  * ``toi``        - the full TESS Object of Interest catalogue, with a TFOPWG
                     disposition (PC/CP/KP/FP/APC/FA) per candidate.
  * ``pscomppars`` - confirmed/composite planet parameters.

We resolve a target to its TIC id, then ask both tables what they know about
it. The result is a structured verdict the UI (and, later, the agent) uses.
"""

from __future__ import annotations

import io
import requests
import pandas as pd

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TIMEOUT = 45

# TFOPWG disposition codes -> human meaning.
DISP_MEANING = {
    "PC": "Planet Candidate",
    "CP": "Confirmed Planet",
    "KP": "Known Planet",
    "FP": "False Positive",
    "FA": "False Alarm",
    "APC": "Ambiguous Planet Candidate",
    "EB": "Eclipsing Binary",
}


def _tap_query(adql: str) -> pd.DataFrame:
    """Run a synchronous ADQL query, return a DataFrame (may be empty)."""
    r = requests.get(
        TAP_URL,
        params={"query": adql, "format": "csv"},
        timeout=TIMEOUT,
        headers={"User-Agent": "ExoScout/0.1 (portfolio project)"},
    )
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def _resolve_tic(tic_id: int | None, toi: float | None) -> tuple[int | None, pd.DataFrame]:
    """Return (tic_id, toi_rows). If only a TOI number is given, look up its TIC."""
    if tic_id is not None:
        rows = _tap_query(
            f"select toi, tid, tfopwg_disp, pl_orbper, pl_trandurh, pl_trandep, "
            f"st_tmag, ra, dec from toi where tid = {int(tic_id)}"
        )
        return int(tic_id), rows

    if toi is not None:
        # match either the exact planet (700.01) or the star prefix (700)
        prefix = int(toi)
        rows = _tap_query(
            f"select toi, tid, tfopwg_disp, pl_orbper, pl_trandurh, pl_trandep, "
            f"st_tmag, ra, dec from toi where toi = {toi:g} or toipfx = {prefix}"
        )
        resolved = int(rows["tid"].iloc[0]) if not rows.empty else None
        return resolved, rows

    return None, pd.DataFrame()


def check_known(tic_id: int | None = None, toi: float | None = None) -> dict:
    """Return a structured verdict on whether a target is already catalogued.

    Keys: ok, tic_id, known (bool|None), disposition, confirmed (bool),
    toi_matches (list of dicts), confirmed_matches (list), summary, source, error.
    """
    source = TAP_URL
    try:
        resolved_tic, toi_rows = _resolve_tic(tic_id, toi)
    except requests.RequestException as e:
        return {
            "ok": False, "error": f"TAP request failed: {e}",
            "known": None, "summary": "Archive unreachable", "source": source,
        }

    confirmed_rows = pd.DataFrame()
    if resolved_tic is not None:
        try:
            confirmed_rows = _tap_query(
                "select pl_name, hostname, tic_id, discoverymethod, disc_year, "
                f"pl_orbper from pscomppars where tic_id = 'TIC {resolved_tic}'"
            )
        except requests.RequestException:
            confirmed_rows = pd.DataFrame()

    toi_matches = toi_rows.to_dict("records") if not toi_rows.empty else []
    confirmed_matches = confirmed_rows.to_dict("records") if not confirmed_rows.empty else []

    # Derive a headline disposition.
    dispositions = [
        str(m.get("tfopwg_disp")).upper()
        for m in toi_matches
        if m.get("tfopwg_disp") and str(m.get("tfopwg_disp")).lower() != "nan"
    ]
    confirmed = len(confirmed_matches) > 0 or any(d in ("CP", "KP") for d in dispositions)
    known = bool(toi_matches or confirmed_matches)

    if confirmed:
        headline = "CONFIRMED / KNOWN PLANET"
    elif any(d in ("FP", "FA") for d in dispositions):
        headline = "FLAGGED FALSE POSITIVE in TOI catalogue"
    elif "PC" in dispositions or "APC" in dispositions:
        headline = "EXISTING PLANET CANDIDATE (TOI already listed)"
    elif known:
        headline = "Listed in TOI catalogue"
    else:
        headline = "NOT found in TOI or confirmed-planet tables"

    disp_human = ", ".join(
        f"{d} ({DISP_MEANING.get(d, 'unknown')})" for d in dict.fromkeys(dispositions)
    ) or "n/a"

    return {
        "ok": True,
        "error": None,
        "tic_id": resolved_tic,
        "known": known,
        "confirmed": confirmed,
        "disposition": disp_human,
        "toi_matches": toi_matches,
        "confirmed_matches": confirmed_matches,
        "summary": headline,
        "source": source,
    }
