"""Target identifier parsing.

Users type things like 'TOI 700', 'TOI-700.01', 'TIC 150428135', or a bare
number. We normalise into a small record the tools can rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Target:
    raw: str
    tic_id: int | None = None       # e.g. 150428135
    toi: float | None = None        # e.g. 700.01 or 700
    search_string: str = ""         # what to hand to Lightkurve's search

    @property
    def label(self) -> str:
        if self.toi is not None:
            return f"TOI {self.toi:g}"
        if self.tic_id is not None:
            return f"TIC {self.tic_id}"
        return self.raw


def parse_target(text: str) -> Target:
    """Best-effort parse of a user-supplied target string."""
    raw = (text or "").strip()
    up = raw.upper().replace("_", " ")

    # TOI 700 / TOI-700.01 / TOI700.01
    m = re.search(r"TOI[\s\-]*([0-9]+(?:\.[0-9]+)?)", up)
    if m:
        toi = float(m.group(1))
        return Target(raw=raw, toi=toi, search_string=f"TOI {toi:g}")

    # TIC 150428135 / TIC-150428135
    m = re.search(r"TIC[\s\-]*([0-9]+)", up)
    if m:
        tic = int(m.group(1))
        return Target(raw=raw, tic_id=tic, search_string=f"TIC {tic}")

    # Bare number -> assume TIC id
    m = re.fullmatch(r"[0-9]+", up)
    if m:
        tic = int(up)
        return Target(raw=raw, tic_id=tic, search_string=f"TIC {tic}")

    # Fall back to passing the raw string straight to the archive/search.
    return Target(raw=raw, search_string=raw)
