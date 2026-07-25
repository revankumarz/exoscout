"""Build a labeled training set for the AstroNet CNN from real TESS data.

Labels come from the TESS Object of Interest catalogue's TFOPWG disposition:
  * positives (planet):        CP (confirmed), KP (known)
  * negatives (false positive): FP (false positive), FA (false alarm)

For each target we download a TESS light curve, flatten it, and build the
global + local views at the catalogued ephemeris (period / epoch / duration).
Failures are skipped. The result is saved as a compressed .npz.

    python -m exoscout.ml.build_dataset --per-class 80 --out data/trainset.npz
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from exoscout.tools import archive
from .preprocess import make_views
from .astronet import GLOBAL_BINS, LOCAL_BINS

BTJD_OFFSET = 2457000.0  # TESS reference epoch


def fetch_targets(per_class: int) -> list[dict]:
    """Return a balanced list of labeled targets with usable ephemerides."""
    cols = "tid, toi, tfopwg_disp, pl_orbper, pl_trandurh, pl_tranmid"
    cond = "pl_orbper is not null and pl_tranmid is not null and pl_trandurh is not null"
    pos = archive._tap_query(
        f"select top {per_class} {cols} from toi where tfopwg_disp in ('CP','KP') and {cond}")
    neg = archive._tap_query(
        f"select top {per_class} {cols} from toi where tfopwg_disp in ('FP','FA') and {cond}")
    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        for _, r in df.iterrows():
            rows.append({"tid": int(r["tid"]), "label": label,
                         "period": float(r["pl_orbper"]),
                         "t0": float(r["pl_tranmid"]),
                         "duration_h": float(r["pl_trandurh"])})
    return rows


def _download_views(target: dict) -> tuple[np.ndarray, np.ndarray] | None:
    import lightkurve as lk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = lk.search_lightcurve(f"TIC {target['tid']}", mission="TESS")
        if len(search) == 0:
            return None
        lc = search[0].download()
        if lc is None:
            return None
        lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401)
        time = np.asarray(lc.time.value, float)
        flux = np.asarray(lc.flux.value, float)

    t0 = target["t0"]
    if t0 > BTJD_OFFSET:            # convert BJD -> BTJD to match TESS time
        t0 -= BTJD_OFFSET
    dur_days = target["duration_h"] / 24.0
    g, l = make_views(time, flux, target["period"], t0, dur_days)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(l))):
        return None
    return g, l


def build(per_class: int, out_path: str) -> dict:
    targets = fetch_targets(per_class)
    G, L, Y, ids = [], [], [], []
    ok = fail = 0
    for i, t in enumerate(targets, 1):
        try:
            views = _download_views(t)
        except Exception:
            views = None
        if views is None:
            fail += 1
        else:
            g, l = views
            G.append(g); L.append(l); Y.append(t["label"]); ids.append(t["tid"])
            ok += 1
        if i % 10 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] ok={ok} fail={fail}", flush=True)

    if not G:
        raise RuntimeError("No usable light curves downloaded.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path,
                        global_view=np.array(G, np.float32),
                        local_view=np.array(L, np.float32),
                        label=np.array(Y, np.int64),
                        tic_id=np.array(ids, np.int64))
    n_pos = int(np.sum(Y))
    print(f"Saved {len(G)} samples ({n_pos} planet / {len(G) - n_pos} FP) -> {out_path}")
    return {"n": len(G), "pos": n_pos, "path": out_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--out", default=os.path.join("data", "trainset.npz"))
    args = ap.parse_args()
    build(args.per_class, args.out)


if __name__ == "__main__":
    main()
