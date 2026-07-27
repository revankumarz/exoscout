"""Evaluation harness - how good is the novelty classifier?

Runs the archive tool against a small labeled set (data/labeled_tois.csv) and
reports accuracy for the two decisions that matter: is the target *known*, and
is it *confirmed*. Borrows the ReplicationBench/AstroVisBench spirit: a project
that measures itself is far more credible than one that just demos.

    python evaluate.py
"""

from __future__ import annotations

import csv

from exoscout.tools import archive
from exoscout.target import parse_target
from exoscout.paths import LABELS_PATH as LABELS


def main() -> None:
    with open(LABELS, newline="") as fh:
        rows = list(csv.DictReader(fh))

    known_hits = conf_hits = 0
    print(f"{'target':16} {'known':>6} {'exp':>4} {'conf':>6} {'exp':>4}  result")
    print("-" * 56)
    for r in rows:
        t = parse_target(r["target"])
        res = archive.check_known(tic_id=t.tic_id, toi=t.toi)
        known = 1 if res.get("known") else 0
        conf = 1 if res.get("confirmed") else 0
        exp_known, exp_conf = int(r["expected_known"]), int(r["expected_confirmed"])
        k_ok, c_ok = known == exp_known, conf == exp_conf
        known_hits += k_ok
        conf_hits += c_ok
        print(f"{r['target']:16} {known:>6} {exp_known:>4} {conf:>6} {exp_conf:>4}  "
              f"{'OK' if (k_ok and c_ok) else 'MISS'}")

    n = len(rows)
    print("-" * 56)
    print(f"known accuracy:     {known_hits}/{n} = {known_hits / n:.0%}")
    print(f"confirmed accuracy: {conf_hits}/{n} = {conf_hits / n:.0%}")


if __name__ == "__main__":
    main()
