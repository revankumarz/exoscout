"""Persistent memory - a SQLite log of every triage.

Gives the agent a memory across runs: past verdicts, when a target was last
seen, and how many times. Best-effort - a storage failure never breaks a triage.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from exoscout.paths import DATA_DIR, DB_PATH

DB_DIR = DATA_DIR


def _connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL,
            tic_id INTEGER,
            ts TEXT NOT NULL,
            transit_real TEXT,
            fp_risk TEXT,
            novelty TEXT,
            recommendation TEXT,
            verdict_json TEXT
        )""")
    return conn


def save_triage(ctx) -> bool:
    try:
        v = ctx.full.get("verdict", {}) or {}
        conn = _connect()
        conn.execute(
            "INSERT INTO triages (target, tic_id, ts, transit_real, fp_risk, "
            "novelty, recommendation, verdict_json) VALUES (?,?,?,?,?,?,?,?)",
            (ctx.target.label, ctx.target.tic_id,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             v.get("transit_real"), v.get("false_positive_risk"),
             v.get("novelty"), v.get("recommendation"), json.dumps(v)),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def recent(limit: int = 25) -> list[dict]:
    try:
        conn = _connect()
        cur = conn.execute(
            "SELECT target, ts, transit_real, fp_risk, novelty, recommendation "
            "FROM triages ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def times_seen(target_label: str) -> int:
    try:
        conn = _connect()
        (n,) = conn.execute("SELECT COUNT(*) FROM triages WHERE target = ?",
                            (target_label,)).fetchone()
        conn.close()
        return int(n)
    except Exception:
        return 0
