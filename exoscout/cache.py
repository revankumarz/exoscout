"""Tiny on-disk response cache with TTL.

Public archives are rate-limited (ADS: 5000/day; arXiv asks for <=1 request per
3 s). Caching identical queries keeps us polite, makes repeat triages fast, and
makes demos resilient to a flaky network. Keyed by a hash of the query string;
values are JSON, expired entries are ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".exoscout_cache"))


def _path(namespace: str, key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"{namespace}_{h}.json")


def get(namespace: str, key: str, ttl_seconds: float) -> object | None:
    """Return the cached value if present and fresh, else None."""
    path = _path(namespace, key)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if time.time() - blob["ts"] > ttl_seconds:
            return None
        return blob["value"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def put(namespace: str, key: str, value: object) -> None:
    """Store a JSON-serialisable value. Best-effort; never raises."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_path(namespace, key), "w", encoding="utf-8") as fh:
            json.dump({"ts": time.time(), "value": value}, fh)
    except (OSError, TypeError):
        pass
