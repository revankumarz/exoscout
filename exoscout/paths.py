"""Project paths, anchored to the package rather than the working directory.

Everything on disk (SQLite memory, HTTP cache, trained model, datasets) used to
be addressed relative to the CWD, so launching Streamlit or the CLI from
anywhere other than the repo root silently lost the model. These are absolute
and overridable by environment variable.
"""

from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.environ.get("EXOSCOUT_DATA_DIR") or os.path.join(ROOT, "data")
MODEL_DIR = os.environ.get("EXOSCOUT_MODEL_DIR") or os.path.join(ROOT, "models")
CACHE_DIR = os.environ.get("EXOSCOUT_CACHE_DIR") or os.path.join(ROOT, ".exoscout_cache")

DB_PATH = os.path.join(DATA_DIR, "exoscout.db")
MODEL_PATH = os.path.join(MODEL_DIR, "astronet.pt")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
TRAINSET_PATH = os.path.join(DATA_DIR, "trainset.npz")
LABELS_PATH = os.path.join(DATA_DIR, "labeled_tois.csv")
