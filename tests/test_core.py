"""Offline unit tests - no network. Run with: pytest -q"""

from __future__ import annotations

import os

import numpy as np

import exoscout.cache as cache
import exoscout.store as store
from exoscout.agent import tools as agent_tools
from exoscout.agent.context import AgentContext
from exoscout.agent.orchestrator import _synthesize_verdict
from exoscout.brief import build_brief
from exoscout.provenance import Provenance
from exoscout.target import parse_target
from exoscout.tools.vetting import run_vetting


# ---- target parsing --------------------------------------------------------
def test_parse_toi():
    t = parse_target("TOI 700.01")
    assert t.toi == 700.01 and t.tic_id is None and t.label == "TOI 700.01"


def test_parse_tic_and_bare():
    assert parse_target("TIC 150428135").tic_id == 150428135
    assert parse_target("307210830").tic_id == 307210830


# ---- provenance ------------------------------------------------------------
def test_provenance_rows():
    p = Provenance()
    p.record("tool.x", "did a thing", "src", ok=True)
    p.record("tool.y", "failed", "src", ok=False)
    rows = p.as_rows()
    assert rows[0]["ok"] == "OK" and rows[1]["ok"] == "FAIL"


# ---- tool registry ---------------------------------------------------------
def test_tool_schemas_complete():
    names = {s["function"]["name"] for s in agent_tools.tool_schemas()}
    assert names == {"fetch_lightcurve", "vet_transit", "archive_check",
                     "stellar_context", "literature_check", "plan_followup"}


def test_unknown_tool_is_safe():
    ctx = AgentContext(target=parse_target("TOI 1.01"))
    out = agent_tools.call_tool("does_not_exist", ctx, {})
    assert out["ok"] is False


# ---- vetting on synthetic light curves --------------------------------------
def _synth_lc(period=3.0, depth=1e-3, dur=0.1, noise=1e-3,
              odd_extra=0.0, secondary=0.0, days=27.0):
    """Boxcar transit + Gaussian noise, at TESS 2-minute cadence."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, days, 2.0 / 60.0 / 24.0)
    phase = ((t + 0.5 * period) % period) / period - 0.5
    epoch = np.round(t / period).astype(int)
    half = (dur / period) / 2.0
    in_tr = np.abs(phase) < half
    sec = np.abs(np.abs(phase) - 0.5) < half

    f = np.ones_like(t)
    f[in_tr] -= depth
    f[in_tr & (epoch % 2 == 1)] -= odd_extra   # unequal depths = EB tell
    f[sec] -= secondary
    f += rng.normal(0.0, noise, t.size)
    return {"ok": True, "time": t.tolist(), "flux": f.tolist(),
            "period": period, "t0": 0.0, "duration": dur}


def test_vetting_passes_a_clean_planet():
    """Depth == per-point noise still yields a high-significance detection,
    because the SNR is on the *mean* depth over many in-transit points."""
    r = run_vetting(_synth_lc())
    assert r["ok"]
    assert r["depth_snr"] > 10
    assert r["summary"].startswith("PASSES")
    assert r["flags"] == []


def test_vetting_catches_odd_even_mismatch():
    r = run_vetting(_synth_lc(odd_extra=5e-4))
    assert r["oddeven_sigma"] > 3.0
    assert "FALSE POSITIVE" in r["summary"]


def test_vetting_catches_secondary_eclipse():
    r = run_vetting(_synth_lc(secondary=4e-4))
    assert r["secondary_sigma"] > 3.0
    assert "FALSE POSITIVE" in r["summary"]


def test_vetting_rejects_pure_noise():
    r = run_vetting(_synth_lc(depth=0.0))
    assert r["depth_snr"] < 7.1
    assert "WEAK" in r["summary"]


def test_vetting_handles_bad_input():
    assert run_vetting({})["ok"] is False
    assert run_vetting({"ok": False})["ok"] is False


def test_vetting_cnn_hook():
    lc = _synth_lc()
    # A working classifier is reported as-is...
    assert run_vetting(lc, cnn_score=lambda _: 0.83)["cnn_score"] == 0.83
    # ...an untrained model returns None, which is not an error...
    r = run_vetting(lc, cnn_score=lambda _: None)
    assert r["cnn_score"] is None and "cnn_error" not in r
    # ...and a broken one is caught, never crashing the triage.
    def boom(_):
        raise RuntimeError("no weights")
    r = run_vetting(lc, cnn_score=boom)
    assert r["ok"] and r["cnn_score"] is None and "no weights" in r["cnn_error"]


# ---- paths are absolute, not CWD-dependent ---------------------------------
def test_paths_are_absolute():
    """Regression: a CWD-relative model path silently disabled the CNN
    whenever the app was launched from outside the repo root."""
    from exoscout import paths
    for p in (paths.DATA_DIR, paths.MODEL_DIR, paths.CACHE_DIR,
              paths.DB_PATH, paths.MODEL_PATH, paths.LABELS_PATH):
        assert os.path.isabs(p), p


# ---- verdict synthesis (fake evidence, no network) -------------------------
def _fake_ctx() -> AgentContext:
    ctx = AgentContext(target=parse_target("TOI 700.01"))
    ctx.full["lightcurve"] = {"ok": True, "period": 3.4, "depth_ppm": 900, "snr": 12,
                              "sector_count": 2, "n_points": 5000}
    ctx.full["vetting"] = {"ok": True, "summary": "PASSES basic vetting (planet-consistent)",
                           "flags": [], "oddeven_sigma": 0.2, "secondary_sigma": 0.1,
                           "depth_snr": 9.0}
    ctx.full["archive"] = {"ok": True, "known": True, "confirmed": True,
                           "disposition": "CP (Confirmed Planet)", "tic_id": 150428135,
                           "summary": "CONFIRMED / KNOWN PLANET"}
    ctx.full["literature"] = {"ok": True, "n_matches": 8, "summary": "8 matches",
                              "arxiv": {"hits": []}, "ads": {"hits": []}}
    return ctx


def test_verdict_confirmed():
    v = _synthesize_verdict(_fake_ctx())
    assert "confirmed" in v["novelty"]
    assert v["false_positive_risk"] == "low"


def test_verdict_eb_flag_forces_high_fp():
    ctx = _fake_ctx()
    ctx.full["stellar"] = {"ok": True, "eclipsing_binary_flag": True}
    v = _synthesize_verdict(ctx)
    assert v["false_positive_risk"] == "high"


def test_verdict_never_claims_novelty_without_a_literature_search():
    """Regression: an absent or failed literature search is not evidence of
    absence, so it must not license a 'novel' call."""
    ctx = AgentContext(target=parse_target("TIC 999"))
    ctx.full["archive"] = {"ok": True, "known": False, "confirmed": False}
    ctx.full["literature"] = {"ok": False, "n_matches": 0}
    v = _synthesize_verdict(ctx)
    assert not v["novelty"].startswith("novel")
    assert "literature" in v["novelty"]


def test_verdict_claims_novelty_when_both_searches_ran_clean():
    ctx = AgentContext(target=parse_target("TIC 999"))
    ctx.full["archive"] = {"ok": True, "known": False, "confirmed": False}
    ctx.full["literature"] = {"ok": True, "n_matches": 0}
    assert _synthesize_verdict(ctx)["novelty"].startswith("novel")


def test_verdict_is_unknown_without_vetting():
    ctx = AgentContext(target=parse_target("TIC 999"))
    ctx.full["archive"] = {"ok": True, "known": False, "confirmed": False}
    v = _synthesize_verdict(ctx)
    assert v["transit_real"] == "unknown"
    assert v["false_positive_risk"] == "unknown"
    assert "Insufficient evidence" in v["recommendation"]


def test_verdict_reasons_are_populated():
    v = _synthesize_verdict(_fake_ctx())
    assert v["reasons"] and all(isinstance(r, str) for r in v["reasons"])


# ---- brief -----------------------------------------------------------------
def test_brief_has_sections():
    ctx = _fake_ctx()
    ctx.full["verdict"] = _synthesize_verdict(ctx)
    md = build_brief(ctx)
    for header in ("# Observing brief", "## Verdict", "## Transit signal",
                   "## Provenance"):
        assert header in md


# ---- store (temp db) -------------------------------------------------------
def test_store_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setattr(store, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(store, "DB_PATH", str(db))
    ctx = _fake_ctx()
    ctx.full["verdict"] = _synthesize_verdict(ctx)
    assert store.save_triage(ctx) is True
    rows = store.recent()
    assert rows and rows[0]["target"] == "TOI 700.01"
    assert store.times_seen("TOI 700.01") == 1


# ---- cache -----------------------------------------------------------------
def test_cache_roundtrip_and_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    cache.put("ns", "key1", {"a": 1})
    assert cache.get("ns", "key1", ttl_seconds=100) == {"a": 1}
    # Expired entries are ignored.
    assert cache.get("ns", "key1", ttl_seconds=-1) is None
    # Missing keys return None.
    assert cache.get("ns", "missing", ttl_seconds=100) is None
