"""Offline unit tests - no network. Run with: pytest -q"""

from __future__ import annotations

import exoscout.store as store
from exoscout.agent import tools as agent_tools
from exoscout.agent.context import AgentContext
from exoscout.agent.orchestrator import _synthesize_verdict
from exoscout.brief import build_brief
from exoscout.provenance import Provenance
from exoscout.target import parse_target


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
