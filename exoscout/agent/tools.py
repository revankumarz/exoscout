"""Agent tool registry.

Wraps the four deterministic tools as LLM-callable functions. Each tool:
  * has an OpenAI-style JSON schema (name/description/parameters),
  * reads what it needs from the shared AgentContext,
  * stores its FULL result in the context (for the UI / provenance),
  * returns a COMPACT summary dict (what the LLM actually sees).
"""

from __future__ import annotations

from typing import Callable

from exoscout.agent.context import AgentContext
from exoscout.tools import archive, lightcurve, literature, vetting


def _t_fetch_lightcurve(ctx: AgentContext, max_period: float | None = None) -> dict:
    mp = float(max_period) if max_period else ctx.max_period
    res = lightcurve.fetch_and_search(ctx.target.search_string, max_period=mp,
                                      max_sectors=ctx.max_sectors)
    ctx.store("lightcurve", res)
    ctx.prov.record("lightcurve.fetch_and_search",
                    res.get("error") or f"period={res.get('period')}", res.get("source", ""),
                    ok=res.get("ok", False))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    return {"ok": True, "period_days": round(res["period"], 4), "depth_ppm": round(res["depth_ppm"]),
            "bls_snr": round(res["snr"], 1), "sectors": res["sector_count"], "n_points": res["n_points"]}


def _t_vet_transit(ctx: AgentContext) -> dict:
    lc = ctx.full.get("lightcurve")
    if not lc or not lc.get("ok"):
        return {"ok": False, "error": "Run fetch_lightcurve first."}
    res = vetting.run_vetting(lc)
    ctx.store("vetting", res)
    ctx.prov.record("vetting.run_vetting", res.get("error") or res.get("summary", ""),
                    res.get("source", ""), ok=res.get("ok", False))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    return {"ok": True, "verdict": res["summary"], "oddeven_sigma": res["oddeven_sigma"],
            "secondary_sigma": res["secondary_sigma"], "depth_snr": res["depth_snr"],
            "flags": res["flags"]}


def _t_archive_check(ctx: AgentContext) -> dict:
    res = archive.check_known(tic_id=ctx.target.tic_id, toi=ctx.target.toi)
    ctx.store("archive", res)
    ctx.prov.record("archive.check_known", res.get("error") or res.get("summary", ""),
                    res.get("source", ""), ok=res.get("ok", False))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    return {"ok": True, "known": res["known"], "confirmed": res["confirmed"],
            "disposition": res["disposition"], "resolved_tic": res.get("tic_id"),
            "summary": res["summary"]}


def _t_literature_check(ctx: AgentContext) -> dict:
    res = literature.check_novelty(tic_id=ctx.target.tic_id, toi=ctx.target.toi)
    ctx.store("literature", res)
    ctx.prov.record("literature.check_novelty", res.get("summary", ""), res.get("source", ""),
                    ok=res.get("ok", False))
    top = [h["title"] for h in (res["arxiv"].get("hits", []) + res["ads"].get("hits", []))[:5]]
    return {"ok": res.get("ok", False), "n_matches": res["n_matches"],
            "summary": res["summary"], "top_titles": top}


# name -> (callable, JSON schema)
_REGISTRY: dict[str, tuple[Callable, dict]] = {
    "fetch_lightcurve": (_t_fetch_lightcurve, {
        "type": "function",
        "function": {
            "name": "fetch_lightcurve",
            "description": "Download the target's TESS light curve and run a BLS transit search. "
                           "Call this first; other tools depend on it.",
            "parameters": {"type": "object", "properties": {
                "max_period": {"type": "number", "description": "Max BLS period in days (default 15)."}
            }},
        }}),
    "vet_transit": (_t_vet_transit, {
        "type": "function",
        "function": {
            "name": "vet_transit",
            "description": "Run false-positive vetting (odd/even depth, secondary eclipse, depth SNR) "
                           "on the fetched light curve to check if the signal is planet-like.",
            "parameters": {"type": "object", "properties": {}},
        }}),
    "archive_check": (_t_archive_check, {
        "type": "function",
        "function": {
            "name": "archive_check",
            "description": "Ask the NASA Exoplanet Archive whether this target is already a known/"
                           "confirmed planet, an existing TOI candidate, or a flagged false positive.",
            "parameters": {"type": "object", "properties": {}},
        }}),
    "literature_check": (_t_literature_check, {
        "type": "function",
        "function": {
            "name": "literature_check",
            "description": "Search arXiv (and ADS if configured) for papers mentioning this target, "
                           "to judge whether it has already been studied.",
            "parameters": {"type": "object", "properties": {}},
        }}),
}


def tool_schemas() -> list[dict]:
    return [schema for _, schema in _REGISTRY.values()]


def call_tool(name: str, ctx: AgentContext, args: dict) -> dict:
    if name not in _REGISTRY:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    fn, _ = _REGISTRY[name]
    try:
        return fn(ctx, **(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}
