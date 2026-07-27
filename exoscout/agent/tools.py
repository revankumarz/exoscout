"""Agent tool registry.

Wraps the four deterministic tools as LLM-callable functions. Each tool:
  * has an OpenAI-style JSON schema (name/description/parameters),
  * reads what it needs from the shared AgentContext,
  * stores its FULL result in the context (for the UI / provenance),
  * returns a COMPACT summary dict (what the LLM actually sees).
"""

from __future__ import annotations

import json
from typing import Callable

from exoscout.agent.context import AgentContext
from exoscout.tools import archive, lightcurve, literature, planning, stellar, vetting


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
    from exoscout.ml.infer import cnn_score
    res = vetting.run_vetting(lc, cnn_score=cnn_score)
    ctx.store("vetting", res)
    ctx.prov.record("vetting.run_vetting", res.get("error") or res.get("summary", ""),
                    res.get("source", ""), ok=res.get("ok", False))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    return {"ok": True, "verdict": res["summary"], "oddeven_sigma": res["oddeven_sigma"],
            "secondary_sigma": res["secondary_sigma"], "depth_snr": res["depth_snr"],
            "cnn_score": res.get("cnn_score"), "flags": res["flags"]}


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


def _t_stellar_context(ctx: AgentContext) -> dict:
    arch = ctx.full.get("archive", {})
    res = stellar.stellar_context(arch.get("ra"), arch.get("dec"))
    ctx.store("stellar", res)
    ctx.prov.record("stellar.stellar_context",
                    res.get("summary") or res.get("reason", "skipped"),
                    res.get("source", ""), ok=res.get("ok", False))
    return {"ok": res.get("ok", False), "found": res.get("found"),
            "otype": res.get("otype"), "eclipsing_binary_flag": res.get("eclipsing_binary_flag"),
            "summary": res.get("summary") or res.get("reason", "skipped")}


def _t_plan_followup(ctx: AgentContext, nights: int | None = None) -> dict:
    arch = ctx.full.get("archive", {})
    res = planning.plan_followup(arch.get("ra"), arch.get("dec"),
                                 ctx.target.label, nights=int(nights or 14))
    ctx.store("planning", res)
    ctx.prov.record("planning.plan_followup", res.get("error") or res.get("summary", ""),
                    res.get("source", ""), ok=res.get("ok", False))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    return {"ok": True, "summary": res["summary"], "best": res.get("best"),
            "observatories": [{"observatory": r["observatory"], "observable": r["observable"],
                               "hours": r["obs_hours_in_window"]} for r in res["observatories"]]}


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
    "stellar_context": (_t_stellar_context, {
        "type": "function",
        "function": {
            "name": "stellar_context",
            "description": "Look up the host star in SIMBAD for its object type - useful to catch "
                           "eclipsing binaries. Needs archive_check to have run first (for coords).",
            "parameters": {"type": "object", "properties": {}},
        }}),
    "plan_followup": (_t_plan_followup, {
        "type": "function",
        "function": {
            "name": "plan_followup",
            "description": "Compute ground-based observability across observatories for the next N "
                           "nights (altitude/airmass/darkness/moon). Needs archive_check first (coords).",
            "parameters": {"type": "object", "properties": {
                "nights": {"type": "integer", "description": "Nights to plan ahead (default 14)."}
            }},
        }}),
}


def tool_schemas() -> list[dict]:
    return [schema for _, schema in _REGISTRY.values()]


def _call_key(name: str, args: dict) -> str:
    return name + "|" + json.dumps(args or {}, sort_keys=True, default=str)


def call_tool(name: str, ctx: AgentContext, args: dict) -> dict:
    """Run a tool, reusing an identical successful call from earlier in the run.

    Small local models happily call ``fetch_lightcurve`` three times in a row;
    that tool re-downloads every TESS sector from MAST and is not disk-cached,
    so the memo below is the difference between a snappy triage and a stalled
    one. Only successful calls are memoised, so a transient failure can retry.
    """
    if name not in _REGISTRY:
        return {"ok": False, "error": f"Unknown tool '{name}'."}

    key = _call_key(name, args)
    prev = ctx.calls.get(key)
    if prev is not None and prev.get("ok"):
        return {**prev, "cached": True,
                "note": f"{name} already ran with these arguments; reusing that result. "
                        "Do not call it again - move on to another tool or write the brief."}

    fn, _ = _REGISTRY[name]
    try:
        out = fn(ctx, **(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}
    ctx.calls[key] = out
    return out
