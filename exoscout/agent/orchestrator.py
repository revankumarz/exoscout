"""The triage orchestrator.

Two ways to run:

  * ``run_agent``  - a real ReAct-style loop. The LLM is given the four tools
                     and decides which to call, in what order, reacting to each
                     result, then writes the final verdict.
  * ``run_deterministic`` - no LLM needed: calls the tools in a sensible fixed
                     order and synthesises the verdict with rules. Always works,
                     so the pipeline is demoable with zero setup.

Both share the same tools, context, provenance log, and verdict synthesiser, so
the outputs are directly comparable.
"""

from __future__ import annotations

import json

from exoscout.agent import tools as agent_tools
from exoscout.agent.context import AgentContext
from exoscout.agent.llm import LLMClient
from exoscout.target import parse_target
from exoscout import store

SYSTEM_PROMPT = """You are ExoScout, an autonomous assistant that triages a single TESS \
planet candidate. Your job: decide whether the candidate is (a) a real transit, \
(b) a likely false positive, and (c) already known / already studied, then give a \
follow-up recommendation.

You have tools: fetch_lightcurve, vet_transit, archive_check, stellar_context, \
literature_check, plan_followup. Always fetch the light curve before vetting, and \
run archive_check before stellar_context or plan_followup (they need coordinates). \
Gather evidence from the archive and the literature before judging novelty, and \
plan follow-up if the candidate looks promising. Call tools until you have enough \
evidence, then STOP calling tools and write a final brief.

The final brief must state: transit_real (yes/no/unclear), false_positive_risk \
(low/medium/high), novelty (novel/known-candidate/confirmed/already-studied), and a \
one-line recommendation. Ground every claim in a tool result. This is \
decision-support with a human in the loop - never assert a discovery."""


def _synthesize_verdict(ctx: AgentContext) -> dict:
    """Rule-based verdict from whatever tool results are in the context."""
    lc = ctx.full.get("lightcurve", {})
    vet = ctx.full.get("vetting", {})
    arch = ctx.full.get("archive", {})
    lit = ctx.full.get("literature", {})
    stel = ctx.full.get("stellar", {})
    plan = ctx.full.get("planning", {})

    transit_real = "unclear"
    if vet.get("ok"):
        transit_real = "no" if "FALSE POSITIVE" in vet.get("summary", "") else \
            ("unclear" if "WEAK" in vet.get("summary", "") else "yes")

    fp_risk = "medium"
    if vet.get("ok"):
        if vet.get("flags"):
            fp_risk = "high"
        elif "PASSES" in vet.get("summary", ""):
            fp_risk = "low"
    # SIMBAD calling it a binary/variable is a strong false-positive signal.
    if stel.get("eclipsing_binary_flag"):
        fp_risk = "high"
    # AstroNet CNN score, when available, sharpens the risk call.
    cnn = vet.get("cnn_score") if vet.get("ok") else None
    if cnn is not None:
        if cnn < 0.3 and fp_risk != "high":
            fp_risk = "high"
        elif cnn > 0.7 and fp_risk == "medium":
            fp_risk = "low"

    if arch.get("confirmed"):
        novelty = "confirmed / known planet"
    elif arch.get("known"):
        novelty = "existing TOI candidate"
    elif lit.get("ok") and lit.get("n_matches", 0) > 0:
        novelty = "already studied in the literature"
    elif arch.get("ok"):
        novelty = "novel (not in archives or literature)"
    else:
        novelty = "unknown"

    where = f" Best placed at {plan['best']}." if plan.get("ok") and plan.get("best") else ""
    if novelty.startswith("novel") and fp_risk == "low" and transit_real == "yes":
        rec = "Strong follow-up target - schedule ground-based confirmation." + where
    elif fp_risk == "high":
        rec = "Likely false positive - deprioritise unless re-vetted."
    elif "confirmed" in novelty or "existing" in novelty or "studied" in novelty:
        rec = "Low novelty - already known/studied; not a new discovery."
    else:
        rec = "Inconclusive - gather more data before deciding." + where

    return {"transit_real": transit_real, "false_positive_risk": fp_risk,
            "novelty": novelty, "recommendation": rec}


def run_deterministic(target_text: str, max_period: float = 15.0,
                      max_sectors: int | None = None) -> AgentContext:
    ctx = AgentContext(target=parse_target(target_text), max_period=max_period,
                       max_sectors=max_sectors)
    ctx.log_step("plan", "Deterministic plan: fetch -> vet -> archive -> stellar "
                         "-> literature -> plan_followup.")
    for name in ("fetch_lightcurve", "vet_transit", "archive_check", "stellar_context",
                 "literature_check", "plan_followup"):
        out = agent_tools.call_tool(name, ctx, {})
        ctx.log_step("tool", json.dumps(out)[:400], tool=name, data=out)
    ctx.full["verdict"] = _synthesize_verdict(ctx)
    ctx.log_step("verdict", json.dumps(ctx.full["verdict"]))
    store.save_triage(ctx)
    return ctx


def run_agent(target_text: str, max_period: float = 15.0, max_steps: int = 8,
              client: LLMClient | None = None, max_sectors: int | None = None) -> AgentContext:
    """LLM-driven ReAct loop. Falls back to deterministic if no LLM is reachable."""
    ctx = AgentContext(target=parse_target(target_text), max_period=max_period,
                       max_sectors=max_sectors)
    client = client or LLMClient()

    if not client.available():
        ctx.log_step("plan", f"No LLM reachable at {client.cfg.describe()} - "
                             "using deterministic planner.")
        det = run_deterministic(target_text, max_period, max_sectors=max_sectors)
        det.trace = ctx.trace + det.trace
        return det

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Triage this TESS candidate: {ctx.target.label}."},
    ]
    schemas = agent_tools.tool_schemas()
    ctx.log_step("plan", f"LLM planner: {client.cfg.describe()}")

    for _ in range(max_steps):
        try:
            resp = client.chat(messages, tools=schemas)
        except Exception as e:  # reachable but model errored (not pulled, etc.)
            ctx.log_step("plan", f"LLM call failed ({type(e).__name__}: {e}) - "
                                 "falling back to deterministic planner.")
            det = run_deterministic(target_text, max_period, max_sectors=max_sectors)
            det.trace = ctx.trace + det.trace
            return det
        tool_calls = resp["tool_calls"]

        if not tool_calls:
            ctx.log_step("verdict", resp.get("content") or "(no content)")
            ctx.full["verdict_text"] = resp.get("content")
            break

        messages.append(resp["raw"])
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out = agent_tools.call_tool(fn, ctx, args)
            ctx.log_step("tool", json.dumps(out)[:400], tool=fn, data=out)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", fn),
                             "name": fn, "content": json.dumps(out)})

    # Always attach a rule-based verdict too, as a grounded cross-check.
    ctx.full["verdict"] = _synthesize_verdict(ctx)
    store.save_triage(ctx)
    return ctx
