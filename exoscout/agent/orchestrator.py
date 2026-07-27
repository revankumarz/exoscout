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
    """Rule-based verdict from whatever tool results are in the context.

    Every field is gated on the evidence that supports it: a tool that did not
    run, or that failed, yields "unknown" rather than a default. In particular
    novelty is only called *novel* when both the archive and the literature
    search actually succeeded and both came back empty - an absent search is
    not evidence of absence. ``reasons`` records what drove each call so the
    brief can show its working.
    """
    vet = ctx.full.get("vetting", {})
    arch = ctx.full.get("archive", {})
    lit = ctx.full.get("literature", {})
    stel = ctx.full.get("stellar", {})
    plan = ctx.full.get("planning", {})
    reasons: list[str] = []

    # ---- is the transit real? ----------------------------------------------
    if not vet.get("ok"):
        transit_real = "unknown"
        reasons.append("Transit reality unknown: vetting did not run.")
    elif "FALSE POSITIVE" in vet.get("summary", ""):
        transit_real = "no"
        reasons.append(f"Vetting: {vet['summary']} ({'; '.join(vet.get('flags', []))}).")
    elif vet.get("alias_suspected"):
        # A real transit folded at half its true period - not a false positive.
        transit_real = "unclear"
        reasons.append(
            f"Transits appear on only one parity: the BLS period is likely half the "
            f"true one. Re-run at {vet.get('suggested_period'):.4f} d before judging.")
    elif "WEAK" in vet.get("summary", ""):
        transit_real = "unclear"
        reasons.append(f"Vetting: weak signal (depth SNR {vet.get('depth_snr')}).")
    else:
        transit_real = "yes"
        reasons.append(f"Vetting: passes odd/even, secondary and SNR checks "
                       f"(depth SNR {vet.get('depth_snr')}).")

    # ---- false-positive risk ------------------------------------------------
    if not vet.get("ok"):
        fp_risk = "unknown"
    elif vet.get("alias_suspected"):
        fp_risk = "medium"          # the ephemeris is wrong, not the signal
    elif vet.get("flags"):
        fp_risk = "high"
    elif "PASSES" in vet.get("summary", ""):
        fp_risk = "low"
    else:
        fp_risk = "medium"

    # SIMBAD calling it a binary/variable is a strong false-positive signal.
    if stel.get("eclipsing_binary_flag"):
        fp_risk = "high"
        reasons.append(f"SIMBAD classifies the host as {stel.get('otype')} "
                       "(eclipsing-binary risk).")

    # AstroNet CNN score, when available, sharpens the risk call.
    cnn = vet.get("cnn_score") if vet.get("ok") else None
    if cnn is not None:
        if cnn < 0.3 and fp_risk in ("low", "medium", "unknown"):
            fp_risk = "high"
            reasons.append(f"CNN P(planet)={cnn:.2f} - below the 0.30 planet threshold.")
        elif cnn > 0.7 and fp_risk in ("medium", "unknown"):
            fp_risk = "low"
            reasons.append(f"CNN P(planet)={cnn:.2f} - above the 0.70 planet threshold.")
        else:
            reasons.append(f"CNN P(planet)={cnn:.2f}.")

    # ---- novelty ------------------------------------------------------------
    lit_searched = bool(lit.get("ok"))
    lit_hits = int(lit.get("n_matches", 0) or 0)
    if not arch.get("ok"):
        novelty = "unknown (archive lookup failed)"
        reasons.append("Novelty unknown: the archive lookup did not succeed.")
    elif arch.get("confirmed"):
        novelty = "confirmed / known planet"
        reasons.append(f"NASA Exoplanet Archive disposition: {arch.get('disposition')}.")
    elif arch.get("known"):
        novelty = "existing TOI candidate"
        reasons.append(f"Already catalogued: {arch.get('disposition')}.")
    elif lit_searched and lit_hits > 0:
        novelty = "already studied in the literature"
        reasons.append(f"{lit_hits} paper(s) mention this target.")
    elif lit_searched:
        novelty = "novel (not in archives or literature)"
        reasons.append("Absent from the archive and from arXiv/ADS.")
    else:
        # Archive is clean but nobody checked the papers - do not claim novelty.
        novelty = "not in archive (literature unchecked)"
        reasons.append("Absent from the archive, but the literature search did "
                       "not succeed - novelty cannot be claimed.")

    # ---- recommendation -----------------------------------------------------
    where = f" Best placed at {plan['best']}." if plan.get("ok") and plan.get("best") else ""
    if vet.get("alias_suspected"):
        rec = (f"Re-run the transit search at {vet.get('suggested_period'):.4f} d - the "
               "current period looks like a half-period alias.")
    elif novelty.startswith("novel") and fp_risk == "low" and transit_real == "yes":
        rec = "Strong follow-up target - schedule ground-based confirmation." + where
    elif fp_risk == "high":
        rec = "Likely false positive - deprioritise unless re-vetted."
    elif "confirmed" in novelty or "existing" in novelty or "studied" in novelty:
        rec = "Low novelty - already known/studied; not a new discovery."
    elif fp_risk == "unknown" or novelty.startswith("unknown"):
        rec = "Insufficient evidence - re-run the missing tools before judging."
    else:
        rec = "Inconclusive - gather more data before deciding." + where

    return {"transit_real": transit_real, "false_positive_risk": fp_risk,
            "novelty": novelty, "recommendation": rec, "reasons": reasons}


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
    else:
        # Step budget exhausted while the model was still calling tools. Ask
        # once more with no tools offered, so the run still ends in a brief
        # rather than silently producing nothing.
        ctx.log_step("plan", f"Step budget ({max_steps}) reached - "
                             "requesting a final brief with tools disabled.")
        messages.append({"role": "user", "content":
                         "You are out of tool calls. Write the final brief now from the "
                         "evidence you already have, and say which checks are missing."})
        try:
            resp = client.chat(messages)
            ctx.log_step("verdict", resp.get("content") or "(no content)")
            ctx.full["verdict_text"] = resp.get("content")
        except Exception as e:
            ctx.log_step("verdict", f"Final LLM call failed ({type(e).__name__}: {e}) - "
                                    "using the rule-based verdict only.")

    # Always attach a rule-based verdict too, as a grounded cross-check.
    ctx.full["verdict"] = _synthesize_verdict(ctx)
    store.save_triage(ctx)
    return ctx
