"""ExoScout agent - command-line demo.

    python agent_cli.py "TOI 700.01"
    python agent_cli.py "TIC 150428135" --deterministic

With an OpenAI-compatible LLM configured (see exoscout/agent/llm.py) it runs the
ReAct loop; otherwise it uses the deterministic planner. Either way you get the
tool trace, the rule-based verdict, and the provenance log.
"""

from __future__ import annotations

import argparse
import json

from exoscout.agent.llm import LLMClient
from exoscout.agent.orchestrator import run_agent, run_deterministic
from exoscout.brief import build_brief


def main() -> None:
    ap = argparse.ArgumentParser(description="ExoScout agentic TESS triage")
    ap.add_argument("target", help="TOI or TIC id, e.g. 'TOI 700.01'")
    ap.add_argument("--deterministic", action="store_true", help="Force the no-LLM planner")
    ap.add_argument("--max-period", type=float, default=15.0)
    ap.add_argument("--max-sectors", type=int, default=4, help="Cap sectors (0 = all)")
    ap.add_argument("--brief", action="store_true", help="Print the full observing brief")
    args = ap.parse_args()

    max_sectors = args.max_sectors or None
    if args.deterministic:
        ctx = run_deterministic(args.target, max_period=args.max_period,
                                max_sectors=max_sectors)
    else:
        client = LLMClient()
        print(f"[llm] {client.cfg.describe()} - "
              f"{'reachable' if client.available() else 'unreachable, falling back'}")
        ctx = run_agent(args.target, max_period=args.max_period, client=client,
                        max_sectors=max_sectors)

    print(f"\n=== Trace for {ctx.target.label} ===")
    for step in ctx.trace:
        tag = step["tool"] or step["kind"]
        print(f"  [{step['kind']:7}] {tag}: {step['text'][:160]}")

    if ctx.full.get("verdict"):
        print("\n=== Verdict (rule-based) ===")
        print(json.dumps(ctx.full["verdict"], indent=2))
    if ctx.full.get("verdict_text"):
        print("\n=== Verdict (LLM) ===")
        print(ctx.full["verdict_text"])

    print("\n=== Provenance ===")
    for row in ctx.prov.as_rows():
        print(f"  [{row['ok']}] {row['tool']}: {row['summary']}  <- {row['source']}")

    if args.brief:
        print("\n" + "=" * 60)
        print(build_brief(ctx))


if __name__ == "__main__":
    main()
