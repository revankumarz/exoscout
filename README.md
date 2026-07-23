# 🔭 ExoScout

Autonomous TESS candidate triage & follow-up planning agent.

ExoScout takes a single TESS candidate (a TOI or TIC id), vets the transit,
checks whether it is **already known or a likely false positive** against public
science archives, and (in later stages) checks the literature and plans
ground-based follow-up. It is being built as an agentic-AI portfolio project:
heavy compute lives in deterministic tools; a small LLM orchestrates.

## What it does

A triage loop with a Streamlit UI, manually orchestrated for now:

1. **Light-curve tool** (`exoscout/tools/lightcurve.py`) - fetches a TESS light
   curve via Lightkurve/MAST, cleans and flattens it, runs a Box Least Squares
   (BLS) transit search, and phase-folds at the best period.
2. **Vetting tool** (`exoscout/tools/vetting.py`) - odd/even depth, secondary
   eclipse, and depth-SNR checks to catch eclipsing binaries and noise. The
   transit-vetting CNN plugs in here via a `cnn_score` hook.
3. **Archive tool** (`exoscout/tools/archive.py`) - queries the NASA Exoplanet
   Archive TAP service (TOI + confirmed-planet tables) to answer
   *"is this candidate already known?"* with a TFOPWG disposition.

Every tool call is written to a **provenance log** so each claim in the verdict
traces back to a tool output.

## Roadmap

- Literature novelty check over ADS + arXiv.
- Single-agent ReAct loop replacing the manual control flow, with memory.
- Split into orchestrator + specialist agents (Data / Vetting / Archive /
  Literature / Planning).
- astroplan follow-up planning + one-page observing brief.
- Evaluation: tool-call success rate, novelty accuracy vs. held-out TOIs,
  hallucination checks.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then enter a target like `TOI 700.01`, `TIC 150428135`, or a bare TIC number.

All data comes from free public APIs (NASA Exoplanet Archive TAP, MAST via
Lightkurve) - no credentials required.

## Design note

Tools are pure functions returning plain dicts and never raise for expected
failures, so they can be registered as LLM-callable tools later without
changing call sites. This is deliberate scaffolding for the agent stages.

> Verdicts are **decision-support with a human in the loop**, not authoritative
> discovery claims.
