# 🔭 ExoScout

Autonomous TESS candidate triage & follow-up planning agent.

ExoScout takes a single TESS candidate (a TOI or TIC id), vets the transit,
checks whether it is **already known or a likely false positive** against public
science archives, and (in later stages) checks the literature and plans
ground-based follow-up. It is being built as an agentic-AI portfolio project:
heavy compute lives in deterministic tools; a small LLM orchestrates.

## What it does

Six deterministic tools, an LLM agent that orchestrates them, a provenance log,
persistent memory, an exportable observing brief, and an evaluation harness.

**Tools** (`exoscout/tools/`)

1. **lightcurve** - fetch a TESS light curve (Lightkurve/MAST), clean/flatten,
   run a Box Least Squares transit search, phase-fold at the best period.
2. **vetting** - odd/even depth, secondary eclipse, depth-SNR checks to catch
   eclipsing binaries and noise. The transit-vetting CNN plugs in via `cnn_score`.
3. **archive** - NASA Exoplanet Archive TAP (TOI + confirmed tables): *"is this
   already known?"* plus coordinates and stellar parameters.
4. **stellar** - SIMBAD object-type lookup (catches known eclipsing binaries).
5. **literature** - arXiv (+ ADS if a token is set): *"has anyone published on it?"*
6. **planning** - astroplan observability across observatories over the next N
   nights (altitude / airmass / darkness / moon separation).

**Around the tools**

- **Provenance log** - every claim traces back to a tool output.
- **Observing brief** (`exoscout/brief.py`) - one-page Markdown deliverable.
- **Memory** (`exoscout/store.py`) - SQLite log of every triage.
- **Evaluation** (`evaluate.py`) - novelty-classifier accuracy on a labeled TOI
  set (`data/labeled_tois.csv`). Currently 100% known/confirmed on the set.
- **Tests** (`tests/`) - offline pytest suite: `pytest -q`.

### Agent layer (`exoscout/agent/`)

An LLM orchestrator turns the four tools into an agentic loop: the model decides
which tool to call, reacts to each result, and writes the final verdict. It is
**LLM-optional** - with an OpenAI-compatible endpoint it runs a ReAct loop;
without one it falls back to a deterministic planner, so it always runs.

```bash
python agent_cli.py "TOI 700.01"                 # LLM if available
python agent_cli.py "TIC 150428135" --deterministic
```

LLM config (env vars, all optional):

```
EXOSCOUT_LLM_BASE_URL   default http://localhost:11434/v1   (local Ollama)
EXOSCOUT_LLM_MODEL      default llama3.2
EXOSCOUT_LLM_API_KEY    default "ollama" (use a real key for hosted APIs)
```

## Roadmap

- Split into orchestrator + specialist sub-agents.
- Wire in the real transit-vetting CNN via the `cnn_score` hook.
- Expand the evaluation set (false positives, hallucination checks per claim).

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
