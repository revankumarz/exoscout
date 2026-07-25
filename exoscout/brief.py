"""Observing-brief generator.

Turns a completed AgentContext into a one-page Markdown brief: the target, the
verdict, the evidence from each tool, and the follow-up plan. This is the human
deliverable at the end of a triage - exportable and copy-pasteable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from exoscout.agent.context import AgentContext


def _fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_brief(ctx: AgentContext) -> str:
    f = ctx.full
    lc, vet, arch = f.get("lightcurve", {}), f.get("vetting", {}), f.get("archive", {})
    lit, stel, plan = f.get("literature", {}), f.get("stellar", {}), f.get("planning", {})
    verdict = f.get("verdict", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Observing brief - {ctx.target.label}",
        f"_Generated {now} by ExoScout. Decision-support, human-in-the-loop._",
        "",
        "## Verdict",
    ]
    if verdict:
        lines += [
            f"- **Transit real:** {verdict.get('transit_real', 'n/a')}",
            f"- **False-positive risk:** {verdict.get('false_positive_risk', 'n/a')}",
            f"- **Novelty:** {verdict.get('novelty', 'n/a')}",
            f"- **Recommendation:** {verdict.get('recommendation', 'n/a')}",
        ]
    else:
        lines.append("- (no verdict synthesised)")

    lines += ["", "## Target & star"]
    if arch.get("ok"):
        lines += [
            f"- Resolved TIC: `{arch.get('tic_id')}`",
            f"- Sky position: RA {_fmt(arch.get('ra'), 4)}, Dec {_fmt(arch.get('dec'), 4)}",
            f"- TESS mag: {_fmt(arch.get('tmag'))}, Teff: {_fmt(arch.get('teff'), 0)} K, "
            f"R*: {_fmt(arch.get('srad'))} Rsun, dist: {_fmt(arch.get('dist_pc'), 1)} pc",
            f"- Archive status: {arch.get('summary')} ({arch.get('disposition')})",
        ]
    if stel.get("found"):
        lines.append(f"- SIMBAD: {stel.get('otype')} (main id {stel.get('main_id')})")

    lines += ["", "## Transit signal"]
    if lc.get("ok"):
        lines += [
            f"- BLS period: {_fmt(lc.get('period'), 4)} d, depth: {_fmt(lc.get('depth_ppm'), 0)} ppm, "
            f"SNR: {_fmt(lc.get('snr'), 1)}",
            f"- Sectors: {lc.get('sector_count')}, points: {lc.get('n_points')}",
        ]
    else:
        lines.append(f"- Light curve unavailable: {lc.get('error', 'n/a')}")

    lines += ["", "## Vetting"]
    if vet.get("ok"):
        lines.append(f"- {vet.get('summary')}")
        if vet.get("cnn_score") is not None:
            lines.append(f"- AstroNet CNN: P(planet) = {_fmt(vet.get('cnn_score'))}")
        for fl in vet.get("flags", []):
            lines.append(f"  - {fl}")
    else:
        lines.append("- not run")

    lines += ["", "## Novelty / literature"]
    if lit.get("ok"):
        lines.append(f"- {lit.get('summary')}")
        hits = lit.get("arxiv", {}).get("hits", []) + lit.get("ads", {}).get("hits", [])
        for h in hits[:5]:
            lines.append(f"  - {h.get('published')} - {h.get('title')}")
    else:
        lines.append("- not run")

    lines += ["", "## Follow-up plan"]
    if plan.get("ok"):
        lines.append(f"- {plan.get('summary')}")
        for r in plan.get("observatories", []):
            mark = "yes" if r["observable"] else "no"
            lines.append(f"  - {r['observatory']}: observable={mark}, "
                         f"~{r.get('obs_hours_in_window', 'n/a')} h in window")
    else:
        lines.append(f"- not available: {plan.get('error', 'n/a')}")

    lines += ["", "## Provenance", "| tool | ok | summary |", "|---|---|---|"]
    for row in ctx.prov.as_rows():
        summ = row["summary"].replace("|", "/")[:80]
        lines.append(f"| {row['tool']} | {row['ok']} | {summ} |")

    return "\n".join(lines)
