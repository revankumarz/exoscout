"""ExoScout - week-1 MVP Streamlit app.

Two-tool triage loop (manually orchestrated for now; the LLM orchestrator
replaces this control flow in a later week):

    target -> [Light-curve tool] -> BLS transit search + phase fold
           -> [Archive tool]     -> "is this already known?"
           -> verdict + provenance log

Run:  streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from exoscout.target import parse_target
from exoscout.provenance import Provenance
from exoscout.tools import archive, lightcurve, literature, vetting

st.set_page_config(page_title="ExoScout", page_icon="🔭", layout="wide")

st.title("🔭 ExoScout")
st.caption(
    "Autonomous TESS candidate triage. "
    "Tools: light-curve search (Lightkurve/MAST), transit vetting "
    "(odd/even, secondary, SNR), archive novelty check (NASA Exoplanet Archive), "
    "and literature check (arXiv + ADS)."
)

with st.sidebar:
    st.header("Target")
    raw = st.text_input("TOI or TIC id", value="TOI 700.01",
                        help="e.g. 'TOI 700.01', 'TIC 150428135', or a bare TIC number")
    max_period = st.slider("Max BLS period (days)", 1.0, 30.0, 15.0, 0.5)
    max_sectors = st.number_input("Max sectors (0 = all)", 0, 200, 4, 1,
                                  help="Cap sectors downloaded; keeps demos fast.")
    run = st.button("Run triage (pipeline)", type="primary", use_container_width=True)
    run_agent_btn = st.button("Run as agent 🤖", use_container_width=True,
                              help="LLM decides which tools to call (falls back to a "
                                   "deterministic planner if no LLM is reachable).")
    st.markdown("---")
    st.caption("All data via free public APIs. No credentials required.")

if run:
    prov = Provenance()
    target = parse_target(raw)
    st.subheader(f"Target: {target.label}")

    col_lc, col_arch = st.columns(2)

    # ---- Tool 1: light-curve fetch + BLS -------------------------------------
    with col_lc:
        st.markdown("### 1 - Light curve & transit search")
        with st.spinner("Fetching TESS light curve from MAST and running BLS..."):
            lc = lightcurve.fetch_and_search(target.search_string, max_period=max_period)
        prov.record(
            "lightcurve.fetch_and_search",
            lc.get("error") or f"period={lc.get('period'):.4f} d, depth={lc.get('depth_ppm'):.0f} ppm",
            lc.get("source", ""),
            ok=lc.get("ok", False),
        )
        if lc.get("ok"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Period (d)", f"{lc['period']:.4f}")
            m2.metric("Depth (ppm)", f"{lc['depth_ppm']:.0f}")
            m3.metric("BLS SNR", f"{lc['snr']:.1f}")
            m4, m5 = st.columns(2)
            m4.metric("Sectors", lc["sector_count"])
            m5.metric("Points", f"{lc['n_points']:,}")

            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(lc["phase"], lc["phase_flux"], ".", ms=1, alpha=0.35, color="#3b82f6")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Normalised flux")
            ax.set_title("Phase-folded light curve")
            ax.set_xlim(-0.5, 0.5)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.error(lc.get("error", "Light-curve step failed."))

    # ---- Tool 2: vetting diagnostics -----------------------------------------
    vet = {}
    if lc.get("ok"):
        from exoscout.ml.infer import cnn_score as _cnn_score
        vet = vetting.run_vetting(lc, cnn_score=_cnn_score)
        prov.record(
            "vetting.run_vetting",
            vet.get("error") or vet.get("summary", ""),
            vet.get("source", ""),
            ok=vet.get("ok", False),
        )
        with col_lc:
            st.markdown("#### Vetting")
            if vet.get("ok"):
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Odd/even Δ (σ)", vet["oddeven_sigma"] if vet["oddeven_sigma"] is not None else "n/a")
                v2.metric("Secondary (σ)", vet["secondary_sigma"] if vet["secondary_sigma"] is not None else "n/a")
                v3.metric("Depth SNR", vet["depth_snr"] if vet["depth_snr"] is not None else "n/a")
                cnn = vet.get("cnn_score")
                v4.metric("CNN P(planet)", f"{cnn:.2f}" if cnn is not None else "n/a")
                if "PASSES" in vet["summary"]:
                    st.success(vet["summary"])
                elif "WEAK" in vet["summary"]:
                    st.warning(vet["summary"])
                else:
                    st.error(vet["summary"])
                for f in vet.get("flags", []):
                    st.write(f"- {f}")
            else:
                st.warning(vet.get("error", "Vetting unavailable."))

    # ---- Tool 3: archive novelty check ---------------------------------------
    with col_arch:
        st.markdown("### 3 - Already known?")
        with st.spinner("Querying NASA Exoplanet Archive (TOI + confirmed tables)..."):
            arch = archive.check_known(tic_id=target.tic_id, toi=target.toi)
        prov.record(
            "archive.check_known",
            arch.get("error") or arch.get("summary", ""),
            arch.get("source", ""),
            ok=arch.get("ok", False),
        )
        if arch.get("ok"):
            if arch["confirmed"]:
                st.error(f"**{arch['summary']}**")
            elif arch["known"]:
                st.warning(f"**{arch['summary']}**")
            else:
                st.success(f"**{arch['summary']}**")

            if arch.get("tic_id"):
                st.write(f"Resolved TIC: `{arch['tic_id']}`")
            st.write(f"TFOPWG disposition: {arch['disposition']}")

            if arch["toi_matches"]:
                st.caption("TOI catalogue matches")
                st.dataframe(arch["toi_matches"], use_container_width=True, hide_index=True)
            if arch["confirmed_matches"]:
                st.caption("Confirmed-planet matches")
                st.dataframe(arch["confirmed_matches"], use_container_width=True, hide_index=True)
        else:
            st.error(arch.get("error", "Archive step failed."))

    # ---- Tool 4: literature novelty check ------------------------------------
    st.markdown("---")
    st.markdown("### 4 - Literature check (has anyone published on it?)")
    with st.spinner("Searching arXiv (and ADS if a token is set)..."):
        litr = literature.check_novelty(tic_id=target.tic_id, toi=target.toi)
    prov.record(
        "literature.check_novelty",
        litr.get("summary", ""),
        litr.get("source", ""),
        ok=litr.get("ok", False),
    )
    if litr.get("ok"):
        if litr["n_matches"] == 0:
            st.success(litr["summary"])
        else:
            st.warning(litr["summary"])
        st.caption(f"Search terms: {', '.join(litr['terms'])}")
        hits = litr["arxiv"].get("hits", []) + litr["ads"].get("hits", [])
        if hits:
            st.dataframe(
                [{"date": h["published"], "title": h["title"], "authors": h["authors"], "link": h["url"]}
                 for h in hits],
                use_container_width=True, hide_index=True,
            )
        if litr["ads"].get("skipped"):
            st.caption("ADS skipped - set the ADS_TOKEN environment variable to include it.")
    else:
        st.error("Literature search unavailable.")

    # ---- Verdict -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Verdict (decision-support - human in the loop)")
    if vet.get("ok"):
        st.write(f"**Vetting:** {vet['summary']}"
                 + (f" - {'; '.join(vet['flags'])}" if vet.get("flags") else ""))
    if litr.get("ok"):
        st.write(f"**Literature:** {litr['summary']}")
    if arch.get("ok"):
        if arch["confirmed"]:
            st.info("This target maps to a **confirmed / known planet**. Low novelty - "
                    "unlikely to be a new discovery.")
        elif "FALSE POSITIVE" in arch["summary"]:
            st.info("The TOI catalogue **flags this as a false positive**. Treat any "
                    "transit signal with caution.")
        elif arch["known"]:
            st.info("An **existing TOI candidate** - not novel, but still an active "
                    "candidate worth following up.")
        else:
            novel_lit = litr.get("ok") and litr.get("n_matches", 0) == 0
            if novel_lit:
                st.info("**Not in the archives and no literature matches** - potentially "
                        "novel. Good follow-up target (verify before any claim).")
            else:
                st.info("**Not found** in the TOI or confirmed tables, but the literature "
                        "mentions it - lower novelty than it first appears.")
    else:
        st.info("Archive check unavailable; cannot assess novelty this run.")

    # ---- Provenance ----------------------------------------------------------
    st.markdown("### Provenance log")
    st.caption("Every claim above traces to one of these tool calls.")
    st.dataframe(prov.as_rows(), use_container_width=True, hide_index=True)

elif run_agent_btn:
    from exoscout.agent.llm import LLMClient
    from exoscout.agent.orchestrator import run_agent

    target = parse_target(raw)
    st.subheader(f"🤖 Agent triage: {target.label}")
    client = LLMClient()
    reachable = client.available()
    st.caption(f"LLM: {client.cfg.describe()} - "
               + ("reachable, running ReAct loop." if reachable
                  else "unreachable, using deterministic planner."))

    with st.spinner("Agent reasoning and calling tools..."):
        ctx = run_agent(raw, max_period=max_period, client=client,
                        max_sectors=(int(max_sectors) or None))

    st.markdown("### Reasoning trace")
    for step in ctx.trace:
        label = step["tool"] or step["kind"]
        if step["kind"] == "tool":
            ok = step["data"].get("ok")
            (st.success if ok else st.warning)(f"**{label}** - {step['text'][:300]}")
        elif step["kind"] == "verdict":
            st.info(f"**Agent verdict:** {step['text'][:1200]}")
        else:
            st.write(f"_{step['text']}_")

    if ctx.full.get("verdict"):
        st.markdown("### Verdict (rule-based cross-check)")
        st.json(ctx.full["verdict"])

    # Follow-up plan table
    plan = ctx.full.get("planning", {})
    if plan.get("ok"):
        st.markdown("### Follow-up plan")
        st.caption(plan["summary"])
        st.dataframe(plan["observatories"], use_container_width=True, hide_index=True)

    # One-page observing brief (exportable)
    from exoscout.brief import build_brief
    md = build_brief(ctx)
    st.markdown("### Observing brief")
    with st.expander("Show / download brief", expanded=False):
        st.markdown(md)
    st.download_button("Download brief (.md)", md,
                       file_name=f"{ctx.target.label.replace(' ', '_')}_brief.md")

    st.markdown("### Provenance log")
    st.dataframe(ctx.prov.as_rows(), use_container_width=True, hide_index=True)

    # Memory: past triages
    from exoscout import store
    hist = store.recent(10)
    if hist:
        st.markdown("### Recent triages (memory)")
        st.dataframe(hist, use_container_width=True, hide_index=True)

else:
    st.info("Enter a target in the sidebar, then **Run triage** (fixed pipeline) "
            "or **Run as agent** (LLM picks the tools).")
