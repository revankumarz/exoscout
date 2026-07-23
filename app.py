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
from exoscout.tools import archive, lightcurve

st.set_page_config(page_title="ExoScout", page_icon="🔭", layout="wide")

st.title("🔭 ExoScout")
st.caption(
    "Autonomous TESS candidate triage - week-1 MVP. "
    "Two tools: light-curve vetting (Lightkurve/MAST) + novelty check "
    "(NASA Exoplanet Archive)."
)

with st.sidebar:
    st.header("Target")
    raw = st.text_input("TOI or TIC id", value="TOI 700.01",
                        help="e.g. 'TOI 700.01', 'TIC 150428135', or a bare TIC number")
    max_period = st.slider("Max BLS period (days)", 1.0, 30.0, 15.0, 0.5)
    run = st.button("Run triage", type="primary", use_container_width=True)
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

    # ---- Tool 2: archive novelty check ---------------------------------------
    with col_arch:
        st.markdown("### 2 - Already known?")
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

    # ---- Verdict -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Verdict (decision-support - human in the loop)")
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
            st.info("**Not found** in the TOI or confirmed tables. Potentially novel - "
                    "warrants literature check (next-week agent step) before any claim.")
    else:
        st.info("Archive check unavailable; cannot assess novelty this run.")

    # ---- Provenance ----------------------------------------------------------
    st.markdown("### Provenance log")
    st.caption("Every claim above traces to one of these tool calls.")
    st.dataframe(prov.as_rows(), use_container_width=True, hide_index=True)
else:
    st.info("Enter a target in the sidebar and click **Run triage**.")
