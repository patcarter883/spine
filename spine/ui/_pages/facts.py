"""SPINE Project Facts page — inspect the CAM memory-organ intent log.

Facts are subject→object associations written to the CAM editable memory
served next to the model (write-gated: only what the base model can't already
recall is stored). This page shows spine's authoritative side index — the CAM
banks themselves can't be enumerated — plus the live server occupancy stats
when the CAM provider is reachable.
"""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp

_GATE_BADGE = {True: "🟢 stored", False: "⚪ gate-skipped"}
_PROBE_BADGE = {True: "✅ delivers", False: "❌ probe FAILED", None: ""}


def render(api: UIApi) -> None:
    """Render the project-facts inspector."""
    st.title("📌 Project Facts")
    st.caption(
        "Durable facts pinned into the CAM memory organ (the served model "
        "answers with them in-forward). This is spine's intent log; the write "
        "gate stores only what the base model can't already recall. Manage "
        "server-side state with `spine facts` (sync/delete/freeze)."
    )

    stats = api.facts_stats()
    total = stats.get("total", 0)

    if not total:
        st.info(
            "No facts recorded yet. They accumulate as runs complete when the "
            "active provider carries a `cam:` block with `write: distill`, or "
            "via `spine facts add`."
        )
        return

    # ── Summary metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", total)
    c2.metric("Stored (gate)", stats.get("stored", 0))
    c3.metric("Gate-skipped", stats.get("gate_skipped", 0))
    probe_failed = stats.get("probe_failed", 0)
    c4.metric("Probe failures", probe_failed)
    if probe_failed:
        st.warning(
            f"{probe_failed} stored fact(s) failed the readback probe — the "
            "store may be crowded. Check the server stats below and consider "
            "pruning."
        )

    # ── Live server stats (fail-open) ──
    with st.expander("Live server stats (/cam/stats)"):
        server = api.cam_server_stats()
        if server is None:
            st.caption("CAM server unreachable or not configured — side index only.")
        else:
            st.json(server)

    st.divider()

    # ── Fact list ──
    facts = api.list_facts()
    namespaces = sorted({f.get("namespace") or "—" for f in facts})
    selected_ns = st.selectbox("Namespace", ["(all)"] + namespaces, index=0)
    if selected_ns != "(all)":
        facts = [f for f in facts if (f.get("namespace") or "—") == selected_ns]

    st.subheader(f"{len(facts)} fact(s)")
    for f in facts:
        subject = f.get("subject", "?")
        with st.container(border=True):
            top, btn = st.columns([6, 1])
            with top:
                st.markdown(f"**{subject}** → `{f.get('object', '')}`")
                badges = " · ".join(
                    b
                    for b in (
                        _GATE_BADGE.get(f.get("stored")),
                        _PROBE_BADGE.get(f.get("verified")),
                        f"ns `{f.get('namespace') or '—'}`",
                        # gate_reason (serve rev 3e8c1b3+) is the meaningful
                        # verdict; base_p is 0.0-noise in frontend mode, so
                        # show it only when the reason is absent.
                        f.get("gate_reason") and f"gate: {f['gate_reason']}",
                        None
                        if f.get("base_p") is None or f.get("gate_reason")
                        else f"base_p {f.get('base_p'):.3f}",
                        f.get("mode") and f"mode `{f['mode']}`",
                    )
                    if b
                )
                st.caption(badges)
                st.caption(f"Probe: {f.get('probe_prompt', '')}")
                src = f.get("source", "")
                work_id = f.get("work_id") or ""
                when = (
                    format_timestamp(f.get("created_at"))
                    if f.get("created_at")
                    else ""
                )
                meta = " · ".join(
                    p
                    for p in (src, f"from `{work_id}`" if work_id else "", when)
                    if p
                )
                st.caption(meta)
            with btn:
                if st.button(
                    "Remove",
                    key=f"del_fact_{f.get('namespace')}_{subject}",
                    use_container_width=True,
                    help="Removes the side-index record only; use `spine facts "
                    "delete` to also tombstone it on the server.",
                ):
                    api.delete_fact(subject, namespace=f.get("namespace"))
                    st.toast("Side-index record removed.")
                    st.rerun()
