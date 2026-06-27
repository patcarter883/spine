"""SPINE Experience page — inspect cross-run distilled lessons.

Lessons are captured at the end of each run from the critic / adversarial
feedback of phases that needed revision, then injected into the matching
phase's prompt on future runs. This page lets a human see what SPINE has
learned, audit individual lessons, and prune any that are wrong or stale.
"""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp

# Phase chip → icon, purely cosmetic grouping in the list.
_PHASE_ICON = {
    "specify": "📝",
    "plan": "📐",
    "implement": "🔧",
    "verify": "✅",
    "tasks": "🧩",
    "gap_plan": "🩹",
}

_TIER_BADGE = {
    "agent": "🤖 critic",
    "adversarial": "⚔️ adversarial",
    "human": "👤 human",
}


def render(api: UIApi) -> None:
    """Render the cross-run experience inspector."""
    st.title("🧠 Learned Experience")
    st.caption(
        "Lessons distilled from past runs' critic feedback. Each is injected "
        "into the matching phase's prompt on future runs to head off repeat "
        "defects. Prune anything inaccurate or obsolete."
    )

    stats = api.experience_stats()
    total = stats.get("total", 0)
    by_phase = stats.get("by_phase", {})

    if not total:
        st.info(
            "No lessons captured yet. They accumulate automatically as runs "
            "complete (whenever the critic asked a phase for revision)."
        )
        return

    # ── Summary metrics ──
    cols = st.columns(max(1, len(by_phase) + 1))
    cols[0].metric("Total lessons", total)
    for col, (phase, count) in zip(cols[1:], sorted(by_phase.items())):
        icon = _PHASE_ICON.get(phase, "•")
        col.metric(f"{icon} {phase}", count)

    st.divider()

    # ── Controls ──
    phase_options = ["(all)"] + sorted(by_phase)
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        selected = st.selectbox("Filter by phase", phase_options, index=0)
    with ctrl_right:
        st.write("")  # vertical spacer to align the button with the selectbox
        clear_label = (
            "🗑️ Clear all" if selected == "(all)" else f"🗑️ Clear {selected}"
        )
        if st.button(clear_label, use_container_width=True):
            removed = api.clear_experience(
                phase=None if selected == "(all)" else selected
            )
            st.toast(f"Removed {removed} lesson(s).")
            st.rerun()

    lessons = api.list_experience()
    if selected != "(all)":
        lessons = [le for le in lessons if le.get("phase") == selected]

    st.subheader(f"{len(lessons)} lesson(s)")

    for le in lessons:
        lesson_id = le.get("id", "")
        phase = le.get("phase", "?")
        icon = _PHASE_ICON.get(phase, "•")
        salience = le.get("salience", 1)
        category = le.get("category")
        cat_suffix = f" · {category}" if category else ""
        header = f"{icon} **{phase}**{cat_suffix} · salience {salience}"

        with st.container(border=True):
            top, btn = st.columns([6, 1])
            with top:
                st.markdown(header)
                st.markdown(f"**Lesson:** {le.get('lesson', '')}")
                trigger = le.get("trigger")
                if trigger:
                    st.caption(f"Flagged: {trigger}")
                tier = _TIER_BADGE.get(le.get("source_tier", ""), le.get("source_tier", ""))
                work_id = le.get("work_id", "")
                when = format_timestamp(le.get("created_at")) if le.get("created_at") else ""
                meta = " · ".join(p for p in (tier, f"from `{work_id}`", when) if p)
                st.caption(meta)
            with btn:
                if st.button("Delete", key=f"del_exp_{lesson_id}", use_container_width=True):
                    api.delete_experience_lesson(lesson_id)
                    st.toast("Lesson deleted.")
                    st.rerun()
