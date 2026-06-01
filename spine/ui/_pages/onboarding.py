"""SPINE Onboarding page — analyse a repo (brownfield) or scaffold a new
project (greenfield) and synthesise the four onboarding markdown artifacts.

This page is a thin view over :class:`spine.ui_api.UIApi` only.  It never
imports the dispatcher, the onboarding engine, or the workflow graph directly
— all backend access goes through ``api`` so the zero-duplication principle
holds (CLI and UI share one backend).

The page lets the user:

* toggle Greenfield / Brownfield mode,
* enter the target project path (defaults to the configured workspace root),
* supply a comma-separated tech stack (greenfield seed),
* dispatch onboarding via ``api.enqueue_onboarding(...)``,
* watch live progress through an auto-refreshing phase bar fed by
  ``api.get_queue_overview()``,
* review the four generated ``.md`` documents inline via
  ``api.read_onboarding_doc(work_id, name)``.
"""

from __future__ import annotations

import streamlit as st

from spine.ui.utils import format_duration, status_icon, truncate
from spine.ui_api import UIApi
from spine.work.onboarding.phases import (
    PHASE_ANALYZE,
    PHASE_COMPLETED,
    PHASE_SCAFFOLD,
    PHASE_SYNTHESIZE,
    PHASES_BY_MODE,
)

# ── Fragment refresh interval (seconds) ──
# The progress section auto-refreshes via @st.fragment(run_every=...) so only
# that fragment re-renders while the form inputs above keep their state.
_POLL_INTERVAL = 10

# Fixed phase sequences per mode (shared with the engine via
# spine.work.onboarding.phases so the phase-bar contract cannot drift).
# Brownfield skips the scaffold step; greenfield scaffolds an empty project
# BEFORE analysing/synthesising defaults, so "scaffold" comes first to keep the
# progress bar monotonic (the engine fires "scaffold" pre-graph, then "analyze",
# then "synthesize").
_PHASES_BY_MODE: dict[str, list[str]] = PHASES_BY_MODE

_PHASE_EMOJI: dict[str, str] = {
    PHASE_ANALYZE: "🔍",
    PHASE_SCAFFOLD: "🏗️",
    PHASE_SYNTHESIZE: "📝",
    PHASE_COMPLETED: "✅",
}

# The four documents the onboarding engine always produces, in display order.
# Mirrors the canonical ``ONBOARDING_DOC_NAMES`` set, kept local so this UI page
# imports no onboarding-engine internals (zero-duplication: all backend access
# goes through ``api``).
_ONBOARDING_DOCS: tuple[str, ...] = (
    "PROJECT_DEFINITION.md",
    "CODING_GUIDELINES.md",
    "ARCHITECTURE_MAP.md",
    "SPINE_ASSISTANCE_REQUIREMENTS.md",
)

# The work_type string for onboarding jobs (used to match the active queue row).
_ONBOARDING_PHASE = "onboarding"


# ── Helpers ──


def _render_phase_bar(phases: list[str], current: str) -> None:
    """Render a horizontal progress bar: completed / current / upcoming.

    Local to this page on purpose — it does not import queue.py internals.

    Args:
        phases: The fixed ordered phase sequence for the active mode.
        current: The currently-running phase name (e.g. ``"analyze"``).
    """
    if not phases:
        return
    cols = st.columns(len(phases))

    try:
        current_idx = phases.index(current)
    except ValueError:
        # "completed" (or any unknown terminal phase) means every step is done.
        current_idx = len(phases) if current == PHASE_COMPLETED else -1

    for i, (col, phase) in enumerate(zip(cols, phases)):
        icon = _PHASE_EMOJI.get(phase, "⚙️")
        label = phase.title()
        if current_idx < 0:
            col.caption(f"○ {icon} {label}")
        elif i < current_idx:
            col.caption(f"✓ {icon} {label}")
        elif i == current_idx:
            col.markdown(f"**● {icon} {label}**")
        else:
            col.caption(f"○ {icon} {label}")


def _parse_tech_stack(raw: str) -> list[str]:
    """Split a comma-separated tech-stack string into a clean list."""
    return [item.strip() for item in raw.split(",") if item.strip()]


# ── Fragment sections ──


@st.fragment(run_every=_POLL_INTERVAL)
def _render_progress(api: UIApi) -> None:
    """Live onboarding progress — auto-refreshing fragment.

    Reads ``api.get_queue_overview()`` (the authoritative source, identical to
    queue.py) and, when the active job is an onboarding job, renders a phase
    bar over the fixed sequence for its mode.  Treats the active dict as the
    single source of truth and the ws_bus only as a refresh hint.
    """
    overview = api.get_queue_overview()
    active = overview.get("active") or {}

    if not active or active.get("work_type") != _ONBOARDING_PHASE:
        st.caption("No onboarding job is currently running.")
        return

    work_id = active.get("work_id") or ""
    queue_id = active.get("id")
    display_id = work_id or f"queue-{queue_id}"
    status = active.get("status", "unknown")
    current_phase = active.get("current_phase") or PHASE_ANALYZE
    created_at = active.get("created_at") or ""
    updated_at = active.get("updated_at") or ""

    st.subheader("🔄 Active Onboarding")
    icon = status_icon(active.get("work_status") or status)
    st.caption(
        f"{icon} Work ID: `{display_id}`"
        + (f"  ·  Queue #{queue_id}" if work_id and queue_id else "")
    )

    description = active.get("description", "")
    if description:
        st.markdown(f"**{truncate(description, 200)}**")

    # Phase bar over the fixed sequence for this mode.  The mode is encoded in
    # the active job's current phase set; default to the greenfield superset so
    # the bar shows every possible step when the mode is unknown.
    phases = _phases_for_active(active)
    st.markdown("**Progress:**")
    _render_phase_bar(phases, current_phase)

    if created_at:
        st.caption(
            f"Started {created_at[:19]}  ·  "
            f"Last updated {updated_at[:19] or '—'}  ·  "
            f"Elapsed: {format_duration(created_at)}"
        )

    if status == "failed":
        result = active.get("result", "")
        if result:
            st.error("**Error:**")
            st.code(result)


def _phases_for_active(active: dict[str, object]) -> list[str]:
    """Pick the phase sequence for an active onboarding job.

    The queue row does not carry the onboarding mode directly, so infer it:
    if the engine has already moved past "analyze" without a "scaffold" step
    we cannot tell, so default to the greenfield superset which is a strict
    superset of the brownfield sequence and still renders correctly.
    """
    mode = active.get("mode")
    if isinstance(mode, str) and mode in _PHASES_BY_MODE:
        return _PHASES_BY_MODE[mode]
    return _PHASES_BY_MODE["greenfield"]


def _render_artifacts(api: UIApi, work_id: str) -> None:
    """Render the four onboarding documents inline for review.

    Shows each known onboarding doc via ``api.read_onboarding_doc(work_id, name)``
    in an expander + markdown block, mirroring human_review.py's review pattern.
    Reads are done exclusively through the UI gateway (no filesystem access here);
    a doc that returns no content is simply skipped.

    Args:
        api: The UI gateway.
        work_id: The completed onboarding work item ID.
    """
    if not work_id:
        return

    st.subheader("📚 Onboarding Documents")
    rendered_any = False
    for name in _ONBOARDING_DOCS:
        content = api.read_onboarding_doc(work_id, name)
        if not content:
            continue
        rendered_any = True
        with st.expander(f"📄 {name}", expanded=False):
            st.markdown(content)

    if not rendered_any:
        st.caption("No onboarding documents are available yet for this work item.")


# ── Page ──


def render(api: UIApi) -> None:
    """Render the Onboarding page.

    Args:
        api: The sole UI gateway (:class:`spine.ui_api.UIApi`).
    """
    st.title("🚀 Onboarding")
    st.caption(
        "Analyse an existing repository (brownfield) or scaffold a new project "
        "(greenfield) and synthesise the four Spine onboarding documents."
    )

    # ── Dispatch form ──
    mode_label = st.radio(
        "Mode",
        ["Brownfield", "Greenfield"],
        horizontal=True,
        key="onboarding_mode",
        help=(
            "Brownfield analyses an existing codebase. "
            "Greenfield scaffolds a new project from a tech-stack seed."
        ),
    )
    mode = "greenfield" if mode_label == "Greenfield" else "brownfield"

    default_path = getattr(api._config, "workspace_root", "") or ""
    workspace_root = st.text_input(
        "Project path",
        value=default_path,
        key="onboarding_path",
        help="Absolute path to the target project.",
    )

    tech_stack: list[str] = []
    if mode == "greenfield":
        tech_stack_raw = st.text_input(
            "Tech stack (comma-separated)",
            value="",
            key="onboarding_tech_stack",
            placeholder="python, langgraph, streamlit",
            help="Seed technologies used to synthesise best-practice defaults.",
        )
        tech_stack = _parse_tech_stack(tech_stack_raw)

    if st.button("▶ Execute Onboarding", key="onboarding_execute", type="primary"):
        if not workspace_root.strip():
            st.error("Please provide a project path.")
        else:
            result = api.enqueue_onboarding(
                workspace_root.strip(),
                mode,
                tech_stack or None,
            )
            queue_id = result.get("queue_id") if isinstance(result, dict) else None
            st.success(
                f"Onboarding enqueued ({mode}) — queue id `{queue_id}`. "
                "Watch progress below."
            )

    # ── Live progress ──
    st.divider()
    st.subheader("▸ Progress")
    _render_progress(api)

    # ── Artifact review ──
    st.divider()
    review_work_id = st.text_input(
        "Review work ID",
        value="",
        key="onboarding_review_id",
        help="Enter a completed onboarding work ID to review its documents.",
    )
    if review_work_id.strip():
        _render_artifacts(api, review_work_id.strip())
