"""Shared reusable Streamlit components."""

import streamlit as st

from spine.ui.utils import format_phase_icon, format_phase_color


def phase_badge(phase: str) -> str:
    """Return a colored badge for a phase.

    Uses Streamlit's colored text syntax for display.

    Args:
        phase: Phase name (e.g., "PLANNING", "COMPLETE").

    Returns:
        Formatted string with color and phase name.
    """
    icon = format_phase_icon(phase)
    color = format_phase_color(phase)
    return f"[{color}]**{icon} {phase}**[/]"


def phase_progress_bar(progress: float, label: str = "") -> None:
    """Render a Streamlit progress bar with optional label.

    Args:
        progress: Progress value from 0.0 to 1.0.
        label: Optional label to display above the progress bar.
    """
    if label:
        st.write(f"**{label}**")
    st.progress(progress)


def status_badge(status: str) -> str:
    """Return a colored status badge.

    Args:
        status: Status string (e.g., "success", "failed", "running").

    Returns:
        Formatted string with color and status.
    """
    color_map: dict[str, str] = {
        "success": "green",
        "running": "blue",
        "pending": "gray",
        "failed": "red",
        "blocked": "red",
        "error": "red",
    }
    color = color_map.get(status.lower(), "gray")
    return f"[{color}]**{status}**[/]"


def empty_state(message: str, icon: str = "ℹ️") -> None:
    """Show an empty state with icon and message.

    Args:
        message: Message to display.
        icon: Icon emoji to show.
    """
    col1, col2 = st.columns([1, 3])
    col1.write(icon)
    col2.info(message)


def work_item_card(item: dict) -> None:
    """Render a single work item card with status and progress.

    Args:
        item: Work item dict with thread_id, requirement, phase, progress, etc.
    """
    icon = format_phase_icon(item.get("phase", "INIT"))
    color = format_phase_color(item.get("phase", "INIT"))

    with st.container():
        col_icon, col_title, col_phase, col_progress, col_actions = st.columns(
            [1, 3, 2, 3, 2]
        )

        col_icon.write(icon)
        col_title.write(f"**{item.get('requirement', 'Untitled')}**")
        col_phase.write(f"[{color}]**{item.get('phase', '')}**[/]")

        progress = item.get("progress", 0)
        col_progress.progress(progress)

        if col_actions.button("View", key=f"card_view_{item.get('thread_id', '')}"):
            st.session_state.selected_work_id = item.get("thread_id")
            st.session_state.page = "Work Detail"
            st.rerun()


def phase_roadmap(current_phase: str, detail: dict = None) -> None:
    """Render a visual phase roadmap with current phase highlighted.

    Args:
        current_phase: The current phase name.
        detail: Optional work item detail for critic gate info.
    """
    phases = [
        ("INIT", "⚙️", "Initialize"),
        ("PLANNING", "📋", "Plan"),
        ("EXECUTION", "🔨", "Execute"),
        ("VERIFICATION", "✅", "Verify"),
        ("COMPLETE", "🏁", "Complete"),
    ]

    phase_names = [p[0] for p in phases]
    current_idx = phase_names.index(current_phase) if current_phase in phase_names else 0

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Phase Roadmap")

        # Build visual roadmap
        parts = []
        for i, (name, icon, label) in enumerate(phases):
            if i < current_idx:
                parts.append(f"{icon} **{name}** ✓")
            elif i == current_idx:
                parts.append(f"{icon} **{name}** ● current")
            else:
                parts.append(f"{icon} {name} ○")

        st.markdown(" → ".join(parts))

    with col2:
        if detail and detail.get("critic_gate_result"):
            st.subheader("Gate Status")
            result = detail["critic_gate_result"]
            if result == "APPROVED":
                st.success("✅ Approved")
            elif result == "REJECTED":
                st.error("✗ Rejected")
            else:
                st.warning(f"⏳ {result}")
