"""Agent resources page — manage AGENTS.md, rules, and knowledge files."""

import streamlit as st

from spine.ui.utils import (
    get_agent_resources,
    save_agent_resource,
    regenerate_agent_resource,
)


# Categories in order for tab display
CATEGORIES = [
    "Agent Memory",
    "Project Rules",
    "MCP Servers",
    "Coding Style",
    "Knowledge Base",
]


def render_agent_resources() -> None:
    """Render the agent resources management page."""
    st.title("📁 Agent Resources")

    st.info(
        "Manage agent resource files that guide AI behavior. "
        "These files are read by agents during work execution."
    )

    # Fetch all resources
    resources = get_agent_resources()

    # Group resources by category
    by_category: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for r in resources:
        cat = r.get("category", "Knowledge Base")
        if cat in by_category:
            by_category[cat].append(r)

    # Create tabs for each category
    tabs = st.tabs(CATEGORIES)

    for tab, category in zip(tabs, CATEGORIES):
        with tab:
            cat_resources = by_category.get(category, [])

            if not cat_resources:
                st.info(f"No resources configured for {category}.")
                continue

            for resource in cat_resources:
                _render_resource_card(resource)


def _render_resource_card(resource: dict) -> None:
    """Render a single resource card with editor and actions."""
    key = resource["key"]
    label = resource["label"]
    path = resource["path"]
    description = resource["description"]
    content = resource["content"]
    exists = resource["exists"]

    # Status indicator
    status_icon = "✅" if exists else "⚠️"
    status_text = "Exists" if exists else "Not Found"

    with st.container():
        # Header with metadata
        st.markdown(f"### {status_icon} {label}")
        st.caption(f"{description} • `{path}` • {status_text}")

        # Text area for content editing
        edited_content = st.text_area(
            "Content",
            value=content,
            height=200,
            key=f"content_{key}",
            label_visibility="collapsed",
        )

        # Action buttons
        col_save, col_regenerate, _ = st.columns([1, 1, 4])

        with col_save:
            if st.button("💾 Save", key=f"save_{key}"):
                success = save_agent_resource(key, edited_content)
                if success:
                    st.toast(f"Saved {label}", icon="✅")
                else:
                    st.toast(f"Failed to save {label}", icon="❌")

        with col_regenerate:
            if st.button("🔄 Regenerate", key=f"regen_{key}"):
                result = regenerate_agent_resource(key)
                if result is not None:
                    # Update the text area with regenerated content
                    st.session_state[f"content_{key}"] = result
                    st.toast(f"Regenerated {label}", icon="✅")
                else:
                    st.toast(f"Could not regenerate {label}", icon="⚠️")

        st.divider()