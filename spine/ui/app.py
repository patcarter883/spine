"""Main Streamlit entry point with sidebar navigation."""

import streamlit as st

from spine.ui.dashboard import render_dashboard
from spine.ui.work_detail import render_work_detail
from spine.ui.work_new import render_new_work
from spine.ui.providers_config import render_providers_config
from spine.ui.settings import render_settings

PAGE_NAMES = ["Dashboard", "New Work", "Work Detail", "Providers", "Settings"]
PAGE_ICONS = {
    "Dashboard": "📊",
    "New Work": "➕",
    "Work Detail": "🔍",
    "Providers": "🔌",
    "Settings": "⚙️",
}


def _render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.title("⚡ SPINE")
        st.caption("Agent Harness Dashboard")
        st.markdown("---")
        st.markdown("### Navigation")

        selected = st.radio(
            "page_selection",
            PAGE_NAMES,
            key="nav_page",
            index=PAGE_NAMES.index(st.session_state.page),
        )
        st.session_state.page = selected

        # Show currently selected work item
        if st.session_state.selected_work_id:
            st.markdown("---")
            st.caption(f"Work: {st.session_state.selected_work_id}")


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="SPINE Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize navigation state
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "selected_work_id" not in st.session_state:
        st.session_state.selected_work_id = None

    # Render sidebar navigation
    _render_sidebar()

    # Route to page handler
    page = st.session_state.page

    if page == "Dashboard":
        render_dashboard()
    elif page == "New Work":
        render_new_work()
    elif page == "Work Detail":
        render_work_detail()
    elif page == "Providers":
        render_providers_config()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
