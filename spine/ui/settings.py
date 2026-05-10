"""Settings page — global UI configuration."""

import streamlit as st


def render_settings() -> None:
    """Render the settings page."""
    st.title("⚙️ Settings")

    # Display tab
    tab_display, tab_data, tab_about = st.tabs(
        ["Display", "Data", "About"]
    )

    with tab_display:
        st.subheader("Display")
        dark_mode = st.toggle("Dark mode", value=False, help="Toggle dark/light theme")
        refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=1,
            max_value=30,
            value=3,
            help="How often to poll for updates",
        )
        st.caption("Streamlit handles theme and refresh. This setting is for reference.")

    with tab_data:
        st.subheader("Data Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear session state"):
                for key in list(st.session_state.keys()):
                    delattr(st.session_state, key)
                st.success("Session state cleared. The app will restart.")
                st.rerun()

        with col2:
            if st.button("Reset navigation"):
                st.session_state.page = "Dashboard"
                st.session_state.selected_work_id = None
                st.success("Navigation reset to Dashboard.")
                st.rerun()

        st.divider()
        st.subheader("Data Sources")
        st.write("**Checkpoint Database:** `.spine/spine.db` (LangGraph SQLiteSaver)")
        st.write("**Config File:** `.spine/config.yaml`")
        st.write("**Events Directory:** `.spine/events/`")
        st.write("**Artifacts Directory:** `.spine/artifacts/`")

    with tab_about:
        st.subheader("SPINE UI")
        st.markdown("""
## SPINE Dashboard v0.1.0

Streamlit dashboard for the SPINE agent harness.

**Features:**
- Real-time work item monitoring
- State machine visualization
- Agent output inspection
- Provider configuration management
- Critic gate approval workflow

**Architecture:**
- Single-process Streamlit app
- Reads SPINE's SQLite checkpoint DB directly
- CLI and UI share the same backend

**Dependencies:**
- Streamlit ≥ 1.30.0
- SPINE core (langgraph, pyyaml, rich)
        """)

        st.divider()
        st.caption("Built with Streamlit • Python-native • No build step • 2024")
