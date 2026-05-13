"""SPINE Streamlit app — main entry point and navigation."""

from __future__ import annotations

import os

import streamlit as st

from spine.ui_api import UIApi

st.set_page_config(
    page_title="SPINE Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize API singleton ──
if "api" not in st.session_state:
    st.session_state.api = UIApi()

api: UIApi = st.session_state.api

# ── LLM debug logging ──
if os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes"):
    from spine.agents.debug_callback import install_global

    install_global()

# ── Navigation ──
PAGES = {
    "Dashboard": "spine.ui._pages.dashboard",
    "Submit Work": "spine.ui._pages.work_submit",
    "Work Status": "spine.ui._pages.work_status",
    "Work History": "spine.ui._pages.work_history",
    "Artifacts": "spine.ui._pages.artifacts",
    "Config": "spine.ui._pages.config_view",
    "Audit Log": "spine.ui._pages.audit_log",
    "Human Review": "spine.ui._pages.human_review",
}

st.sidebar.title("🦴 SPINE")
st.sidebar.caption("Deterministic AI Agent Harness")

selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

# ── Load and render selected page ──
module_name = PAGES[selection]
try:
    import importlib

    page_module = importlib.import_module(module_name)
    page_module.render(api)
except Exception as e:
    st.error(f"Error loading page: {e}")
    with st.expander("Details"):
        __import__("traceback").print_exc()
