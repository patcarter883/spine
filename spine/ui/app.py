"""SPINE Streamlit app — main entry point and navigation.

Uses Streamlit's built-in ``st.navigation`` / ``st.Page`` multipage
routing for deep-linkable URLs (e.g. ``/work-detail?work_id=abc``).

A lightweight WebSocket server runs in a daemon thread for push events.
Pages that need live data refreshes use ``@st.fragment(run_every=...)``
for isolated re-renders that preserve widget state (no full-page reloads).
"""

from __future__ import annotations

import os

import streamlit as st

from spine.ui.pages import register
from spine.ui_api import UIApi
from spine.ui.ws_server import start_ws_server

st.set_page_config(
    page_title="SPINE Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize API singleton ──
if "api" not in st.session_state:
    st.session_state.api = UIApi()

# ── Start WebSocket server (idempotent — safe to call every re-run) ──
start_ws_server()

api: UIApi = st.session_state.api

# ── Ensure the queue worker loop is alive (idempotent) ──
# Boot it here so the queue is always being serviced and the worker
# status the Queue page reports reflects reality rather than "not running
# until the first job is enqueued".
api.ensure_worker_running()

# ── LLM debug logging ──
if os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes"):
    from spine.agents.debug_callback import install_global

    install_global()


# ── Page definitions ──


def _dashboard() -> None:
    from spine.ui._pages.dashboard import render

    render(api)


def _submit_work() -> None:
    from spine.ui._pages.work_submit import render

    render(api)


def _queue() -> None:
    from spine.ui._pages.queue import render

    render(api)


def _work_detail() -> None:
    from spine.ui._pages.work_detail import render

    render(api)


def _work_history() -> None:
    from spine.ui._pages.work_history import render

    render(api)


def _config() -> None:
    from spine.ui._pages.config_view import render

    render(api)


def _audit_log() -> None:
    from spine.ui._pages.audit_log import render

    render(api)


def _human_review() -> None:
    from spine.ui._pages.human_review import render

    render(api)


def _spec_planning_render(api: UIApi) -> None:
    from spine.ui._pages.spec_planning import render

    render(api)


def _onboarding() -> None:
    from spine.ui._pages.onboarding import render

    render(api)


def _gate_run() -> None:
    from spine.ui._pages.gate_run import render

    render(api)


pages = {
    "": [
        st.Page(_dashboard, title="Dashboard", icon="🏠", url_path="dashboard"),
        st.Page(_submit_work, title="Submit Work", icon="📝", url_path="submit"),
        st.Page(_queue, title="Queue", icon="🚦", url_path="queue"),
    ],
    "Planning": [
        st.Page(
            lambda: _spec_planning_render(api),
            title="Spec & Planning",
            icon="📐",
            url_path="spec-planning",
        ),
        st.Page(
            _onboarding,
            title="Project Onboarding",
            icon="🚀",
            url_path="onboarding",
        ),
    ],
    "Work": [
        st.Page(_work_detail, title="Work Details", icon="🔍", url_path="work-detail"),
        st.Page(_work_history, title="Work History", icon="📜", url_path="work-history"),
        st.Page(_gate_run, title="Git Gate", icon="🔒", url_path="git-gate"),
    ],
    "System": [
        st.Page(_human_review, title="Human Review", icon="👤", url_path="human-review"),
        st.Page(_audit_log, title="Audit Log", icon="📋", url_path="audit-log"),
        st.Page(_config, title="Config", icon="⚙️", url_path="config"),
    ],
}

# ── Navigation (renders sidebar, returns selected page) ──
page = st.navigation(pages)

# ── Register page objects for cross-page navigation ──
for section_pages in pages.values():
    for p in section_pages:
        register(p.url_path, p)

# ── Run the selected page ──
page.run()
