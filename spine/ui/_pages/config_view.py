"""SPINE Config page — view current configuration."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the config page."""
    st.title("⚙️ Configuration")

    config = api.get_config()

    st.subheader("Current Configuration")
    st.json(config)

    st.divider()

    st.subheader("Configuration Reference")

    config_help = {
        "checkpoint_path": "Path to the SQLite database for LangGraph checkpoints.",
        "artifact_path": "Directory where workflow artifacts are stored.",
        "max_critic_retries": (
            "Maximum times a phase can be sent back for rework "
            "before being flagged for human review."
        ),
        "work_type": (
            "Default workflow type when not specified. Options: "
            "quick, critical_quick, spec, critical_spec."
        ),
        "queue_backend": "Queue backend for RalphLoopWorker (sqlite or redis).",
    }

    for key, description in config_help.items():
        st.write(f"**{key}**: {description}")
