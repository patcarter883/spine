"""SPINE Artifacts page — browse and view work item artifacts."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the artifacts page."""
    st.title("📁 Artifacts")

    work_id = st.text_input("Work ID", placeholder="Enter work item ID to browse artifacts")

    if not work_id:
        st.info("Enter a work item ID to view its artifacts.")
        return

    artifacts = api.get_artifacts(work_id)

    if not artifacts:
        st.warning(f"No artifacts found for work item '{work_id}'.")
        return

    st.subheader(f"Artifacts for {work_id}")

    for artifact in artifacts:
        phase = artifact.get("phase", "unknown")
        name = artifact.get("name", "unknown")
        size = artifact.get("size", 0)
        modified = artifact.get("modified", "N/A")

        with st.expander(f"📄 {phase}/{name} ({size} bytes)"):
            st.write(f"**Phase:** {phase}")
            st.write(f"**Name:** {name}")
            st.write(f"**Size:** {size} bytes")
            st.write(f"**Modified:** {modified}")

            # Load and display content
            content = api.read_artifact(work_id, phase, name)
            if content:
                if name.endswith(".md") or name.endswith(".txt"):
                    st.markdown(content)
                elif name.endswith(".json"):
                    st.json(content)
                else:
                    st.code(content, language="text")
