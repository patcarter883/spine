"""Spec Driven Development (SDD) page — manage SDD projects.

Shows SDD phase lifecycle: SPEC -> DESIGN -> PLAN -> IMPLEMENT -> REVIEW -> VERIFY
with progress indicators, projects list, and controls.
"""

from pathlib import Path

import streamlit as st

from spine.ui.utils import (
    get_sdd_projects,
    start_sdd_project,
    update_sdd_project_phase,
    SDD_PHASES,
    SDD_PHASE_ICONS,
    navigate_to_work,
)


def render_sdd() -> None:
    """Render the SDD management page."""
    st.title("🧭 Spec Driven Development")

    # Description
    st.info(
        "SPINE's Spec Driven Development (SDD) workflow guides you through "
        "6 phases for greenfield development: Specification, Design, Planning, "
        "Implementation, Review, and Verification."
    )

    # Phase lifecycle visualization
    _render_phase_lifecycle()

    st.divider()

    # New project form
    _render_new_project_form()

    st.divider()

    # List existing projects
    _render_projects_list()


def _render_phase_lifecycle() -> None:
    """Render the SDD phase lifecycle with progress indicators."""
    st.subheader("Phase Lifecycle")

    # Build visual indicator
    parts = []
    for i, phase in enumerate(SDD_PHASES):
        icon = SDD_PHASE_ICONS.get(phase, "•")
        if i < len(SDD_PHASES) - 1:
            parts.append(f"{icon} **{phase}** →")
        else:
            parts.append(f"{icon} **{phase}**")

    st.markdown(" ".join(parts))

    # Phase descriptions
    phase_info = {
        "SPEC": "Gather requirements and write formal specification",
        "DESIGN": "Design architecture, interfaces, and data flow",
        "PLAN": "Create detailed plan with FeatureSlices",
        "IMPLEMENT": "Execute implementation tasks in parallel",
        "REVIEW": "Code review and swarm gate validation",
        "VERIFY": "Test execution and final validation",
    }

    with st.expander("Phase Details"):
        for phase in SDD_PHASES:
            icon = SDD_PHASE_ICONS.get(phase, "•")
            desc = phase_info.get(phase, "")
            st.write(f"**{icon} {phase}**: {desc}")


def _render_new_project_form() -> None:
    """Render the form to start a new SDD project."""
    st.subheader("Start New SDD Project")

    # Session state for submission lifecycle
    if "sdd_submit_error" not in st.session_state:
        st.session_state.sdd_submit_error = None
    if "sdd_submitting" not in st.session_state:
        st.session_state.sdd_submitting = False

    # Display error if any
    if st.session_state.sdd_submit_error:
        st.error(st.session_state.sdd_submit_error)
        if st.button("Dismiss", key="dismiss_sdd_error"):
            st.session_state.sdd_submit_error = None
            st.rerun()

    is_submitting = st.session_state.sdd_submitting

    with st.form("sdd_new_project_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Project Name",
                placeholder="e.g., my-awesome-api",
                disabled=is_submitting,
                help="Unique name for this SDD project",
            )

        with col2:
            method = st.selectbox(
                "Method",
                ["Full Spec Project", "Full Spec Work", "Quick Work"],
                index=0,
                disabled=is_submitting,
                help="Automation level for the project",
            )

        requirement = st.text_area(
            "Requirement",
            placeholder="Describe what needs to be built...",
            height=100,
            disabled=is_submitting,
            help="Detailed description of the project requirements",
        )

        col3, col4 = st.columns(2)

        with col3:
            project_type = st.radio(
                "Project Type",
                ["Greenfield", "Brownfield"],
                index=0,
                disabled=is_submitting,
                help="Greenfield = new project. Brownfield = existing codebase.",
            )

        with col4:
            use_worktrees = st.toggle(
                "Use Worktrees",
                value=False,
                disabled=is_submitting,
                help="Enable git worktrees for parallel implementation",
            )

        submitted = st.form_submit_button(
            "▶ Start SDD Project" if not is_submitting else "⏳ Starting...",
            type="primary",
            disabled=is_submitting,
        )

        if submitted:
            if not name.strip():
                st.error("Please enter a project name.")
            elif not requirement.strip():
                st.error("Please enter a requirement description.")
            else:
                st.session_state.sdd_submitting = True

                with st.spinner("Starting SDD project..."):
                    result = start_sdd_project(
                        name=name.strip(),
                        requirement=requirement.strip(),
                        method=method,
                        project_type=project_type,
                        llm_provider="",
                        use_worktrees=use_worktrees,
                    )

                if result and "project_id" in result:
                    st.session_state.sdd_submitting = False
                    st.session_state.sdd_submit_error = None
                    st.success(f"SDD project created: {name}")
                    st.rerun()
                else:
                    st.session_state.sdd_submitting = False
                    error_msg = result.get("error", "Unknown error") if result else "Failed to create project"
                    st.session_state.sdd_submit_error = f"Failed to start SDD project: {error_msg}"
                    st.error(st.session_state.sdd_submit_error)


def _render_projects_list() -> None:
    """Render the list of existing SDD projects."""
    st.subheader("Existing SDD Projects")

    projects = get_sdd_projects()

    if not projects:
        st.info("No SDD projects yet. Use the form above to start your first project.")
        return

    for project in projects:
        _render_project_card(project)


def _render_project_card(project: dict) -> None:
    """Render a single SDD project card with controls."""
    project_id = project.get("id", "unknown")
    name = project.get("name", project_id)
    requirement = project.get("requirement", "")
    current_phase = project.get("current_phase", "SPEC")
    status = project.get("status", "unknown")
    phases = project.get("phases", {})

    icon = SDD_PHASE_ICONS.get(current_phase, "•")

    with st.container():
        col_icon, col_name, col_phase, col_status, col_actions = st.columns([1, 3, 2, 2, 3])

        col_icon.write(f"{icon}")

        col_name.write(f"**{name}**")
        if requirement:
            req_preview = requirement[:60] + ("..." if len(requirement) > 60 else "")
            col_name.caption(req_preview)

        col_phase.write(f"**{current_phase}**")

        # Status with color
        if status == "running":
            col_status.write(f":orange[{status}]")
        elif status == "completed":
            col_status.write(f":green[{status}]")
        elif status == "failed":
            col_status.write(f":red[{status}]")
        else:
            col_status.write(status)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        # View artifact button
        if btn_col1.button("View", key=f"view_sdd_{project_id}"):
            navigate_to_work(project_id)
            st.rerun()

        # Force complete button
        if btn_col2.button("Force Complete", key=f"force_complete_{project_id}"):
            idx = SDD_PHASES.index(current_phase) if current_phase in SDD_PHASES else 0
            if idx < len(SDD_PHASES) - 1:
                next_phase = SDD_PHASES[idx + 1]
                update_sdd_project_phase(project_id, next_phase, "success")
                st.rerun()
            else:
                update_sdd_project_phase(project_id, current_phase, "success")
                st.rerun()

        # Abort button
        if btn_col3.button("Abort", key=f"abort_sdd_{project_id}"):
            update_sdd_project_phase(project_id, current_phase, "failed")
            st.rerun()

    # Show phase artifacts if they exist
    _render_phase_artifacts(project_id, current_phase)


def _render_phase_artifacts(project_id: str, current_phase: str) -> None:
    """Render SDD phase artifacts (spec.md, architecture.md, plan.md) if they exist."""
    artifacts = []

    # Check for spec.md
    spec_path = Path(f".spine/sdd/projects/{project_id}/spec.md")
    if spec_path.exists():
        artifacts.append(("spec.md", spec_path))

    # Check for architecture.md
    arch_path = Path(f".spine/sdd/projects/{project_id}/architecture.md")
    if arch_path.exists():
        artifacts.append(("architecture.md", arch_path))

    # Check for plan.md
    plan_path = Path(f".spine/sdd/projects/{project_id}/plan.md")
    if plan_path.exists():
        artifacts.append(("plan.md", plan_path))

    if artifacts:
        with st.expander("Phase Artifacts", expanded=False):
            for art_name, art_path in artifacts:
                try:
                    content = art_path.read_text(encoding="utf-8")
                    st.markdown(f"**{art_name}**")
                    st.code(content, language="markdown")
                except Exception:
                    st.write(f"Could not read {art_name}")