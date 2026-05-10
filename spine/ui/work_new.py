"""Work creation form — create new work items with options."""

import streamlit as st

from spine.ui.utils import start_work


def render_new_work() -> None:
    """Render the new work item creation form."""
    st.title("➕ New Work Item")

    st.info(
        "Create a new work item to start a multi-agent workflow. "
        "The SPINE harness will orchestrate planning, execution, and verification phases."
    )

    with st.form("new_work_form"):
        st.subheader("Work Details")
        title = st.text_input(
            "Working Title",
            placeholder="e.g., Build authentication system",
            help="Short, descriptive title for this work item",
        )
        description = st.text_area(
            "Description",
            placeholder="Detailed description of what needs to be done...",
            height=100,
            help="Optional: provide more context for better planning",
        )

        st.subheader("Method")
        method = st.radio(
            "Automation Level",
            ["Quick Work", "Full Spec Work", "Full Spec Project"],
            index=0,
            help=(
                "**Quick Work**: Plan → Implement → Verify. "
                "**Full Spec**: Adds Design and Spec phases before planning. "
                "**Full Spec Project**: Adds additional requirements analysis."
            ),
        )

        st.subheader("Project Type")
        project_type = st.radio(
            "Environment",
            ["Greenfield", "Brownfield"],
            index=0,
            help="Greenfield = new project from scratch. Brownfield = existing codebase.",
        )

        st.subheader("Execution")

        # LLM provider selection
        providers = _get_provider_options()
        llm_provider = st.selectbox(
            "LLM Provider",
            providers,
            index=0,
            help="The LLM model used for agent decision-making",
        )

        parallel_agents = st.slider(
            "Max Parallel Agents",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum agents to run in parallel within a phase. "
                 "Higher values enable more concurrency but use more API calls.",
        )

        submitted = st.form_submit_button("▶ Start Work → ", type="primary")

        if submitted:
            if not title.strip():
                st.error("Please enter a working title.")
            else:
                with st.spinner("Starting work item..."):
                    result = start_work(
                        requirement=title.strip(),
                        method=method,
                        project_type=project_type,
                        llm_provider=llm_provider,
                        parallel_agents=parallel_agents,
                    )
                if result:
                    st.success("Work item created! Redirecting to details...")
                    st.session_state.selected_work_id = result.get("thread_id", "default")
                    st.session_state.page = "Work Detail"
                    st.rerun()
                else:
                    st.error("Failed to start work item. Check the console for details.")


def _get_provider_options() -> list[str]:
    """Get list of configured LLM providers for the selectbox.

    Returns:
        List of provider name strings, or default options if none configured.
    """
    from .utils import get_llm_providers

    providers = get_llm_providers()
    options = []
    for p in providers:
        name = p.get("name", "unnamed")
        model = p.get("model", "")
        provider_type = p.get("type", "")
        if model:
            label = f"{name}: {model} ({provider_type})"
        else:
            label = f"{name} ({provider_type})"
        options.append(label)

    if not options:
        options = [
            "qwen3:32b (ollama)",
            "gpt-4.1 (openai)",
            "local-model (local-openai)",
        ]

    return options
