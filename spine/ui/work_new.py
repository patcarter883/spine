"""Work creation form — create new work items with options."""

import uuid

import streamlit as st

from spine.ui.utils import start_work


def render_new_work() -> None:
    """Render the new work item creation form."""
    st.title("➕ New Work Item")

    st.info(
        "Create a new work item to start a multi-agent workflow. "
        "The SPINE harness will orchestrate planning, execution, and verification phases."
    )

    # ── Session state guards for submission lifecycle ──
    if "submitting_new_work" not in st.session_state:
        st.session_state.submitting_new_work = False
    if "submit_error" not in st.session_state:
        st.session_state.submit_error = None
    if "submit_idempotency_key" not in st.session_state:
        st.session_state.submit_idempotency_key = None

    # Display error banner from a previous failed submission
    if st.session_state.submit_error:
        st.error(st.session_state.submit_error)
        if st.button("Dismiss"):
            st.session_state.submit_error = None
            st.session_state.submitting_new_work = False
            st.rerun()

    is_submitting = st.session_state.submitting_new_work

    with st.form("new_work_form"):
        st.subheader("Work Details")
        title = st.text_input(
            "Working Title",
            placeholder="e.g., Build authentication system",
            help="Short, descriptive title for this work item",
            disabled=is_submitting,
        )
        _description = st.text_area(
            "Description",
            placeholder="Detailed description of what needs to be done...",
            height=100,
            help="Optional: provide more context for better planning",
            disabled=is_submitting,
        )

        st.subheader("Method")
        method = st.radio(
            "Automation Level",
            ["Quick Work", "Full Spec Work", "Full Spec Project"],
            index=0,
            disabled=is_submitting,
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
            disabled=is_submitting,
            help="Greenfield = new project from scratch. Brownfield = existing codebase.",
        )

        st.subheader("Execution")

        # LLM provider selection
        providers = _get_provider_options()
        llm_provider = st.selectbox(
            "LLM Provider",
            providers,
            index=0,
            disabled=is_submitting,
            help="The LLM model used for agent decision-making",
        )

        parallel_agents = st.slider(
            "Max Parallel Agents",
            min_value=1,
            max_value=10,
            value=3,
            disabled=is_submitting,
            help="Maximum agents to run in parallel within a phase. "
                 "Higher values enable more concurrency but use more API calls.",
        )

        submitted = st.form_submit_button(
            "▶ Start Work → " if not is_submitting else "⏳ Starting...",
            type="primary",
            disabled=is_submitting,
        )

        if submitted:
            if not title.strip():
                st.error("Please enter a working title.")
            else:
                # Guard: reject if already submitting
                if st.session_state.submitting_new_work:
                    st.warning("Submission already in progress.")
                    return

                st.session_state.submitting_new_work = True

                # Generate idempotency key for this submission
                idempotency_key = str(uuid.uuid4())
                st.session_state.submit_idempotency_key = idempotency_key

                with st.spinner("Starting work item..."):
                    result = start_work(
                        requirement=title.strip(),
                        method=method,
                        project_type=project_type,
                        llm_provider=llm_provider,
                        parallel_agents=parallel_agents,
                        idempotency_key=idempotency_key,
                    )

                if result and "thread_id" in result:
                    st.session_state.submitting_new_work = False
                    st.session_state.submit_error = None
                    st.session_state.submit_idempotency_key = None
                    st.success("Work item created! Redirecting to details...")
                    st.session_state.selected_work_id = result.get("thread_id", "default")
                    st.session_state.page = "Work Detail"
                    st.rerun()
                else:
                    # Failure: roll back UI state, show error banner
                    error_msg = result.get("error", "Unknown error") if result else "Could not reach backend"
                    friendly_msg = _format_submit_error(error_msg)
                    st.session_state.submit_error = friendly_msg
                    st.session_state.submitting_new_work = False
                    st.session_state.submit_idempotency_key = None
                    st.error(friendly_msg)


def _format_submit_error(raw: str) -> str:
    """Format a raw error into a user-friendly message.

    Args:
        raw: Raw error string from the backend.

    Returns:
        User-friendly error message.
    """
    raw_lower = raw.lower()
    if "timeout" in raw_lower or "timed out" in raw_lower:
        return "The request timed out. Please check that the backend is running and try again."
    if "connection" in raw_lower or "refused" in raw_lower:
        return "Could not connect to the backend. Make sure the SPINE service is running."
    if "provider" in raw_lower and ("not" in raw_lower or "configure" in raw_lower):
        return "No LLM provider is configured. Go to the Providers page and add one before starting work."
    return f"Failed to start work item: {raw}"


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
