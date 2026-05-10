"""Provider configuration page — CRUD for LLM providers."""

import streamlit as st

from spine.ui.utils import get_llm_providers, set_llm_providers


LLM_TYPES = ["openai", "ollama", "openrouter", "local-openai"]


def render_providers_config() -> None:
    """Render the provider configuration page."""
    st.title("🔌 Provider Configuration")

    st.info(
        "Manage LLM provider configurations. Changes are saved to "
        ".spine/config.yaml and affect both the CLI and the UI."
    )

    # Tab for LLM providers
    tab_llm, tab_memory, tab_storage, tab_notify = st.tabs(
        ["LLM Providers", "Memory", "Storage", "Notifications"]
    )

    with tab_llm:
        render_llm_providers_tab()

    with tab_memory:
        st.info("Memory provider configuration (future feature).")
        st.caption("Currently using built-in LangGraph memory state.")

    with tab_storage:
        st.info("Storage provider configuration (future feature).")
        st.caption("Currently using local file system.")

    with tab_notify:
        render_notify_tab()


def render_llm_providers_tab() -> None:
    """Render the LLM providers configuration tab."""
    providers = get_llm_providers()

    if not providers:
        st.warning("No LLM providers configured. Add one below.")

    st.subheader("Configured Providers")

    for i, provider in enumerate(providers):
        with st.container():
            col_remove = st.columns([1, 4, 4, 4, 4, 4, 1])[0]

            st.write(f"**Provider {i + 1}**")
            name = st.text_input(
                "Name",
                value=provider.get("name", ""),
                key=f"llm_name_{i}",
            )
            ltype = st.selectbox(
                "Type",
                LLM_TYPES,
                index=min(LLM_TYPES.index(provider.get("type", "ollama")), len(LLM_TYPES) - 1)
                if provider.get("type", "") in LLM_TYPES
                else 1,
                key=f"llm_type_{i}",
            )
            model = st.text_input(
                "Model",
                value=provider.get("model", "qwen3:32b"),
                key=f"llm_model_{i}",
            )
            api_key = st.text_input(
                "API Key",
                value=provider.get("api_key", ""),
                type="password",
                key=f"llm_key_{i}",
            )
            base_url = st.text_input(
                "Base URL",
                value=provider.get("base_url", _default_base_url(ltype)),
                key=f"llm_url_{i}",
            )
            priority = st.number_input(
                "Priority",
                min_value=0,
                max_value=10,
                value=provider.get("priority", i),
                key=f"llm_pri_{i}",
            )

            if col_remove.button("✕", key=f"llm_rm_{i}"):
                providers.pop(i)
                st.session_state.providers_dirty = True
                st.rerun()

            st.divider()

    # Add provider button
    col_add, _ = st.columns([1, 3])
    if col_add.button("+ Add Provider"):
        providers.append({
            "name": f"provider_{len(providers) + 1}",
            "type": LLM_TYPES[0],
            "model": "",
            "enabled": True,
            "priority": len(providers),
        })
        st.session_state.providers_dirty = True
        st.rerun()

    # Save button
    st.divider()
    col_save, _ = st.columns([1, 3])
    if col_save.button("💾 Save Configuration"):
        new_providers = []
        for i in range(len(providers)):
            new_providers.append({
                "name": st.session_state.get(f"llm_name_{i}", f"provider_{i + 1}"),
                "type": st.session_state.get(f"llm_type_{i}", "ollama"),
                "model": st.session_state.get(f"llm_model_{i}", ""),
                "api_key": st.session_state.get(f"llm_key_{i}", ""),
                "base_url": st.session_state.get(f"llm_url_{i}", ""),
                "priority": st.session_state.get(f"llm_pri_{i}", i),
                "enabled": True,
            })

        set_llm_providers(new_providers)
        st.success("Configuration saved!")
        st.session_state.providers_dirty = False


def render_notify_tab() -> None:
    """Render notification provider configuration tab."""
    from .utils import load_config

    config = load_config()
    notify_providers = config.get("providers", {}).get("notify", [])

    st.subheader("Notification Providers")

    if not notify_providers:
        st.info("No notification providers configured.")
    else:
        for i, provider in enumerate(notify_providers):
            with st.container():
                st.write(f"**{provider.get('name', 'unnamed')}** ({provider.get('type', '')})")
                st.json(provider)

    st.info("Notification providers (Discord, Slack, Email) are configured via .spine/config.yaml.")


def _default_base_url(provider_type: str) -> str:
    """Get the default base URL for a provider type."""
    defaults = {
        "ollama": "http://localhost:11434",
        "local-openai": "http://localhost:8000/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    return defaults.get(provider_type, "")
