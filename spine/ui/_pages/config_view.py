"""SPINE Config page — view and manage configuration including MCP servers."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the config page."""
    st.title("⚙️ Configuration")

    config = api.get_config()

    # ── Core settings ──────────────────────────────────────────────────
    st.subheader("Core Settings")
    with st.expander("View core configuration", expanded=False):
        st.json({k: v for k, v in config.items() if k != "mcp_servers"})

    # ── LLM Providers ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 LLM Providers")

    providers_data = api.get_providers()
    llm_providers = providers_data.get("llm", [])

    if not llm_providers:
        st.info("No LLM providers configured. Add one below.")

    for provider in llm_providers:
        name = provider.get("name", "unnamed")
        enabled = provider.get("enabled", True)
        badge = "✅ Enabled" if enabled else "❌ Disabled"

        with st.expander(f"🤖 {name} — {badge}", expanded=True):
            col1, col2, col3 = st.columns([3, 1, 0.5])

            with col1:
                st.markdown(f"**Model:** `{provider.get('model', 'N/A')}`")
                base_url = provider.get('base_url')
                st.markdown(f"**Base URL:** `{base_url if base_url else 'default'}`")
                if provider.get('temperature') is not None:
                    st.markdown(f"**Temperature:** `{provider['temperature']}`")
                if provider.get('max_tokens') is not None:
                    st.markdown(f"**Max Tokens:** `{provider['max_tokens']}`")

            with col2:
                if st.button("🗑️ Remove", key=f"remove_provider_{name}"):
                    if api.remove_llm_provider(name):
                        st.toast(f"Removed provider '{name}'", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("Failed to remove provider")

            with col3:
                is_editing = st.session_state.get(f"editing_{name}", False)
                if is_editing:
                    if st.button("Cancel", key=f"cancel_edit_{name}"):
                        st.session_state[f"editing_{name}"] = False
                        st.rerun()
                else:
                    if st.button("✏️ Edit", key=f"edit_{name}"):
                        st.session_state[f"editing_{name}"] = True
                        st.rerun()

            # Edit form (shown when editing this provider)
            if st.session_state.get(f"editing_{name}", False):
                with st.form(f"edit_provider_{name}"):
                    edit_model = st.text_input("Model", value=provider.get("model", ""), key=f"edit_{name}_model")
                    edit_base_url = st.text_input("Base URL", value=provider.get("base_url", ""), key=f"edit_{name}_base_url")
                    edit_api_key = st.text_input("API Key", value=provider.get("api_key", ""), type="password", key=f"edit_{name}_api_key")
                    edit_temp = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(provider.get("temperature", 0.7)), step=0.1, key=f"edit_{name}_temp")
                    edit_max_tokens = st.number_input("Max Tokens", min_value=1, value=int(provider.get("max_tokens", 4096)), step=1, key=f"edit_{name}_max_tokens")
                    edit_enabled = st.checkbox("Enabled", value=provider.get("enabled", True), key=f"edit_{name}_enabled")

                    save_submitted = st.form_submit_button("💾 Save", type="primary")
                    if save_submitted:
                        if not edit_model:
                            st.error("Model name is required")
                        else:
                            update_cfg = {"model": edit_model}
                            if edit_base_url:
                                update_cfg["base_url"] = edit_base_url
                            if edit_api_key:
                                update_cfg["api_key"] = edit_api_key
                            update_cfg["temperature"] = edit_temp
                            update_cfg["max_tokens"] = edit_max_tokens
                            update_cfg["enabled"] = edit_enabled

                            if api.update_llm_provider(name, update_cfg):
                                st.toast(f"Updated provider '{name}'", icon="✅")
                                st.session_state[f"editing_{name}"] = False
                                st.rerun()
                            else:
                                st.error("Failed to update provider")

    # ── Add LLM provider ───────────────────────────────────────────────
    st.divider()
    st.subheader("➕ Add LLM Provider")

    with st.form("add_llm_provider_form"):
        new_name = st.text_input(
            "Provider Name",
            key="new_provider_name",
            placeholder="e.g., openrouter-gateway",
        )
        new_model = st.text_input(
            "Model",
            key="new_provider_model",
            placeholder="e.g., openrouter:deepseek/deepseek-v4-pro",
        )
        new_base_url = st.text_input(
            "Base URL (optional)",
            key="new_provider_base_url",
            placeholder="e.g., https://api.openai.com/v1",
        )
        new_api_key = st.text_input(
            "API Key (optional)",
            key="new_provider_api_key",
            type="password",
            placeholder="sk-...",
        )
        new_temp = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key="new_provider_temp",
        )
        new_max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            value=4096,
            step=1,
            key="new_provider_max_tokens",
        )
        new_enabled = st.checkbox(
            "Enabled",
            value=True,
            key="new_provider_enabled",
        )

        submitted = st.form_submit_button("Add Provider", type="primary")
        if submitted:
            if not new_name or not new_model:
                st.warning("Provider name and model are required")
            else:
                provider_cfg = {
                    "model": new_model,
                }
                if new_base_url:
                    provider_cfg["base_url"] = new_base_url
                if new_api_key:
                    provider_cfg["api_key"] = new_api_key
                provider_cfg["temperature"] = new_temp
                provider_cfg["max_tokens"] = new_max_tokens
                provider_cfg["enabled"] = new_enabled

                if api.add_llm_provider(new_name, provider_cfg):
                    st.toast(f"Added provider '{new_name}'", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to add provider")

    # ── MCP Servers ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔌 MCP Servers")

    mcp_servers = config.get("mcp_servers", {})

    if not mcp_servers:
        st.info("No MCP servers configured. Add one below to give agents access to external tools.")

    # Display each configured server
    for server_name, server_cfg in mcp_servers.items():
        with st.expander(f"🖥️ {server_name}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                transport = server_cfg.get("transport", "stdio")
                st.markdown(f"**Transport:** `{transport}`")
                st.markdown(f"**Command:** `{server_cfg.get('command', '?')}`")
                args = server_cfg.get("args", [])
                if args:
                    st.markdown(f"**Args:** `{' '.join(args)}`")
                env_vars = server_cfg.get("env", {})
                if env_vars:
                    st.markdown("**Environment:**")
                    for k, v in env_vars.items():
                        st.markdown(f"- `{k}` = `{v}`")

            with col2:
                if st.button("🧪 Test Connection", key=f"test_{server_name}"):
                    with st.spinner(f"Testing connection to {server_name}..."):
                        result = api.test_mcp_connection(server_name)
                    if result["connected"]:
                        st.success(f"✅ Connected — {result['tool_count']} tools discovered")
                        with st.expander("View tools"):
                            for t in result["tool_names"]:
                                st.markdown(f"- `{t}`")
                    else:
                        st.error(f"❌ Connection failed: {result['error']}")

            with col3:
                if st.button("🗑️ Remove", key=f"remove_{server_name}"):
                    if api.remove_mcp_server(server_name):
                        st.toast(f"Removed {server_name}", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("Failed to remove server")

    # ── Add new MCP server ─────────────────────────────────────────────
    st.divider()
    st.subheader("➕ Add MCP Server")

    with st.form("add_mcp_server"):
        new_name = st.text_input("Server Name", placeholder="e.g., my-codebase-index")
        new_transport = st.selectbox(
            "Transport",
            options=["stdio", "http"],
            index=0,
            help="stdio: local subprocess. http: remote server.",
        )
        new_command = st.text_input(
            "Command",
            placeholder="e.g., mcp-codebase-index",
            help="The executable to run. Use 'python' for module-based servers.",
        )
        new_args = st.text_input(
            "Arguments (space-separated)",
            placeholder="e.g., -m mcp_codebase_index.server",
        )
        new_project_root = st.text_input(
            "PROJECT_ROOT (optional)",
            placeholder="/path/to/project",
            help="The directory to index. Defaults to workspace root if empty.",
        )

        submitted = st.form_submit_button("Add Server", type="primary")
        if submitted and new_name and new_command:
            server_cfg = {
                "transport": new_transport,
                "command": new_command,
                "args": new_args.split() if new_args else [],
                "env": {"PROJECT_ROOT": new_project_root} if new_project_root else {},
            }
            if api.update_mcp_server(new_name, server_cfg):
                st.toast(f"Added MCP server '{new_name}'", icon="✅")
                st.rerun()
            else:
                st.error("Failed to save MCP server configuration")
        elif submitted:
            st.warning("Server name and command are required")

    # ── Phase Provider Configuration ───────────────────────────────────
    st.divider()
    st.subheader("📋 Phase Provider Configuration")

    providers_data = api.get_providers()
    llm_providers = providers_data.get("llm", [])

    if not llm_providers:
        st.info("Configure at least one LLM provider first.")
    else:
        # Build provider names list
        provider_names = ["(default)"] + [p.get("name", "unnamed") for p in llm_providers]

        phase_providers = api.get_phase_providers()
        phase_config = {
            "specify": ("📐 Specify", "specify"),
            "plan": ("📝 Plan", "plan"),
            "implement": ("🔨 Implement", "implement"),
            "verify": ("🔍 Verify", "verify"),
            "critic": ("🧐 Critic", "critic"),
            "gap_plan": ("🔀 Gap Plan", "gap_plan"),
        }

        for display_name, phase_key in phase_config.values():
            current_provider = phase_providers.get(phase_key, {}).get("provider", "")

            col1, col2, col3 = st.columns([1, 2, 2])

            with col1:
                st.markdown(f"**{display_name}**")

            with col2:
                # Find index of current provider
                if current_provider and current_provider in provider_names:
                    index = provider_names.index(current_provider)
                else:
                    index = 0  # default

                st.selectbox(
                    "Provider",
                    options=provider_names,
                    index=index,
                    key=f"phase_{phase_key}_provider",
                )

            with col3:
                # Show temperature override if phase has one
                phase_cfg = phase_providers.get(phase_key, {})
                if "temperature" in phase_cfg:
                    st.number_input(
                        "Temperature Override",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(phase_cfg["temperature"]),
                        step=0.1,
                        key=f"phase_{phase_key}_temp",
                    )
                else:
                    st.text_input("Override", value="", disabled=True, key=f"phase_{phase_key}_override")

        # Save button
        if st.button("💾 Save Phase Configuration", type="primary"):
            saved_count = 0
            for display_name, phase_key in phase_config.values():
                selected_provider = st.session_state.get(f"phase_{phase_key}_provider", "(default)")

                if selected_provider == "(default)":
                    # Remove phase config (use default provider)
                    api.set_phase_provider(phase_key, {})
                else:
                    # Set phase to use this provider
                    api.set_phase_provider(phase_key, {"provider": selected_provider})
                saved_count += 1

            st.toast(f"Phase configuration saved ({saved_count} phases)", icon="✅")
            st.rerun()

    # ── Configuration Reference ────────────────────────────────────────
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
        "mcp_servers": (
            "Model Context Protocol servers for external tool integration. "
            "Uses langchain-mcp-adapters (MultiServerMCPClient). "
            "mcp-codebase-index provides 18 structural codebase query tools "
            "(symbol lookup, dependency analysis, change impact assessment)."
        ),
    }

    for key, description in config_help.items():
        st.write(f"**{key}**: {description}")
