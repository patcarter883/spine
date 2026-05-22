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
