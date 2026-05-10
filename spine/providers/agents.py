"""Agent Provider implementations for external coding agents.

Provides a unified interface for delegating implementation work to external
coding agents (OpenCode, Codex CLI, Claude Code).  The default provider is
OpenCode because it supports any LLM backend (Anthropic, OpenAI, Ollama,
OpenRouter, custom endpoints, etc.), preserving SPINE's model-agnostic design.

Codex CLI (OpenAI-only) and Claude Code (Anthropic-only) are available as
optional providers that users opt into via spine.yaml.

Integration modes for OpenCode:
  - ``run``  -- subprocess ``opencode run "prompt"`` (simplest)
  - ``serve`` -- HTTP API via ``opencode serve`` (production, parallel)
  - ``acp``  -- JSON-RPC over stdio via ``opencode acp`` (recommended)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import Provider, ProviderType

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result from an agent execution."""

    output: str = ""
    exit_code: int = 0
    files_changed: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "exit_code": self.exit_code,
            "files_changed": self.files_changed,
            "error": self.error,
            "metadata": self.metadata,
            "success": self.success,
        }


# ── Base AgentProvider ─────────────────────────────────────────────────

class AgentProvider(Provider):
    """Base class for agent providers that delegate work to external tools.

    Agent providers are used by implementation agents (coder, test_engineer,
    reviewer) to actually write and modify code.  Decision-making agents
    (planner, critic, explorer) use LLMProvider directly instead.

    Subclasses must implement:
      - execute()  -- run a prompt through the external agent
      - is_available() -- check if the agent binary / SDK is installed
    """

    provider_type = ProviderType.AGENT

    @abstractmethod
    def execute(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a prompt via the external agent.

        Args:
            prompt: The instruction for the agent.
            workdir: Working directory for the agent (defaults to cwd).
            files: Optional list of file paths the agent should focus on.
            timeout: Max seconds to wait for completion.
            **kwargs: Provider-specific options.

        Returns:
            AgentResult with output, exit code, changed files, etc.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the agent binary / SDK is installed and usable."""
        ...

    # -- Provider interface (configure / validate / name / enabled) ------

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config

    def validate(self) -> bool:
        return self.is_available()

    @property
    def enabled(self) -> bool:
        cfg = getattr(self, "_config", None)
        if cfg and not cfg.get("enabled", True):
            return False
        return self.is_available()


# ── Helper: find changed files via git ─────────────────────────────────

def _git_changed_files(workdir: str | Path) -> list[str]:
    """Return list of files changed (including untracked) in *workdir*."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, cwd=str(workdir), timeout=10,
        )
        changed = result.stdout.strip().splitlines() if result.stdout.strip() else []

        # Also pick up untracked files
        result2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=str(workdir), timeout=10,
        )
        untracked = result2.stdout.strip().splitlines() if result2.stdout.strip() else []
        return sorted(set(changed + untracked))
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
#  OpenCode Agent Provider  (PRIMARY — model-agnostic)
# ═══════════════════════════════════════════════════════════════════════

class OpenCodeAgentProvider(AgentProvider):
    """Agent provider backed by anomalyco/opencode.

    OpenCode supports any LLM provider (Anthropic, OpenAI, Ollama,
    OpenRouter, Azure, Bedrock, custom endpoints, etc.), making it the
    default choice that preserves SPINE's model-agnostic design.

    Three integration modes:

    * ``run``   -- subprocess ``opencode run "prompt"`` (default)
    * ``serve`` -- HTTP POST to an ``opencode serve`` instance
    * ``acp``   -- JSON-RPC over stdio via ``opencode acp`` (future)

    Configuration (spine.yaml)::

        providers:
          agent:
            - name: opencode
              type: opencode
              config:
                mode: run            # run | serve | acp
                model: openrouter/google/gemini-2.5-flash
                serve_url: http://localhost:4096   # for serve mode
                agent: build          # build | plan | general
                auto_approve: true    # --auto-approve flag
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "opencode"

    # -- Availability check ----------------------------------------------

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    # -- Mode selection --------------------------------------------------

    @property
    def mode(self) -> str:
        return self._config.get("mode", "run")

    # -- Execute ---------------------------------------------------------

    def execute(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        mode = kwargs.pop("mode", None) or self.mode
        if mode == "serve":
            return self._execute_serve(prompt, workdir, files, timeout, **kwargs)
        if mode == "acp":
            return self._execute_acp(prompt, workdir, files, timeout, **kwargs)
        return self._execute_run(prompt, workdir, files, timeout, **kwargs)

    # -- Run mode (subprocess) -------------------------------------------

    def _execute_run(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        cmd = self._build_run_command(prompt, files, **kwargs)
        cwd = str(workdir) if workdir else os.getcwd()

        logger.info("OpenCode run: %s (cwd=%s, timeout=%ds)", " ".join(cmd[:4]), cwd, timeout)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            changed = _git_changed_files(cwd)
            result = AgentResult(
                output=proc.stdout,
                exit_code=proc.returncode,
                files_changed=changed,
                error=proc.stderr.strip() if proc.returncode != 0 else None,
                metadata={"mode": "run", "command": " ".join(cmd[:6])},
            )
            if proc.returncode != 0:
                logger.warning("OpenCode exited %d: %s", proc.returncode, proc.stderr[:200])
            return result

        except subprocess.TimeoutExpired:
            logger.error("OpenCode timed out after %ds", timeout)
            return AgentResult(
                output="",
                exit_code=-1,
                error=f"Timeout: opencode did not complete within {timeout}s",
                metadata={"mode": "run", "timeout": timeout},
            )
        except FileNotFoundError:
            return AgentResult(
                output="",
                exit_code=-1,
                error="opencode binary not found. Install: sudo pacman -S opencode",
                metadata={"mode": "run"},
            )

    def _build_run_command(
        self,
        prompt: str,
        files: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        cmd = ["opencode", "run"]

        # Model selection
        model = kwargs.get("model") or self._config.get("model")
        if model:
            cmd.extend(["--model", model])

        # Agent type (build=write code, plan=read-only, general=subagent)
        agent = kwargs.get("agent") or self._config.get("agent")
        if agent:
            cmd.extend(["--agent", agent])

        # Auto-approve tool calls (skip confirmation prompts)
        auto_approve = kwargs.get("auto_approve", self._config.get("auto_approve", True))
        if auto_approve:
            cmd.append("--dangerously-skip-permissions")

        # Use default format for reliable agent execution
        # JSON mode causes timeouts with local models due to different
        # internal processing; default mode uses efficient streaming
        cmd.extend(["--format", "default"])

        # File attachments
        if files:
            for f in files:
                cmd.extend(["--file", f])

        # Working directory
        workdir = kwargs.get("workdir") or self._config.get("workdir")
        if workdir:
            cmd.extend(["--dir", str(workdir)])

        # The prompt is the last positional argument
        cmd.append(prompt)
        return cmd

    # -- Serve mode (HTTP API) -------------------------------------------

    def _execute_serve(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        url = kwargs.get("serve_url") or self._config.get(
            "serve_url", "http://localhost:4096"
        )
        model = kwargs.get("model") or self._config.get("model")
        agent = kwargs.get("agent") or self._config.get("agent", "build")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "agent": agent,
        }
        if model:
            payload["model"] = model
        if files:
            payload["files"] = files

        try:
            import httpx

            resp = httpx.post(
                f"{url}/api/session",
                json=payload,
                timeout=float(timeout),
            )
            resp.raise_for_status()
            data = resp.json()

            session_id = data.get("session_id", data.get("id", ""))
            output = data.get("output", data.get("result", json.dumps(data)))

            cwd = str(workdir) if workdir else os.getcwd()
            changed = _git_changed_files(cwd)

            return AgentResult(
                output=output,
                exit_code=0,
                files_changed=changed,
                metadata={"mode": "serve", "session_id": session_id},
            )
        except ImportError:
            return AgentResult(
                output="",
                exit_code=-1,
                error="httpx not installed. Add to dependencies or use mode=run.",
                metadata={"mode": "serve"},
            )
        except Exception as exc:
            logger.error("OpenCode serve error: %s", exc)
            return AgentResult(
                output="",
                exit_code=-1,
                error=f"OpenCode serve error: {exc}",
                metadata={"mode": "serve"},
            )

    # -- ACP mode (JSON-RPC over stdio) ------------------------------------

    def _execute_acp(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        """ACP mode — structured agent-to-agent communication.

        Uses OpenCode's Agent Client Protocol (JSON-RPC over stdio).
        More reliable than ``opencode run`` subprocess mode because it
        uses a persistent connection with proper event streaming.
        """
        cwd = str(workdir) if workdir else os.getcwd()
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)

        model = kwargs.get("model") or self._config.get("model")

        proc = None
        session_id = None
        try:
            cmd = ["opencode", "acp"]
            env = {**os.environ, "TERM": "dumb"}

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            # 1. Initialize
            init_resp = self._acp_send_and_recv(proc, 1, "initialize", {
                "protocolVersion": 1,
                "capabilities": {},
                "clientInfo": {"name": "spine", "version": "0.1.0"},
            })
            if "error" in init_resp:
                return AgentResult(
                    output="", exit_code=-1,
                    error=f"ACP initialize failed: {init_resp['error']}",
                    metadata={"mode": "acp"},
                )

            # 2. Initialized notification (no response expected)
            self._acp_send(proc, "notifications/initialized", {})

            # 3. Create session
            sess_resp = self._acp_send_and_recv(proc, 2, "session/new", {
                "cwd": cwd, "mcpServers": [],
            })
            if "error" in sess_resp:
                return AgentResult(
                    output="", exit_code=-1,
                    error=f"ACP session/new failed: {sess_resp['error']}",
                    metadata={"mode": "acp"},
                )
            session_id = sess_resp.get("result", {}).get("sessionId")
            if not session_id:
                return AgentResult(
                    output="", exit_code=-1,
                    error="ACP session/new returned no session ID",
                    metadata={"mode": "acp"},
                )
            logger.info("ACP session created: %s", session_id)

            # 4. Build prompt params with model option
            prompt_parts = [{"type": "text", "text": prompt}]
            if files:
                for f in files:
                    try:
                        with open(f, "r") as fh:
                            content = fh.read()
                        prompt_parts.append({
                            "type": "resource",
                            "resource": {
                                "uri": f"file://{os.path.abspath(f)}",
                                "mimeType": "text/plain",
                                "text": content,
                            },
                        })
                    except Exception:
                        pass

            prompt_params: dict[str, Any] = {
                "sessionId": session_id,
                "prompt": prompt_parts,
            }
            if model:
                prompt_params["options"] = {"model": model}

            prompt_id = 3
            self._acp_send(proc, "session/prompt", prompt_params, id=prompt_id)

            # 5. Collect streaming events until prompt completes or timeout
            text_parts = []
            deadline = time.monotonic() + timeout
            completed = False
            error_msg = None

            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                line = self._acp_readline(proc, timeout=max(1, int(remaining)))
                if line is None:
                    if proc.poll() is not None:
                        error_msg = f"ACP process exited with code {proc.returncode}"
                        break
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Prompt response (matches request id) = completion
                if msg.get("id") == prompt_id:
                    if "error" in msg:
                        error_msg = f"ACP prompt error: {msg['error']}"
                    completed = True
                    break

                # Process session/update notifications
                method = msg.get("method", "")
                if method == "session/update":
                    update = msg.get("params", {}).get("update", {})
                    su = update.get("sessionUpdate", "")

                    if su == "agent_message_chunk":
                        content = update.get("content", {})
                        if content.get("type") == "text":
                            text = content.get("text", "")
                            if text:
                                text_parts.append(text)
                    elif su == "tool_call":
                        logger.debug(
                            "ACP tool_call: %s", update.get("title", "?"))
                    # Skip: agent_thought_chunk, usage_update,
                    # available_commands_update, tool_call_update

            # 6. Close session
            if session_id:
                try:
                    self._acp_send(proc, "session/close",
                                   {"sessionId": session_id}, id=99)
                except Exception:
                    pass

            # Determine result
            changed = _git_changed_files(cwd)
            output = "".join(text_parts).strip()

            if error_msg:
                return AgentResult(
                    output=output, exit_code=1,
                    error=error_msg,
                    files_changed=changed,
                    metadata={"mode": "acp", "session": session_id},
                )

            if not completed and time.monotonic() >= deadline:
                return AgentResult(
                    output=output, exit_code=-1,
                    error=f"ACP timed out after {timeout}s",
                    files_changed=changed,
                    metadata={"mode": "acp", "session": session_id,
                              "timeout": True},
                )

            return AgentResult(
                output=output, exit_code=0,
                files_changed=changed,
                metadata={"mode": "acp", "session": session_id},
            )

        except Exception as exc:
            return AgentResult(
                output="", exit_code=-1,
                error=f"ACP execution error: {exc}",
                metadata={"mode": "acp"},
            )
        finally:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def _acp_send(
        self,
        proc: subprocess.Popen,
        method: str,
        params: dict,
        id: int | None = None,
    ) -> None:
        """Send a JSON-RPC message to the ACP subprocess."""
        msg: dict = {"jsonrpc": "2.0", "method": method, "params": params}
        if id is not None:
            msg["id"] = id
        data = json.dumps(msg) + "\n"
        proc.stdin.write(data.encode())
        proc.stdin.flush()

    def _acp_send_and_recv(
        self,
        proc: subprocess.Popen,
        id: int,
        method: str,
        params: dict,
    ) -> dict:
        """Send a JSON-RPC request and wait for its response."""
        self._acp_send(proc, method, params, id=id)
        while True:
            line = self._acp_readline(proc, timeout=30)
            if line is None:
                raise RuntimeError("ACP subprocess closed while waiting for response")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == id:
                return msg
            # Skip notifications — they'll be processed during prompt streaming

    def _acp_readline(
        self,
        proc: subprocess.Popen,
        timeout: int = 30,
    ) -> str | None:
        """Read one line from ACP subprocess stdout with timeout."""
        import select
        if proc.stdout is None:
            return None
        fd = proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        return line.decode().strip()


# ═══════════════════════════════════════════════════════════════════════
#  Codex CLI Agent Provider  (OPTIONAL — OpenAI only)
# ═══════════════════════════════════════════════════════════════════════

class CodexAgentProvider(AgentProvider):
    """Agent provider backed by OpenAI Codex CLI.

    Codex CLI provides excellent sandboxing (read-only, workspace-write,
    full-access) and native ``codex exec`` for non-interactive execution.
    However, it only works with OpenAI models, violating SPINE's
    model-agnostic constraint.  It is therefore an **optional** provider
    that users must explicitly enable in spine.yaml.

    Configuration::

        providers:
          agent:
            - name: codex
              type: codex
              config:
                model: o3
                sandbox: workspace-write   # read-only | workspace-write | danger-full-access
                approval_mode: full-auto
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "codex"

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def execute(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        cmd = self._build_command(prompt, files, **kwargs)
        cwd = str(workdir) if workdir else os.getcwd()

        logger.info("Codex exec: %s (cwd=%s, timeout=%ds)", " ".join(cmd[:4]), cwd, timeout)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            changed = _git_changed_files(cwd)
            return AgentResult(
                output=proc.stdout,
                exit_code=proc.returncode,
                files_changed=changed,
                error=proc.stderr.strip() if proc.returncode != 0 else None,
                metadata={"mode": "exec", "command": " ".join(cmd[:6])},
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                output="",
                exit_code=-1,
                error=f"Timeout: codex did not complete within {timeout}s",
                metadata={"mode": "exec", "timeout": timeout},
            )
        except FileNotFoundError:
            return AgentResult(
                output="",
                exit_code=-1,
                error="codex binary not found. Install: npm install -g @openai/codex",
                metadata={"mode": "exec"},
            )

    def _build_command(
        self,
        prompt: str,
        files: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        cmd = ["codex", "exec"]

        # Model selection
        model = kwargs.get("model") or self._config.get("model")
        if model:
            cmd.extend(["--model", model])

        # Sandbox level
        sandbox = kwargs.get("sandbox") or self._config.get(
            "sandbox", "workspace-write"
        )
        cmd.extend(["--sandbox", sandbox])

        # Approval mode
        approval = kwargs.get("approval_mode") or self._config.get(
            "approval_mode", "full-auto"
        )
        cmd.extend(["--approval-mode", approval])

        # The prompt
        cmd.append(prompt)
        return cmd


# ═══════════════════════════════════════════════════════════════════════
#  Claude Code Agent Provider  (OPTIONAL — Anthropic only)
# ═══════════════════════════════════════════════════════════════════════

class ClaudeCodeAgentProvider(AgentProvider):
    """Agent provider backed by Anthropic Claude Code.

    Claude Code provides the best code quality and has a Python SDK with
    streaming, hooks, and subagent support.  However, it only works with
    Anthropic models, violating SPINE's model-agnostic constraint.  It is
    therefore an **optional** provider that users must explicitly enable.

    Two integration paths:
      - SDK mode: Uses the ``claude-agent-sdk`` Python package (preferred)
      - CLI mode: Falls back to the ``claude`` CLI binary

    Configuration::

        providers:
          agent:
            - name: claude-code
              type: claude-code
              config:
                mode: sdk                # sdk | cli
                model: claude-sonnet-4-20250514
                allowed_tools:
                  - read
                  - write
                  - bash
                max_turns: 50
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "claude-code"

    def is_available(self) -> bool:
        # Check for either SDK or CLI
        return self._sdk_available() or shutil.which("claude") is not None

    def _sdk_available(self) -> bool:
        try:
            import claude_agent_sdk  # noqa: F401
            return True
        except ImportError:
            return False

    def execute(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        mode = kwargs.get("mode") or self._config.get("mode")
        if mode == "cli":
            return self._execute_cli(prompt, workdir, files, timeout, **kwargs)
        # Prefer SDK, fall back to CLI
        if self._sdk_available():
            return self._execute_sdk(prompt, workdir, files, timeout, **kwargs)
        if shutil.which("claude"):
            return self._execute_cli(prompt, workdir, files, timeout, **kwargs)
        return AgentResult(
            output="",
            exit_code=-1,
            error="Neither claude-agent-sdk nor claude CLI found. "
                  "Install: pip install claude-agent-sdk",
            metadata={"mode": "none"},
        )

    # -- SDK mode --------------------------------------------------------

    def _execute_sdk(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        try:
            from claude_agent_sdk import ClaudeAgent

            model = kwargs.get("model") or self._config.get(
                "model", "claude-sonnet-4-20250514"
            )
            allowed_tools = kwargs.get("allowed_tools") or self._config.get(
                "allowed_tools", ["read", "write", "bash"]
            )
            max_turns = kwargs.get("max_turns") or self._config.get("max_turns", 50)
            cwd = str(workdir) if workdir else os.getcwd()

            agent = ClaudeAgent(
                model=model,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                cwd=cwd,
            )

            # Run synchronously — SDK manages the conversation loop
            output_text = ""
            for event in agent.run(prompt):
                if hasattr(event, "text"):
                    output_text += event.text
                elif hasattr(event, "content"):
                    output_text += str(event.content)

            changed = _git_changed_files(cwd)
            return AgentResult(
                output=output_text,
                exit_code=0,
                files_changed=changed,
                metadata={"mode": "sdk", "model": model},
            )
        except ImportError:
            return AgentResult(
                output="",
                exit_code=-1,
                error="claude-agent-sdk not installed. pip install claude-agent-sdk",
                metadata={"mode": "sdk"},
            )
        except Exception as exc:
            logger.error("Claude Code SDK error: %s", exc)
            return AgentResult(
                output="",
                exit_code=-1,
                error=f"Claude Code SDK error: {exc}",
                metadata={"mode": "sdk"},
            )

    # -- CLI mode --------------------------------------------------------

    def _execute_cli(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        cmd = self._build_cli_command(prompt, files, **kwargs)
        cwd = str(workdir) if workdir else os.getcwd()

        logger.info("Claude CLI: %s (cwd=%s, timeout=%ds)", " ".join(cmd[:4]), cwd, timeout)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            changed = _git_changed_files(cwd)
            return AgentResult(
                output=proc.stdout,
                exit_code=proc.returncode,
                files_changed=changed,
                error=proc.stderr.strip() if proc.returncode != 0 else None,
                metadata={"mode": "cli", "command": " ".join(cmd[:6])},
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                output="",
                exit_code=-1,
                error=f"Timeout: claude did not complete within {timeout}s",
                metadata={"mode": "cli", "timeout": timeout},
            )
        except FileNotFoundError:
            return AgentResult(
                output="",
                exit_code=-1,
                error="claude CLI not found. Install: npm install -g @anthropic-ai/claude-code",
                metadata={"mode": "cli"},
            )

    def _build_cli_command(
        self,
        prompt: str,
        files: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        cmd = ["claude"]

        # Model selection
        model = kwargs.get("model") or self._config.get("model")
        if model:
            cmd.extend(["--model", model])

        # Max turns
        max_turns = kwargs.get("max_turns") or self._config.get("max_turns")
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        # Allowed tools
        allowed = kwargs.get("allowed_tools") or self._config.get("allowed_tools")
        if allowed:
            for tool in allowed:
                cmd.extend(["--allowedTools", tool])

        # Print mode for non-interactive output
        cmd.extend(["--print", prompt])
        return cmd


# ═══════════════════════════════════════════════════════════════════════
#  Agent Fallback Chain
# ═══════════════════════════════════════════════════════════════════════

class AgentFallbackChain:
    """Manages multiple AgentProviders with priority-based fallback.

    Similar to ProviderFallbackChain but tailored for agent execution:
    tries providers in priority order, failing over on errors or
    unavailability.

    Usage::

        chain = AgentFallbackChain()
        chain.add(OpenCodeAgentProvider(), priority=1)
        chain.add(CodexAgentProvider(), priority=2)  # fallback

        result = chain.execute("Add a login page", workdir="/path/to/project")
    """

    def __init__(self) -> None:
        self._providers: list[tuple[int, AgentProvider]] = []

    def add(self, provider: AgentProvider, priority: int = 0) -> None:
        """Add a provider with a priority (lower = preferred)."""
        self._providers.append((priority, provider))
        self._providers.sort(key=lambda x: x[0])

    @property
    def active_provider(self) -> Optional[AgentProvider]:
        """Return the first available provider."""
        for _, provider in self._providers:
            if provider.enabled:
                return provider
        return None

    def execute(
        self,
        prompt: str,
        workdir: str | Path | None = None,
        files: list[str] | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute using the best available provider, with fallback.

        Tries each provider in priority order until one succeeds.
        If all fail, returns the last error result.
        """
        last_result: Optional[AgentResult] = None

        for _, provider in self._providers:
            if not provider.enabled:
                logger.debug("Skipping disabled provider: %s", provider.name)
                continue

            logger.info("Trying agent provider: %s", provider.name)
            try:
                result = provider.execute(prompt, workdir, files, timeout, **kwargs)
                if result.success:
                    return result
                last_result = result
                logger.warning(
                    "Provider %s failed (exit %d): %s",
                    provider.name, result.exit_code,
                    result.error or "unknown error",
                )
            except Exception as exc:
                logger.error("Provider %s raised exception: %s", provider.name, exc)
                last_result = AgentResult(
                    output="",
                    exit_code=-1,
                    error=f"Provider {provider.name} exception: {exc}",
                    metadata={"provider": provider.name},
                )

        if last_result is not None:
            return last_result

        return AgentResult(
            output="",
            exit_code=-1,
            error="No agent providers available. Install opencode, codex, or claude-code.",
            metadata={},
        )


# ═══════════════════════════════════════════════════════════════════════
#  Factory: create agent provider from config
# ═══════════════════════════════════════════════════════════════════════

_PROVIDER_MAP: dict[str, type[AgentProvider]] = {
    "opencode": OpenCodeAgentProvider,
    "codex": CodexAgentProvider,
    "claude-code": ClaudeCodeAgentProvider,
}


def create_agent_provider(name: str, config: dict[str, Any] | None = None) -> AgentProvider:
    """Create an AgentProvider by name with optional config.

    Args:
        name: Provider type name (opencode, codex, claude-code).
        config: Optional configuration dict passed to provider.configure().

    Returns:
        Configured AgentProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    cls = _PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown agent provider: '{name}'. "
            f"Available: {sorted(_PROVIDER_MAP.keys())}"
        )
    provider = cls()
    if config:
        provider.configure(config)
    return provider


def create_agent_chain_from_config(configs: list[dict[str, Any]]) -> AgentFallbackChain:
    """Build an AgentFallbackChain from spine.yaml provider configs.

    Expects a list of dicts like::

        [
            {"name": "opencode", "type": "opencode", "priority": 0, "config": {...}},
            {"name": "codex", "type": "codex", "priority": 1, "config": {...}},
        ]

    Returns:
        Configured AgentFallbackChain with all providers added.
    """
    chain = AgentFallbackChain()
    for cfg in configs:
        provider_type = cfg.get("type", cfg.get("name", ""))
        provider = create_agent_provider(provider_type, cfg.get("config"))
        chain.add(provider, priority=cfg.get("priority", 0))
    return chain


__all__ = [
    "AgentResult",
    "AgentProvider",
    "OpenCodeAgentProvider",
    "CodexAgentProvider",
    "ClaudeCodeAgentProvider",
    "AgentFallbackChain",
    "create_agent_provider",
    "create_agent_chain_from_config",
]
