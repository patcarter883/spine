"""SPINE configuration — load and validate .spine/config.yaml.

Environment variables are loaded from ``.env`` (project root) on first
import so that ``LANGSMITH_*`` and other runtime vars are available to
LangGraph, Deep Agents, and LangSmith tracing without manual sourcing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Load .env on import ──
# This ensures LANGSMITH_API_KEY, LANGSMITH_TRACING, OPENROUTER_API_KEY,
# etc. are set before any LangGraph or Deep Agents code reads them.
# It's safe to call multiple times (no-op if already loaded).

def _load_dotenv() -> None:
    """Load .env from the project root if python-dotenv is available.

    Search order:
      1. CWD and its parents (works when launched from the project root)
      2. The directory containing the spine package and its parents
         (works when Streamlit or another runner changes CWD away
         from the project root)

    ``override=False`` ensures manually-set env vars always win.
    """
    try:
        from dotenv import load_dotenv

        # Strategy 1: walk up from CWD (includes CWD itself)
        loaded = False
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / ".env").is_file():
                loaded = load_dotenv(dotenv_path=candidate / ".env", override=False)
                break

        # Strategy 2: walk up from the package directory — this handles
        # Streamlit which may launch with a CWD like $HOME or /tmp.
        if not loaded:
            pkg_dir = Path(__file__).resolve().parent
            for candidate in pkg_dir.parents:
                if (candidate / ".env").is_file():
                    load_dotenv(dotenv_path=candidate / ".env", override=False)
                    break
    except ImportError:
        # python-dotenv not installed — env vars must be set manually
        pass

_load_dotenv()


@dataclass
class SpineConfig:
    """Runtime configuration for SPINE.

    Loads from ``.spine/config.yaml`` with sensible defaults for missing keys.
    Environment variables override individual settings when set.
    """

    checkpoint_path: str = ".spine/spine.db"
    artifact_path: str = ".spine/artifacts"
    max_critic_retries: int = 3
    work_type: str = "spec"
    providers: dict = field(default_factory=dict)
    queue_backend: str = "sqlite"
    queue_path: str = ".spine/queue.db"
    workspace_root: str = ""
    interpreter_enabled: bool = False
    tool_schema_validation: bool = True
    phase_timeouts: dict = field(default_factory=lambda: {
        "specify": 600,
        "plan": 600,
        "tasks": 900,
        "implement": 1800,
        "verify": 600,
        "critic": 300,
    })
    default_timeout: int = 900

    @staticmethod
    def _find_workspace_root() -> str:
        """Auto-detect workspace root by searching upward for ``.spine/``.

        Search order:
          1. Walk up from CWD (works when launched from the project root)
          2. Walk up from the spine package directory (handles Streamlit,
             systemd, or other runners that change CWD away from the project)
          3. Fall back to CWD if neither search finds ``.spine/``

        This mirrors the ``_load_dotenv`` strategy for ``.env`` discovery.
        """
        # Strategy 1: walk up from CWD
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / ".spine").is_dir():
                return str(candidate)

        # Strategy 2: walk up from the package directory — this handles
        # Streamlit, systemd, or other runners that change CWD away from
        # the project root (e.g. to /root or /tmp).  Without this, the
        # workspace_root resolves to an inaccessible directory like /root,
        # causing LocalShellBackend to fail with Permission denied.
        pkg_dir = Path(__file__).resolve().parent
        for candidate in pkg_dir.parents:
            if (candidate / ".spine").is_dir():
                return str(candidate)

        return str(cwd)

    @classmethod
    def load(cls, path: str = ".spine/config.yaml") -> SpineConfig:
        """Load configuration from a YAML file, falling back to defaults.

        When *path* is relative and doesn't exist relative to CWD, also
        searches upward from the spine package directory for the config
        file.  This ensures Streamlit, systemd, and other runners that
        change CWD away from the project root can still find the config.

        Args:
            path: Path to the configuration YAML file.

        Returns:
            A SpineConfig instance with values from the file or defaults.
        """
        config = {}
        resolved_path = path

        if os.path.exists(path):
            resolved_path = path
        else:
            # Search from the package directory for the config file
            # (same strategy as _find_workspace_root and _load_dotenv).
            pkg_dir = Path(__file__).resolve().parent
            for candidate in pkg_dir.parents:
                candidate_path = candidate / path
                if candidate_path.is_file():
                    resolved_path = str(candidate_path)
                    break

        if os.path.exists(resolved_path):
            try:
                with open(resolved_path) as f:
                    config = yaml.safe_load(f) or {}
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                # If YAML is invalid, fall back to empty config (defaults will be used)
                config = {}

        spine = config.get("spine", {})

        # Resolve workspace_root: use Path.resolve() to get the canonical
        # (case-correct) absolute path.  On case-sensitive Linux, a typo
        # like /home/pat/projects vs /home/pat/Projects would silently
        # point at a different (or non-existent) directory, causing the
        # deep agent to write files to the wrong place.
        #
        # Auto-detect by searching upward for .spine/ when neither the env
        # var nor the config file explicitly set a value.
        raw_root = os.getenv(
            "SPINE_WORKSPACE_ROOT", spine.get("workspace_root", None)
        )
        if raw_root is None:
            raw_root = cls._find_workspace_root()
        resolved_root = str(Path(raw_root).resolve())

        # Sanity check: if workspace_root points to a directory the agent
        # can't write to (e.g. /root when not running as root), log a
        # warning.  This is a common failure mode when CWD is wrong and
        # auto-detection falls back to an inaccessible path.
        root_path = Path(resolved_root)
        if not os.access(resolved_root, os.W_OK):
            import logging
            logging.getLogger(__name__).warning(
                "workspace_root %s is not writable — agents will fail. "
                "Set SPINE_WORKSPACE_ROOT or add 'workspace_root' to "
                ".spine/config.yaml to fix this.",
                resolved_root,
            )
        elif not (root_path / ".spine").is_dir():
            import logging
            logging.getLogger(__name__).warning(
                "workspace_root %s has no .spine/ directory — auto-detection "
                "may have resolved to the wrong path. Consider setting "
                "SPINE_WORKSPACE_ROOT explicitly.",
                resolved_root,
            )

        return cls(
            checkpoint_path=os.getenv(
                "SPINE_CHECKPOINT_PATH", spine.get("checkpoint_path", ".spine/spine.db")
            ),
            artifact_path=os.getenv(
                "SPINE_ARTIFACT_PATH", spine.get("artifact_path", ".spine/artifacts")
            ),
            max_critic_retries=int(
                os.getenv("SPINE_MAX_CRITIC_RETRIES", spine.get("max_critic_retries", 3))
            ),
            work_type=os.getenv("SPINE_WORK_TYPE", spine.get("work_type", "spec")),
            providers=config.get("providers", {}),
            queue_backend=os.getenv("SPINE_QUEUE_BACKEND", spine.get("queue_backend", "sqlite")),
            queue_path=os.getenv("SPINE_QUEUE_PATH", spine.get("queue_path", ".spine/queue.db")),
            workspace_root=resolved_root,
            interpreter_enabled=os.getenv(
                "SPINE_INTERPRETER", str(spine.get("interpreter_enabled", False)).lower()
            ) in ("1", "true", "yes"),
            tool_schema_validation=os.getenv(
                "SPINE_TOOL_SCHEMA_VALIDATION",
                str(spine.get("tool_schema_validation", True)).lower(),
            ) not in ("0", "false", "no"),
        )

    def resolve_model(self, phase: str | None = None) -> str:
        """Resolve the LLM model identifier from provider config.

        Supports per-phase and per-subagent model overrides via the
        ``providers.phases`` section of ``.spine/config.yaml``.  Resolution
        order:

        1. ``providers.phases.<phase>.model`` (e.g. ``implement``)
        2. ``providers.phases.<phase/subagents/name>.model``
           (e.g. ``implement/subagents/slice-implementer``)
        3. First enabled LLM provider's ``model`` field
        4. ``SPINE_MODEL`` env var
        5. ``ValueError`` if none of the above are set

        The path-style key (``phase/subagents/name``) is checked **before**
        the bare phase key so that subagent overrides take priority over the
        phase default.

        Args:
            phase: Optional phase or phase/subagent path (e.g. ``"implement"``
                or ``"implement/subagents/slice-implementer"``).  When
                ``None``, only the default provider and env var are consulted.

        Returns:
            A model string like ``openrouter:z-ai/glm-4.5-air:free``.

        Raises:
            ValueError: If no model is configured anywhere.
        """
        # Check phase-specific overrides first (more specific key wins)
        if phase:
            phases = self.providers.get("phases", {})
            # Exact key first (e.g. "implement/subagents/slice-implementer")
            phase_cfg = phases.get(phase, {})
            if isinstance(phase_cfg, dict) and phase_cfg.get("model"):
                return phase_cfg["model"]
            # Fall back to parent phase key (e.g. "implement" from
            # "implement/subagents/slice-implementer")
            parent_phase = phase.split("/")[0]
            if parent_phase != phase:
                parent_cfg = phases.get(parent_phase, {})
                if isinstance(parent_cfg, dict) and parent_cfg.get("model"):
                    return parent_cfg["model"]

        # Default provider resolution
        provider = self.resolve_active_provider()
        if provider:
            return provider["model"]

        env_model = os.getenv("SPINE_MODEL")
        if env_model:
            return env_model

        raise ValueError(
            "No LLM model configured. Set 'providers.llm[].model' in "
            ".spine/config.yaml or set the SPINE_MODEL environment variable."
        )

    # ── Provider keys that phases can override locally ────────────────
    _PROVIDER_KEYS: tuple[str, ...] = (
        "base_url", "api_key", "temperature", "max_tokens",
        "max_completion_tokens", "request_timeout", "max_retries",
    )

    def resolve_active_provider(self) -> dict | None:
        """Return the full config dict for the first enabled LLM provider.

        This exposes ``base_url``, ``api_key``, ``temperature``, and other
        provider-specific fields that ``resolve_model()`` alone discards.
        Returns ``None`` when no enabled provider is found.

        Returns:
            The provider config dict, or ``None``.
        """
        llm_providers = self.providers.get("llm", [])
        for provider in llm_providers:
            if provider.get("enabled", True) and provider.get("model"):
                return provider
        return None

    def _lookup_provider_by_name(self, name: str) -> dict | None:
        """Find a named provider in ``providers.llm[]``.

        Args:
            name: The ``"name"`` field of the provider entry to find.

        Returns:
            The full provider config dict, or ``None`` if not found.
        """
        for provider in self.providers.get("llm", []):
            if provider.get("name") == name:
                return provider
        return None

    def resolve_provider_config(self, phase: str | None = None) -> dict:
        """Resolve provider-level settings for a given phase.

        Unlike :meth:`resolve_model` (which returns only the model string),
        this returns the full provider config dict — ``base_url``,
        ``api_key``, ``temperature``, ``max_tokens``,
        ``max_completion_tokens``, ``request_timeout``, ``max_retries`` —
        after applying any per-phase overrides.

        Resolution order (most specific wins, values are merged):

        1. Phase config's direct provider keys (``base_url``,
           ``temperature``, etc.) — take priority
        2. Phase config's ``provider`` reference — look up
           ``providers.llm[name]`` and inherit its settings
        3. First enabled provider in ``providers.llm[]``

        Args:
            phase: Optional phase or phase/subagent path (e.g.
                ``"implement"`` or
                ``"implement/subagents/slice-implementer"``).  When
                ``None``, only the default provider is consulted.

        Returns:
            A provider config dict containing ``base_url``, ``api_key``,
            and any other provider-level fields.  May be empty if no
            enabled provider is found.

        Example config::

            providers:
              llm:
                - name: vllm-local
                  model: openai:qwen3.6
                  base_url: http://localhost:8000/v1
                  api_key: vllm
                  temperature: 0.7
                  enabled: true
                - name: openrouter-gateway
                  model: openrouter:deepseek/deepseek-v4-pro
                  enabled: true
              phases:
                implement:
                  provider: vllm-local           # inherit vllm-local settings
                  temperature: 0.3               # but override temp
                verify:
                  base_url: http://other:8000/v1  # fully custom
                  api_key: other-key
        """
        # ── Step 1: resolve base provider (from reference or default) ──
        base: dict = {}
        if phase:
            phases = self.providers.get("phases", {})
            for key in (phase, phase.split("/")[0] if "/" in phase else None):
                if key is None:
                    continue
                phase_cfg = phases.get(key, {})
                if not isinstance(phase_cfg, dict):
                    continue
                provider_ref = phase_cfg.get("provider")
                if provider_ref:
                    named = self._lookup_provider_by_name(provider_ref)
                    if named:
                        base = dict(named)
                        break

        if not base:
            default = self.resolve_active_provider()
            if default:
                base = dict(default)

        # ── Step 2: apply phase-level overrides on top ──
        if phase:
            phases = self.providers.get("phases", {})
            for key in (phase, phase.split("/")[0] if "/" in phase else None):
                if key is None:
                    continue
                phase_cfg = phases.get(key, {})
                if not isinstance(phase_cfg, dict):
                    continue
                for k in self._PROVIDER_KEYS:
                    if k in phase_cfg:
                        base[k] = phase_cfg[k]

        return base

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        for p in [self.checkpoint_path, self.artifact_path, self.queue_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
