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
    """Load .env from the project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        # Walk up from CWD to find .env — prefer project root
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
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

    @classmethod
    def load(cls, path: str = ".spine/config.yaml") -> SpineConfig:
        """Load configuration from a YAML file, falling back to defaults.

        Args:
            path: Path to the configuration YAML file.

        Returns:
            A SpineConfig instance with values from the file or defaults.
        """
        config = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
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
        raw_root = os.getenv(
            "SPINE_WORKSPACE_ROOT", spine.get("workspace_root", os.getcwd())
        )
        resolved_root = str(Path(raw_root).resolve())

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

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        for p in [self.checkpoint_path, self.artifact_path, self.queue_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
