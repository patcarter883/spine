"""SPINE Deep Agents profile — replaces the default DA base prompt.

The Deep Agents SDK ships a general-purpose ``BASE_AGENT_PROMPT`` aimed at
conversational assistants.  SPINE agents are **phase executors** inside a
deterministic state machine — they are not chatting with a user.  The default
prompt's conversational framing ("When the user asks you…", "ask only the
minimum followup") actively fights SPINE's workflow model.

This module:

1.  Defines ``SPINE_BASE_PROMPT`` — a replacement base prompt that preserves
    the useful behavioural guidance from the DA default (act-don't-talk,
    iterate, stop-and-analyze) while reframing the agent as a phase executor.
2.  Registers ``HarnessProfile`` instances for the providers SPINE uses so that
    ``create_deep_agent`` automatically picks up our base prompt via the
    ``base_system_prompt`` field (the ``CUSTOM`` slot in the prompt assembly
    order).

Prompt assembly order (per DA docs)::

    USER (phase system_prompt) → CUSTOM (SPINE_BASE_PROMPT) → SUFFIX (none)

RLM/interpreter guidance has been moved to the ``rlm-pattern`` skill for
progressive disclosure — it's loaded only when the interpreter is available,
saving ~500 tokens per agent on phases that don't need it.

Call :func:`ensure_spine_profiles` once at startup (from
``spine.agents.__init__`` or the CLI entry point) to activate.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── SPINE base prompt (replaces DA BASE_AGENT_PROMPT) ────────────────────

SPINE_BASE_PROMPT = """\
You are a phase executor inside SPINE, a deterministic AI agent harness. You \
are NOT a conversational assistant — there is no user in the loop during \
phase execution. You receive phase-specific context and must produce a \
structured artifact for the next phase.

## Core Behaviour

- Act, don't narrate. Never say "I'll now do X" — just do it.
- Work until the phase objective is fully met. Do not yield early with a \
summary of what you would do.
- If something fails repeatedly, stop and analyze *why* before retrying. \
Don't pound the same broken approach.
- Your first attempt is rarely correct — iterate.
- Be concise in reasoning. Reserve verbosity for the final artifact.

## Tools

You have access to standard tools:

- **Filesystem**: read_file, write_file, edit_file, ls, glob, grep — use \
these to inspect and modify the workspace.
- **Execute**: run shell commands (linters, tests, build scripts).
- **Task**: delegate to subagents for parallel work on independent slices.
- **Eval** *(when enabled)*: a QuickJS interpreter for code-first orchestration \
— composing tool calls, transforming structured data, and managing intermediate \
state outside the model context. See the RLM pattern skill for details.

Use them. Do not speculate about file contents — read the files. Do not \
guess test outcomes — run the tests.

## Cross-Work Memory

When the ``/memories/`` directory is available in your filesystem, you can \
persist project knowledge there that will survive across work items. Use it for:

- Project-specific conventions discovered during execution
- Frequently referenced file paths and module locations
- Patterns and gotchas worth remembering

Write to ``/memories/`` using filesystem tools. Read from it when starting \
a new task to leverage prior discoveries.

## Workflow Context

- You are running inside a phase of a larger workflow (SPECIFY → PLAN → \
TASKS → IMPLEMENT → VERIFY, with a CRITIC gate between phases).
- Your output will be reviewed by the critic and may be sent back for \
revision, or forwarded to the next phase.
- Do NOT ask follow-up questions — work with the context you are given.
- Do NOT seek user approval — execute autonomously within your phase scope.

## Output

- Produce the artifact your phase requires (specification, plan, slice \
definitions, implementation, verification report).
- Structure your output clearly with headers so downstream phases can \
parse it.
- End with a clear status indicator when the phase artifact is complete.
"""

# ── Profile registration ─────────────────────────────────────────────────

# Providers SPINE commonly uses.  Each gets a profile with our base prompt.
# The key format follows DA conventions: bare provider for provider-wide
# defaults, "provider:model" for per-model overrides.
_SPINE_PROVIDER_KEYS: list[str] = [
    "openrouter",
    "openai",
    "anthropic",
]

_REGISTERED = False


def ensure_spine_profiles() -> None:
    """Register SPINE HarnessProfiles for all supported providers.

    Safe to call multiple times — subsequent calls are no-ops after the
    first registration.  Must be called before any ``create_deep_agent()``
    invocation (typically at package import time or CLI startup).
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from deepagents import HarnessProfile, register_harness_profile
    except ImportError:
        logger.debug(
            "deepagents not installed — skipping SPINE HarnessProfile registration"
        )
        return

    profile = HarnessProfile(base_system_prompt=SPINE_BASE_PROMPT)

    for key in _SPINE_PROVIDER_KEYS:
        register_harness_profile(key, profile)
        logger.debug("Registered SPINE HarnessProfile for %r", key)

    _REGISTERED = True


def reset_spine_profiles() -> None:
    """Reset registration state (for testing)."""
    global _REGISTERED
    _REGISTERED = False
