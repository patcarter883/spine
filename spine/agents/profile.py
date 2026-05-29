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
2.  Registers ``HarnessProfile`` instances for the providers SPINE uses so
that the agent factory can resolve the profile and apply the
``base_system_prompt`` field (the ``CUSTOM`` slot in the prompt assembly
order).

Prompt assembly order (in ``build_phase_agent``)::

    USER (phase system_prompt) → CUSTOM (SPINE_BASE_PROMPT from profile) → SUFFIX (none)

Note: The factory now uses ``create_agent`` directly instead of
``create_deep_agent``, so the HarnessProfile is resolved manually
via ``_resolve_profile()`` rather than automatically by DA.  The profile
still serves its purpose: it holds ``base_system_prompt`` and
``tool_description_overrides`` that the factory reads.

Phase agents do not have the ``eval`` interpreter or the ``task`` subagent
dispatcher. Parallel work is dispatched by the per-phase subgraph routers
via the LangGraph ``Send`` API.

Token budget for 128K context models (target: <60K prompt tokens at peak):

    ┌───────────────────────────────────────────────────┬──────────┐
    │ Component                                         │ Tokens   │
    ├───────────────────────────────────────────────────┼──────────┤
    │ SPINE_BASE_PROMPT (CUSTOM slot)                   │ ~550     │
    │ Phase system_prompt (USER slot)                    │ ~800     │
    │ Tool schemas (DA auto-injected)                    │ ~2,000   │
    │ Skills + memory (DA auto-injected)                 │ ~500     │
    │ Subagent specs (DA auto-injected)                  │ ~400     │
    │ Conversation history (grows, evicted by trimmer)   │ 0–50K    │
    │ Read cache hits (replaces full content w/ summary) │ ~50/ea   │
    │ Trimmed tool results (structured metadata)         │ ~100/ea  │
    │ AI arg trimming (write_file/edit_file content)     │ ~50/ea   │
    ├───────────────────────────────────────────────────┼──────────┤
    │ Fixed overhead (prompt + schemas + skills)         │ ~4,250   │
    │ Available for conversation before compaction       │ ~56K     │
    │ Read cache prevents re-read amplification          │ —        │
    └───────────────────────────────────────────────────┴──────────┘

Budget rules:
1. Read cache prevents duplicate file reads, keeping context growth linear.
2. ToolOutputTrimmer was REMOVED (2026-05 directive). Context now managed via prompts + read cache.
3. AI arg trimming removes write_file/edit_file content from history.
4. codebase-map.md (produced by tasks phase) eliminates re-exploration.
5. Each phase starts with a fresh agent — no cross-phase history bloat.

Call :func:`ensure_spine_profiles` once at startup (from
``spine.agents.__init__`` or the CLI entry point) to activate.
"""

from __future__ import annotations

import logging

from spine.agents.prompt_format import Tag, xml_block

logger = logging.getLogger(__name__)

# ── SPINE base prompt (replaces DA BASE_AGENT_PROMPT) ────────────────────

SPINE_BASE_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are a phase executor inside SPINE, a deterministic AI agent "
        "harness. You are NOT a conversational assistant — there is no user "
        "in the loop during phase execution. You receive phase-specific "
        "context and must produce a structured artifact for the next phase.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "Core behaviour:\n"
        "- Act, don't narrate. Never say \"I'll now do X\" — just do it.\n"
        "- Work until the phase objective is fully met. Do not yield early "
        "with a summary of what you would do.\n"
        "- If something fails repeatedly, stop and analyze *why* before "
        "retrying. Don't pound the same broken approach.\n"
        "- Your first attempt is rarely correct — iterate.\n"
        "- Be concise in reasoning. Reserve verbosity for the final artifact.\n"
        "- Batch independent operations. When you need to read ≥2 files or "
        "run ≥2 searches, make all calls in one response instead of "
        "sequentially.",
    )
    + "\n\n"
    + xml_block(
        Tag.TOOLS,
        "Tool descriptions are provided by the runtime. Follow these "
        "principles:\n"
        "- Read before write — inspect existing code before modifying it.\n"
        "- Test after write — run tests immediately after making changes.\n"
        "- Do not re-read a file already read this phase. The runtime keeps "
        "a read cache; rely on its summary (line count + symbols) instead "
        "of calling read_file again.",
    )
    + "\n\n"
    + xml_block(
        Tag.WORKFLOW,
        "- You are running inside a phase of a larger workflow (SPECIFY → "
        "PLAN → TASKS → IMPLEMENT → VERIFY, with a CRITIC gate between "
        "phases).\n"
        "- Your output will be reviewed by the critic and may be sent back "
        "for revision, or forwarded to the next phase.\n"
        "- Do NOT ask follow-up questions — work with the context you are "
        "given.\n"
        "- Do NOT seek user approval — execute autonomously within your "
        "phase scope.",
    )
    + "\n\n"
    + xml_block(
        Tag.OUTPUT_SCHEMA,
        "- Produce the artifact your phase requires (specification, plan, "
        "slice definitions, implementation, verification report).\n"
        "- Structure your output clearly with headers so downstream phases "
        "can parse it.\n"
        "- End with a clear status indicator when the phase artifact is "
        "complete.",
    )
)

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
    first registration.  Must be called before any agent construction
    (typically at package import time or CLI startup) so that
    ``_resolve_profile()`` in the factory can find the registered profile.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from deepagents import HarnessProfile, register_harness_profile
    except ImportError:
        logger.debug("deepagents not installed — skipping SPINE HarnessProfile registration")
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
