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

RLM/interpreter guidance has been moved to the ``rlm-pattern`` skill for
progressive disclosure — it's loaded only when the interpreter is available,
saving ~500 tokens per agent on phases that don't need it.

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
    │ Summarization trigger                             │ 60K      │
    │ Trimmed tool results (structured metadata)         │ ~100/ea  │
    │ AI arg trimming (write_file/edit_file content)     │ ~50/ea   │
    ├───────────────────────────────────────────────────┼──────────┤
    │ Fixed overhead (prompt + schemas + skills)         │ ~4,250   │
    │ Available for conversation before summarization     │ ~56K     │
    │ Summarization keeps last 20 messages                │ ~15K     │
    └───────────────────────────────────────────────────┴──────────┘

Budget rules:
1. Summarization triggers at 60K tokens (keeps KV cache <50%).
2. ToolOutputTrimmer was REMOVED (2026-05 directive). Context now managed via prompts + summarization only.
3. AI arg trimming removes write_file/edit_file content from history.
4. codebase-map.md (produced by tasks phase) eliminates re-exploration.
5. Each phase starts with a fresh agent — no cross-phase history bloat.

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
- **Batch independent operations.** When you need to read ≥2 files or run ≥2 \
searches, make all calls in one response instead of sequentially.
- **Use the interpreter (eval) for orchestration.** When processing ≥3 files \
or dispatching ≥2 subagents, write a JS program in eval that reads files, \
dispatches work, and returns only the synthesis. \
PTC tool names are camelCase (`tools.readFile`), arguments are snake_case \
(`{file_path: '...'}`), and return values are native JS types — \
`readFile` returns a string, not an object.

## Tools

Tool descriptions are provided by the runtime. Follow these principles:
- Read before write — inspect existing code before modifying it.
- Test after write — run tests immediately after making changes.
- Use `task` subagents for parallel work on independent slices.
- Use `eval` to orchestrate multi-step workflows in code, not conversation.
- **Context is L1 cache; conversation history is swap.** If a compaction \
summary references an offloaded history file at \
`/conversation_history/{thread_id}.md`, you can read_file that path to \
page back specific details. Do NOT re-read source files just because they \
were evicted — cache them in eval instead.
- **Never re-read a file in the same phase.** If context editing evicts a \
prior read result, recover from eval: \
`globalThis.files = globalThis.files || {}; globalThis.files['path'] = content;`. \
Retrieve from eval instead of calling read_file again. \
Remember: `tools.readFile(...)` returns a string directly — store the string, \
not an object with `.content`.
- **Token budget: 60K prompt token target.** After 60K tokens, \
summarization compresses your conversation. Work efficiently: batch reads, \
use eval for multi-step orchestration, and produce compact artifacts. \
Evicted tool results appear as structured metadata like \
`[read: path (N lines) — symbols]` — use these hints instead of re-reading.

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
