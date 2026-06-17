"""Window-aware token budgeting for the SPECIFY/PLAN synthesizer calls.

The synthesize nodes assemble a single large prompt (objective + research
findings + recalled code + feedback) and request a completion on top. The
individual blocks each had a fixed token budget, but nothing coordinated
them against the model's context window — on trace 019eb3dd the specify
synthesizer sent a ~33K-token prompt plus a 30K completion request to a
60K-window model and 400'd.

This module derives one coherent ledger from the provider's declared
``context_window``::

    input_budget = window
                 - completion_cap        (clamped synth output request)
                 - fixed_cost            (system prompt + objective + feedback
                                          + scratchpad + instruction tail)
                 - tool_payload_reserve  (what the agent's first tool call
                                          returns on turn 2 — the prompt is
                                          re-sent each turn, so turn 2 is
                                          strictly larger than turn 1)
                 - overhead margin       (tool schemas, chat-template framing,
                                          tokenizer drift)

Providers without ``context_window`` declared get legacy behaviour: the
historical fixed budgets and no completion clamp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from spine.agents._tokens import count_tokens
from spine.config import SpineConfig

logger = logging.getLogger(__name__)

# Never allocate less evidence budget than this — a synthesizer with zero
# findings/recall context produces vacuous specs. If the fixed costs leave
# less than this, we log loudly and accept the overflow risk rather than
# silently dropping all evidence.
MIN_INPUT_BUDGET = 4000

# When evidence must shrink, findings keep at least this fraction of the
# input budget — findings are the primary evidence; recall is supplementary.
FINDINGS_FLOOR_FRAC = 0.6


@dataclass(frozen=True)
class SynthesisBudget:
    """Resolved token ledger for one synthesizer invocation."""

    window: int        # model context window (0 = unknown)
    completion_cap: int  # clamp for max_completion_tokens (0 = don't clamp)
    input_budget: int  # tokens available for findings + recall evidence
    legacy: bool       # True when no context_window is declared


@dataclass(frozen=True)
class EvidenceAllocation:
    """Split of ``input_budget`` across the two evidence blocks."""

    findings: int
    recall: int


def window_hard_ceiling(
    window: int, overhead: int, completion_floor: int = 512
) -> int:
    """Max prompt tokens that still leave room for overhead + a floor completion.

    Used as the last-resort eviction target so a single prompt can never be
    sent larger than the model's context window minus framing overhead and a
    minimum completion reservation. Returns 0 when ``window`` is unknown
    (cloud/legacy providers), which callers treat as "no hard guard".
    """
    if window <= 0:
        return 0
    return max(0, window - max(0, overhead) - max(0, completion_floor))


def synthesis_completion_cap(phase: str, phase_cap: int | None = None) -> int:
    """Completion-token clamp for a synth call, or 0 when legacy.

    Takes the tightest of the provider/global ``max_completion_tokens``
    and ``synthesize_max_completion_tokens`` — but only when the provider
    declares a ``context_window`` (finite-window local models). Cloud
    providers without a declared window keep their configured behaviour.

    Args:
        phase: Phase name for provider resolution.
        phase_cap: Optional phase-specific clamp used INSTEAD of the
            config's ``synthesize_max_completion_tokens`` — e.g. the
            implement phase passes ``implement_max_completion_tokens``
            so edit payloads get more headroom than spec/plan JSON.
    """
    cfg = SpineConfig.load()
    provider_cfg = cfg.resolve_provider_config(phase=phase)
    window = int(provider_cfg.get("context_window") or 0)
    if window <= 0:
        return 0
    candidates = [
        int(provider_cfg.get("max_completion_tokens") or 0),
        int(provider_cfg.get("max_tokens") or 0),
        cfg.max_completion_tokens,
        phase_cap if phase_cap and phase_cap > 0 else cfg.synthesize_max_completion_tokens,
    ]
    positive = [c for c in candidates if c > 0]
    return min(positive) if positive else 0


def escalated_completion_cap(
    budget: SynthesisBudget,
    *,
    prompt_tokens: int,
) -> int:
    """Raised completion clamp for a length-truncated synthesis retry.

    The synth clamp exists to keep prompt + completion inside a finite
    window — but when the structured artifact legitimately needs more than
    the clamp, the forced tool call truncates mid-arguments and an identical
    retry truncates identically (trace 019eb940: three plan-synthesize calls
    each burned exactly 8K completion tokens and produced no parseable
    ``write_structured_plan``). Doubling the clamp, bounded by the window
    room left above the MEASURED prompt, gives the retry a real chance
    without re-risking the overflow the clamp was added for.

    Args:
        budget: The ledger the truncated call ran under.
        prompt_tokens: Measured size of the actual synthesis prompt
            (system + user), not the worst-case reservation.

    Returns:
        The raised clamp, or 0 when escalation is impossible (legacy
        provider, no clamp, or no window headroom above the current clamp).
    """
    if budget.legacy or budget.window <= 0 or budget.completion_cap <= 0:
        return 0
    cfg = SpineConfig.load()
    room = budget.window - prompt_tokens - cfg.synthesize_overhead_tokens
    if room <= budget.completion_cap:
        return 0
    return min(budget.completion_cap * 2, room)


def window_aware_completion_cap(
    *,
    window: int,
    prompt_tokens: int,
    base_cap: int,
    overhead: int,
    floor: int = 512,
) -> int:
    """Per-turn completion clamp that keeps prompt+completion inside ``window``.

    ``synthesis_completion_cap`` reserves a *fixed* slice of the window for
    generation, computed once when the agent is built. But an agentic loop
    (the slice-implementer) grows its prompt turn by turn as it reads files,
    so a static reservation eventually leaves the request as
    ``prompt + base_cap > window`` and the provider 400s with "Context size
    has been exceeded" (trace 019ece87: a 27K prompt + 12K cap against a 32K
    window). Recomputing the clamp against the *measured* prompt each turn
    makes the overflow structurally impossible.

    Args:
        window: Model context window. ``<= 0`` (legacy/cloud) → no-op,
            returns ``base_cap`` unchanged.
        prompt_tokens: Measured size of the current request (system + history).
        base_cap: The completion ceiling to never exceed (the statically
            bound ``max_tokens``). The result is never larger than this.
        overhead: Safety margin for tool schemas, chat-template framing and
            tokenizer drift.
        floor: Smallest generation the call will still request. When the
            prompt has already eaten the window the provider may still reject,
            but we never request a zero/negative completion — eviction
            (TokenBudgetCompactor) is what reclaims the room.

    Returns:
        ``min(base_cap, window - prompt_tokens - overhead)``, floored at
        ``floor``; or ``base_cap`` when no finite window is declared.
    """
    if window <= 0 or base_cap <= 0:
        return base_cap
    room = window - prompt_tokens - overhead
    if room >= base_cap:
        return base_cap
    return max(floor, room)


def window_aware_compaction_threshold(
    *,
    window: int,
    configured_threshold: int,
    reserve: int,
) -> int:
    """Clamp a configured compaction trigger below the finite-window ceiling.

    A fixed threshold (e.g. 30000) is meaningless if the window is smaller
    than it: at a 32K window with a 12K generation reserve the usable prompt
    ceiling is ~20K, so a 30K trigger never fires before the provider 400s.
    This caps the trigger at ``window - reserve`` so eviction always engages
    with working room to spare.

    Args:
        window: Model context window. ``<= 0`` → return ``configured_threshold``
            unchanged (legacy/cloud providers keep their behaviour).
        configured_threshold: The threshold from config (0 disables).
        reserve: Room to keep free below the trigger for generation + overhead.

    Returns:
        The window-aware threshold, never below a small positive floor when a
        finite window is declared and the configured threshold is enabled.
    """
    if window <= 0 or configured_threshold <= 0:
        return configured_threshold
    ceiling = window - reserve
    if ceiling <= 0:
        # Pathologically small window — trim aggressively but never to 0.
        return max(1, window // 2)
    return min(configured_threshold, ceiling)


def resolve_synthesis_budget(
    phase: str,
    *,
    fixed_texts: list[str],
    tool_payload_reserve: int = 0,
) -> SynthesisBudget:
    """Compute the evidence input budget for one synthesizer invocation.

    Args:
        phase: Phase name for provider resolution (``"specify"``/``"plan"``).
        fixed_texts: Prompt pieces that are sent verbatim regardless of
            evidence size — system prompt, objective/description, rendered
            feedback, scratchpad, instruction tail. Measured exactly.
        tool_payload_reserve: Measured size of what the agent's first tool
            call (``read_work_context``/``read_prior_artifacts``) will
            append to the conversation, so the turn-2 request also fits.

    Returns:
        A :class:`SynthesisBudget`. ``legacy=True`` (with the historical
        fixed budgets) when the provider declares no ``context_window``.
    """
    cfg = SpineConfig.load()
    provider_cfg = cfg.resolve_provider_config(phase=phase)
    window = int(provider_cfg.get("context_window") or 0)

    if window <= 0:
        return SynthesisBudget(
            window=0,
            completion_cap=0,
            input_budget=(
                cfg.synthesize_findings_token_budget
                + cfg.specify_context_token_budget
            ),
            legacy=True,
        )

    completion_cap = synthesis_completion_cap(phase)
    fixed_cost = sum(count_tokens(t) for t in fixed_texts if t)
    overhead = cfg.synthesize_overhead_tokens
    input_budget = (
        window - completion_cap - fixed_cost - int(tool_payload_reserve) - overhead
    )
    floored = input_budget < MIN_INPUT_BUDGET
    if floored:
        logger.warning(
            "[%s] synthesis budget floored: window=%d completion_cap=%d "
            "fixed=%d reserve=%d overhead=%d → input_budget=%d < %d — "
            "fixed prompt content alone is near the window; the request "
            "may still overflow",
            phase, window, completion_cap, fixed_cost,
            tool_payload_reserve, overhead, input_budget, MIN_INPUT_BUDGET,
        )
        input_budget = MIN_INPUT_BUDGET
    logger.info(
        "[%s] synthesis budget ledger: window=%d completion_cap=%d fixed=%d "
        "reserve=%d overhead=%d → input_budget=%d%s",
        phase, window, completion_cap, fixed_cost, tool_payload_reserve,
        overhead, input_budget, " (floored)" if floored else "",
    )
    return SynthesisBudget(
        window=window,
        completion_cap=completion_cap,
        input_budget=input_budget,
        legacy=False,
    )


def allocate_evidence(
    budget: SynthesisBudget,
    *,
    findings_tokens: int,
    recall_tokens: int = 0,
) -> EvidenceAllocation:
    """Split ``input_budget`` across findings and recall blocks.

    Pass-through when both rendered blocks already fit; otherwise a
    proportional squeeze with a findings floor (findings are the primary
    evidence). Legacy budgets return the historical fixed constants so
    behaviour is unchanged for providers without ``context_window``.
    """
    if budget.legacy:
        cfg = SpineConfig.load()
        return EvidenceAllocation(
            findings=cfg.synthesize_findings_token_budget,
            recall=cfg.specify_context_token_budget,
        )

    total = findings_tokens + recall_tokens
    if total <= budget.input_budget:
        return EvidenceAllocation(findings=findings_tokens, recall=recall_tokens)

    findings_alloc = min(
        findings_tokens,
        max(
            int(budget.input_budget * FINDINGS_FLOOR_FRAC),
            budget.input_budget - recall_tokens,
        ),
    )
    recall_alloc = max(budget.input_budget - findings_alloc, 0)
    logger.info(
        "evidence over budget (%d > %d): findings %d→%d, recall %d→%d",
        total, budget.input_budget,
        findings_tokens, findings_alloc, recall_tokens, recall_alloc,
    )
    return EvidenceAllocation(findings=findings_alloc, recall=recall_alloc)


def estimate_tool_payload_reserve(
    *,
    workspace_root: str,
    artifact_dirs: list[str],
    description: str,
    feedback: list[str] | None = None,
) -> int:
    """Measure what the synth agent's first tool call will return.

    ``read_work_context`` (SPECIFY) returns description + feedback + the
    prior specification.md on rework; ``read_prior_artifacts`` (PLAN)
    returns every file under the prior phases' artifact directories. Both
    payloads land in the conversation as a ToolMessage and ride along in
    every subsequent request, so they must be budgeted up front.
    """
    total = count_tokens(description or "")
    for item in feedback or []:
        total += count_tokens(str(item))
    for rel in artifact_dirs:
        root = Path(workspace_root) / rel
        if not root.is_dir():
            continue
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            try:
                total += count_tokens(fp.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return total
