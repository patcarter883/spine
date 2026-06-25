"""Map-reduce evidence compression for the SPECIFY/PLAN synthesizer prompts.

When the rendered research findings / recalled code exceed the evidence
allocation computed by :mod:`spine.agents.synthesis_budget`, hard truncation
(the historical behaviour) silently drops whole findings off the tail. This
module degrades more gracefully, and only does LLM work when actually over
budget:

Recall chunks — structural degrade, zero LLM calls:
    Strip ``raw_code`` from the largest chunks (largest first) so
    ``_format_retrieved_context`` falls back to each chunk's existing
    ``enriched_summary``. File/symbol identity is fully preserved.

Findings — map-reduce digest:
    Partition the finding blocks into batches and compress each batch with
    a parallel call to the ``summarization`` phase model (tightly capped,
    reasoning suppressed). Digests must preserve file paths, symbol names,
    and concrete facts. Any batch failure falls back to that batch's
    original findings — compression never fails synthesis; the budgeted
    ``format_findings`` truncation remains the final backstop.

Gated by ``evidence_compression_enabled`` (config kill switch).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents._tokens import count_tokens
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.config import SpineConfig

logger = logging.getLogger(__name__)

# Per-batch input size for the map stage. Small enough that batch + digest
# prompt fits comfortably in the summarization model's window, large enough
# to amortise the call overhead.
_BATCH_INPUT_TOKENS = 7000

# Never ask for a digest smaller than this — below it the model drops the
# concrete facts the synthesizer needs.
_MIN_DIGEST_TOKENS = 512

# Rough token overhead of a chunk's markdown header/fences in
# _format_retrieved_context, used when ranking chunks for degrade.
_CHUNK_HEADER_TOKENS = 20


def compress_recall_chunks(
    chunks: list[dict],
    *,
    budget_tokens: int,
) -> list[dict]:
    """Structurally degrade recall chunks to fit ``budget_tokens``.

    Strips ``raw_code`` from the largest chunks first, so the renderer's
    existing ``enriched_summary`` fallback kicks in for those chunks. Pure
    and deterministic — no LLM calls. Returns new dicts; input unmodified.
    """
    if not chunks or budget_tokens <= 0:
        return chunks

    def _cost(chunk: dict) -> int:
        raw = chunk.get("raw_code", "") or ""
        if raw:
            return count_tokens(raw) + _CHUNK_HEADER_TOKENS
        return count_tokens(chunk.get("enriched_summary", "") or "") + _CHUNK_HEADER_TOKENS

    def _summary_cost(chunk: dict) -> int:
        return count_tokens(chunk.get("enriched_summary", "") or "") + _CHUNK_HEADER_TOKENS

    out = [dict(c) for c in chunks]
    total = sum(_cost(c) for c in out)
    if total <= budget_tokens:
        return out

    # Largest raw bodies first — each swap buys the most headroom.
    by_size = sorted(
        (i for i, c in enumerate(out) if c.get("raw_code")),
        key=lambda i: count_tokens(out[i]["raw_code"]),
        reverse=True,
    )
    swapped = 0
    for i in by_size:
        if total <= budget_tokens:
            break
        saving = _cost(out[i]) - _summary_cost(out[i])
        if saving <= 0:
            continue
        out[i]["raw_code"] = ""
        total -= saving
        swapped += 1
    logger.info(
        "compress_recall_chunks: swapped raw_code→summary on %d/%d chunks "
        "(~%d tokens vs budget %d)",
        swapped, len(out), total, budget_tokens,
    )
    return out


def _batch_findings(
    findings: list[dict],
    render_one: Any,
) -> list[tuple[list[dict], str]]:
    """Greedily pack findings into batches of ~_BATCH_INPUT_TOKENS each."""
    batches: list[tuple[list[dict], str]] = []
    current: list[dict] = []
    current_tokens = 0
    for f in findings:
        text = render_one([f])
        size = count_tokens(text)
        if current and current_tokens + size > _BATCH_INPUT_TOKENS:
            batches.append((current, render_one(current)))
            current, current_tokens = [], 0
        current.append(f)
        current_tokens += size
    if current:
        batches.append((current, render_one(current)))
    return batches


# Largest finding set we will attempt to merge in a single call. Above this
# the consolidated prompt risks the summarization window; we skip aggregation
# and let the budgeted compress_findings map-reduce handle sizing instead.
_AGGREGATE_MAX_INPUT_TOKENS = 24000


async def aggregate_findings(
    findings: list[dict],
    *,
    phase: str,
    work_id: str = "",
    config: RunnableConfig | None = None,
) -> list[dict]:
    """Merge accumulated findings into one consolidated, deduped set.

    This is the Graph-of-Thoughts ``Aggregate`` operation (distinct from
    ``compress_findings``, which is budget-driven truncation): a single LLM
    pass that reconciles overlapping findings, drops redundancy, and ranks
    what remains by relevance — improving evidence *quality* before synthesis
    rather than merely fitting a token budget. Reducing the duplicate, raw
    append-only findings the synthesizer would otherwise ingest also shrinks
    its context, mitigating the read-spiral / context-blowup aborts.

    Returns the findings unchanged when there is nothing worth merging (<2
    findings), when the rendered set is too large for a single merge call, or
    when the merge call fails — aggregation never blocks synthesis.
    """
    from spine.agents.exploration_agents import format_findings

    if not findings or len(findings) < 2:
        return findings

    rendered = format_findings(findings)
    total = count_tokens(rendered)
    if total > _AGGREGATE_MAX_INPUT_TOKENS:
        logger.info(
            "[%s] findings too large to aggregate in one pass (%d > %d tokens) "
            "— leaving for budgeted compression",
            work_id, total, _AGGREGATE_MAX_INPUT_TOKENS,
        )
        return findings

    cfg = SpineConfig.load()
    try:
        from spine.agents.helpers import (
            cap_completion_tokens,
            resolve_chat_model,
            suppress_reasoning,
        )

        model = resolve_chat_model(
            config, session_id=work_id or None, phase="summarization"
        )
        model = cap_completion_tokens(model, cfg.summarise_max_completion_tokens)
        model = suppress_reasoning(model)
    except Exception:
        logger.warning(
            "[%s] could not build summarization model for findings "
            "aggregation — using raw findings",
            work_id, exc_info=True,
        )
        return findings

    prompt = hostage_layout(
        xml_blocks((Tag.FINDINGS, rendered)),
        (
            f"Consolidate the research findings above into a single deduplicated "
            f"evidence set for a {phase} synthesizer. MERGE findings that cover "
            "the same file or symbol; RECONCILE any contradictions (state the "
            "resolved fact); RANK the result so the most decision-relevant "
            "evidence comes first. You MUST preserve every file path, symbol "
            "name, data shape, and concrete technical fact — drop only "
            "duplication, narrative repetition, and hedging. Output consolidated "
            "markdown bullet points directly, no preamble."
        ),
    )
    try:
        result = await model.ainvoke(prompt)
        text = result.content if isinstance(result.content, str) else str(result.content)
        if not text.strip():
            raise ValueError("empty aggregation")
    except Exception:
        logger.warning(
            "[%s] findings aggregation call failed — using raw findings",
            work_id, exc_info=True,
        )
        return findings

    merged = [{"topic": "Aggregated findings (deduped, ranked)", "summary": text}]
    logger.info(
        "[%s] findings aggregation: %d findings (%d tokens) → 1 consolidated "
        "entry (%d tokens)",
        work_id, len(findings), total, count_tokens(format_findings(merged)),
    )
    return merged


async def compress_findings(
    findings: list[dict],
    *,
    budget_tokens: int,
    phase: str,
    work_id: str = "",
    config: RunnableConfig | None = None,
) -> list[dict]:
    """Compress findings to ~``budget_tokens`` via map-reduce digests.

    Returns the findings unchanged when they already fit, when compression
    is disabled, or when every digest call fails. Failed batches keep their
    original findings; the caller's budgeted ``format_findings`` render is
    the final truncation backstop either way.
    """
    from spine.agents.exploration_agents import format_findings

    if not findings or budget_tokens <= 0:
        return findings

    rendered = format_findings(findings)
    total = count_tokens(rendered)
    if total <= budget_tokens:
        return findings

    cfg = SpineConfig.load()
    if not cfg.evidence_compression_enabled:
        logger.info(
            "[%s] findings over budget (%d > %d) but evidence compression "
            "is disabled — falling back to truncation",
            work_id, total, budget_tokens,
        )
        return findings

    batches = _batch_findings(findings, format_findings)
    target = max(_MIN_DIGEST_TOKENS, budget_tokens // max(len(batches), 1))
    logger.info(
        "[%s] compressing %d findings (%d tokens) into %d digest batches "
        "of ~%d tokens each (budget %d)",
        work_id, len(findings), total, len(batches), target, budget_tokens,
    )

    try:
        from spine.agents.helpers import (
            cap_completion_tokens,
            resolve_chat_model,
            suppress_reasoning,
        )

        model = resolve_chat_model(config, session_id=work_id or None, phase="summarization")
        model = cap_completion_tokens(model, cfg.summarise_max_completion_tokens)
        model = suppress_reasoning(model)
    except Exception:
        logger.warning(
            "[%s] could not build summarization model for findings "
            "compression — falling back to truncation",
            work_id, exc_info=True,
        )
        return findings

    async def _digest(batch_text: str) -> str:
        prompt = hostage_layout(
            xml_blocks((Tag.FINDINGS, batch_text)),
            (
                f"Compress the research findings above into at most "
                f"~{target} tokens of markdown bullet points for a {phase} "
                "synthesizer. You MUST preserve every file path, symbol "
                "name, data shape, and concrete technical fact — drop only "
                "narrative repetition and hedging. Output the bullet "
                "points directly with no preamble."
            ),
        )
        result = await model.ainvoke(prompt)
        text = result.content if isinstance(result.content, str) else str(result.content)
        if not text.strip():
            raise ValueError("empty digest")
        return text

    results = await asyncio.gather(
        *(_digest(text) for _, text in batches), return_exceptions=True
    )

    out: list[dict] = []
    failed = 0
    for (batch, _text), result in zip(batches, results):
        if isinstance(result, BaseException):
            failed += 1
            out.extend(batch)
            continue
        topics = ", ".join(
            str(f.get("topic", "")) for f in batch if f.get("topic")
        )
        out.append(
            {
                "topic": f"Compressed digest — {topics}" if topics else "Compressed digest",
                "summary": result,
            }
        )
    if failed:
        logger.warning(
            "[%s] %d/%d digest batches failed — kept originals for those "
            "batches (budget truncation will apply)",
            work_id, failed, len(batches),
        )
    compressed_total = count_tokens(format_findings(out))
    logger.info(
        "[%s] findings compression: %d → %d tokens (%d findings → %d entries)",
        work_id, total, compressed_total, len(findings), len(out),
    )
    return out
