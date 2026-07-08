"""SPINE project facts — distil runs into CAM memory-organ writes.

Complement to :mod:`spine.agents.experience`: that loop captures *prose
lessons* injected into prompts; this one captures **fact-shaped knowledge**
(``subject → short object``) and writes it to the CAM editable memory served
next to the model (minisgl ``/cam/*`` plane), where the served model answers
with it in-forward — no prompt tokens, no retrieval step.

The pipeline at run finalization (:func:`capture_run_facts`):

1. **Distil** — one LLM pass over the run's material proposes at most
   :data:`_MAX_FACTS_PER_RUN` durable, project-level facts with single-phrase
   objects. Most runs yield none; the prompt says so explicitly.
2. **Gate** — each candidate goes to ``POST /cam/remember``; the *server's*
   base-uncertainty write gate decides (stores only what the base model can't
   already recall) and reports ``base_p`` either way.
3. **Record** — every attempt (stored or gate-skipped) lands in the
   ``facts.jsonl`` side index (:class:`spine.persistence.facts_store.FactsStore`)
   — the CAM banks can't be enumerated, so this is the authoritative intent
   log for reconciliation and rebuild.
4. **Persist** — one ``POST /cam/save`` after any accepted write.

Everything is best-effort and fail-open: no CAM provider, an unreachable
server, or a capacity stop all degrade to a no-op. Capture never raises into
run finalisation (the ``capture_run_experience`` contract).

Anchoring rule: like the experience store, the side index lives at the MAIN
repo root, never a run's worktree (see :mod:`spine.agents.experience` module
docstring for why).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spine.persistence.facts_store import FactsStore
from spine.models.types import ProjectFact

logger = logging.getLogger(__name__)

# One run should pin at most a handful of facts — the CAM store's comfortable
# capacity is ~128 per namespace and eviction is LRU (a flood here can push
# out durable knowledge). Selectivity is the feature.
_MAX_FACTS_PER_RUN = 5
_MAX_MATERIAL_CHARS = 6000
_MAX_OBJECT_WORDS = 4

# Crash/abort runs produce no trustworthy facts (same set the experience loop
# skips).
_SKIP_CAPTURE_STATUSES = {"failed", "cancelled", "stalled"}


def facts_store_for(config: Any) -> FactsStore:
    """Build a :class:`FactsStore` rooted at the project's main repo.

    Same anchoring as :func:`spine.agents.experience.experience_store_for` —
    the side index must survive worktree rollback.
    """
    base = getattr(config, "experience_path", None) or ".spine/experience"
    path = Path(base)
    root = getattr(config, "workspace_root", "") or ""
    if not path.is_absolute() and root:
        path = Path(root) / base
    return FactsStore(str(path))


# ── Distillation (LLM pass) ──────────────────────────────────────────────────
class _FactCandidate(BaseModel):
    """One proposed fact, in the exact shape /cam/remember consumes."""

    subject: str = Field(default="", description="Subject key, e.g. 'spine default branch'")
    probe_prompt: str = Field(
        default="",
        description=(
            "Cloze sentence ending right before the object, e.g. "
            "'The default branch of the spine repository is'"
        ),
    )
    object: str = Field(
        default="", description="The answer — a single word or very short phrase"
    )


class _FactDistillResult(BaseModel):
    facts: list[_FactCandidate] = Field(default_factory=list)


def _distill_system_prompt() -> str:
    from spine.agents.prompt_format import Tag, xml_block

    return (
        xml_block(
            Tag.ROLE,
            "You mine a completed software-engineering run for durable, "
            "project-level FACTS worth pinning into an editable model memory. "
            "A fact is a stable subject→object association (a name, a value, a "
            "decision outcome) that will still be true on future runs.",
        )
        + "\n\n"
        + xml_block(
            Tag.CONSTRAINTS,
            "- Each fact: a `subject` key, a cloze `probe_prompt` ending right "
            "before the answer, and an `object` of ONE word or a very short "
            "phrase (max "
            f"{_MAX_OBJECT_WORDS} words). Long answers do not fit the store.\n"
            "- Only durable project knowledge: pinned decisions, canonical "
            "names, fixed values, tool/branch/config identities. NEVER "
            "run-specific state (slice ids, temporary paths, this run's "
            "verdicts).\n"
            "- Never store secrets, tokens, or credentials.\n"
            f"- At most {_MAX_FACTS_PER_RUN} facts; MOST RUNS YIELD NONE — an "
            "empty list is the expected answer unless something genuinely "
            "durable was established.",
        )
        + "\n\n"
        + xml_block(
            Tag.OUTPUT_SCHEMA,
            'Return JSON {"facts": [{"subject": str, "probe_prompt": str, '
            '"object": str}, ...]} — empty list when nothing qualifies.',
        )
    )


def _run_material(result: dict[str, Any]) -> str:
    """Assemble the distillation input from the run's text-bearing fields."""
    parts: list[str] = []
    desc = result.get("description")
    if desc:
        parts.append(f"WORK DESCRIPTION:\n{desc}")
    spec = result.get("specification_json")
    if spec:
        parts.append(f"SPECIFICATION:\n{spec}")
    findings = result.get("verification_findings") or []
    lines = [str(f) for f in findings[:10]]
    if lines:
        parts.append("VERIFICATION FINDINGS:\n" + "\n".join(lines))
    material = "\n\n".join(parts)
    return material[:_MAX_MATERIAL_CHARS]


def _valid_candidate(c: _FactCandidate) -> bool:
    return bool(
        c.subject.strip()
        and c.probe_prompt.strip()
        and c.object.strip()
        and len(c.object.split()) <= _MAX_OBJECT_WORDS
    )


async def distill_run_facts(
    result: dict[str, Any], config: Any
) -> list[_FactCandidate]:
    """Propose fact candidates from a run's material via one LLM call.

    Best-effort: returns ``[]`` on any failure or when the material yields
    nothing durable (the common case).
    """
    material = _run_material(result)
    if not material.strip():
        return []
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from spine.agents.helpers import bind_structured_output, resolve_chat_model
        from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

        # Shares the `experience` phase routing override — same distillation
        # family, same cheap-model preference.
        model = resolve_chat_model(None, phase="experience")
        bound = bind_structured_output(model, _FactDistillResult)
        prompt = hostage_layout(
            xml_blocks((Tag.FINDINGS, material)),
            "Extract the durable project facts per the constraints — or an "
            "empty list if none qualify.",
        )
        res = await bound.ainvoke(
            [SystemMessage(content=_distill_system_prompt()), HumanMessage(content=prompt)]
        )
        if not isinstance(res, _FactDistillResult):
            res = _FactDistillResult.model_validate(res)
        return [c for c in res.facts if _valid_candidate(c)][:_MAX_FACTS_PER_RUN]
    except Exception:  # noqa: BLE001 — distillation is best-effort
        logger.debug("fact distillation failed (non-fatal)", exc_info=True)
        return []


# ── Capacity guard ───────────────────────────────────────────────────────────
def _store_count(stats: dict[str, Any] | None, facts: list | None) -> int | None:
    """Best-effort current fact count from /cam/stats (shape may evolve)."""
    if isinstance(stats, dict):
        for key in ("total_facts", "n_facts", "total_edits", "total"):
            v = stats.get(key)
            if isinstance(v, (int, float)):
                return int(v)
    if isinstance(facts, list):
        return len(facts)
    return None


# ── Capture (write path) ─────────────────────────────────────────────────────
async def capture_run_facts(
    result: dict[str, Any],
    config: Any,
    final_status: str,
) -> int:
    """Distil, gate-write, and record a run's project facts.

    Best-effort — never raises. Returns the number of facts the server's
    write gate accepted. No-op unless the active provider carries a ``cam:``
    block with ``write: distill``.
    """
    client = None
    try:
        if final_status in _SKIP_CAPTURE_STATUSES:
            return 0
        provider_cfg = None
        resolver = getattr(config, "resolve_active_provider", None)
        if callable(resolver):
            provider_cfg = resolver()
        if not provider_cfg or not provider_cfg.get("cam"):
            return 0

        from spine.services.cam_client import CAMClient, resolve_cam_settings

        settings = resolve_cam_settings(
            provider_cfg, workspace_root=getattr(config, "workspace_root", None)
        )
        if settings is None or settings.write != "distill":
            return 0

        candidates = await distill_run_facts(result, config)
        if not candidates:
            return 0

        client = CAMClient(settings)

        # Capacity guard (F2.4): the store knee is ~128/namespace and server
        # eviction is LRU — writing past the alert threshold risks pushing out
        # durable facts silently. Skip the batch instead.
        count = _store_count(await client.stats(), await client.facts())
        if count is not None and count >= settings.capacity_alert:
            logger.warning(
                "CAM store at %d facts (alert=%d) — skipping %d distilled write(s); "
                "prune with `/cam/facts` or raise capacity_alert",
                count,
                settings.capacity_alert,
                len(candidates),
            )
            return 0

        created = datetime.now().isoformat()
        work_id = result.get("work_id", "unknown")
        records: list[ProjectFact] = []
        accepted = 0
        for cand in candidates:
            resp = await client.remember(
                cand.subject, cand.probe_prompt, cand.object
            )
            if resp is None:
                # Server unreachable / CAM unloaded — nothing happened server-
                # side; stop attempting and record nothing for this candidate.
                break
            stored = bool(resp.get("stored"))
            accepted += int(stored)
            records.append(
                ProjectFact(
                    id=uuid.uuid4().hex[:12],
                    work_id=work_id,
                    subject=cand.subject,
                    probe_prompt=cand.probe_prompt,
                    object=cand.object,
                    namespace=settings.namespace,
                    stored=stored,
                    base_p=resp.get("base_p"),
                    source="distilled",
                    created_at=created,
                )
            )
        if accepted:
            await client.save()  # F2.3: persist across server restarts
        if records:
            facts_store_for(config).add_many(records)
            logger.info(
                "[%s] CAM facts: %d attempted, %d stored by write gate",
                work_id,
                len(records),
                accepted,
            )
        return accepted
    except Exception:  # noqa: BLE001 — capture must never break run finalisation
        logger.debug("fact capture failed (non-fatal)", exc_info=True)
        return 0
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
