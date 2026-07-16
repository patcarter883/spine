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
# out durable knowledge). Selectivity is also a delivery-quality lever, not
# just capacity discipline: a small curated block keeps the in-context
# mechanism in its high-accuracy regime (plan §6.2).
_MAX_FACTS_PER_RUN = 5
_MAX_MATERIAL_CHARS = 6000
# Object length: the tap store holds a first-token/pooled latent, so objects
# must stay near-atomic. Pointer delivery (hybrid serving, `cam.mode` set) is
# lossless multi-token — the cap relaxes but stays short: these are pinned
# facts, not prose.
_MAX_OBJECT_WORDS = 4
_MAX_OBJECT_WORDS_POINTER = 12
# How many existing subject keys to show the distiller for canonical reuse.
_KNOWN_SUBJECTS_LIMIT = 40

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


def _cam_provider(config: Any) -> dict[str, Any] | None:
    """The provider whose ``cam:`` block governs the memory organ.

    Prefers :meth:`SpineConfig.resolve_cam_provider` — the organ is not
    necessarily on the first-enabled ("active") provider when phase routing
    sends the workhorse traffic to a later entry (e.g. openrouter listed
    first, CAM on the minisgl lane). Falls back to the active provider for
    config objects that predate the method.
    """
    for name in ("resolve_cam_provider", "resolve_active_provider"):
        resolver = getattr(config, name, None)
        if callable(resolver):
            provider = resolver()
            if provider and provider.get("cam"):
                return provider
    return None


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


# Role paragraph per material source: runs yield facts rarely (the prompt
# says so); curated project docs are dense with pinnable ground truth.
_ROLE_BY_SOURCE = {
    "run": (
        "You mine a completed software-engineering run for durable, "
        "project-level FACTS worth pinning into an editable model memory. "
        "A fact is a stable subject→object association (a name, a value, a "
        "decision outcome) that will still be true on future runs."
    ),
    "docs": (
        "You mine a project's curated documentation for durable, "
        "project-level FACTS worth pinning into an editable model memory. "
        "A fact is a stable subject→object association (a name, a value, a "
        "decision outcome) that will still be true on future runs. Prefer "
        "identities an agent would otherwise have to look up: entry points, "
        "canonical commands, key file locations, pinned conventions."
    ),
}


def _distill_system_prompt(
    max_object_words: int = _MAX_OBJECT_WORDS,
    known_subjects: list[str] | None = None,
    source: str = "run",
) -> str:
    from spine.agents.prompt_format import Tag, xml_block

    # Subject canonicalization (plan §6.2): the routed bank's retrieval misses
    # are same-structure aliases — spine controls the subject strings it
    # writes, so the distiller must never coin a near-alias for an entity
    # that already has a key.
    known_block = ""
    if known_subjects:
        known_block = (
            xml_block(
                Tag.KNOWN_FACTS,
                "Subject keys already in the project store — when a fact "
                "concerns one of these entities, reuse the EXACT string:\n"
                + "\n".join(f"- {s}" for s in known_subjects),
            )
            + "\n\n"
        )
    return (
        xml_block(Tag.ROLE, _ROLE_BY_SOURCE.get(source, _ROLE_BY_SOURCE["run"]))
        + "\n\n"
        + known_block
        + xml_block(
            Tag.CONSTRAINTS,
            "- Each fact: a `subject` key, a cloze `probe_prompt` ending right "
            "before the answer, and an `object` of ONE word or a very short "
            "phrase (max "
            f"{max_object_words} words). Long answers do not fit the store.\n"
            "- Subject keys are CANONICAL: one fixed name per entity, in the "
            "form '<project noun> <attribute>' (e.g. 'spine default branch'). "
            "Never coin a paraphrase, synonym, or re-ordering of a subject "
            "that already exists — aliases silently split the memory.\n"
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


def _valid_candidate(
    c: _FactCandidate, max_object_words: int = _MAX_OBJECT_WORDS
) -> bool:
    return bool(
        c.subject.strip()
        and c.probe_prompt.strip()
        and c.object.strip()
        and len(c.object.split()) <= max_object_words
    )


async def _distill_material(
    material: str,
    max_object_words: int = _MAX_OBJECT_WORDS,
    known_subjects: list[str] | None = None,
    source: str = "run",
) -> list[_FactCandidate]:
    """One LLM pass proposing fact candidates from arbitrary material.

    Best-effort: returns ``[]`` on any failure or empty material.
    """
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
        system = _distill_system_prompt(
            max_object_words=max_object_words,
            known_subjects=known_subjects,
            source=source,
        )
        res = await bound.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        if not isinstance(res, _FactDistillResult):
            res = _FactDistillResult.model_validate(res)
        return [
            c for c in res.facts if _valid_candidate(c, max_object_words)
        ][:_MAX_FACTS_PER_RUN]
    except Exception:  # noqa: BLE001 — distillation is best-effort
        logger.debug("fact distillation failed (non-fatal)", exc_info=True)
        return []


async def distill_run_facts(
    result: dict[str, Any],
    config: Any,
    max_object_words: int = _MAX_OBJECT_WORDS,
    known_subjects: list[str] | None = None,
) -> list[_FactCandidate]:
    """Propose fact candidates from a run's material via one LLM call.

    ``known_subjects`` are existing store keys shown to the distiller so it
    reuses canonical subject strings instead of coining near-aliases.
    Best-effort: returns ``[]`` on any failure or when the material yields
    nothing durable (the common case).
    """
    return await _distill_material(
        _run_material(result),
        max_object_words=max_object_words,
        known_subjects=known_subjects,
        source="run",
    )


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


# ── Shared write-path helpers ────────────────────────────────────────────────
def _known_subjects(config: Any, namespace: str | None) -> list[str]:
    """Existing subject keys for a namespace, newest first (canonical reuse)."""
    known: list[str] = []
    try:
        seen: set[str] = set()
        existing = facts_store_for(config).all()
        existing.sort(key=lambda f: f.created_at or "", reverse=True)
        for f in existing:
            if f.namespace != namespace:
                continue
            key = " ".join(f.subject.lower().split())
            if key and key not in seen:
                seen.add(key)
                known.append(f.subject)
            if len(known) >= _KNOWN_SUBJECTS_LIMIT:
                break
    except Exception:  # noqa: BLE001 — the reuse list is best-effort
        logger.debug("known-subjects gather failed (non-fatal)", exc_info=True)
    return known


async def _gate_write_and_record(
    client: Any,
    settings: Any,
    candidates: list[_FactCandidate],
    work_id: str,
    source: str,
    write_mode: str | None,
) -> tuple[int, list[ProjectFact]]:
    """Push candidates through the server's write gate, probing each accept.

    Returns ``(accepted, records)``. Stops (without a record) at the first
    ``None`` remember response — the server never saw that write, so a record
    would be a phantom. Every gate verdict that DID happen server-side is
    recorded, stored or skipped alike.
    """
    created = datetime.now().isoformat()
    records: list[ProjectFact] = []
    accepted = 0
    for cand in candidates:
        resp = await client.remember(
            cand.subject, cand.probe_prompt, cand.object, mode=write_mode
        )
        if resp is None:
            break
        stored = bool(resp.get("stored"))
        accepted += int(stored)
        # F3.3 readback probe: /cam/ask is the only ground-truth check that
        # the store actually delivers the fact (a crowded bank degrades
        # silently). One short generation per accepted write.
        verified: bool | None = None
        if stored:
            probe = await client.ask_full(
                cand.probe_prompt, cand.subject, mode=write_mode
            )
            text = probe.get("text") if isinstance(probe, dict) else None
            if text is not None:
                verified = cand.object.strip().lower() in text.lower()
                if not verified:
                    logger.warning(
                        "CAM readback probe failed for %r — store may be "
                        "crowded (check /cam/stats)",
                        cand.subject,
                    )
                mode_served = probe.get("mode_served")
                if write_mode and mode_served and mode_served != write_mode:
                    logger.warning(
                        "CAM served %r via %s (asked for %s) — check the "
                        "hybrid routing for %r",
                        cand.subject,
                        mode_served,
                        write_mode,
                        settings.namespace,
                    )
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
                verified=verified,
                source=source,
                created_at=created,
                mode=write_mode,
            )
        )
    return accepted, records


# ── Capture (write path) ─────────────────────────────────────────────────────
async def capture_run_facts(
    result: dict[str, Any],
    config: Any,
    final_status: str,
) -> int:
    """Distil, gate-write, and record a run's project facts.

    Best-effort — never raises. Returns the number of facts the server's
    write gate accepted. No-op unless an enabled provider carries a ``cam:``
    block with ``write: distill``.
    """
    client = None
    try:
        if final_status in _SKIP_CAPTURE_STATUSES:
            return 0
        provider_cfg = _cam_provider(config)
        if not provider_cfg or not provider_cfg.get("cam"):
            return 0

        from spine.services.cam_client import CAMClient, resolve_cam_settings

        settings = resolve_cam_settings(
            provider_cfg, workspace_root=getattr(config, "workspace_root", None)
        )
        if settings is None or settings.write != "distill":
            return 0

        # Distilled facts always go to the pointer store on a hybrid server —
        # they are exactly what pointer delivery is for (never opt them into
        # tap, even when cam.mode is "tap"/"both"). Pre-hybrid: no mode field.
        write_mode = "pointer" if settings.mode else None
        max_object_words = (
            _MAX_OBJECT_WORDS_POINTER if write_mode else _MAX_OBJECT_WORDS
        )

        # Show the distiller the namespace's existing subject keys so it
        # reuses canonical strings instead of coining near-aliases (the routed
        # bank's retrieval misses are same-structure aliases).
        known_subjects = _known_subjects(config, settings.namespace)

        candidates = await distill_run_facts(
            result,
            config,
            max_object_words=max_object_words,
            known_subjects=known_subjects,
        )
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

        work_id = result.get("work_id", "unknown")
        accepted, records = await _gate_write_and_record(
            client,
            settings,
            candidates,
            work_id=work_id,
            source="distilled",
            write_mode=write_mode,
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


# ── Seeding (curated docs → gate-filtered store writes) ─────────────────────
# Deliberate store population from the project's onboarding docs, so the very
# first run gets a useful <known_facts> block instead of waiting weeks of
# ≤5-facts-per-run organic fill. Same gate, same side index, same probes as
# run capture — only the material differs.
_SEED_DEFAULT_DIR = ".spine/onboarding"
_MAX_SEED_FACTS = 20


def _chunk_paragraphs(text: str, size: int = _MAX_MATERIAL_CHARS) -> list[str]:
    """Pack paragraphs into ~``size``-char chunks (hard-splitting oversized ones)."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        if len(para) > size:  # pathological single paragraph: flush, hard-split
            if current:
                chunks.append(current)
                current = ""
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size:]
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _seed_paths(config: Any, paths: list[str] | None) -> list[Path]:
    """Resolve seed sources: given files/dirs, or the onboarding dir's *.md."""
    root = Path(getattr(config, "workspace_root", "") or ".")
    if not paths:
        default = root / _SEED_DEFAULT_DIR
        return sorted(default.glob("*.md")) if default.is_dir() else []
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            out.extend(sorted(path.glob("*.md")))
        elif path.is_file():
            out.append(path)
    return out


async def seed_project_facts(
    config: Any,
    paths: list[str] | None = None,
    max_facts: int = _MAX_SEED_FACTS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Distil project docs into gate-filtered CAM facts.

    Chunks each doc, distils chunks with the accumulated subject list (so
    later chunks reuse canonical keys instead of coining aliases), dedupes
    against the side index, and — unless ``dry_run`` — pushes candidates
    through the server's write gate, never past the capacity alert.

    Returns a summary dict: ``docs`` (paths read), ``candidates`` (proposed
    after dedupe/cap), ``records`` (side-index records written),
    ``accepted`` (gate-stored count), and ``blocked`` (candidates the
    capacity guard refused to attempt).
    """
    from spine.services.cam_client import CAMClient, resolve_cam_settings

    provider_cfg = _cam_provider(config)
    if not provider_cfg:
        raise RuntimeError("No enabled provider carries a `cam:` block")
    settings = resolve_cam_settings(
        provider_cfg, workspace_root=getattr(config, "workspace_root", None)
    )
    if settings is None:
        raise RuntimeError("CAM is configured but disabled")

    write_mode = "pointer" if settings.mode else None
    max_object_words = (
        _MAX_OBJECT_WORDS_POINTER if write_mode else _MAX_OBJECT_WORDS
    )

    docs = _seed_paths(config, paths)
    known = _known_subjects(config, settings.namespace)
    seen = {" ".join(s.lower().split()) for s in known}
    candidates: list[_FactCandidate] = []
    for doc in docs:
        if len(candidates) >= max_facts:
            break
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("seed: cannot read %s — skipped", doc)
            continue
        for chunk in _chunk_paragraphs(f"SOURCE DOC: {doc.name}\n\n{text}"):
            if len(candidates) >= max_facts:
                break
            found = await _distill_material(
                chunk,
                max_object_words=max_object_words,
                # Accumulate: later chunks see earlier chunks' subjects too.
                known_subjects=known + [c.subject for c in candidates],
                source="docs",
            )
            for cand in found:
                key = " ".join(cand.subject.lower().split())
                if key in seen or len(candidates) >= max_facts:
                    continue
                seen.add(key)
                candidates.append(cand)

    summary: dict[str, Any] = {
        "docs": [str(d) for d in docs],
        "candidates": candidates,
        "records": [],
        "accepted": 0,
        "blocked": 0,
    }
    if dry_run or not candidates:
        return summary

    client = CAMClient(settings)
    try:
        # Capacity discipline (F2.4): never write past the alert threshold —
        # crowding degrades OTHER facts silently. Trim to the headroom.
        count = _store_count(await client.stats(), await client.facts())
        headroom = (
            max(0, settings.capacity_alert - count)
            if count is not None
            else len(candidates)
        )
        to_write, blocked = candidates[:headroom], candidates[headroom:]
        summary["blocked"] = len(blocked)
        if blocked:
            logger.warning(
                "CAM store at %s facts (alert=%d) — %d seed candidate(s) not "
                "attempted",
                count,
                settings.capacity_alert,
                len(blocked),
            )
        accepted, records = await _gate_write_and_record(
            client,
            settings,
            to_write,
            work_id="",
            source="seeded",
            write_mode=write_mode,
        )
        if accepted:
            await client.save()
        if records:
            facts_store_for(config).add_many(records)
        summary["accepted"] = accepted
        summary["records"] = records
        return summary
    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


# ── Read path (F1.3: deterministic prompt-side rendering) ────────────────────
# How many facts to inject — the store itself is capped (~128/namespace) but a
# prompt block should stay small; injection favours the most recent records.
_INJECT_FACTS_LIMIT = 20


def format_known_facts_block(facts: list[ProjectFact]) -> str:
    """Render stored facts as a ``<known_facts>`` system-prompt block."""
    if not facts:
        return ""
    from spine.agents.prompt_format import Tag, xml_block

    lines = ["Established facts about this project (treat as ground truth):"]
    lines.extend(f"- {f.subject}: {f.object}" for f in facts)
    return xml_block(Tag.KNOWN_FACTS, "\n".join(lines))


def resolve_known_facts_block(config: Any | None = None) -> str:
    """Return the injectable ``<known_facts>`` block (best-effort).

    Active only when the provider's ``cam.read`` mode is ``facts_block`` or
    ``both``. Renders from the LOCAL side index (gate-accepted facts for the
    resolved namespace), not a live ``/cam/facts`` call: the agent-build path
    must not block on the network, and the side index is spine's authoritative
    record of what it wrote. The block is the principled read mechanism, not a
    fallback: the frozen-base scorecard (plan §6.2) shows in-forward injection
    cannot participate in multi-hop reasoning — in-context delivery is the
    quality ceiling. Under ``both`` it coexists with transparent delivery,
    which then serves as a token saving for recall-shaped queries.
    Returns ``""`` when CAM is off, the mode doesn't inject, or anything fails.
    """
    try:
        if config is None:
            from spine.config import SpineConfig

            config = SpineConfig.load()
        provider_cfg = _cam_provider(config)
        if not provider_cfg or not provider_cfg.get("cam"):
            return ""

        from spine.services.cam_client import resolve_cam_settings

        settings = resolve_cam_settings(
            provider_cfg, workspace_root=getattr(config, "workspace_root", None)
        )
        if settings is None or settings.read not in ("facts_block", "both"):
            return ""
        facts = facts_store_for(config).stored(namespace=settings.namespace)
        facts.sort(key=lambda f: f.created_at or "", reverse=True)
        return format_known_facts_block(facts[:_INJECT_FACTS_LIMIT])
    except Exception:  # noqa: BLE001 — injection is best-effort
        logger.debug("known-facts injection failed (non-fatal)", exc_info=True)
        return ""
