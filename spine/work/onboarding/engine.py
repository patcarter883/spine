"""Onboarding engine entrypoint — orchestrates analyse → scaffold → synthesise.

This is the dispatch coroutine the RalphLoopWorker queue routes to when a work
item carries ``work_type == "onboarding"``. It mirrors the dispatcher's own
``submit_work(...) -> dict[str, Any]`` shape: ``config`` defaults to
``SpineConfig.load()``, ``work_id`` defaults to an 8-char uuid4 prefix, and the
return dict carries ``work_id``/``status``/``work_type``.

Flow (design Revision 2, §2.1, §4.2, §5): the engine builds the composed
onboarding ``StateGraph`` (:func:`spine.work.onboarding.onboarding_graph.build_onboarding_graph`),
compiles it with a per-work :class:`AsyncSqliteSaver`, and ``ainvoke``s it.

- **Brownfield**: the graph's Phase A (deterministic analysis map-reduce)
  assembles + persists the :class:`~spine.work.onboarding.manifest.RepoManifest`
  as ``repo_manifest.json`` ONCE, then Phase B (the two-tier documentation
  manager → section-worker synthesis hierarchy) writes the four markdown
  documents — **no LLM ever receives the whole manifest**.
- **Greenfield**: :func:`spine.work.onboarding.scaffold.scaffold_project` lays
  out a minimal project deterministically BEFORE the graph runs; the graph then
  seeds a greenfield manifest and synthesises best-practice defaults.

Progress is tracked through the same mechanism the LangGraph dispatcher uses —
:func:`spine.work.dispatcher._update_work_progress` and
:func:`spine.work.dispatcher.update_work_phase_started`. The engine streams the
graph's node updates (``astream(stream_mode="updates")``) and maps the
``analysis_manager`` / ``doc_manager`` node boundaries to phases, so progress
reporting lives entirely in the engine (the graph nodes stay pure). The UI's
``get_queue_overview()`` polling and the ws_bus events work unchanged. The
recorded ``current_phase`` values (see :mod:`spine.work.onboarding.phases`) are
``"scaffold"`` (greenfield, pre-graph), ``"analyze"`` (at analysis_manager),
``"synthesize"`` (at doc_manager), then ``"completed"``.

Idempotent: re-running for the same ``work_id`` overwrites ``repo_manifest.json``
and the four ``.md`` artifacts cleanly (ArtifactStore overwrites, scaffold writes
are idempotent).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.persistence.artifacts import ArtifactStore
from spine.work.onboarding.manifest import RepoManifest
from spine.work.onboarding.onboarding_graph import build_onboarding_graph
from spine.work.onboarding.phases import (
    PHASE_ANALYZE,
    PHASE_COMPLETED,
    PHASE_SCAFFOLD,
    PHASE_SYNTHESIZE,
)
from spine.work.onboarding.scaffold import scaffold_project
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES, ONBOARDING_PHASE

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "repo_manifest.json"

# The four onboarding document filenames (display order) — derived from the
# canonical doc-name constant so the set is defined in exactly one place.
_ONBOARDING_DOCS: tuple[str, ...] = tuple(f"{name}.md" for name in ONBOARDING_DOC_NAMES)


def _ensure_work_entry(
    db: Any,
    work_id: str,
    description: str,
    config: SpineConfig,
) -> None:
    """Insert a work_entries row for the onboarding job if one is absent.

    The queue worker pre-generates ``work_id`` and routes here before any
    work_entries row exists, so the engine seeds the row itself (mirroring the
    insert ``submit_work`` performs for graph work). If a row already exists
    (e.g. a direct re-run), it is left in place and only updated by the
    progress calls.
    """
    try:
        existing = db["work_entries"].get(work_id)
    except Exception:
        existing = None

    now = datetime.now().isoformat()
    if existing is None:
        db["work_entries"].insert(
            {
                "id": work_id,
                "description": description,
                "work_type": "onboarding",
                "status": TaskStatus.RUNNING.value,
                "current_phase": "",
                "created_at": now,
                "updated_at": now,
                "result": "{}",
            },
            pk="id",
            alter=True,
        )
    else:
        db["work_entries"].update(
            work_id,
            {"status": TaskStatus.RUNNING.value, "updated_at": now},
        )


def _persist_manifest(
    manifest: RepoManifest,
    workspace_root: str,
    work_id: str,
) -> str:
    """Write ``repo_manifest.json`` under the onboarding artifact dir.

    Persisted via :class:`ArtifactStore` (phase ``"onboarding"``) so re-runs
    overwrite idempotently and the synthesis stage's ``read_repo_manifest`` tool
    can load it from the same ``.spine/artifacts/<work_id>/onboarding`` path.

    Returns the absolute path the manifest was written to.
    """
    out_dir = str(Path(workspace_root) / ".spine" / "artifacts")
    store = ArtifactStore(base_path=out_dir)
    content = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    path = store.save_artifact(
        work_id=work_id,
        phase=ONBOARDING_PHASE,
        name=_MANIFEST_NAME,
        content=content,
        overwrite_shorter=True,
    )
    return str(path)


async def _compile_onboarding_graph(
    workspace_root: str,
    work_id: str,
) -> Any:
    """Compile :func:`build_onboarding_graph` with a per-work checkpointer.

    Reuses :func:`spine.workflow.subgraph_wrapper._get_phase_checkpointer`
    (phase :data:`ONBOARDING_PHASE`) so onboarding shares ONE checkpointer
    implementation with the phase subgraphs: the same
    ``<workspace_root>/.spine/checkpoints/<work_id>/onboarding.db`` path scheme,
    the same context-manager cache that keeps the aiosqlite connection thread
    alive, AND its package-dir fallback when ``workspace_root`` is unset (so a
    ``/root`` or ``/tmp`` CWD doesn't cause permission errors). On any failure
    opening the checkpointer we fall back to compiling WITHOUT one — strictly
    >= today's non-resumable behaviour (design Risk #8).

    Returns the compiled graph.
    """
    graph = build_onboarding_graph()
    try:
        from spine.workflow.subgraph_wrapper import _get_phase_checkpointer

        checkpointer = await _get_phase_checkpointer(
            work_id,
            ONBOARDING_PHASE,
            workspace_root=workspace_root or None,
        )
        return graph.compile(checkpointer=checkpointer)
    except Exception as exc:  # noqa: BLE001 - fall back to no checkpointer
        logger.warning(
            "[%s] onboarding checkpointer failed (%s) — compiling without one",
            work_id,
            type(exc).__name__,
        )
        return graph.compile()


async def run_onboarding(
    workspace_root: str,
    mode: str,
    tech_stack: list[str] | None = None,
    config: SpineConfig | None = None,
    work_id: str | None = None,
) -> dict[str, Any]:
    """Orchestrate the onboarding engine end-to-end.

    Brownfield: slice1 analyse -> RepoManifest -> slice3 synthesise 4 .md artifacts.
    Greenfield: slice2 scaffold dirs/config -> seed RepoManifest(mode="greenfield") -> slice3 synthesise.
    Idempotent: re-running overwrites repo_manifest.json + the 4 .md artifacts cleanly.

    Args:
        workspace_root: Absolute path to the project to onboard.
        mode: ``"greenfield"`` or ``"brownfield"``.
        tech_stack: Optional stack tags (seed for greenfield, merged for
            brownfield).
        config: Optional :class:`SpineConfig`; defaults to ``SpineConfig.load()``.
        work_id: Optional pre-generated work item ID; defaults to an 8-char
            uuid4 prefix.

    Returns:
        ``{"work_id", "status", "work_type": "onboarding", "workspace_root",
        "mode", "artifacts": [...names], "manifest_path": ...}``. ``workspace_root``
        and ``mode`` are also persisted into ``work_entries.result`` so the UI can
        (a) select the correct phase bar (``mode``) and (b) resolve the artifact
        base path for an EXTERNAL onboarding target (``workspace_root``).
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    mode = mode if mode in ("greenfield", "brownfield") else "brownfield"
    work_id = work_id or str(uuid.uuid4())[:8]
    seed_stack = list(tech_stack or [])

    # Lazy import to avoid a circular import at module load
    # (dispatcher imports the engine inside submit_work).
    from spine.work.dispatcher import (
        get_work_db,
        update_work_phase_started,
        _update_work_progress,
    )

    db = get_work_db(config)
    description = json.dumps(
        {"workspace_root": workspace_root, "mode": mode, "tech_stack": seed_stack}
    )
    _ensure_work_entry(db, work_id, description, config)

    # Progress callback threaded into the graph via ``RunnableConfig``. Graph
    # nodes fire it at phase boundaries with the SAME ``current_phase`` strings
    # the UI expects: ``analyze`` (at analysis_manager), ``synthesize`` (at
    # doc_manager). ``scaffold`` is fired here pre-graph (greenfield only), and
    # ``completed`` is recorded after the graph finishes.
    _fired_phases: set[str] = set()

    def _progress(phase: str) -> None:
        # Idempotent per phase: the node wrappers (via the config callback) and
        # the engine's stream loop may both report the same boundary; we record
        # each ``current_phase`` exactly once.
        if phase in _fired_phases:
            return
        _fired_phases.add(phase)
        update_work_phase_started(db, work_id, phase)
        _update_work_progress(db, work_id, phase, "running")

    try:
        # ── Phase: scaffold (greenfield only, deterministic, BEFORE graph) ─
        if mode == "greenfield":
            _progress(PHASE_SCAFFOLD)
            scaffold_project(workspace_root, seed_stack)

        # ── Build + compile the composed onboarding graph (per-work CP) ────
        compiled = await _compile_onboarding_graph(workspace_root, work_id)

        initial_state: dict[str, Any] = {
            "work_id": work_id,
            "workspace_root": workspace_root,
            "mode": mode,
            "tech_stack": seed_stack,
        }
        runnable_config: dict[str, Any] = {
            "configurable": {
                "spine_config": config,
                "work_id": work_id,
                "thread_id": work_id,
            }
        }

        # ── Run analysis (Phase A) → synthesis (Phase B) ───────────────────
        # The analysis aggregator persists ``repo_manifest.json`` exactly once
        # (via ``_persist_manifest`` using ``spine_config`` from config); the
        # synthesis tier writes the four ``.md`` artifacts idempotently.
        #
        # Progress reporting lives ONLY here: we stream node updates so the
        # ``current_phase`` progression fires at node boundaries (``analyze`` at
        # analysis_manager, ``synthesize`` at doc_manager). The graph nodes stay
        # pure — there is no in-graph progress wrapper — so this is the single
        # progress mechanism. ``_progress`` is idempotent per phase.
        _node_phase = {
            "analysis_manager": PHASE_ANALYZE,
            "doc_manager": PHASE_SYNTHESIZE,
        }
        final_state: dict[str, Any] = {}
        async for chunk in compiled.astream(
            initial_state, config=runnable_config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, node_update in chunk.items():
                phase = _node_phase.get(node_name)
                if phase:
                    _progress(phase)  # idempotent per phase
                if isinstance(node_update, dict):
                    final_state.update(node_update)

        manifest_dict = dict(final_state.get("manifest", {}) or {})
        manifest = RepoManifest.from_dict(manifest_dict) if manifest_dict else None
        manifest_path = final_state.get("manifest_path", "")
        written = dict(final_state.get("written", {}) or {})

        artifact_names = [f"{name}.md" for name in written] or list(_ONBOARDING_DOCS)

        # ── Finalise ─────────────────────────────────────────────────────
        final_status = TaskStatus.COMPLETED.value
        db["work_entries"].update(
            work_id,
            {
                "status": final_status,
                "current_phase": PHASE_COMPLETED,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps(
                    {
                        "mode": mode,
                        "workspace_root": workspace_root,
                        "artifacts": artifact_names,
                        "manifest_path": manifest_path,
                        "symbol_count": manifest.symbol_count if manifest else 0,
                        "file_count": manifest.file_count if manifest else 0,
                    }
                ),
            },
        )
        _update_work_progress(db, work_id, PHASE_COMPLETED, final_status)

        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_completed", {"work_id": work_id, "status": final_status}
            )
        except Exception:
            pass

        return {
            "work_id": work_id,
            "status": final_status,
            "work_type": "onboarding",
            "workspace_root": workspace_root,
            "mode": mode,
            "artifacts": artifact_names,
            "manifest_path": manifest_path,
        }

    except Exception as exc:
        logger.error("Onboarding %s failed: %s", work_id, exc, exc_info=True)
        last_phase = ""
        try:
            row = db["work_entries"].get(work_id)
            last_phase = row.get("current_phase", "") if row else ""
        except Exception:
            pass
        try:
            db["work_entries"].update(
                work_id,
                {
                    "status": TaskStatus.FAILED.value,
                    "current_phase": last_phase,
                    "updated_at": datetime.now().isoformat(),
                    "result": json.dumps(
                        {
                            "error": str(exc),
                            "workspace_root": workspace_root,
                            "mode": mode,
                        }
                    ),
                },
            )
        except Exception:
            logger.warning("Could not record onboarding failure for %s", work_id)

        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_failed", {"work_id": work_id, "error": str(exc)}
            )
        except Exception:
            pass

        return {
            "work_id": work_id,
            "status": TaskStatus.FAILED.value,
            "work_type": "onboarding",
            "error": str(exc),
        }
