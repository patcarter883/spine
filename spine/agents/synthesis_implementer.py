"""Synthesis + placement editor — the two-pure-node IMPLEMENT path.

The tool-using slice implementer reads too much before it writes: Laguna ran
79 tool calls and produced 0 edits; North spirals; even Qwen3-Coder finished a
full implement with an empty diff (bench 0625). The root cause is structural,
not a model defect (see the survey-trap analysis): the editor is handed
filesystem tools and a "go survey then edit" loop, so a weak model surveys
forever and never commits an edit.

This module removes the loop. Editing is split into two side-effect-shaped
nodes:

* **SYNTHESIS** (:func:`synthesize_slice_code`) — a structured-output call with
  *no filesystem tools*. It receives the edit intent, the API surface (the
  current source of every reference symbol), and the current body to rewrite,
  and returns structured :class:`SynthesizedEdit` data ``{file, symbol, action,
  code}``. It cannot read, so it cannot spiral.

* **PLACEMENT** (:func:`apply_synthesized`) — applies each edit deterministically
  through :class:`ReadEditLintTool` in ``ast_edit`` mode (apply → lint →
  revert-on-fail in memory, write only on a clean lint). Lint is the oracle.

Because synthesis is side-effect-free, sampling ``n`` candidates is cheap, and
placement gives each candidate an objective score (how many edits apply and
lint clean). :func:`place_best_candidate` is the IMPLEMENT-side mirror of
``spine.workflow.plan_score`` — Graph-of-Thoughts *Score + KeepBest*, with the
linter as the deterministic scorer instead of the structural plan heuristics.

The whole path is flag-gated (``implement_synthesis_placement``, default off);
the node wiring and ``_route_slices`` gate live in ``implement_subgraph.py``.
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.decomposer import _ainvoke_structured_escalating
from spine.agents.helpers import resolve_chat_model
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.agents.tools.read_edit_lint import ReadEditLintTool

logger = logging.getLogger(__name__)

# Statuses ReadEditLintTool returns for a *successful* placement. Everything
# else (syntax_error, no_match, ambiguous_match, reference_only, …) is a
# failure the synthesizer must fix on the retry pass.
_OK_STATUSES = frozenset({"ok", "already_applied"})


def _is_stub_body(code: str) -> bool:
    """True when a synthesized def/method's body is a bare stub, not real logic.

    Placement's linter oracle only checks syntax + style, so an edit whose
    body is just ``pass``/``...``/``raise NotImplementedError`` lints clean
    and applies — then verify finds it a phase later (019f1bed:
    ui-provider-controls was marked "implemented" with edit/remove handlers
    stubbed to ``pass``). Judges only single function/method defs; classes,
    constants, and other edit shapes are not stub-checked.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return False
    body = list(tree.body[0].body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop a leading docstring — not itself a stub marker
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            continue
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            name = exc.func.id if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) else (
                exc.id if isinstance(exc, ast.Name) else None
            )
            if name == "NotImplementedError":
                continue
        return False  # a real statement exists — not a stub
    return True


class SynthesizedEdit(BaseModel):
    """One symbol-anchored edit the synthesizer wants placed.

    Mirrors :class:`spine.agents.tools.read_edit_lint.AstEdit` with an explicit
    ``file`` field, so placement can call ``ReadEditLintTool._run(file_path=…,
    ast_edit={symbol, action, code})`` directly with no translation layer.
    """

    file: str = Field(
        description=(
            "Path of the file to edit, exactly as it appears in the slice's "
            "target_files (e.g. 'spine/agents/api.py')."
        )
    )
    symbol: str = Field(
        description=(
            "Qualified name of the target definition, e.g. "
            "'SpineConfig.resolve_model' or 'baseline_config_yaml'. For action "
            "'replace' this is the definition you are rewriting; for "
            "'insert_before'/'insert_after' it is the anchor you insert next to."
        )
    )
    action: str = Field(
        default="replace",
        description=(
            "'replace' the whole definition with `code`, or 'insert_before' / "
            "'insert_after' to add a new top-level construct adjacent to the "
            "anchor symbol."
        ),
    )
    code: str = Field(
        description=(
            "Complete new source. For 'replace': the entire new definition "
            "(signature + body). For insert: a complete construct (def/class). "
            "Never a fragment — placement replaces or inserts whole definitions."
        )
    )


class SynthesizedSlice(BaseModel):
    """A full candidate edit-set for one slice — the synthesizer's output."""

    edits: list[SynthesizedEdit] = Field(
        default_factory=list,
        description=(
            "Every edit needed to implement the slice, one per change site. "
            "Empty only if the slice genuinely requires no code change."
        ),
    )
    summary: str = Field(
        default="",
        description="One-line description of what these edits accomplish.",
    )


@dataclass
class PlacementResult:
    """Outcome of applying one candidate's edits through the linter."""

    applied: list[dict] = field(default_factory=list)   # [{file, symbol, ...}]
    failures: list[dict] = field(default_factory=list)  # [{file, symbol, status, detail}]
    ruff_issues: int = 0

    @property
    def n_applied(self) -> int:
        return len(self.applied)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    @property
    def clean(self) -> bool:
        """True when every edit placed and nothing failed."""
        return self.n_failures == 0 and self.n_applied > 0

    def score(self) -> tuple[int, int, int]:
        """Sortable KeepBest key — higher is better.

        ``(applied - failures, -failures, -ruff)``: prefer the candidate that
        lands the most edits, breaking ties toward fewer hard failures and then
        fewer ruff lint diagnostics. The linter is the entire scoring oracle.
        """
        return (self.n_applied - self.n_failures, -self.n_failures, -self.ruff_issues)


_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a code SYNTHESIZER. You have NO filesystem access — you cannot "
    "read, search, or survey. Everything you need is already in front of you: "
    "the slice to implement, the reference symbols it calls (with their current "
    "source), and the edit plan (each target's current source inlined).\n\n"
    "Return a SynthesizedSlice: a list of complete, symbol-anchored edits. For "
    "each edit name the file and the qualified target symbol, choose an action "
    "('replace' the whole definition, or 'insert_before'/'insert_after' a new "
    "construct next to an anchor), and write the COMPLETE new source — full "
    "signature and body, never a fragment or a diff. Rewrite from the inlined "
    "current source; do not invent symbols or files that are not shown. Emit one "
    "edit per change site. If the edit plan lists entries, produce exactly one "
    "edit per entry.\n\n"
    "The target_files block shows the CURRENT content of each file you may edit "
    "— the live file on disk, which MAY ALREADY CONTAIN edits from an earlier "
    "slice of this same feature. It is authoritative: everything already in it "
    "must survive your edit. Implement your slice by INSERTING new definitions "
    "(insert_before/insert_after an existing anchor) or by REPLACING only the "
    "specific definition your slice changes. Never regenerate a whole file, and "
    "never replace or delete a definition your slice does not need to touch — "
    "that clobbers the earlier slice's work. If a target file's current content "
    "is not shown, it does not exist yet and you are creating it.\n\n"
    "Placement is purely textual: an 'insert_before'/'insert_after' edit lands "
    "your new definition as a SIBLING at the anchor's own nesting level — same "
    "indentation, same scope. It gets NO implicit access to another function's "
    "locals, and no `self` unless the anchor is itself a method of the class "
    "you are joining. Never reference `self.<attr>` or a bare name from an "
    "enclosing function unless it is one of your own parameters or already "
    "module-global in the inlined source — write the full parameter list the "
    "edit plan's intent specifies instead of assuming closure capture. If a new "
    "helper needs to be called for the slice to work, that call must appear in "
    "a 'replace' edit on the calling function — inserting a helper does not "
    "wire it in by itself."
)


async def synthesize_slice_code(
    *,
    slice_json: str,
    refs_body: str,
    plan_body: str,
    config: RunnableConfig | None,
    session_id: str | None,
    files_body: str = "",
    n: int = 1,
    feedback: str = "",
    escalation_level: int = 0,
) -> list[SynthesizedSlice]:
    """Generate ``n`` candidate edit-sets for a slice via a no-tool structured call.

    Takes the prompt blocks already built by ``implement_subgraph`` helpers
    (``_target_files_body`` / ``_reference_symbols_body`` / ``_edit_plan_body``)
    so this module never imports the subgraph — no circular import. ``files_body``
    is the current on-disk content of the slice's target files; it is shown first
    so a serialized same-file slice edits additively rather than clobbering the
    prior slice's work. Candidates are generated
    sequentially (local providers cap ``max_concurrent_calls`` at 1–2, so
    parallel sampling buys no wall-clock); each is a self-contained
    :class:`SynthesizedSlice`. ``feedback`` (lint errors from a prior placement)
    is injected so the retry pass corrects exactly what failed.

    Returns the list of successfully-synthesized candidates (may be shorter than
    ``n`` if some calls error — callers place whatever came back).
    """
    from spine.config import SpineConfig

    cfg = SpineConfig.load()
    phase_path = "implement/synthesis"
    model = resolve_chat_model(
        config, session_id=session_id, phase=phase_path, escalation_level=escalation_level
    )
    base_cap = cfg.implement_max_completion_tokens
    try:
        window = int(
            (
                cfg.resolve_provider_config(
                    phase=phase_path, escalation_level=escalation_level
                )
                or {}
            ).get("context_window")
            or 0
        )
    except Exception:  # noqa: BLE001
        window = 0

    block_pairs = [(Tag.FINDINGS, f"```json\n{slice_json}\n```")]
    if files_body:
        # Current on-disk content of the slice's target files — shown FIRST so
        # the synthesizer edits additively on top of any earlier same-file
        # slice's work instead of regenerating the file from scratch.
        block_pairs.append((Tag.TARGET_FILES, files_body))
    block_pairs += [
        (Tag.REFERENCE_SYMBOLS, refs_body),
        (Tag.EDIT_PLAN, plan_body),
    ]
    if feedback:
        block_pairs.append((Tag.ERRORS, f"```\n{feedback}\n```"))
        tail = (
            "Your previous edits FAILED to place (errors above). Return a "
            "corrected SynthesizedSlice that fixes exactly those failures — keep "
            "the edits that were fine, repair the ones the linter rejected."
        )
    else:
        tail = (
            "Return a SynthesizedSlice implementing the slice above. Write "
            "complete definitions; do not survey, you have everything you need."
        )
    human_content = hostage_layout(xml_blocks(*block_pairs), tail)
    messages = [
        SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    candidates: list[SynthesizedSlice] = []
    for i in range(max(1, n)):
        try:
            response = await _ainvoke_structured_escalating(
                model,
                SynthesizedSlice,
                messages,
                label=f"synthesis-implementer[{i + 1}/{max(1, n)}]",
                base_cap=base_cap,
                window=window,
            )
        except Exception as exc:  # noqa: BLE001 — one bad sample must not sink the rest
            logger.warning(
                "synthesis candidate %d/%d failed: %s", i + 1, max(1, n), exc
            )
            continue
        candidate = _coerce_candidate(response)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _coerce_candidate(response: Any) -> SynthesizedSlice | None:
    """Normalize a structured response into a :class:`SynthesizedSlice`."""
    if isinstance(response, SynthesizedSlice):
        return response
    if isinstance(response, BaseModel):
        try:
            return SynthesizedSlice.model_validate(response.model_dump())
        except Exception:  # noqa: BLE001
            return None
    if isinstance(response, dict):
        try:
            return SynthesizedSlice.model_validate(response)
        except Exception:  # noqa: BLE001
            return None
    return None


def apply_synthesized(
    candidate: SynthesizedSlice,
    *,
    workspace_root: str,
    target_files: list[str],
    reference_only_files: list[str] | None = None,
) -> PlacementResult:
    """Place every edit in *candidate* deterministically through the linter.

    Each edit becomes one ``ReadEditLintTool._run(file_path, ast_edit=…)`` call:
    the tool resolves the symbol anchor, applies the edit in memory, runs the
    syntax/lint check, and only writes on a clean result — reverting otherwise.
    No model is in this loop; the linter is the sole arbiter.

    Edits whose ``file`` is not one of the slice's ``target_files`` are rejected
    locally (a synthesizer hallucinating an out-of-scope file must not get a
    write), mirroring the tool's own reference-only guard.
    """
    tool = ReadEditLintTool(
        workspace_root=workspace_root,
        target_files=list(target_files or []),
        reference_only_files=list(reference_only_files or []),
    )
    allowed = {str(f).strip() for f in (target_files or []) if str(f).strip()}
    result = PlacementResult()

    for edit in candidate.edits:
        rec = {"file": edit.file, "symbol": edit.symbol, "action": edit.action}
        if allowed and edit.file not in allowed:
            result.failures.append(
                {**rec, "status": "out_of_scope",
                 "detail": f"{edit.file} is not in the slice's target_files."}
            )
            continue
        try:
            raw = tool._run(
                file_path=edit.file,
                ast_edit={
                    "symbol": edit.symbol,
                    "action": edit.action or "replace",
                    "code": edit.code,
                },
            )
        except Exception as exc:  # noqa: BLE001 — a tool crash is a placement failure
            result.failures.append({**rec, "status": "tool_error", "detail": str(exc)})
            continue

        payload = _parse_tool_result(raw)
        status = payload.get("status", "")
        if status in _OK_STATUSES:
            result.applied.append({**rec, "status": status})
            result.ruff_issues += _count_ruff(payload.get("ruff"))
            if _is_stub_body(edit.code):
                # Placed and lint-clean, but empty of real behaviour — count it
                # against the score/retry-trigger without reverting the write
                # (a corrective retry re-resolves the same symbol and overwrites
                # it in place; see _synthesis_implementer_node).
                result.failures.append(
                    {**rec, "status": "stub_body",
                     "detail": (
                         "placed but the body is a bare stub (pass/…/"
                         "NotImplementedError) — implement the real behaviour "
                         "the slice's acceptance criteria describe for this "
                         "symbol."
                     )}
                )
        else:
            # Preserve `target` (a resolvable anchor/location) and `next_action`
            # (the concrete corrective call) alongside `detail` — the ast_edit
            # creation-anchor guard builds these specifically so a weak model
            # can self-correct in one retry (read_edit_lint._edit_feedback).
            # Collapsing them into detail-or-next_action dropped the actual
            # fix-it instruction, so the retry kept re-emitting the same
            # broken self-anchor for two full implement/verify cycles
            # (019f1c10: UIApi.add_embedding_provider anchored to itself
            # instead of the suggested insert_after target).
            failure = {**rec, "status": status or "unknown", "detail": payload.get("detail", "")}
            if payload.get("target"):
                failure["target"] = payload["target"]
            if payload.get("next_action"):
                failure["next_action"] = payload["next_action"]
            result.failures.append(failure)
    return result


def _parse_tool_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"status": "unknown"}
        except (json.JSONDecodeError, TypeError):
            return {"status": "unknown", "detail": raw[:200]}
    return {"status": "unknown"}


def _count_ruff(ruff: Any) -> int:
    """Best-effort count of lint diagnostics in a tool ``ruff`` field."""
    if isinstance(ruff, list):
        return len(ruff)
    if isinstance(ruff, str):
        return 0 if not ruff.strip() else ruff.count("\n") + 1
    if isinstance(ruff, dict):
        return int(ruff.get("count", 0) or 0)
    return 0


def _stage_files(workspace_root: str, files: list[str]) -> str:
    """Copy *files* into a fresh temp dir at their relative paths; return its root.

    Only files that exist in the live tree are copied — a candidate that creates
    a new file just creates it inside the staging dir. The caller owns cleanup.
    """
    src_root = Path(workspace_root)
    staging = tempfile.mkdtemp(prefix="spine-place-")
    dst_root = Path(staging)
    for f in files:
        src = src_root / f
        if not src.is_file():
            continue
        dst = dst_root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as exc:  # noqa: BLE001
            logger.warning("could not stage %s for scoring: %s", f, exc)
    return staging


def place_best_candidate(
    candidates: list[SynthesizedSlice],
    *,
    workspace_root: str,
    target_files: list[str],
    reference_only_files: list[str] | None = None,
) -> tuple[SynthesizedSlice | None, PlacementResult]:
    """Score + KeepBest over synthesized candidates, with the linter as scorer.

    Scoring is done in throwaway temp COPIES of the touched files — never on the
    live tree — so concurrent sibling slices editing the same file cannot revert
    one another's committed edits (the 019efd92 best-of-N snapshot/restore stomp
    that turned a same-file race into a non-terminating rework loop). Each
    candidate gets a pristine staging copy, is applied + linted there, and scored;
    only the winner is then applied ONCE to the live ``workspace_root``. ``n == 1``
    skips staging entirely and applies the single candidate directly (one write,
    no revert).

    Returns ``(winner, placement)`` — ``(None, empty)`` when there are no
    candidates to place.
    """
    real = [c for c in candidates if c is not None]
    if not real:
        return None, PlacementResult()

    if len(real) == 1:
        only = real[0]
        return only, apply_synthesized(
            only,
            workspace_root=workspace_root,
            target_files=target_files,
            reference_only_files=reference_only_files,
        )

    best: SynthesizedSlice | None = None
    best_key: tuple[int, int, int] | None = None
    for i, cand in enumerate(real):
        files = sorted({e.file for e in cand.edits if e.file})
        staging = _stage_files(workspace_root, files)
        try:
            res = apply_synthesized(
                cand,
                workspace_root=staging,
                target_files=target_files,
                reference_only_files=reference_only_files,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        key = res.score()
        logger.info(
            "synthesis best-of-%d: candidate %d/%d applied=%d failed=%d ruff=%d",
            len(real), i + 1, len(real), res.n_applied, res.n_failures, res.ruff_issues,
        )
        if best_key is None or key > best_key:
            best_key, best = key, cand

    assert best is not None  # len(real) >= 2 → the loop set a winner

    # Apply the winner ONCE to the live tree — the only mutation of the real
    # workspace, and the authoritative placement we report.
    placement = apply_synthesized(
        best,
        workspace_root=workspace_root,
        target_files=target_files,
        reference_only_files=reference_only_files,
    )
    return best, placement
