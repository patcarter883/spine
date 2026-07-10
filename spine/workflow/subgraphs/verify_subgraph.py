"""VERIFY phase as a LangGraph subgraph with Send API dispatch.

The subgraph uses the same manager/router/call/aggregate pattern as the
exploration and implement subgraphs: it dispatches one ``slice-verifier``
per slice in parallel via the LangGraph ``Send`` API, then aggregates
the verdicts.

Nodes:
- ``verify_router``: conditional edge — reads ``execution_waves`` from
  state, returns ``[Send("run_slice_verifier", ...)]`` or
  ``"synthesize_verification"``
- ``run_slice_verifier``: builds a ``slice-verifier`` subagent per slice
  and invokes it. Runs in parallel via Send API.
- ``aggregate_verification``: deterministic fan-in point after all
  parallel slice-verifier nodes complete.
- ``synthesize_verification``: writes ``verification.md`` and
  ``verification.json`` from accumulated verdicts, determines
  ``overall_status``.
- ``save_artifacts``: scans disk, materializes to state, determines
  phase status from ``verification.json``.

Edges::

    START → verify_router
    verify_router → Send("run_slice_verifier", {slice}) × N  OR  → synthesize_verification
    run_slice_verifier → aggregate_verification
    aggregate_verification → synthesize_verification → save_artifacts → END
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from spine.agents.artifacts import (
    artifact_path,
    materialize_phase_artifacts,
    scan_artifact_dir,
)
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.agents.plan_do import (
    directive_from_state,
    format_directive_for_prompt,
    run_plan_node,
)
from spine.agents.retry import ainvoke_with_retry, MaxTokenBudgetExceeded
from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import VerifySubgraphState

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


# ── Router: START → run_slice_verifier (Send) or synthesize ─────────────


async def _seed_prior_results_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Carry forward last cycle's VERIFIED verdicts for untouched slices.

    Per-slice convergence (run 019f20e0: with gap feedback flowing, one slice
    reached 1 remaining gap while the other was re-verified from scratch every
    cycle): a slice that was VERIFIED last cycle and whose ``target_files``
    were NOT rewritten by the rework keeps its verdict — its prior finding is
    seeded into ``verification_results`` (marked reused) and its id recorded
    in ``reverify_skipped_ids`` so the router does not dispatch a verifier for
    it. A verified slice whose files WERE touched since is re-verified — that
    is exactly the regression case (trace 019f2040) skipping must not mask.
    """
    prior = state.get("prior_verification_findings") or []
    if not prior:
        return {}
    verified_by_id = {
        f.get("slice_name"): f
        for f in prior
        if isinstance(f, dict) and f.get("verdict") == "VERIFIED" and f.get("slice_name")
    }
    if not verified_by_id:
        return {}
    rewritten = {str(p) for p in (state.get("files_written") or []) if p}

    seeded: list[dict] = []
    skipped_ids: list[str] = []
    for wave in state.get("execution_waves") or []:
        if not isinstance(wave, list):
            continue
        for sl in wave:
            if not isinstance(sl, dict):
                continue
            sid = sl.get("id")
            finding = verified_by_id.get(sid)
            if finding is None:
                continue
            targets = {str(t) for t in (sl.get("target_files") or []) if t}
            if targets & rewritten:
                continue  # touched since it passed — re-verify for regressions
            seeded.append({**finding, "reused_from_prior_cycle": True})
            skipped_ids.append(sid)
    if not seeded:
        return {}
    logger.info(
        "VERIFY seed: carrying forward %d VERIFIED verdict(s) for untouched "
        "slice(s): %s",
        len(seeded),
        skipped_ids,
    )
    return {
        "verification_results": seeded,
        "reverify_skipped_ids": skipped_ids,
    }


def _verify_router(
    state: VerifySubgraphState,
) -> list[Send] | Literal["synthesize_verification"]:
    """Fan-out to slice-verifier nodes via Send API.

    Reads ``execution_waves`` from state, flattens all waves into a single
    dispatch list (dependencies resolved by the scheduler during PLAN).
    Slices whose prior VERIFIED verdict was carried forward by
    ``seed_prior_results`` are not re-dispatched; when every slice is covered
    by a carried-forward verdict the router goes straight to synthesis.

    Raises ``CriticalContractFailure`` if ``execution_waves`` is missing
    or empty — this is a structural invariant violation.
    """
    execution_waves = state.get("execution_waves")

    if not execution_waves:
        raise CriticalContractFailure(
            phase="verify",
            reason="execution_waves is missing or empty in state — "
                   "the PLAN phase did not produce structured data transfer. "
                   "The prerequisite gate should have caught this before "
                   "VERIFY ran; check artifact_gate.py.",
        )

    all_slices: list[dict] = []
    for wave in execution_waves:
        if isinstance(wave, list):
            for sl in wave:
                if isinstance(sl, dict) and sl.get("id"):
                    all_slices.append(sl)

    if not all_slices:
        raise CriticalContractFailure(
            phase="verify",
            reason="execution_waves is present but contains zero valid "
                   "slice dicts with 'id' fields. The PLAN phase produced "
                   "malformed structured data.",
        )

    skipped = set(state.get("reverify_skipped_ids") or [])
    if skipped:
        all_slices = [s for s in all_slices if s.get("id") not in skipped]
        if not all_slices:
            # Every slice's verdict was carried forward — nothing to re-run.
            logger.info(
                "VERIFY router: all slices covered by carried-forward "
                "verdicts (%s) — skipping dispatch", sorted(skipped),
            )
            return "synthesize_verification"

    logger.info(
        "VERIFY router: dispatching %d slice-verifier(s): %s%s",
        len(all_slices),
        [s.get("id", "?") for s in all_slices],
        f" (carried forward: {sorted(skipped)})" if skipped else "",
    )

    base_state = {
        "phase": state.get("phase", "verify"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
    }
    return [
        # Two-node verifier branch: plan_slice_verifier (no tools) →
        # run_slice_verifier (tools). Each parallel branch carries its
        # own active_slice_directive through the chain.
        Send("plan_slice_verifier", {**base_state, "slice": s})
        for s in all_slices
    ]


# ── Node: plan_slice_verifier (no tools) ────────────────────────────────


async def _plan_slice_verifier_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> Command:
    """No-tool plan step for one slice's verification.

    Produces a per-branch SubagentDirective and dispatches a Send to
    run_slice_verifier carrying both the slice and the directive on the
    per-branch payload. Returning ``Command(goto=Send(...))`` — rather
    than writing the directive to a shared channel — is required
    because parallel Send branches share the subgraph's channel space,
    so N concurrent writes to ``active_slice_directive`` would crash
    apply_writes with ``InvalidUpdateError``.
    """
    work_id = state.get("work_id", "unknown")
    slice_data: dict = state.get("slice", {}) or {}
    slice_id = slice_data.get("id", "unknown")
    title = slice_data.get("title", "")
    target_files = slice_data.get("target_files") or []
    criteria = slice_data.get("acceptance_criteria") or []

    crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "(none provided)"
    file_lines = "\n".join(f"- {p}" for p in target_files) if target_files else "(none provided)"
    task = (
        f"Plan a verification pass for slice {slice_id!r} (title: {title!r}). "
        "The do node will read the files, run any lint/tests it needs, and emit a "
        "VerificationResult (verdict, checklist, gaps, recommendations).\n\n"
        f"## Acceptance criteria\n{crit_lines}\n\n"
        f"## Target files\n{file_lines}"
    )
    directive = await run_plan_node(
        state=dict(state),
        config=config,
        phase_path=f"{PhaseName.VERIFY.value}/subagents/slice-verifier",
        task_description=task,
        role_hint=f"slice-verifier for slice {slice_id!r}",
    )
    logger.info(
        "[%s] plan_slice_verifier: slice=%r approach=%r",
        work_id, slice_id, directive.approach[:80],
    )
    send_payload: dict[str, Any] = {
        "phase": state.get("phase", "verify"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
        "slice": slice_data,
        "active_slice_directive": directive.model_dump(),
    }
    return Command(goto=Send("run_slice_verifier", send_payload))


# ── Node: run_slice_verifier ────────────────────────────────────────────


_DIFF_MAX_CHARS = 24000


def _worktree_diff(workspace_root: str, paths: list[str] | None) -> str:
    """Return the sandbox worktree's git diff, scoped to ``paths``.

    The implement sandbox leaves its edits UNCOMMITTED in this worktree, so a
    diff against HEAD plus any new untracked files is the exact, ground-truth
    record of what changed. Handing this to the verifier replaces "read the
    codebase to discover what was edited" — which cost ~1M input tokens on an
    empty diff (it surveyed the tree to confirm nothing changed). Returns ''
    on any git failure so the verifier falls back to reading files.
    """
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", workspace_root, *args],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception:  # noqa: BLE001 — diff is best-effort
            return ""

    clean_paths = [p for p in (paths or []) if p]
    pathspec = ["--", *clean_paths] if clean_paths else []
    diff = _git("--no-pager", "diff", "HEAD", *pathspec)
    # git diff HEAD omits untracked (new) files — append them as add-only diffs.
    others = _git("ls-files", "--others", "--exclude-standard", *pathspec).split()
    for f in others:
        try:
            body = (Path(workspace_root) / f).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        diff += f"\n--- /dev/null\n+++ b/{f}\n" + "".join(
            f"+{ln}\n" for ln in body.splitlines()
        )
    if len(diff) > _DIFF_MAX_CHARS:
        diff = diff[:_DIFF_MAX_CHARS] + "\n...[diff truncated — read the file for the rest]"
    return diff


_CHECKS_MAX_CHARS = 8000


# Per-section budgets. A single global cap let one verbose check starve
# the rest: run 0eabad7d's ruff output (full code-frame format) consumed
# the whole 8000-char budget and the pytest section — carrying the
# decisive ModuleNotFoundError — was truncated away, so the judge
# VERIFIED a test file the landing gate then rejected. Each section is
# bounded on its own; test runners keep their TAIL because they print the
# failure summary last.
def _clip_head(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + "\n…[output truncated]"


def _clip_tail(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return "…[output truncated]…\n" + text[-cap:]


def _custom_check_specs() -> list[dict]:
    """``verify_checks`` entries from ./spine-gate.yaml, ``[]`` on any failure.

    The built-in checks below know Python; other stacks declare their
    evidence commands per target repo (the file is read from the dispatcher
    process CWD, the same convention the landing-gate config uses). Each
    entry: ``name``, ``files`` (fnmatch patterns over changed paths),
    ``command`` (shell line; ``{files}`` expands to the quoted matches),
    optional ``hard`` (default true — failure forces NOT_VERIFIED),
    ``timeout_seconds`` (default 300). Loaded directly rather than via
    spine.git.orchestrator.load_gate_config to avoid an import cycle
    (orchestrator imports the dispatcher at module level).
    """
    try:
        import yaml

        path = Path("spine-gate.yaml")
        if not path.is_file():
            return []
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = cfg.get("verify_checks") or []
        return [
            s for s in specs
            if isinstance(s, dict) and s.get("command") and s.get("files")
        ]
    except Exception:  # noqa: BLE001 — evidence config is best-effort
        return []


def _is_test_path(f: str) -> bool:
    """Test-shaped path: under a tests root or a test_-prefixed filename."""
    return Path(f).name.startswith("test_") or f.split("/", 1)[0] == "tests"


def _automated_checks(
    workspace_root: str,
    target_files: list[str] | None,
    feature_test_files: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Run deterministic checks over changed files as verifier evidence.

    This is the evidence-then-judge counterpart of ``_worktree_diff``: instead
    of letting the verifier spend an unbounded ReAct loop shelling ``py_compile``
    / ``ruff`` / ``inspect.signature`` probes (trace 019f16cf: the loop never
    converged and crashed on the token budget), we run the checks that
    actually matter ONCE, here, and hand the verifier the results. Python
    checks (py_compile + ruff + pytest) are built in; other stacks declare
    theirs via ``verify_checks`` in spine-gate.yaml (PHP/pest through the
    Sail stack for the agripath clone; TypeScript next). Best-effort:
    a missing runner or a subprocess error is reported as such, never raised —
    the verifier still has the diff and source to judge from.

    Returns ``(rendered_block, hard_failures)``. ``hard_failures`` lists
    HARD check failures (py_compile / pytest / hard custom checks — ruff
    stays advisory for the judge to weigh) so the caller can OVERRIDE a
    VERIFIED verdict: run e95c1bc4's judge was handed a pytest collection
    ImportError in its prompt and still verified the slice with zero gaps —
    an LLM verdict that contradicts deterministic ground truth is corrected
    mechanically, not argued with.
    """
    existing_files = [
        f for f in (target_files or [])
        if f and (Path(workspace_root) / f).exists()
    ]
    py_files = [f for f in existing_files if f.endswith(".py")]

    def _run(args: list[str]) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                args, cwd=workspace_root,
                capture_output=True, text=True, timeout=60,
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except FileNotFoundError:
            return 127, f"{args[0]}: not found in this sandbox (check skipped)"
        except Exception as exc:  # noqa: BLE001 — checks are best-effort evidence
            return 1, f"{args[0]} could not run: {exc}"

    def _run_shell(command: str, timeout: int) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=workspace_root,
                capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except Exception as exc:  # noqa: BLE001 — checks are best-effort evidence
            return 1, f"check could not run: {exc}"

    sections: list[str] = []
    hard_failures: list[str] = []
    if py_files:
        compile_rc, compile_out = _run(
            [sys.executable, "-m", "py_compile", *py_files]
        )
        sections.append(
            f"$ python -m py_compile {' '.join(py_files)}\n"
            + ("OK — all target files compile." if compile_rc == 0
               else _clip_head(compile_out, 1500) or f"FAILED (exit {compile_rc})")
        )
        if compile_rc != 0:
            hard_failures.append(
                f"py_compile failed (exit {compile_rc}): "
                + _clip_head(compile_out, 400)
            )
        ruff_rc, ruff_out = _run(
            [sys.executable, "-m", "ruff", "check",
             "--output-format=concise", *py_files]
        )
        sections.append(
            f"$ ruff check {' '.join(py_files)}\n"
            + ("OK — no lint findings." if ruff_rc == 0
               else _clip_head(ruff_out, 2500) or f"exit {ruff_rc}")
        )
        # Slices that author tests get their tests EXECUTED, not just read.
        # Evidence-only judging cannot ground criteria like "pytest exits 0"
        # — run 019f40e0/ce6f887d parked on exactly that gap ("no pytest run
        # was performed or its results provided") — and a real run also
        # surfaces collection errors (duplicate defs, broken fixtures) that
        # reading the source misses (545264cc landed 9 collection-erroring
        # tests as VERIFIED). The sandbox worktree's code wins imports over
        # the editable install (pytest prepends the rootdir; the .pth path
        # comes later), so the results reflect the patch under verification.
        test_files = [
            f for f in py_files
            if Path(f).name.startswith("test_") or f.split("/", 1)[0] == "tests"
        ]
        if test_files:
            pytest_rc, pytest_out = _run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", *test_files]
            )
            sections.append(
                f"$ pytest -q {' '.join(test_files)}\n"
                + ("OK — all tests pass." if pytest_rc == 0
                   else _clip_tail(pytest_out, 3500) or f"exit {pytest_rc}")
            )
            if pytest_rc != 0:
                hard_failures.append(
                    f"pytest failed (exit {pytest_rc}): "
                    + _clip_tail(pytest_out, 600)
                )

    # PHP: PSR-4 ground truth as evidence. The judge hallucinated a wrong
    # "correct" namespace against a PSR-4-compliant file (probe 9: flagged
    # Database\Factories — exactly what composer.json dictates for
    # database/factories/ — demanding App\Database\Factories). Print the
    # expectation per changed file so the judge anchors on computed truth,
    # and make a real mismatch (one the write-exit repair could not fix) a
    # hard failure.
    php_files = [f for f in existing_files if f.endswith(".php")]
    if php_files:
        try:
            from spine.agents.tools.read_edit_lint import (
                _psr4_map,
                _psr4_namespace_for,
            )
            import re as _re

            psr4 = _psr4_map(workspace_root)
        except Exception:  # noqa: BLE001 — evidence is best-effort
            psr4 = {}
        if psr4:
            lines: list[str] = []
            for f in php_files:
                expected = _psr4_namespace_for(f, psr4)
                if not expected:
                    continue
                try:
                    src = (Path(workspace_root) / f).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                m = _re.search(
                    r"^namespace\s+([A-Za-z0-9_\\]+)\s*;", src, _re.MULTILINE
                )
                declared = m.group(1) if m else "(none)"
                if m is None and not _re.search(
                    r"^\s*(?:final\s+|abstract\s+|readonly\s+)*"
                    r"(?:class|interface|trait|enum)\s+\w+",
                    src,
                    _re.MULTILINE,
                ):
                    # PSR-4 governs autoloadable TYPES; a file that declares
                    # none (procedural Pest tests — this repo's own tests/Unit
                    # files all omit the namespace) has nothing to autoload.
                    # Probe 20 hard-failed a correct test file on '(none)'.
                    lines.append(
                        f"OK {f}: no class declared — PSR-4 not applicable "
                        "(procedural file)"
                    )
                    continue
                ok = declared == expected
                lines.append(
                    f"{'OK ' if ok else 'MISMATCH '}{f}: declared "
                    f"`{declared}` — PSR-4 (composer.json) dictates `{expected}`"
                )
                if not ok:
                    hard_failures.append(
                        f"PSR-4 namespace mismatch in {f}: declared "
                        f"{declared!r}, composer.json dictates {expected!r}"
                    )
            if lines:
                sections.append(
                    "$ [psr4] namespace ground truth (computed from "
                    "composer.json autoload maps — these ARE the correct "
                    "namespaces; do not dispute them):\n" + "\n".join(lines)
                )

    # Project-declared checks (spine-gate.yaml verify_checks) — how non-Python
    # stacks get executed evidence in front of the judge. Same lesson as the
    # Python path: without real check output, PHP/TS slices would be judged
    # evidence-starved and the pre-2a2d9a2 failure modes return.
    import fnmatch
    import shlex

    for spec in _custom_check_specs():
        patterns = spec.get("files") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        matched = [
            f for f in existing_files
            if any(fnmatch.fnmatch(f, p) for p in patterns)
        ]
        if not matched:
            continue
        name = str(spec.get("name") or "custom_check")
        command = str(spec["command"]).replace(
            "{files}", " ".join(shlex.quote(f) for f in matched)
        )
        timeout = int(spec.get("timeout_seconds", 300))
        rc, out = _run_shell(command, timeout)
        sections.append(
            f"$ [{name}] {command}\n"
            + (f"OK — {name} passed." if rc == 0
               else _clip_tail(out, 3500) or f"exit {rc}")
        )
        if rc != 0 and spec.get("hard", True):
            hard_failures.append(
                f"{name} failed (exit {rc}): " + _clip_tail(out, 600)
            )

    # Feature-wide test evidence (ADVISORY). A slice's defect often only
    # surfaces when the FEATURE's tests run: probe 12 (ed2c9f85) verified
    # the migration slice clean while its TypeError exploded in the test
    # slice's pest run — the gap landed on the wrong slice and the fix loop
    # would have reworked correct test code. Run the sibling slices' test
    # files as evidence for THIS slice too, advisory rather than hard: the
    # judge attributes failures ('migration TypeError at line 15' → the
    # migration slice's criterion fails), while a merely-broken sibling
    # test file cannot hard-fail an implementation slice.
    own = set(target_files or [])
    sibling_tests = [
        f for f in (feature_test_files or [])
        if f not in own and (Path(workspace_root) / f).exists()
    ]
    if sibling_tests:
        py_sibs = [f for f in sibling_tests if f.endswith(".py")]
        if py_sibs:
            rc, out = _run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", *py_sibs]
            )
            sections.append(
                f"$ [feature tests — ADVISORY: attribute any failure to the "
                f"slice whose FILES cause it] pytest -q {' '.join(py_sibs)}\n"
                + ("OK — the feature's tests pass." if rc == 0
                   else _clip_tail(out, 3000) or f"exit {rc}")
            )
        for spec in _custom_check_specs():
            patterns = spec.get("files") or []
            if isinstance(patterns, str):
                patterns = [patterns]
            matched = [
                f for f in sibling_tests
                if any(fnmatch.fnmatch(f, p) for p in patterns)
            ]
            if not matched:
                continue
            name = str(spec.get("name") or "custom_check")
            command = str(spec["command"]).replace(
                "{files}", " ".join(shlex.quote(f) for f in matched)
            )
            rc, out = _run_shell(command, int(spec.get("timeout_seconds", 300)))
            sections.append(
                f"$ [feature tests via {name} — ADVISORY: attribute any "
                f"failure to the slice whose FILES cause it] {command}\n"
                + ("OK — the feature's tests pass." if rc == 0
                   else _clip_tail(out, 3000) or f"exit {rc}")
            )

    if not sections:
        return "", []
    body = "\n\n".join(sections)
    if len(body) > _CHECKS_MAX_CHARS:
        body = body[:_CHECKS_MAX_CHARS] + "\n…[checks output truncated]"
    block = (
        "<automated_checks>\nDeterministic checks already run for you over the "
        "changed files (you have no tools — judge from these, do not ask to "
        "re-run them):\n\n" + body + "\n</automated_checks>"
    )
    return block, hard_failures


def _reconcile_verdict(
    verification_result: dict,
    checks_failures: list[str],
    work_id: str,
    slice_id: str,
) -> None:
    """Derive the slice verdict from ground truth instead of trusting it.

    The judge's verdict field has now been wrong in BOTH directions:

    * run e95c1bc4 — VERIFIED over a pytest collection ImportError sitting
      in its own prompt (broken file reached the landing gate);
    * run 28b62d1e — NOT_VERIFIED with every checklist item passed, zero
      gaps and green checks: an inconsistent verdict the gap-fix loop can
      do nothing with, so the run parked contentlessly.

    Rules, in order: any HARD check failure ⇒ NOT_VERIFIED (failures
    appended as checklist entries + gaps); else a non-empty all-passed
    checklist with no gaps ⇒ VERIFIED. Mutates in place.
    """
    verdict = verification_result.get("verdict")
    if checks_failures:
        if verdict == "VERIFIED":
            logger.warning(
                "[%s] Slice-verifier %r: judge said VERIFIED but %d hard "
                "check(s) failed — overriding to NOT_VERIFIED",
                work_id, slice_id, len(checks_failures),
            )
        verification_result["verdict"] = "NOT_VERIFIED"
        checklist = verification_result.setdefault("checklist", [])
        gaps = verification_result.setdefault("gaps", [])
        for failure in checks_failures:
            entry = f"Automated check failed: {failure}"
            if entry not in gaps:
                checklist.append({
                    "criterion": "Automated checks pass (deterministic)",
                    "passed": False,
                    "detail": failure,
                })
                gaps.append(entry)
        return

    checklist = verification_result.get("checklist") or []
    all_passed = bool(checklist) and all(
        c.get("passed") for c in checklist if isinstance(c, dict)
    )
    if all_passed and not verification_result.get("gaps") and verdict != "VERIFIED":
        logger.warning(
            "[%s] Slice-verifier %r: judge said %r but every checklist item "
            "passed, no gaps, checks clean — overriding to VERIFIED",
            work_id, slice_id, verdict,
        )
        verification_result["verdict"] = "VERIFIED"


# Pre-loaded source bounds. The verifier reads its target files to check them
# against the criteria; handing the current source up-front means it starts
# grounded at ZERO reads instead of paging each file in (trace 019f10bf read
# api.py 8×). Bounded so a god-class file cannot dominate the prompt — the rest
# is one bounded read_file away.
_PRELOAD_MAX_LINES_PER_FILE = 400
_PRELOAD_MAX_CHARS = 24000


def _target_source_block(workspace_root: str, target_files: list[str] | None) -> str:
    """Render the current source of the slice's target files, line-numbered.

    Bounded per-file and overall; a file longer than the per-file cap is shown
    head-first with a note steering the verifier to ``read_file`` for the rest.
    Returns ``""`` when there is nothing to show.
    """
    files = [f for f in (target_files or []) if f]
    if not files:
        return ""
    root = Path(workspace_root)
    rendered: list[str] = []
    budget = _PRELOAD_MAX_CHARS
    for rel in files:
        path = root / rel.lstrip("/")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        lines = text.splitlines()
        total = len(lines)
        shown = lines[:_PRELOAD_MAX_LINES_PER_FILE]
        body = "\n".join(f"{i}| {ln}" for i, ln in enumerate(shown, start=1))
        note = (
            ""
            if len(shown) >= total
            else f"\n… ({total - len(shown)} more lines — read_file '{rel}' "
            f"offset={len(shown)} for the rest)"
        )
        chunk = f"### {rel} ({total} lines)\n{body}{note}"
        if len(chunk) > budget:
            chunk = chunk[:budget] + "\n…[pre-load truncated — read_file for the rest]"
            rendered.append(chunk)
            break
        rendered.append(chunk)
        budget -= len(chunk)
    if not rendered:
        return ""
    return (
        "<target_source>\nCurrent source of this slice's target files "
        "(already loaded — do NOT re-read these with read_file unless you need "
        "a region beyond what is shown):\n\n" + "\n\n".join(rendered) + "\n</target_source>"
    )


async def _run_slice_verifier_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a slice-verifier subagent for one feature slice.

    The slice is injected into state by the Send API via
    ``Send("run_slice_verifier", {"slice": {...}})``.  The subagent
    receives the slice definition, acceptance criteria, and access
    to the filesystem to inspect the implemented code.
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec

    work_id = state.get("work_id", "unknown")
    slice_data: dict = state.get("slice", {})
    slice_id = slice_data.get("id", "unknown")

    logger.info(
        "[%s] Slice-verifier node: slice=%r (title=%r)",
        work_id,
        slice_id,
        slice_data.get("title", ""),
    )

    try:
        subagent_spec = build_subagent_spec(
            name="slice-verifier",
            phase=PhaseName.VERIFY,
            state=state,
            config=config,
        )

        extra_tools = list(subagent_spec.get("tools", []))
        agent = build_phase_agent(
            state=state,
            config=config,
            phase=PhaseName.VERIFY,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=extra_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
            # The subagent_spec already curated the verifier's tool surface
            # (read + execute tools only — slice-verifier is not in the MCP
            # injection set) — they live in ``extra_tools`` above.
        )

        slice_json = json.dumps(slice_data, indent=2, ensure_ascii=False)
        directive_block = format_directive_for_prompt(
            directive_from_state(dict(state), "active_slice_directive")
        )
        # The worktree diff is the ground truth of what the implementer changed
        # for this slice. Hand it over so the verifier checks the CHANGES
        # against the criteria instead of reading the codebase to find them.
        diff_text = _worktree_diff(
            state.get("workspace_root", "."), slice_data.get("target_files")
        )
        diff_block = (
            "<worktree_diff>\nThis git diff is EVERY change made for this slice "
            "(empty ⇒ nothing was implemented — fail the slice):\n```diff\n"
            f"{diff_text or '(no changes in the working tree)'}\n```\n</worktree_diff>"
        )
        # Pre-load the target files' current source so the verifier starts
        # grounded at zero reads instead of paging each file in (trace 019f10bf).
        source_block = _target_source_block(
            state.get("workspace_root", "."), slice_data.get("target_files")
        )
        # Evidence-then-judge: when the no-tool judge is on, run the checks the
        # ReAct verifier used to spend an unbounded loop on (py_compile + ruff)
        # ONCE, here, and inline the results. The judge has no tools, so this is
        # the only place those checks can run.
        from spine.config import SpineConfig

        judge_mode = SpineConfig.load().verify_evidence_then_judge
        # The feature's test files (from every slice's target_files) run as
        # ADVISORY evidence for this slice — a migration/model defect often
        # only surfaces in the sibling test slice's run (probe 12).
        feature_tests = [
            f
            for wave in (state.get("execution_waves") or [])
            if isinstance(wave, list)
            for sl in wave
            if isinstance(sl, dict)
            for f in (sl.get("target_files") or [])
            if f and _is_test_path(f)
        ]
        checks_block, checks_failures = (
            _automated_checks(
                state.get("workspace_root", "."),
                slice_data.get("target_files"),
                feature_test_files=feature_tests,
            )
            if judge_mode
            else ("", [])
        )
        # Hostage layout: data blocks first, plain-text directive at the
        # absolute tail. The directive_block from format_directive_for_prompt
        # is already wrapped in <directive> — splice it after xml_blocks
        # rather than re-wrapping.
        tail = (
            (
                "Judge each acceptance criterion in the slice JSON using ONLY "
                "the evidence above — <worktree_diff> (ground truth of the "
                "change; empty ⇒ NOT_VERIFIED), <target_source> (current "
                "source), and <automated_checks> (py_compile + ruff results). "
                "You have NO tools; do not ask to read or run anything. Emit the "
                "structured VerificationResult now."
            )
            if judge_mode
            else (
                "Check the worktree_diff above against the acceptance_criteria "
                "in the slice JSON. The diff is the ground truth of what "
                "changed: if it is empty, the slice was NOT implemented — fail "
                "it. The current source of the target files is in "
                "<target_source> — read it there, NOT with read_file. Read a "
                "file only when you need a region beyond what is shown; do NOT "
                "survey the codebase."
            )
        )
        prompt = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"Verify slice: {slice_id}"),
                (Tag.FINDINGS, f"```json\n{slice_json}\n```"),
            )
            + "\n\n" + diff_block
            + ("\n\n" + source_block if source_block else "")
            + ("\n\n" + checks_block if checks_block else "")
            + ("\n\n" + directive_block if directive_block else ""),
            tail,
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="verify-slice",
            work_id=work_id,
        )

        verification_result = _extract_verification_result(result, slice_id)
        if isinstance(verification_result, dict):
            _reconcile_verdict(
                verification_result, checks_failures, work_id, slice_id
            )

    except MaxTokenBudgetExceeded as budget_exc:
        # The judge's one-shot call already produced a verdict before the
        # cumulative budget check raised — salvage it instead of discarding
        # paid-for work as a "crash". Falls through to the generic handler only
        # if no parseable verdict was attached.
        salvaged = (
            _extract_verification_result(budget_exc.result, slice_id)
            if getattr(budget_exc, "result", None) is not None
            else None
        )
        if salvaged is not None:
            logger.warning(
                "[%s] Slice-verifier %r: budget tripped post-call (%s) — "
                "salvaged the computed verdict.",
                work_id, slice_id, budget_exc,
            )
            verification_result = salvaged
        else:
            logger.error(
                "[%s] Slice-verifier %r: budget exceeded with no salvageable "
                "verdict: %s", work_id, slice_id, budget_exc,
            )
            verification_result = {
                "slice_name": slice_id,
                "verdict": "NOT_VERIFIED",
                "checklist": [{
                    "criterion": "Subagent execution",
                    "passed": False,
                    "detail": f"Token budget exceeded before a verdict: {budget_exc}",
                }],
                "gaps": [f"Verification could not complete: {budget_exc}"],
                "recommendations": ["Re-run verification for this slice"],
            }

    except Exception as e:
        logger.error(
            "[%s] Slice-verifier failed for %r: %s",
            work_id,
            slice_id,
            e,
            exc_info=True,
        )
        verification_result = {
            "slice_name": slice_id,
            "verdict": "NOT_VERIFIED",
            "checklist": [
                {
                    "criterion": "Subagent execution",
                    "passed": False,
                    "detail": f"Verifier subagent crashed: {e}",
                }
            ],
            "gaps": [f"Verification could not complete: {e}"],
            "recommendations": ["Re-run verification for this slice"],
        }

    return {"verification_results": [verification_result]}


def _strip_json_fence(text: str) -> str:
    """Strip a leading/trailing Markdown code fence around a JSON payload.

    The slice-verifier prompt shows a ```json fenced example, so models often
    wrap their verdict in a fence. A bare ``json.loads`` fails on the fence and
    the verdict is discarded as unparseable → spurious NOT_VERIFIED (finding
    #12). Tolerate ```json … ``` and bare ``` … ``` wrappers.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence line (``` or ```json) and the trailing fence.
    first_newline = s.find("\n")
    if first_newline == -1:
        return s
    s = s[first_newline + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[: -3]
    return s.strip()


def _extract_verification_result(result: dict, slice_id: str) -> dict:
    """Extract a VerificationResult dict from an agent result.

    If the agent returned structured output via ``response_format``,
    it'll be in the ``structured_response`` key.  Falls back to the
    last assistant message content.

    The ``slice_name`` field is overridden with the actual slice_id
    from the router to guarantee consistency.
    """
    structured = result.get("structured_response")
    if structured:
        if isinstance(structured, dict):
            structured["slice_name"] = slice_id
            return structured
        if hasattr(structured, "model_dump"):
            d = structured.model_dump()
            d["slice_name"] = slice_id
            return d

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(_strip_json_fence(content))
                if isinstance(parsed, dict):
                    parsed["slice_name"] = slice_id
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {
                "slice_name": slice_id,
                "verdict": "NOT_VERIFIED",
                "checklist": [
                    {
                        "criterion": "Agent output",
                        "passed": False,
                        "detail": "Subagent produced unstructured output — verify manually",
                    }
                ],
                "gaps": ["Unstructured output from subagent"],
                "recommendations": [],
            }

    return {
        "slice_name": slice_id,
        "verdict": "NOT_VERIFIED",
        "checklist": [
            {
                "criterion": "Agent output",
                "passed": False,
                "detail": "(no output from subagent)",
            }
        ],
        "gaps": ["Subagent produced no output"],
        "recommendations": [],
    }


# ── Node: aggregate_verification ───────────────────────────────────────


async def _aggregate_verification_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Fan-in point after all parallel slice-verifier nodes complete.

    Results are already accumulated via ``operator.add`` on the
    ``verification_results`` field — no manual merging needed.
    """
    results = state.get("verification_results", [])
    verdicts = [r.get("verdict", "?") for r in results]
    logger.info(
        "VERIFY aggregate: %d verification result(s) — verdicts: %s",
        len(results),
        verdicts,
    )
    return {}


# ── Node: synthesize_verification ───────────────────────────────────────


async def _synthesize_verification_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Write verification.md and verification.json from accumulated verdicts.

    Uses the existing ``WriteVerificationReportTool`` logic (via a
    utility function) to produce both the human-readable markdown and
    the structured JSON artifact.
    """
    from spine.agents.verify_tools import write_verification_files

    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    verification_results = state.get("verification_results", [])

    if not verification_results:
        logger.warning("[%s] VERIFY synthesize: zero slice verification results", work_id)
        return {
            "agent_response": "",
            "artifacts_output": {},
            "phase_status": "needs_review",
            "verification_attempted": False,
            "verification_passed": False,
        }

    verify_dir = artifact_path(work_id, PhaseName.VERIFY.value)
    summary = _build_verification_summary(verification_results)

    try:
        write_verification_files(verification_results, summary, workspace_root, verify_dir)
    except Exception as e:
        logger.error(
            "[%s] VERIFY synthesize: failed to write artifacts: %s",
            work_id,
            e,
        )
        return {
            "agent_response": summary,
            "artifacts_output": {},
            "phase_status": "error",
            "verification_attempted": True,
            "verification_passed": False,
        }

    all_verified = all(
        r.get("verdict") == "VERIFIED" for r in verification_results
    )

    logger.info(
        "[%s] VERIFY synthesize: wrote %d slice verdicts to %s/ (all_verified=%s)",
        work_id,
        len(verification_results),
        verify_dir,
        all_verified,
    )

    return {
        "agent_response": summary,
        "artifacts_output": {"verification.md": summary[:_MAX_ARTIFACT_STATE_CHARS]},
        "phase_status": "success" if all_verified else "needs_review",
        "verification_attempted": True,
        "verification_passed": all_verified,
    }


def _build_verification_summary(verification_results: list[dict]) -> str:
    """Build a human-readable verification summary from slice verdicts."""
    total = len(verification_results)
    verdicts: dict[str, int] = {}
    for r in verification_results:
        v = r.get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1

    verified = verdicts.get("VERIFIED", 0)
    not_verified = total - verified

    parts = [
        f"Verification complete for {total} feature slice(s).",
        f"- VERIFIED: {verified}",
    ]
    if not_verified:
        parts.append(f"- NOT_VERIFIED: {not_verified}")

    if not_verified == 0:
        parts.append("All slices passed verification.")
    else:
        parts.append(f"{not_verified} slice(s) did not pass — see verification.md for details.")

    return "\n".join(parts)


# ── Node: save_artifacts ────────────────────────────────────────────────


def _load_verification_results(workspace_root: str, work_id: str) -> list[dict]:
    """Load per-slice verdicts from ``verification.json``, if present.

    Returns the list of slice result dicts (``slice_name``, ``verdict``,
    ``gaps``, ``recommendations``) so downstream consumers — the
    needs_review feedback reason and the gap-plan phase — can name concrete
    gaps instead of pointing at an artifact that may have been cleared.
    Returns ``[]`` when the JSON is missing or unreadable.
    """
    verify_json = (
        Path(workspace_root)
        / artifact_path(work_id, PhaseName.VERIFY.value)
        / "verification.json"
    )
    if not verify_json.exists():
        return []
    try:
        data = json.loads(verify_json.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    results = data.get("verification_results", [])
    return results if isinstance(results, list) else []


# The verify report must cross the sandbox→durable boundary whole: gap_plan
# reads verification.md, restart/UI read both, and a 500-char preview leaves the
# persisted file truncated mid-record. Bounded so a pathological report can't
# blow up state; the mapper's _FULL_PERSIST_ARTIFACTS keeps these untruncated.
_FULL_REPORT_FILES = ("verification.json", "verification.md")
_MAX_FULL_REPORT_CHARS = 200_000


def _verify_artifacts_for_state(workspace_root: str, work_id: str) -> dict[str, str]:
    """Scan the verify artifact dir, but carry the report files at full fidelity.

    Preview truncation is right for incidental files but loses the verification
    report, whose only copy after sandbox teardown is what we return here.
    """
    return scan_artifact_dir(
        workspace_root, work_id, PhaseName.VERIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        full_fidelity=_FULL_REPORT_FILES,
        max_full_chars=_MAX_FULL_REPORT_CHARS,
    )


async def _save_verify_artifacts(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the verify phase to disk and state.

    Reads ``verification.json`` for authoritative phase status,
    falling back to string-matching on ``verification.md`` content if
    JSON is not available.
    """
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        # Even when verification did not pass, carry the report the
        # synthesize node wrote to disk back into state. Restarted/resumed
        # runs execute in a worktree sandbox that is torn down on finalize,
        # so anything NOT returned in artifacts_output is lost — leaving the
        # reviewer staring at "needs_review" with no explanation. Surface the
        # per-slice findings too, so the feedback reason can name the gaps.
        disk_artifacts = _verify_artifacts_for_state(workspace_root, work_id)
        return {
            "artifacts_output": disk_artifacts,
            "phase_status": existing_phase_status,
            "verification_findings": _load_verification_results(workspace_root, work_id),
        }

    disk_artifacts = _verify_artifacts_for_state(workspace_root, work_id)

    if not disk_artifacts:
        verify_content = agent_response
        if not verify_content or len(verify_content.strip()) < 20:
            verify_content = (
                "Verification could not produce a meaningful report. "
                "The agent returned insufficient output. Manual review required."
            )
        materialize_phase_artifacts(
            PhaseName.VERIFY.value,
            {"verification.md": verify_content},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]}

    # Determine status from verification.json (authoritative source).
    verify_dir = Path(workspace_root) / artifact_path(work_id, PhaseName.VERIFY.value)
    verify_json_path = verify_dir / "verification.json"
    is_verified = False
    if verify_json_path.exists():
        try:
            vdata = json.loads(verify_json_path.read_text())
            is_verified = vdata.get("overall_status") == "VERIFIED"
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "[%s] verification.json exists but could not be parsed; "
                "defaulting to unverified",
                work_id,
            )

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if is_verified else "needs_review",
        "verification_findings": _load_verification_results(workspace_root, work_id),
    }


# ── Builder ──────────────────────────────────────────────────────────────


def build_verify_subgraph() -> Any:
    """Build the VERIFY phase subgraph with Send API dispatch.

    Returns a compiled StateGraph with six nodes:
    1. seed_prior_results — carries forward VERIFIED verdicts for slices
       untouched since the last cycle (per-slice convergence)
    2. verify_router — conditional edge dispatching Send objects
    3. run_slice_verifier — per-slice subagent invocation (parallel)
    4. aggregate_verification — fan-in checkpoint
    5. synthesize_verification — writes verification artifacts
    6. save_artifacts — scans disk, materializes to state
    """
    builder = StateGraph(VerifySubgraphState)

    builder.add_node("seed_prior_results", _seed_prior_results_node)
    builder.add_node("plan_slice_verifier", _plan_slice_verifier_node)
    builder.add_node("run_slice_verifier", _run_slice_verifier_node)
    builder.add_node("aggregate_verification", _aggregate_verification_node)
    builder.add_node("synthesize_verification", _synthesize_verification_node)
    builder.add_node("save_artifacts", _save_verify_artifacts)

    # seed_prior_results runs before the router so carried-forward VERIFIED
    # verdicts are in verification_results (and their ids in
    # reverify_skipped_ids) before dispatch decisions are made.
    builder.add_edge(START, "seed_prior_results")
    builder.add_conditional_edges(
        "seed_prior_results",
        _verify_router,
        {
            # Send targets dispatch to plan_slice_verifier; each parallel
            # branch then chains plan → do before fan-in.
            "plan_slice_verifier": "plan_slice_verifier",
            "synthesize_verification": "synthesize_verification",
        },
    )

    # plan_slice_verifier dispatches to run_slice_verifier dynamically
    # via Command(goto=Send) (see the node) so each parallel branch
    # carries its own directive without colliding on a shared LastValue
    # channel. run_slice_verifier → aggregate_verification is a plain
    # fan-in edge that runs once on the merged verification_results.
    builder.add_edge("run_slice_verifier", "aggregate_verification")
    builder.add_edge("aggregate_verification", "synthesize_verification")
    builder.add_edge("synthesize_verification", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder