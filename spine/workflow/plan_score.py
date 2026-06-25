"""Plan scoring for best-of-N synthesis (Graph-of-Thoughts ``Score`` + ``KeepBest``).

The PLAN synthesizer historically produced *one* plan per round, then the
critic loop ground it toward acceptance one rework cycle at a time (commit
32adfb9 raised ``max_critic_retries`` 3→5 precisely because the GLM planner
needed another round to converge). Best-of-N front-loads that convergence:
generate several candidate plans, score each cheaply here, and keep the best
*before* the critic ever sees it — turning sequential rework into parallel
selection.

This module is the ``Score`` operation. It is deliberately **deterministic and
LLM-free** so the selector is cheap and reproducible: it rewards the exact
properties the critic rejects on (empty ``target_files``, the duplicate-path
explosion of trace 019eddd3, dependency cycles that block IMPLEMENT, and
spec-file coverage). An optional LLM-judge can be layered on top by the caller
(``research_synth_scorer="hybrid"``) but is not required.

``score_plan`` returns a :class:`PlanScore` whose ``total`` is in ``[0, 1]``;
higher is better. A plan whose waves do not compute (a cycle, an unknown
dependency) scores ``0`` on the ``schedulable`` gate — it cannot start
IMPLEMENT, so it can never win against any schedulable candidate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sane slice-count band. A plan with one mega-slice defeats the point of
# wave-based parallelism; a plan with 40 slices for a small task is the
# over-decomposition failure mode. Outside the band we taper rather than
# hard-fail — a coarse plan is still usable.
_MIN_SLICES = 1
_IDEAL_LOW = 2
_IDEAL_HIGH = 8
_MAX_REASONABLE_SLICES = 20

# Weights for the component scores. ``schedulable`` is a hard multiplicative
# gate (below), so it carries no additive weight here.
_W_TARGET_FILES = 0.40   # every slice names the files it touches
_W_NO_DUP = 0.20         # no within-slice duplicate paths (the 250-dup blowup)
_W_COVERAGE = 0.25       # plan touches the files the spec implicates
_W_SIZE = 0.15           # slice count sits in the sane band


@dataclass
class PlanScore:
    """Result of scoring a single candidate plan."""

    total: float
    schedulable: bool
    target_files_score: float
    dup_penalty_score: float
    coverage_score: float
    size_score: float
    n_slices: int
    reason: str = ""
    components: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        gate = "schedulable" if self.schedulable else "UNSCHEDULABLE"
        return (
            f"PlanScore(total={self.total:.3f} {gate} slices={self.n_slices} "
            f"target={self.target_files_score:.2f} dup={self.dup_penalty_score:.2f} "
            f"cov={self.coverage_score:.2f} size={self.size_score:.2f})"
        )


def _slices(plan_data: dict) -> list[dict]:
    raw = plan_data.get("feature_slices")
    return [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []


def _is_schedulable(plan_data: dict, work_id: str) -> bool:
    """True when the plan's slices form a valid, acyclic dependency DAG.

    Reuses the production wave computation so the gate is exactly the
    constraint IMPLEMENT enforces — never a looser proxy.
    """
    try:
        from spine.workflow.slice_scheduler import (
            compute_execution_waves,
            validate_feature_slices,
        )
        from spine.models.types import FeatureSlice
    except ImportError:  # pragma: no cover - scheduler always present in prod
        logger.debug("[%s] slice_scheduler unavailable — skipping gate", work_id)
        return True

    slices = _slices(plan_data)
    if not slices:
        return False
    try:
        fs = [FeatureSlice.from_dict(s) for s in slices]
        validate_feature_slices(fs)
        compute_execution_waves(fs)
        return True
    except (ValueError, KeyError, TypeError) as exc:
        logger.info("[%s] plan unschedulable: %s", work_id, exc)
        return False


def _target_files_score(slices: list[dict]) -> float:
    """Fraction of slices that name at least one concrete target file.

    Empty ``target_files`` is the single most common PLAN-critic rejection
    (trace 019ec997: three rework cycles, all flagging it). A plan that
    leaves it blank everywhere scores 0 here.
    """
    if not slices:
        return 0.0
    named = sum(1 for s in slices if [f for f in (s.get("target_files") or []) if str(f).strip()])
    return named / len(slices)


def _dup_penalty_score(slices: list[dict]) -> float:
    """1.0 when no slice repeats a path; degrades with duplication.

    Trace 019eddd3 produced a single slice carrying 250+ duplicate
    ``target_files`` — a degenerate synthesis the critic cannot repair. We
    score the mean within-slice uniqueness ratio across slices that list
    files.
    """
    ratios: list[float] = []
    for s in slices:
        files = [str(f).strip() for f in (s.get("target_files") or []) if str(f).strip()]
        if not files:
            continue
        ratios.append(len(set(files)) / len(files))
    return sum(ratios) / len(ratios) if ratios else 1.0


_PATH_RE = re.compile(r"[`'\"]?([\w./-]+\.[A-Za-z][\w]*)[`'\"]?")


def _spec_paths(spec_body: str) -> set[str]:
    """Best-effort extraction of file-like tokens mentioned in the spec.

    Heuristic, not authoritative — used only to reward plans whose
    ``target_files`` overlap what the spec talks about. We compare on
    basenames to tolerate path-prefix differences.
    """
    found: set[str] = set()
    for m in _PATH_RE.finditer(spec_body or ""):
        tok = m.group(1)
        if "/" in tok or tok.count(".") == 1:
            found.add(tok.rsplit("/", 1)[-1])
    return found


def _coverage_score(slices: list[dict], spec_body: str) -> float:
    """Fraction of spec-implicated files that appear in some slice.

    Returns 1.0 when the spec names no files (nothing to cover) — absence of
    signal must not penalize an otherwise-fine plan.
    """
    spec_files = _spec_paths(spec_body)
    if not spec_files:
        return 1.0
    plan_basenames: set[str] = set()
    for s in slices:
        for f in (s.get("target_files") or []):
            f = str(f).strip()
            if f:
                plan_basenames.add(f.rsplit("/", 1)[-1])
    if not plan_basenames:
        return 0.0
    hit = sum(1 for sf in spec_files if sf in plan_basenames)
    return hit / len(spec_files)


def _size_score(n: int) -> float:
    """Triangular preference for a sane slice count."""
    if n < _MIN_SLICES:
        return 0.0
    if _IDEAL_LOW <= n <= _IDEAL_HIGH:
        return 1.0
    if n < _IDEAL_LOW:  # n == 1: usable but not parallelizable
        return 0.6
    if n <= _MAX_REASONABLE_SLICES:
        # Linear taper from ideal-high down to the reasonable ceiling.
        span = _MAX_REASONABLE_SLICES - _IDEAL_HIGH
        return max(0.3, 1.0 - (n - _IDEAL_HIGH) / span * 0.7)
    return 0.2


def score_plan(plan_data: dict, *, spec_body: str = "", work_id: str = "unknown") -> PlanScore:
    """Score a candidate plan in ``[0, 1]`` (higher is better).

    Args:
        plan_data: Parsed ``plan.json`` content (the ``feature_slices`` list
            is what matters).
        spec_body: The specification text, used for file-coverage scoring.
            Pass ``""`` to skip coverage (it then contributes a neutral 1.0).
        work_id: For logging only.

    Returns:
        A :class:`PlanScore`. Unschedulable plans (cycles, dangling deps,
        zero slices) return ``total=0.0`` regardless of other components.
    """
    slices = _slices(plan_data)
    n = len(slices)
    schedulable = _is_schedulable(plan_data, work_id)

    tf = _target_files_score(slices)
    dup = _dup_penalty_score(slices)
    cov = _coverage_score(slices, spec_body)
    size = _size_score(n)

    additive = (
        _W_TARGET_FILES * tf
        + _W_NO_DUP * dup
        + _W_COVERAGE * cov
        + _W_SIZE * size
    )
    total = additive if schedulable else 0.0

    components = {
        "target_files": tf,
        "no_dup": dup,
        "coverage": cov,
        "size": size,
        "schedulable": 1.0 if schedulable else 0.0,
    }
    reason = (
        f"{'schedulable' if schedulable else 'UNSCHEDULABLE'}; "
        f"{n} slices; target_files={tf:.2f} dup={dup:.2f} "
        f"coverage={cov:.2f} size={size:.2f}"
    )
    score = PlanScore(
        total=round(total, 4),
        schedulable=schedulable,
        target_files_score=tf,
        dup_penalty_score=dup,
        coverage_score=cov,
        size_score=size,
        n_slices=n,
        reason=reason,
        components=components,
    )
    logger.info("[%s] %s", work_id, score)
    return score
