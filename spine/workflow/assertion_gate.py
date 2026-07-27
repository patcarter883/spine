"""Ground an equality/persistence criterion in the test source that claims it.

A passing test suite proves that assertions did not fail. It says nothing
about *what was asserted* — a weak test is exactly as green as a strong one.
The judge has no way to tell those apart from a check-runner line, and when
it tries, it gets it wrong in the direction of shipping.

The incident (probe 25, agripath UnitOfMeasure, landed as dca6ebc). The
criterion read "The test asserts that the `name` of the created
UnitOfMeasure matches the expected factory value". The test read
``expect($unitOfMeasure->name)->not->toBeEmpty()``. The judge marked it
PASSED, reasoning: *"The automated checks (pest_changed_tests) passed,
indicating the generated value was non-empty and valid."* Two verify rounds
earlier the same judge, on the same source, had failed the same criterion
with a correct diagnosis; the gap plan had spelled out the fix. The run
landed with the weak test.

The prose rule the judge over-generalised was added for a different defect
(run f788042e: the judge failing criteria by asserting framework defaults it
had misremembered). That rule ends "when no executed check demonstrates a
violation, the criterion passes on that point" — sound for a framework-
default claim, catastrophic as a general licence. Rather than re-balance
prose against prose, this module answers the question deterministically: an
equality criterion needs an equality assertion to exist in the source.

Deliberately conservative — it demotes only when the slice's test sources
contain *no* equality-shaped assertion at all. A test with any real
comparison is left entirely to the judge.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Criterion phrasings that demand a value comparison or a persistence
# round-trip. "matches", "equals", "round-trip", "persisted".
_EQUALITY_CRITERION_RE = re.compile(
    r"\b("
    r"match(?:es|ing|ed)?|equals?|equal\s+to|identical|"
    r"round[\s-]?trips?|persisted|persists?|"
    r"same\s+(?:value|as)|correct\s+value"
    r")\b",
    re.IGNORECASE,
)

# Assertion forms that actually compare against an expected value, per
# language family. Substring match on source — cheap and parser-free, which
# matters because this must never throw inside verify.
_EQUALITY_ASSERTIONS: dict[str, tuple[str, ...]] = {
    # NOTE: toBeTrue()/toBeFalse()/toBeEmpty() are deliberately ABSENT. They
    # compare against a constant, not against an expected value derived from
    # the subject — and probe 25's file shipped a boilerplate
    # `expect(true)->toBeTrue()` alongside the weak assertions, so admitting
    # them here would let exactly the defect this gate exists for slip past.
    ".php": (
        "->tobe(", "->toequal(", "->tomatch",
        "assertsame(", "assertequals(", "assertdatabasehas(",
        "assertdatabasecount(", "->tocontain(",
    ),
    ".py": (
        "assertequal", "assertis", "assertin", "assertdictequal",
        "assertlistequal", "== ", "assert_frame_equal", "approx(",
    ),
    ".ts": (
        "tobe(", "toequal(", "tostrictequal(", "tomatchobject(",
        "assert.equal", "assert.strictequal", "tocontain(",
    ),
}
_EQUALITY_ASSERTIONS[".tsx"] = _EQUALITY_ASSERTIONS[".ts"]
_EQUALITY_ASSERTIONS[".js"] = _EQUALITY_ASSERTIONS[".ts"]
_EQUALITY_ASSERTIONS[".jsx"] = _EQUALITY_ASSERTIONS[".ts"]

_TEST_HINTS = ("test", "spec")


def is_equality_criterion(text: str) -> bool:
    """True when *text* demands a value comparison or persistence round-trip."""
    return bool(_EQUALITY_CRITERION_RE.search(text or ""))


def is_test_file(path: str) -> bool:
    """True when *path* looks like a test file."""
    low = str(path or "").lower()
    if Path(low).suffix not in _EQUALITY_ASSERTIONS:
        return False
    return any(h in low for h in _TEST_HINTS)


def has_equality_assertion(source: str, suffix: str) -> bool:
    """True when *source* contains any comparison-style assertion.

    Unknown suffix ⇒ True (never demote on a language we cannot read).
    """
    tokens = _EQUALITY_ASSERTIONS.get(suffix.lower())
    if not tokens:
        return True
    low = source.lower()
    return any(t in low for t in tokens)


def find_assertionless_tests(
    target_files: list[str], workspace_root: str | Path
) -> list[str]:
    """Slice test files that contain no equality-shaped assertion.

    Returns the offending relative paths. Empty when the slice has no test
    files, when every test asserts something comparable, or on any read
    failure — this gate accuses only on positive evidence.
    """
    root = Path(workspace_root or ".")
    offenders: list[str] = []
    for rel in target_files or []:
        if not rel or not is_test_file(str(rel)):
            continue
        fpath = root / str(rel)
        try:
            if not fpath.is_file():
                continue
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not source.strip():
            continue
        if not has_equality_assertion(source, Path(str(rel)).suffix):
            offenders.append(str(rel))
    return offenders


def demote_unsupported_equality_criteria(
    verification_result: dict,
    target_files: list[str],
    workspace_root: str | Path,
    work_id: str = "unknown",
    slice_id: str = "unknown",
) -> int:
    """Fail PASSED equality criteria whose tests assert no equality.

    Mutates *verification_result* in place: flips the checklist entries,
    appends matching gaps and recommendations, and drops the verdict to
    NOT_VERIFIED. Returns how many criteria were demoted.
    """
    checklist = verification_result.get("checklist") or []
    if not checklist:
        return 0
    passed_equality = [
        c
        for c in checklist
        if isinstance(c, dict)
        and c.get("passed")
        and is_equality_criterion(str(c.get("criterion", "")))
    ]
    if not passed_equality:
        return 0

    offenders = find_assertionless_tests(target_files, workspace_root)
    if not offenders:
        return 0

    gaps = verification_result.setdefault("gaps", [])
    recs = verification_result.setdefault("recommendations", [])
    listed = ", ".join(offenders)
    for entry in passed_equality:
        criterion = str(entry.get("criterion", ""))
        entry["passed"] = False
        entry["detail"] = (
            f"DEMOTED by assertion gate: this criterion requires a value "
            f"comparison, but {listed} contains no equality assertion "
            f"(no toBe/toEqual/assertSame/assertEquals/assertDatabaseHas or "
            f"equivalent). A green check run does not evidence an assertion "
            f"the test never makes. Previous detail: {entry.get('detail', '')}"
        )
        gap = f"{criterion} — {listed} asserts no compared value."
        if gap not in gaps:
            gaps.append(gap)
        rec = (
            f"In {listed}, assert the actual value against the expected one "
            f"(e.g. toBe/assertSame on the persisted attribute), not merely "
            f"that it is non-empty or of the right type."
        )
        if rec not in recs:
            recs.append(rec)

    verification_result["verdict"] = "NOT_VERIFIED"
    logger.warning(
        "[%s] Slice-verifier %r: assertion gate demoted %d equality "
        "criterion/criteria — %s has no comparison assertion",
        work_id, slice_id, len(passed_equality), listed,
    )
    return len(passed_equality)
