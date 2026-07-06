"""Tests for the deterministic reference-symbol gate at critic_plan time.

Trace 019f2077: four critic rounds churned on plans whose reference_symbols
named UIApi methods that don't exist (`UIApi.get_llm_providers` for the real
`UIApi.get_providers`), and the true blocker — the spec requires embedding /
reranker provider UI while its scope_exclusions forbid the UIApi changes that
needs — never fired the spec_contradiction escape hatch. The gate detects both
deterministically:

* dangling reference_symbols → NEEDS_REVISION with a "did you mean" near-miss;
* a dangling symbol whose owner a scope_exclusions bullet protects, with no
  near-equivalent (or persisting across rounds) → NEEDS_REVIEW with
  blocker_category='spec_contradiction', which the critic result mapper routes
  to a SPECIFY amendment.
"""

from __future__ import annotations

import json

import pytest

from spine.models.enums import PhaseName, ReviewStatus
from spine.workflow.plan_reference_gate import check_reference_symbols

DB = "fake.db"  # any truthy path — lookups are monkeypatched


def _plan(*slices: dict) -> dict:
    return {"feature_slices": list(slices)}


def _spec(exclusions: list[str] | None = None) -> str:
    return json.dumps(
        {
            "scope_inclusions": ["Embedding provider management UI."],
            "scope_exclusions": exclusions or [],
        }
    )


@pytest.fixture()
def index(monkeypatch: pytest.MonkeyPatch):
    """Install a fake codebase index into the gate.

    ``symbols`` maps file_path -> qualified symbol names. Existence checks and
    near-miss candidate lookups both resolve against it.
    """

    def _install(symbols: dict[str, list[str]]) -> None:
        all_syms = {s for syms in symbols.values() for s in syms}

        def exists(db_path, sym):
            s = (sym or "").strip()
            leaf = s.rsplit(".", 1)[-1]
            return any(
                c == s or c.rsplit(".", 1)[-1] == leaf for c in all_syms
            )

        def find_files(db_path, name):
            # Mirrors the real index's _NAME_MATCH: exact symbol_name OR a
            # dotted-suffix match (find_symbol('get_providers') hits
            # 'UIApi.get_providers'), plus owner-class matching for near-miss
            # candidate discovery.
            return [
                fp
                for fp, syms in symbols.items()
                if any(
                    c == name
                    or c.endswith("." + name)
                    or c.split(".")[0] == name
                    for c in syms
                )
            ]

        monkeypatch.setattr(
            "spine.workflow.plan_reference_gate._symbol_exists_in_index", exists
        )
        monkeypatch.setattr(
            "spine.workflow.plan_reference_gate._find_symbol_files", find_files
        )
        monkeypatch.setattr(
            "spine.workflow.plan_reference_gate._list_file_symbols",
            lambda db_path, fp: symbols.get(fp, []),
        )

    return _install


UIAPI_FILE = "spine/ui_api/api.py"
UIAPI_SYMS = [
    "UIApi",
    "UIApi.get_providers",
    "UIApi.add_llm_provider",
    "UIApi.remove_llm_provider",
    "UIApi._save_config",
]


def test_all_resolved_returns_none(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {"id": "a", "reference_symbols": ["UIApi.get_providers", "st.form"]}
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


def test_no_index_is_permissive() -> None:
    plan = _plan({"id": "a", "reference_symbols": ["UIApi.invented"]})
    assert check_reference_symbols(plan, _spec(), None, db_path=None) is None


def test_slice_provided_symbol_is_exempt(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {"id": "api", "provides": ["UIApi.add_embedding_provider"]},
        {"id": "ui", "reference_symbols": ["UIApi.add_embedding_provider"]},
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


def test_dangling_symbol_needs_revision_with_near_miss(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {
            "id": "embed-ui",
            "target_files": ["spine/ui/_pages/config_view.py"],
            "reference_symbols": ["UIApi.get_llm_providers"],
        }
    )
    result = check_reference_symbols(plan, _spec(), None, db_path=DB)
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value
    assert result["tier"] == "structural"
    assert result["blocker_category"] is None
    assert "embed-ui" in result["reason"]
    assert "UIApi.get_llm_providers" in result["reason"]
    assert "did you mean" in result["reason"]
    assert result["dangling_leafs"] == ["get_llm_providers"]
    assert any("UIApi.get_" in s for s in result["suggestions"])


def test_exclusion_protected_escalates_only_after_persisting(index) -> None:
    # Owner class exists but nothing similar to the needed method does, and
    # the spec forbids changing UIApi — unresolvable by plan rework. Round 1
    # is still a revision (with an explicit escalation warning): a single
    # round's exclusion match is not enough evidence to park the run (run
    # 019f2104's false positive). Round 2, same symbol still dangling →
    # spec contradiction.
    index({UIAPI_FILE: ["UIApi", "UIApi.run_project_verify"]})
    exclusion = "Changes to SpineConfig or UIApi schemas."
    plan = _plan(
        {"id": "embed-ui", "reference_symbols": ["UIApi.add_embedding_provider"]}
    )
    result = check_reference_symbols(plan, _spec([exclusion]), None, db_path=DB)
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value
    assert result["blocker_category"] is None
    assert "will escalate a spec contradiction" in result["reason"]

    result2 = check_reference_symbols(plan, _spec([exclusion]), result, db_path=DB)
    assert result2 is not None
    assert result2["status"] == ReviewStatus.NEEDS_REVIEW.value
    assert result2["blocker_category"] == "spec_contradiction"
    assert result2["cited_exclusions"] == [exclusion]
    assert "amend" in result2["reason"].lower()


def test_module_qualified_external_alias_is_skipped(index) -> None:
    # Run 019f2104: 'spine.ui._pages.config_view.st' is a reference to the
    # streamlit import alias inside config_view — an external-library name,
    # not a code contract. Must not be treated as dangling at all.
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {"id": "s1", "reference_symbols": ["spine.ui._pages.config_view.st"]}
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


def test_builtins_and_logger_refs_are_skipped(index) -> None:
    # Run 019f34b7: planners copy research "Calls:" lists ('open',
    # 'logger.exception') into reference_symbols; the gate flagged both as
    # dangling for two consecutive rounds, driving the stagnation streak
    # toward a park on a false-positive class.
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {
            "id": "s1",
            "reference_symbols": [
                "open",
                "logger.exception",
                "logging.getLogger",
                "print",
                "dict.setdefault",
            ],
        }
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


def test_generic_package_prefix_never_matches_exclusion(index) -> None:
    # Run 019f2104: the exclusion "Core spine settings (checkpoint_path, ...)"
    # matched the 'spine' package prefix of every module path. Only the
    # DIRECT owner segment may match, and even persistent dangling must not
    # escalate on an intermediate-segment match.
    index({UIAPI_FILE: UIAPI_SYMS})
    exclusion = "Core spine settings (checkpoint_path, artifact_path, etc.)."
    plan = _plan(
        {"id": "s1", "reference_symbols": ["spine.ui.widgets.render_all_panels"]}
    )
    first = check_reference_symbols(plan, _spec([exclusion]), None, db_path=DB)
    assert first is not None
    assert first["status"] == ReviewStatus.NEEDS_REVISION.value
    second = check_reference_symbols(plan, _spec([exclusion]), first, db_path=DB)
    assert second is not None
    assert second["status"] == ReviewStatus.NEEDS_REVISION.value
    assert second["blocker_category"] is None


def test_exclusion_with_near_miss_gets_one_revision_round_first(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    exclusion = "Changes to SpineConfig or UIApi schemas."
    plan = _plan(
        {"id": "embed-ui", "reference_symbols": ["UIApi.get_llm_providers"]}
    )
    # Round 1: a near-miss exists, so the planner gets a chance to adopt it.
    result = check_reference_symbols(plan, _spec([exclusion]), None, db_path=DB)
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value
    assert result["blocker_category"] is None

    # Round 2: the same leaf is still dangling despite exact feedback —
    # escalate as a spec contradiction instead of churning to stagnation.
    result2 = check_reference_symbols(
        plan, _spec([exclusion]), result, db_path=DB
    )
    assert result2 is not None
    assert result2["status"] == ReviewStatus.NEEDS_REVIEW.value
    assert result2["blocker_category"] == "spec_contradiction"
    assert result2["cited_exclusions"] == [exclusion]


def test_unqualified_dangling_never_matches_exclusion(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan({"id": "a", "reference_symbols": ["add_embedding_provider"]})
    result = check_reference_symbols(
        plan, _spec(["Changes to SpineConfig or UIApi schemas."]), None, db_path=DB
    )
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value
    assert result["blocker_category"] is None


def test_persistence_without_exclusion_stays_revision(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan({"id": "a", "reference_symbols": ["Widget.render_all"]})
    first = check_reference_symbols(plan, _spec(), None, db_path=DB)
    assert first is not None
    second = check_reference_symbols(plan, _spec(), first, db_path=DB)
    assert second is not None
    # No scope_exclusions implicates the spec → generic non-convergence is the
    # stagnation detector's job, not a SPECIFY amendment.
    assert second["status"] == ReviewStatus.NEEDS_REVISION.value


# ── Format normalization + provides-of-existing (run 019f20e0) ───────────────


def test_colon_format_refs_resolve_after_normalization(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    # Run 019f20e0 emitted 'file.py:symbol' — the part after ':' is the symbol.
    plan = _plan(
        {"id": "a", "reference_symbols": ["spine/ui_api/api.py:UIApi.get_providers"]}
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


def test_colon_format_mispathed_ref_still_dangles(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {"id": "a", "reference_symbols": ["spine/ui/api/api.py:UIApi.get_llm_providers"]}
    )
    result = check_reference_symbols(plan, _spec(), None, db_path=DB)
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value


def test_provides_existing_symbol_flagged(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    # Run 019f20e0: the plan "provided" UIApi.get_providers, which already
    # exists — its workers then invented acceptance criteria contradicting the
    # live implementation. provides is for NEW symbols only.
    plan = _plan(
        {"id": "api-slice", "provides": ["spine/ui_api/api.py:get_providers"]}
    )
    result = check_reference_symbols(plan, _spec(), None, db_path=DB)
    assert result is not None
    assert result["status"] == ReviewStatus.NEEDS_REVISION.value
    assert "ALREADY EXISTS" in result["reason"]
    assert UIAPI_FILE in result["reason"]
    assert any("reference_symbols" in s for s in result["suggestions"])


def test_provides_genuinely_new_symbol_not_flagged(index) -> None:
    index({UIAPI_FILE: UIAPI_SYMS})
    plan = _plan(
        {
            "id": "api-slice",
            "provides": ["UIApi.add_embedding_provider", "render_new_form"],
        }
    )
    assert check_reference_symbols(plan, _spec(), None, db_path=DB) is None


# ── Subgraph fold: validation verdict overrides the agent vote ───────────────


@pytest.mark.asyncio
async def test_agent_check_propagates_spec_contradiction(monkeypatch) -> None:
    from spine.workflow.subgraphs import critic_subgraph

    async def fake_agent_check(state, reviewed_phase, config=None):
        return {
            "status": ReviewStatus.PASSED.value,
            "tier": "agent",
            "reason": "looks fine to me",
            "suggestions": [],
        }

    monkeypatch.setattr(critic_subgraph, "agent_critic_check", fake_agent_check)

    gate_verdict = {
        "status": ReviewStatus.NEEDS_REVIEW.value,
        "tier": "structural",
        "reason": "plan needs UIApi.add_embedding_provider; spec forbids it",
        "suggestions": [],
        "blocker_category": "spec_contradiction",
        "cited_exclusions": ["Changes to SpineConfig or UIApi schemas."],
        "dangling_leafs": ["add_embedding_provider"],
    }
    out = await critic_subgraph._agent_check_node(
        {
            "reviewed_phase": PhaseName.PLAN.value,
            "work_id": "w1",
            "validation_result": gate_verdict,
            "specification_json": _spec(),
            "plan_json": "{}",
        }
    )
    assert out["phase_status"] == ReviewStatus.NEEDS_REVIEW.value
    merged = out["agent_result"]
    assert merged["status"] == ReviewStatus.NEEDS_REVIEW.value
    assert merged["blocker_category"] == "spec_contradiction"
    assert merged["cited_exclusions"] == gate_verdict["cited_exclusions"]
    assert "UIApi.add_embedding_provider" in merged["reason"]


@pytest.mark.asyncio
async def test_agent_check_still_forces_revision_for_plain_failures(
    monkeypatch,
) -> None:
    from spine.workflow.subgraphs import critic_subgraph

    async def fake_agent_check(state, reviewed_phase, config=None):
        return {
            "status": ReviewStatus.PASSED.value,
            "tier": "agent",
            "reason": "fine",
            "suggestions": [],
        }

    monkeypatch.setattr(critic_subgraph, "agent_critic_check", fake_agent_check)
    out = await critic_subgraph._agent_check_node(
        {
            "reviewed_phase": PhaseName.PLAN.value,
            "work_id": "w1",
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "dangling reference",
                "suggestions": ["fix it"],
            },
            "specification_json": _spec(),
            "plan_json": "{}",
        }
    )
    assert out["phase_status"] == ReviewStatus.NEEDS_REVISION.value
    assert out["agent_result"].get("blocker_category") is None


# ── Parent mappers: round-trip of the gate result and prior verdict ──────────


def test_critic_state_mapper_forwards_last_critic_review() -> None:
    from spine.workflow.compose import _critic_state_mapper

    prior = {"phase": "plan", "status": "needs_revision", "reference_gate": {}}
    mapped = _critic_state_mapper(PhaseName.PLAN.value)(
        {"work_id": "w1", "last_critic_review": prior, "retry_count": {}}, None
    )
    assert mapped["last_critic_review"] == prior


def test_critic_result_mapper_attaches_reference_gate() -> None:
    from spine.workflow.compose import _critic_result_mapper

    gate = {"status": "needs_revision", "dangling_leafs": ["x"]}
    result = _critic_result_mapper(PhaseName.PLAN.value)(
        {
            "agent_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "agent",
                "reason": "r",
                "suggestions": [],
            },
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "reference_gate_result": gate,
        },
        {"work_id": "w1", "retry_count": {}, "max_retries": 5},
    )
    assert result["last_critic_review"]["reference_gate"] == gate


def test_critic_result_mapper_routes_spec_contradiction_to_specify() -> None:
    from spine.workflow.compose import _critic_result_mapper

    result = _critic_result_mapper(PhaseName.PLAN.value)(
        {
            "agent_result": {
                "status": ReviewStatus.NEEDS_REVIEW.value,
                "tier": "structural",
                "reason": "spec forbids the needed API",
                "suggestions": [],
                "blocker_category": "spec_contradiction",
            },
            "phase_status": ReviewStatus.NEEDS_REVIEW.value,
            "reference_gate_result": {"dangling_leafs": ["x"]},
        },
        {"work_id": "w1", "retry_count": {}, "max_retries": 5},
    )
    assert result["status"] == "needs_review"
    assert result["needs_review_phase"] == PhaseName.SPECIFY.value
    assert result["needs_review_kind"] == "spec_amendment"
    assert result["last_critic_review"]["escalate"] is True
