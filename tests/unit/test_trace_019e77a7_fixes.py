"""Regression tests for the fixes driven by trace 019e77a7.

Trace 019e77a7 (`task` / "Add a --verbose flag to the CLI entrypoint to
toggle debug logging") looped PLAN ↔ critic three times on a one-line task
and escalated to a `human_review` interrupt that an autonomous run can never
resume. Six fixes landed:

1. Rework cap lowered to 2 (config).
2. Autonomous work types terminate needs_review instead of interrupting.
3. The critic sees its own prior verdict on rework (convergence).
4. The critic no longer invents schema/field requirements.
5. PLAN critic enforces proportionality / minimalism.
6. SearchCodebaseTool self-corrects on empty `queries`.
7. SPECIFY recall gate has a trivial-task fast path that fires on 0 hits.
"""

from __future__ import annotations

from spine.config import SpineConfig


# ── 1. Rework cap ───────────────────────────────────────────────────────────


def test_max_critic_retries_capped_at_two():
    """Default rework budget is 2 cycles, not the old 3."""
    assert SpineConfig().max_critic_retries == 2


# ── 2. Autonomous types terminate needs_review; reviewed types interrupt ─────


def test_autonomous_work_types_get_terminal_needs_review_node():
    from spine.workflow.compose import build_workflow_graph

    for wt in ("task", "critical_task"):
        nodes = set(build_workflow_graph(wt).get_graph().nodes)
        assert "flag_needs_review" in nodes, wt


def test_reviewed_work_types_have_no_terminal_flag_node():
    from spine.workflow.compose import build_workflow_graph

    for wt in ("reviewed_task", "critical_reviewed_task"):
        nodes = set(build_workflow_graph(wt).get_graph().nodes)
        assert "flag_needs_review" not in nodes, wt
        # The human gate remains the designed checkpoint for reviewed types.
        assert "human_review" in nodes, wt


def test_flag_needs_review_terminal_sets_status():
    from spine.workflow.compose import _flag_needs_review_terminal

    out = _flag_needs_review_terminal(
        {"current_phase": "plan", "feedback": [{"reason": "x"}]}
    )
    assert out["status"] == "needs_review"
    assert out["needs_review_phase"] == "plan"


# ── 3. Critic sees its own prior verdict on rework ──────────────────────────


def test_review_prompt_injects_prior_verdict_on_rework():
    from spine.workflow.critic_review import _build_review_prompt

    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload='{"feature_slices": []}',
        description="Add a --verbose flag",
        prior_review={
            "phase": "plan",
            "status": "needs_revision",
            "reason": "target_files reference non-existent files",
            "suggestions": ["Fix target_files to point at spine/cli/__init__.py"],
            "attempt": 1,
        },
    )
    assert "<critic_feedback>" in prompt
    assert "target_files reference non-existent files" in prompt
    assert "REWORK review" in prompt
    # Convergence instruction: confirm prior asks before raising new issues.
    assert "ADDRESSES it" in prompt or "addresses it" in prompt.lower()


def test_review_prompt_first_pass_has_no_prior_block():
    from spine.workflow.critic_review import _build_review_prompt

    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload='{"feature_slices": []}',
        description="Add a --verbose flag",
        prior_review=None,
    )
    assert "<critic_feedback>" not in prompt
    assert "REWORK review" not in prompt


# ── 4 & 5. Critic prompt: no schema-field invention + proportionality ───────


def test_critic_base_prompt_forbids_schema_field_invention():
    from spine.critic.agent import build_critic_agent  # noqa: F401
    from spine.critic import agent as critic_agent

    # The base system prompt is assembled inside build_critic_agent; assert the
    # guardrail text exists in the module so the clause can't silently drop.
    import inspect

    src = inspect.getsource(critic_agent)
    assert "Schema conformance is NOT your job" in src
    assert "execution_requirements" in src  # named as a thing NOT to flag


def test_plan_review_instructions_enforce_proportionality():
    from spine.critic.agent import _PLAN_REVIEW_INSTRUCTIONS

    assert "Proportionality & minimalism" in _PLAN_REVIEW_INSTRUCTIONS
    assert "Prefer extending existing patterns over creating new modules" in (
        _PLAN_REVIEW_INSTRUCTIONS
    )
    assert "Never request scope expansion" in _PLAN_REVIEW_INSTRUCTIONS


# ── 6. SearchCodebaseTool self-corrects on empty queries ────────────────────


def test_search_codebase_empty_queries_raises_teaching_error():
    from langchain_core.tools import ToolException
    from spine.agents.plan_tools import SearchCodebaseTool

    tool = SearchCodebaseTool(workspace_root=".")
    for bad in ([], None, ["", "  "]):
        try:
            tool._run(queries=bad)
        except ToolException as exc:
            assert "queries" in str(exc)
            assert "queries=[" in str(exc)  # worked example present
        else:  # pragma: no cover - must raise
            raise AssertionError(f"expected ToolException for queries={bad!r}")


def test_search_codebase_input_queries_optional_no_min_length():
    """Empty `{}` calls must pass schema validation (then teach in _run),
    not 400 at the pydantic layer as before."""
    from spine.agents.plan_tools import _SearchCodebaseInput

    parsed = _SearchCodebaseInput()  # no args
    assert parsed.queries == []


# ── 7. SPECIFY recall gate trivial-task fast path ───────────────────────────


def _gate(description, *, confidence=0.8, hits=0, phase="specify", retry=0):
    from spine.workflow.subgraphs.exploration_subgraph import _gate_router

    return _gate_router(
        {
            "phase": phase,
            "description": description,
            "classification_confidence": confidence,
            "retrieved_context": [{"x": 1}] * hits,
            "retry_count": retry,
        }
    )


def test_gate_trivial_task_skips_exploration_with_zero_hits():
    # 0 recall hits would normally force exploration; the trivial fast path
    # fires on a short, high-confidence, non-architectural description.
    assert _gate("Add a --verbose flag to the CLI entrypoint") == "skip_to_synth"


def test_gate_architectural_verb_still_explores():
    assert _gate("Refactor the CLI entrypoint to a new command framework") == (
        "explore"
    )


def test_gate_long_description_still_explores():
    long_desc = "Add a flag " + "and also handle many edge cases " * 10
    assert len(long_desc) > 150
    assert _gate(long_desc) == "explore"


def test_gate_low_confidence_still_explores():
    assert _gate("Add a --verbose flag", confidence=0.4) == "explore"


def test_gate_plan_phase_never_uses_trivial_fast_path():
    assert _gate("Add a --verbose flag", phase="plan") == "explore"


# ── Follow-ups from the 019e77fe validation re-run ──────────────────────────


def test_plan_synthesizer_prompt_renders_real_newlines_and_proportionality():
    """The synth prompt had a literal-`\\n` bug (run-on blob) and no
    minimalism guidance; both are fixed."""
    from spine.agents.plan_agent import _build_plan_synthesizer_prompt

    p = _build_plan_synthesizer_prompt()
    assert "\n" in p and "\\n" not in p  # real newlines, no literal backslash-n
    # Minimalism: prefer extending existing patterns over adding new modules.
    assert "Extend existing patterns" in p
    assert "adding modules" in p
    # Dependency-integrity guard (the 019e77fe critic flagged a dangling dep).
    assert "MUST be the id of another slice" in p


def test_plan_synthesizer_prompt_demands_minimal_but_spec_complete():
    """Minimal slice COUNT must not drop spec requirements (019e783e: the
    minimal single-slice plan dropped the spec's test-targeting requirement,
    so the critic oscillated scope-too-big → tests-missing)."""
    from spine.agents.plan_agent import _build_plan_synthesizer_prompt

    p = _build_plan_synthesizer_prompt()
    # The two-axis framing: minimal count, complete coverage.
    assert "FEWEST slices" in p
    assert "cover EVERY spec requirement" in p
    # Tests must be folded into the slice, not dropped or split off.
    assert "test file" in p.lower()
    assert "target_files" in p
    assert "never drop a requirement" in p.lower()
    assert "every requirement is covered by a slice" in p.lower()


def test_summarise_max_completion_tokens_config_default():
    from spine.config import SpineConfig

    assert SpineConfig().summarise_max_completion_tokens == 4096


def test_findings_structured_model_caps_completion_tokens():
    """The ResearchFindings summarisation call is rebuilt with a tight token
    cap so a degenerate generation fails fast instead of running to the 16K
    window cap (trace 019e77fe: 207s LengthFinishReasonError)."""
    from spine.agents.exploration_agents import _findings_structured_model

    captured = {}

    class FakeModel:
        max_completion_tokens = 16000
        max_tokens = None

        def model_copy(self, *, update):
            captured.update(update)
            new = FakeModel()
            for k, v in update.items():
                setattr(new, k, v)
            return new

        def with_structured_output(self, schema):
            return ("structured", self.max_completion_tokens)

    out = _findings_structured_model(FakeModel())
    assert out == ("structured", 4096)
    assert captured == {"max_completion_tokens": 4096}


def test_findings_structured_model_none_passthrough():
    from spine.agents.exploration_agents import _findings_structured_model

    assert _findings_structured_model(None) is None


# ── Follow-ups from the 019e784c IMPLEMENT-reaching trace ───────────────────


def test_bool_or_reducer_composes_parallel_true_writes():
    """slices_dispatched / implementation_files_written are written True by
    every parallel slice-implementer Send branch in one super-step; the OR
    reducer composes them instead of crashing (InvalidUpdateError)."""
    from spine.workflow.subgraph_state import _bool_or

    assert _bool_or(None, True) is True
    assert _bool_or(True, None) is True
    assert _bool_or(False, True) is True
    assert _bool_or(False, False) is False
    # Sequential application over a super-step's updates (LangGraph folds them).
    from functools import reduce

    assert reduce(_bool_or, [True, True], False) is True


def test_implement_invariant_bools_have_reducer_annotation():
    """The two IMPLEMENT completion-invariant bools must carry a reducer, or
    concurrent Send writes crash (trace 019e784c). Resolve the PEP-563 string
    annotations the way LangGraph does (get_type_hints + include_extras)."""
    import typing

    from spine.workflow.subgraph_state import ImplementSubgraphState, _bool_or

    hints = typing.get_type_hints(ImplementSubgraphState, include_extras=True)
    for field in ("slices_dispatched", "implementation_files_written"):
        meta = getattr(hints[field], "__metadata__", ())
        assert _bool_or in meta, f"{field} is missing the _bool_or reducer"


def test_search_codebase_tolerates_invalid_glob_pattern():
    """A bare '**' (or other malformed) file_pattern used to raise a raw
    ValueError that crashed the whole search (trace 019e784c). It must now be
    skipped/handled, falling back to a full walk, returning a result string."""
    from spine.agents.plan_tools import SearchCodebaseTool

    tool = SearchCodebaseTool(workspace_root=".")
    # Should not raise.
    out = tool._run(queries=["def "], file_patterns=["**"])
    assert isinstance(out, str) and out
