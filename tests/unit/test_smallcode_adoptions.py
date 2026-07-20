"""Tests for the smallcode-inspired small-model adaptations.

Covers four additions, all designed to be no-ops unless explicitly configured:

1. Per-model behavioural profiles (``model_profiles``)
2. Adaptive failure-driven model escalation (``providers.phases.*.escalation``)
3. Forgiving multi-format tool-call parsing (``ToolCallNormalizer``)
4. LLM error diagnosis on command failure (``ExecuteErrorDiagnoser``)
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from spine.config import SpineConfig


def _provider(name: str, model: str, **extra) -> dict:
    return {"name": name, "model": model, "enabled": True, **extra}


# ── 1. Per-model behavioural profiles ─────────────────────────────────────────


class TestModelProfiles:
    def _cfg(self) -> SpineConfig:
        return SpineConfig(
            providers={
                "llm": [_provider("local", "openai:qwen3.6", base_url="http://x/v1",
                                  api_key="k")],
                "phases": {"implement": {"provider": "local", "temperature": 0.3}},
            },
            model_profiles={
                "openai:qwen3.6": {"context_window": 32768, "tool_format": "hermes",
                                   "temperature": 0.9},
            },
        )

    def test_profile_fills_defaults(self) -> None:
        cfg = self._cfg().resolve_provider_config("implement")
        assert cfg["context_window"] == 32768
        assert cfg["tool_format"] == "hermes"

    def test_explicit_phase_setting_wins_over_profile(self) -> None:
        # Phase temperature 0.3 must beat profile temperature 0.9.
        assert self._cfg().resolve_provider_config("implement")["temperature"] == 0.3

    def test_no_op_when_profiles_unset(self) -> None:
        cfg = SpineConfig(
            providers={"llm": [_provider("local", "openai:qwen3.6")],
                       "phases": {"implement": {"provider": "local"}}},
        )
        assert "context_window" not in cfg.resolve_provider_config("implement")

    def test_normalized_model_name_key(self) -> None:
        cfg = SpineConfig(
            providers={"llm": [_provider("or", "openrouter:deepseek/deepseek-v4-pro:free")]},
            model_profiles={"deepseek/deepseek-v4-pro": {"context_window": 128000}},
        )
        assert cfg.resolve_provider_config()["context_window"] == 128000


# ── 2. Adaptive failure-driven escalation ─────────────────────────────────────


class TestEscalation:
    def _cfg(self) -> SpineConfig:
        return SpineConfig(
            providers={
                "llm": [
                    _provider("local", "openai:qwen3.6", base_url="http://l/v1", api_key="k"),
                    _provider("flash", "openrouter:deepseek/deepseek-v4-flash"),
                    _provider("pro", "openrouter:deepseek/deepseek-v4-pro"),
                ],
                "phases": {
                    "implement": {
                        "provider": "local", "temperature": 0.6,
                        "escalation": [
                            {"provider": "flash"},
                            {"provider": "pro", "temperature": 0.2},
                        ],
                    },
                    "implement/subagents/slice-implementer": {"provider": "local"},
                },
            },
        )

    def test_resolve_model_ladder(self) -> None:
        c = self._cfg()
        assert c.resolve_model("implement", 0) == "openai:qwen3.6"
        assert c.resolve_model("implement", 1) == "openrouter:deepseek/deepseek-v4-flash"
        assert c.resolve_model("implement", 2) == "openrouter:deepseek/deepseek-v4-pro"

    def test_level_clamps_to_strongest_entry(self) -> None:
        assert self._cfg().resolve_model("implement", 9) == "openrouter:deepseek/deepseek-v4-pro"

    def test_subagent_inherits_phase_ladder(self) -> None:
        c = self._cfg()
        assert (
            c.resolve_model("implement/subagents/slice-implementer", 1)
            == "openrouter:deepseek/deepseek-v4-flash"
        )

    def test_level_0_byte_identical(self) -> None:
        c = self._cfg()
        assert c.resolve_model("implement") == c.resolve_model("implement", 0)

    def test_no_ladder_is_graceful(self) -> None:
        c = SpineConfig(
            providers={"llm": [_provider("local", "openai:qwen3.6")],
                       "phases": {"verify": {"provider": "local"}}},
        )
        assert c.resolve_model("verify", 5) == c.resolve_model("verify", 0)

    def test_provider_config_swaps_base_and_carries_phase_tuning(self) -> None:
        cfg1 = self._cfg().resolve_provider_config("implement", 1)
        assert cfg1["model"] == "openrouter:deepseek/deepseek-v4-flash"
        # Phase temperature 0.6 carries onto the escalated model.
        assert cfg1["temperature"] == 0.6

    def test_provider_config_entry_keys_win(self) -> None:
        cfg2 = self._cfg().resolve_provider_config("implement", 2)
        assert cfg2["model"] == "openrouter:deepseek/deepseek-v4-pro"
        # Escalation entry's own temperature 0.2 beats the phase's 0.6.
        assert cfg2["temperature"] == 0.2


class TestEscalationLevelForPhase:
    def test_int_retry_count(self) -> None:
        from spine.agents.helpers import escalation_level_for_phase
        from spine.models.enums import PhaseName

        assert escalation_level_for_phase({"retry_count": 2}, PhaseName.PLAN) == 2

    def test_dict_retry_count(self) -> None:
        from spine.agents.helpers import escalation_level_for_phase
        from spine.models.enums import PhaseName

        assert escalation_level_for_phase({"retry_count": {"plan": 3}}, PhaseName.PLAN) == 3

    def test_missing_and_junk(self) -> None:
        from spine.agents.helpers import escalation_level_for_phase
        from spine.models.enums import PhaseName

        assert escalation_level_for_phase({}, PhaseName.PLAN) == 0
        assert escalation_level_for_phase({"retry_count": None}, PhaseName.PLAN) == 0


# ── 3. Forgiving multi-format tool-call parsing ───────────────────────────────


class TestToolCallNormalizer:
    def test_hermes_envelope(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, stripped = extract_tool_calls(
            'Reading.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        )
        assert calls == [{"name": "read_file", "args": {"path": "a.py"}}]
        assert stripped == "Reading."

    def test_multiple_hermes(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, _ = extract_tool_calls(
            '<tool_call>{"name":"grep","arguments":{"q":"x"}}</tool_call>'
            '<tool_call>{"name":"read_file","arguments":{"path":"b"}}</tool_call>'
        )
        assert [c["name"] for c in calls] == ["grep", "read_file"]

    def test_fenced_json(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, _ = extract_tool_calls(
            'Call:\n```json\n{"name": "write_file", "arguments": {"path": "x"}}\n```'
        )
        assert calls[0]["name"] == "write_file"

    def test_bare_object(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, stripped = extract_tool_calls('{"name": "execute", "args": {"command": "ls"}}')
        assert calls == [{"name": "execute", "args": {"command": "ls"}}]
        assert stripped == ""

    def test_yaml_in_envelope(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, _ = extract_tool_calls(
            "<tool_call>\nname: read_file\narguments:\n  path: c.py\n</tool_call>"
        )
        assert calls == [{"name": "read_file", "args": {"path": "c.py"}}]

    def test_noarg_hermes_call(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        calls, _ = extract_tool_calls('<tool_call>{"name": "list_dir"}</tool_call>')
        assert calls == [{"name": "list_dir", "args": {}}]

    def test_prose_is_not_a_call(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        assert extract_tool_calls("The user name is John Smith with unknown args.")[0] == []

    def test_plain_code_block_is_not_a_call(self) -> None:
        from spine.agents.tool_call_normalizer import extract_tool_calls

        assert extract_tool_calls("```python\ndef foo():\n    return 1\n```")[0] == []

    def test_middleware_promotes_text_call(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        msg = AIMessage(
            content='<tool_call>{"name":"read_file","arguments":{"path":"z.py"}}</tool_call>'
        )
        resp = SimpleNamespace(result=[msg])
        ToolCallNormalizer()._normalize_response(resp)
        out = resp.result[0]
        assert out.tool_calls[0]["name"] == "read_file"
        assert out.tool_calls[0]["args"] == {"path": "z.py"}
        assert out.content == ""

    def test_middleware_leaves_native_calls_untouched(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        native = AIMessage(
            content="ok",
            tool_calls=[{"name": "read_file", "args": {"path": "a"}, "id": "1",
                         "type": "tool_call"}],
        )
        resp = SimpleNamespace(result=[native])
        ToolCallNormalizer()._normalize_response(resp)
        assert resp.result[0].content == "ok"
        assert len(resp.result[0].tool_calls) == 1


class TestNormalizerStructuredTermination:
    """A recovered structured-output call must TERMINATE the agent.

    create_agent parses structured output inside its base model handler,
    BEFORE wrap_model_call middleware — so a text-emitted call the normalizer
    promotes afterwards never becomes ``structured_response``, and a no-tool
    structured agent's model→model edge loops forever (trace 019f7d50: the
    plan critic spun 31 iterations re-emitting the same CriticReview call,
    including a discarded PASSED verdict on iteration 1). The normalizer now
    finishes the job itself: parse → structured_response + ToolMessage.
    """

    @staticmethod
    def _review_schema():
        from pydantic import BaseModel

        class Review(BaseModel):
            status: str
            reason: str

        return Review

    def test_bare_object_structured_call_sets_structured_response(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        msg = AIMessage(
            content='{"name": "Review", "arguments": '
            '{"status": "PASSED", "reason": "plan is sound"}}'
        )
        resp = SimpleNamespace(result=[msg], structured_response=None)
        ToolCallNormalizer(response_format=Review)._normalize_response(resp)

        out = resp.result[0]
        assert out.tool_calls[0]["name"] == "Review"
        assert isinstance(resp.structured_response, Review)
        assert resp.structured_response.status == "PASSED"
        # Terminating ToolMessage mirrors _handle_model_output's happy path.
        assert isinstance(resp.result[1], ToolMessage)
        assert resp.result[1].tool_call_id == out.tool_calls[0]["id"]

    def test_hermes_structured_call_also_completes(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        msg = AIMessage(
            content='<tool_call>{"name": "Review", "arguments": '
            '{"status": "ok", "reason": "r"}}</tool_call>'
        )
        resp = SimpleNamespace(result=[msg], structured_response=None)
        ToolCallNormalizer(response_format=Review)._normalize_response(resp)
        assert resp.structured_response is not None

    def test_schema_mismatch_appends_corrective_tool_message(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        msg = AIMessage(
            content='{"name": "Review", "arguments": {"wrong_field": 1}}'
        )
        resp = SimpleNamespace(result=[msg], structured_response=None)
        ToolCallNormalizer(response_format=Review)._normalize_response(resp)

        assert resp.structured_response is None
        # The retry is corrective, not a blind re-ask: an error ToolMessage
        # answers the promoted call so the next iteration sees what failed.
        assert isinstance(resp.result[1], ToolMessage)
        assert "Review" in str(resp.result[1].content)

    def test_non_structured_recovered_call_is_left_to_tool_node(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        msg = AIMessage(
            content='<tool_call>{"name": "read_file", "arguments": '
            '{"path": "a.py"}}</tool_call>'
        )
        resp = SimpleNamespace(result=[msg], structured_response=None)
        ToolCallNormalizer(response_format=Review)._normalize_response(resp)

        assert resp.result[0].tool_calls[0]["name"] == "read_file"
        assert resp.structured_response is None
        assert len(resp.result) == 1  # no synthetic ToolMessage

    def test_no_response_format_is_promotion_only(self) -> None:
        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        msg = AIMessage(
            content='{"name": "Review", "arguments": {"status": "ok", "reason": "r"}}'
        )
        resp = SimpleNamespace(result=[msg], structured_response=None)
        ToolCallNormalizer()._normalize_response(resp)
        assert resp.result[0].tool_calls[0]["name"] == "Review"
        assert resp.structured_response is None
        assert len(resp.result) == 1

    def test_provider_strategy_yields_no_bindings(self) -> None:
        from langchain.agents.structured_output import ProviderStrategy

        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        norm = ToolCallNormalizer(response_format=ProviderStrategy(schema=Review))
        assert norm._structured == {}

    def test_tool_strategy_binding_names_match_create_agent(self) -> None:
        from langchain.agents.structured_output import ToolStrategy

        from spine.agents.tool_call_normalizer import ToolCallNormalizer

        Review = self._review_schema()
        # Raw schema and explicit ToolStrategy must resolve to the same
        # tool name create_agent binds (the schema's name).
        assert set(ToolCallNormalizer(response_format=Review)._structured) == {"Review"}
        assert set(
            ToolCallNormalizer(response_format=ToolStrategy(schema=Review))._structured
        ) == {"Review"}


# ── 4. LLM error diagnosis on command failure ─────────────────────────────────


class TestExecuteDiagnoser:
    def _d(self):
        from spine.agents.execute_diagnoser import ExecuteErrorDiagnoser

        return ExecuteErrorDiagnoser()

    def test_detects_status_error(self) -> None:
        assert self._d()._is_execute_failure(
            "execute", ToolMessage(content="boom", tool_call_id="1", status="error")
        )

    def test_detects_nonzero_exit_code(self) -> None:
        assert self._d()._is_execute_failure(
            "execute",
            ToolMessage(content="[Command failed with exit code 127]", tool_call_id="1"),
        )

    def test_ignores_exit_zero(self) -> None:
        assert not self._d()._is_execute_failure(
            "execute", ToolMessage(content="done\nexit code 0", tool_call_id="1")
        )

    def test_ignores_clean_output(self) -> None:
        assert not self._d()._is_execute_failure(
            "execute", ToolMessage(content="hello", tool_call_id="1")
        )

    def test_ignores_non_execute_tool(self) -> None:
        assert not self._d()._is_execute_failure(
            "read_file", ToolMessage(content="x", tool_call_id="1", status="error")
        )

    def test_append_is_additive_and_preserves_status(self) -> None:
        tm = ToolMessage(content="bash: foo: not found", tool_call_id="1", name="execute")
        out = self._d()._append(tm, "foo is missing; install it")
        assert out.content == "bash: foo: not found\n[diagnosis] foo is missing; install it"
        assert out.status == "success"

    def test_disabled_by_default(self) -> None:
        assert SpineConfig().error_diagnosis is False
        assert SpineConfig().tool_call_normalize is False
