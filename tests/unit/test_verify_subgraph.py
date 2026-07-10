"""Tests for the VERIFY phase subgraph.

Updated for Send API dispatch pattern — the verify subgraph now uses
``verify_router`` → ``Send("run_slice_verifier", ...)`` → ``aggregate_verification``
→ ``synthesize_verification`` → ``save_artifacts``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVerifySubgraphCompilation:
    """Tests that the verify subgraph compiles correctly."""

    def test_verify_subgraph_compiles(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        assert graph is not None

    def test_verify_subgraph_has_correct_nodes(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_slice_verifier" in nodes
        assert "aggregate_verification" in nodes
        assert "synthesize_verification" in nodes
        assert "save_artifacts" in nodes

    def test_verify_subgraph_edges(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        mermaid = graph.get_graph().draw_mermaid()
        assert "run_slice_verifier" in mermaid
        assert "aggregate_verification" in mermaid
        assert "synthesize_verification" in mermaid
        assert "save_artifacts" in mermaid
        assert "__end__" in mermaid


class TestVerifyRouter:
    """Tests for the _verify_router conditional edge function."""

    def test_verify_router_raises_on_missing_execution_waves(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from spine.exceptions import CriticalContractFailure

        with pytest.raises(CriticalContractFailure, match="execution_waves"):
            _verify_router(VerifySubgraphState(
                work_id="test", phase="verify", workspace_root=".",
            ))

    def test_verify_router_raises_on_empty_execution_waves(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from spine.exceptions import CriticalContractFailure

        with pytest.raises(CriticalContractFailure, match="execution_waves"):
            _verify_router(VerifySubgraphState(
                work_id="test", phase="verify", workspace_root=".",
                execution_waves=[],
            ))

    def test_verify_router_dispatches_sends(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from langgraph.types import Send

        result = _verify_router(VerifySubgraphState(
            work_id="test", phase="verify", workspace_root="/tmp",
            execution_waves=[[{"id": "s1", "title": "Slice 1"}]],
        ))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Send)
        # Sends now target the per-branch plan node, which dispatches
        # to run_slice_verifier dynamically via Command(goto=Send).
        # See the plan→do split.
        assert result[0].node == "plan_slice_verifier"
        assert result[0].arg["slice"]["id"] == "s1"


class TestPlanSliceVerifierCommand:
    """Regression for the InvalidUpdateError crash.

    Parallel ``Send("plan_slice_verifier", ...)`` branches must hand
    off to run_slice_verifier via per-branch ``Command(goto=Send(...))``,
    not by writing the directive to a shared LastValue channel.
    """

    @pytest.mark.asyncio
    async def test_returns_command_with_send_to_run_slice_verifier(self, monkeypatch):
        from langgraph.types import Command, Send
        from spine.agents.plan_do import SubagentDirective
        from spine.workflow.subgraphs.verify_subgraph import _plan_slice_verifier_node

        async def _fake_plan(*, state, config, phase_path, task_description, role_hint=""):
            return SubagentDirective(approach="check tests", acceptance=["tests green"])

        monkeypatch.setattr(
            "spine.workflow.subgraphs.verify_subgraph.run_plan_node", _fake_plan
        )

        state = {
            "phase": "verify",
            "work_id": "w1",
            "work_type": "task",
            "workspace_root": "/tmp",
            "slice": {"id": "s1", "title": "Slice 1"},
        }
        out = await _plan_slice_verifier_node(state, None)
        assert isinstance(out, Command)
        assert isinstance(out.goto, Send)
        assert out.goto.node == "run_slice_verifier"
        assert out.goto.arg["slice"]["id"] == "s1"
        assert "active_slice_directive" in out.goto.arg
        assert out.goto.arg["active_slice_directive"]["acceptance"] == ["tests green"]
        # Nothing problematic on update.
        assert "active_slice_directive" not in (out.update or {})


class TestRunSliceVerifier:
    """Tests for the _run_slice_verifier_node."""

    @pytest.mark.asyncio
    async def test_run_verifier_node_success(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_slice_verifier_node

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(
                    content='{"verdict": "VERIFIED", "checklist": [{"criterion":"test","passed":true,"detail":"ok"}], "gaps": [], "recommendations": []}'
                ),
            ]
        }

        with patch(
            "spine.agents.subagents.build_subagent_spec",
            return_value={"system_prompt": "test prompt", "tools": [], "response_format": None},
        ):
            with patch(
                "spine.agents.factory.build_phase_agent",
                return_value=mock_agent,
            ):
                state = {
                    "work_id": "test123",
                    "work_type": "task",
                    "phase": "verify",
                    "workspace_root": "/tmp",
                    "slice": {"id": "s1", "title": "Test Slice"},
                    "messages": [],
                }
                result = await _run_slice_verifier_node(state)
                assert "verification_results" in result
                assert len(result["verification_results"]) == 1
                assert result["verification_results"][0]["verdict"] == "VERIFIED"
                assert result["verification_results"][0]["slice_name"] == "s1"

    @pytest.mark.asyncio
    async def test_run_verifier_node_error_returns_not_verified(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_slice_verifier_node

        with patch(
            "spine.agents.subagents.build_subagent_spec",
            side_effect=RuntimeError("boom"),
        ):
            state = {
                "work_id": "test",
                "work_type": "task",
                "phase": "verify",
                "workspace_root": ".",
                "slice": {"id": "s-err"},
            }
            result = await _run_slice_verifier_node(state)
            assert "verification_results" in result
            assert result["verification_results"][0]["verdict"] == "NOT_VERIFIED"
            assert result["verification_results"][0]["slice_name"] == "s-err"


class TestSaveVerifyArtifacts:
    """Tests for the _save_verify_artifacts node within the subgraph."""

    @pytest.mark.asyncio
    async def test_save_artifacts_with_disk_files(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        art_dir = tmp_path / ".spine" / "artifacts" / "test123" / "verify"
        art_dir.mkdir(parents=True)
        (art_dir / "verification.md").write_text("VERIFIED all slices")
        (art_dir / "verification.json").write_text(
            '{"overall_status": "VERIFIED", "summary": "All good"}'
        )

        state = {
            "work_id": "test123",
            "workspace_root": str(tmp_path),
            "agent_response": "",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "success"
        assert "verification.md" in result["artifacts_output"]

    @pytest.mark.asyncio
    async def test_save_artifacts_not_verified_sets_needs_review(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        art_dir = tmp_path / ".spine" / "artifacts" / "test456" / "verify"
        art_dir.mkdir(parents=True)
        (art_dir / "verification.md").write_text("Some issues found, not complete")
        (art_dir / "verification.json").write_text(
            '{"overall_status": "FAILED", "summary": "Issues found"}'
        )

        state = {
            "work_id": "test456",
            "workspace_root": str(tmp_path),
            "agent_response": "",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_save_artifacts_falls_back_to_agent_response(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test789",
            "workspace_root": str(tmp_path),
            "agent_response": "VERIFIED everything looks good",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        # Without verification.json on disk, authoritative status defaults
        # to unverified; phase_status becomes needs_review.
        assert result["phase_status"] == "needs_review"
        assert "verification.md" in result["artifacts_output"]

    @pytest.mark.asyncio
    async def test_save_artifacts_preserves_error_status(self):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test000",
            "workspace_root": "/tmp",
            "agent_response": "",
            "phase_status": "error",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "error"

    @pytest.mark.asyncio
    async def test_save_artifacts_empty_response_fallback(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test111",
            "workspace_root": str(tmp_path),
            "agent_response": "  ",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "needs_review"
        assert "insufficient output" in result["artifacts_output"]["verification.md"].lower()


class TestVerifyStateAndResult:
    """Tests state mapping between parent and verify subgraph."""

    def test_verify_state_mapper_includes_execution_waves(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "abc",
            "work_type": "task",
            "description": "fix bug",
            "workspace_root": "/projects/spine",
            "retry_count": {"verify": 0},
            "feedback": [],
        }
        result = _verify_state_mapper(parent, None)
        assert result["work_id"] == "abc"
        assert result["work_type"] == "task"
        assert result["spec_path"] == ".spine/artifacts/abc/specify"
        assert result["plan_path"] == ".spine/artifacts/abc/plan"
        assert result["execution_waves"] == []

    def test_verify_state_mapper_passes_execution_waves(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "def",
            "work_type": "task",
            "description": "build feature",
            "workspace_root": "/projects/spine",
            "execution_waves": [[{"id": "s1"}, {"id": "s2"}]],
        }
        result = _verify_state_mapper(parent, None)
        assert result["execution_waves"] == [[{"id": "s1"}, {"id": "s2"}]]

    def test_verify_result_mapper_success(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {"verification.md": "VERIFIED"},
            "phase_status": "success",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "running"
        assert result["current_phase"] == "verify"
        assert result["phase_results"]["verify"]["status"] == "success"

    def test_verify_result_mapper_needs_gap_fix(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "needs_gap_fix"
        assert result["verify_attempts"] == 1
        assert any(f.get("status") == "needs_review" for f in result["feedback"])

    def test_verify_result_mapper_needs_review_after_max_gaps(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test", "verify_attempts": 2})
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "verify"
        assert any(f.get("status") == "needs_review" for f in result["feedback"])

    def test_verify_result_mapper_second_gap_fix(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test", "verify_attempts": 1})
        assert result["status"] == "needs_gap_fix"
        assert result["verify_attempts"] == 2

    def test_verify_result_mapper_error(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "error",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "failed"

class TestTargetSourcePreload:
    """The verifier starts grounded on pre-loaded target source (trace 019f10bf)."""

    def test_preload_renders_line_numbered_source(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _target_source_block

        (tmp_path / "a.py").write_text("import os\nx = 1\n", encoding="utf-8")
        block = _target_source_block(str(tmp_path), ["a.py"])
        assert "<target_source>" in block
        assert "1| import os" in block
        assert "2| x = 1" in block
        assert "do NOT re-read" in block

    def test_preload_truncates_long_file_with_pointer(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import (
            _PRELOAD_MAX_LINES_PER_FILE,
            _target_source_block,
        )

        n = _PRELOAD_MAX_LINES_PER_FILE + 50
        (tmp_path / "big.py").write_text(
            "\n".join(f"x{i} = {i}" for i in range(n)), encoding="utf-8"
        )
        block = _target_source_block(str(tmp_path), ["big.py"])
        assert "more lines" in block
        assert "read_file 'big.py'" in block

    def test_preload_empty_when_no_targets(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _target_source_block

        assert _target_source_block(str(tmp_path), []) == ""
        assert _target_source_block(str(tmp_path), None) == ""

    def test_preload_skips_missing_files(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _target_source_block

        assert _target_source_block(str(tmp_path), ["ghost.py"]) == ""


class TestAutomatedChecksRunTests:
    """Slices that author tests must have those tests EXECUTED as evidence.

    Regression (runs ce6f887d / 545264cc): the evidence-only judge cannot
    ground "pytest exits 0" criteria by reading source, and collection
    errors (duplicate defs, nonexistent fixtures) are invisible to it —
    broken tests were landed as VERIFIED.
    """

    def test_passing_test_file_is_executed_and_reported(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_ok.py").write_text(
            "def test_truth():\n    assert True\n", encoding="utf-8"
        )
        out, failures = _automated_checks(str(tmp_path), ["tests/test_ok.py"])
        assert "$ pytest -q tests/test_ok.py" in out
        assert "OK — all tests pass." in out
        assert failures == []

    def test_failing_test_file_output_is_evidence(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_bad.py").write_text(
            "def test_lie():\n    assert False\n", encoding="utf-8"
        )
        out, failures = _automated_checks(str(tmp_path), ["tests/test_bad.py"])
        assert "$ pytest -q tests/test_bad.py" in out
        assert "OK — all tests pass." not in out
        assert "test_lie" in out  # the failure detail reaches the judge
        # ...and the failure is reported structurally so the caller can
        # OVERRIDE a judge that verifies over it (run e95c1bc4).
        assert failures and "pytest failed" in failures[0]

    def test_non_test_files_do_not_trigger_pytest(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        out, failures = _automated_checks(str(tmp_path), ["mod.py"])
        assert "pytest" not in out
        assert failures == []

    def test_pytest_evidence_survives_verbose_ruff_output(self, tmp_path):
        """One noisy check must not starve the others (run 0eabad7d: ruff's
        code-frame output ate the whole budget and the pytest section
        carrying the decisive ModuleNotFoundError was truncated away)."""
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        # A file that is syntactically valid, drowning in lint findings, and
        # broken at import time — the exact 0eabad7d shape.
        lines = ["import nonexistent_module_xyz_abc\n"]
        lines += [f"import os as unused_alias_{i}\n" for i in range(200)]
        lines += ["def test_x():\n", "    assert True\n"]
        (tests_dir / "test_noisy.py").write_text("".join(lines), encoding="utf-8")
        out, _failures = _automated_checks(str(tmp_path), ["tests/test_noisy.py"])
        assert "$ pytest -q tests/test_noisy.py" in out
        assert "nonexistent_module_xyz_abc" in out.split("$ pytest")[1]


class TestReconcileVerdict:
    """The judge's verdict field is derived from ground truth, not trusted —
    it has been wrong in BOTH directions (e95c1bc4: VERIFIED over a pytest
    ImportError; 28b62d1e: NOT_VERIFIED with an all-passed checklist, zero
    gaps, and green checks — a contentless park)."""

    def _result(self, verdict, checklist, gaps=None):
        return {
            "slice_name": "s",
            "verdict": verdict,
            "checklist": checklist,
            "gaps": gaps or [],
        }

    def test_hard_failure_forces_not_verified(self):
        from spine.workflow.subgraphs.verify_subgraph import _reconcile_verdict

        r = self._result("VERIFIED", [{"criterion": "c", "passed": True, "detail": ""}])
        _reconcile_verdict(r, ["pytest failed (exit 2): ImportError"], "w", "s")
        assert r["verdict"] == "NOT_VERIFIED"
        assert any("pytest failed" in g for g in r["gaps"])

    def test_all_passed_checklist_forces_verified(self):
        from spine.workflow.subgraphs.verify_subgraph import _reconcile_verdict

        r = self._result(
            "NOT_VERIFIED",
            [{"criterion": "c1", "passed": True, "detail": ""},
             {"criterion": "c2", "passed": True, "detail": ""}],
        )
        _reconcile_verdict(r, [], "w", "s")
        assert r["verdict"] == "VERIFIED"

    def test_failed_item_keeps_not_verified(self):
        from spine.workflow.subgraphs.verify_subgraph import _reconcile_verdict

        r = self._result(
            "NOT_VERIFIED",
            [{"criterion": "c1", "passed": True, "detail": ""},
             {"criterion": "c2", "passed": False, "detail": "missing"}],
            gaps=["c2 missing"],
        )
        _reconcile_verdict(r, [], "w", "s")
        assert r["verdict"] == "NOT_VERIFIED"

    def test_gaps_block_the_upgrade(self):
        # All-passed checklist but gaps present: contradictory, keep the
        # failure verdict — the gaps at least give the loop something to do.
        from spine.workflow.subgraphs.verify_subgraph import _reconcile_verdict

        r = self._result(
            "NOT_VERIFIED",
            [{"criterion": "c1", "passed": True, "detail": ""}],
            gaps=["something is off"],
        )
        _reconcile_verdict(r, [], "w", "s")
        assert r["verdict"] == "NOT_VERIFIED"

    def test_empty_checklist_never_upgrades(self):
        from spine.workflow.subgraphs.verify_subgraph import _reconcile_verdict

        r = self._result("NOT_VERIFIED", [])
        _reconcile_verdict(r, [], "w", "s")
        assert r["verdict"] == "NOT_VERIFIED"


class TestCustomVerifyChecks:
    """spine-gate.yaml verify_checks give non-Python stacks executed
    evidence — PHP through the Sail stack for the agripath clone, and
    TypeScript next. Without real check output the judge is
    evidence-starved for those slices and the pre-2a2d9a2 failure modes
    (verifying broken files, contentless parks) return."""

    def _gate(self, tmp_path, monkeypatch, specs):
        import yaml

        (tmp_path / "spine-gate.yaml").write_text(
            yaml.safe_dump({"verify_checks": specs}), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

    def test_matching_files_run_the_command(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        ws = tmp_path / "ws"
        (ws / "app").mkdir(parents=True)
        (ws / "app" / "Thing.php").write_text("<?php\n", encoding="utf-8")
        self._gate(tmp_path, monkeypatch, [
            {"name": "php_lint", "files": ["*.php"], "command": "echo linted {files}"},
        ])
        block, failures = _automated_checks(str(ws), ["app/Thing.php"])
        assert "$ [php_lint]" in block
        assert "linted" in block and "app/Thing.php" in block
        assert failures == []

    def test_hard_failure_is_reported(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "x.php").write_text("<?php\n", encoding="utf-8")
        self._gate(tmp_path, monkeypatch, [
            {"name": "pest", "files": ["*.php"],
             "command": "echo 1 test failed; false"},
        ])
        block, failures = _automated_checks(str(ws), ["x.php"])
        assert failures and "pest failed" in failures[0]
        assert "1 test failed" in failures[0]

    def test_advisory_failure_is_shown_but_not_hard(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "x.php").write_text("<?php\n", encoding="utf-8")
        self._gate(tmp_path, monkeypatch, [
            {"name": "style", "files": ["*.php"], "hard": False,
             "command": "echo style nit; false"},
        ])
        block, failures = _automated_checks(str(ws), ["x.php"])
        assert "style nit" in block
        assert failures == []

    def test_failing_check_keeps_exception_head_and_summary_tail(
        self, tmp_path, monkeypatch
    ):
        """Head+tail clipping: the exception line prints near the HEAD of a
        pest failure and the summary at the TAIL; tail-only clipping erased
        the decisive SQLSTATE line (probe 23, run f788042e)."""
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "x.php").write_text("<?php\n", encoding="utf-8")
        self._gate(tmp_path, monkeypatch, [
            {"name": "pest", "files": ["*.php"],
             "command": (
                 "echo 'SQLSTATE[23502]: Not null violation: column x'; "
                 "for i in $(seq 400); do echo \"  at vendor/frame_$i.php:10\"; done; "
                 "echo 'Tests: 1 failed'; false"
             )},
        ])
        block, failures = _automated_checks(str(ws), ["x.php"])
        assert "SQLSTATE[23502]" in block          # head survived
        assert "Tests: 1 failed" in block          # tail survived
        assert "[middle truncated]" in block
        assert failures and "SQLSTATE[23502]" in failures[0]

    def test_unmatched_patterns_skip_the_command(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "x.ts").write_text("export {}\n", encoding="utf-8")
        self._gate(tmp_path, monkeypatch, [
            {"name": "pest", "files": ["*.php"], "command": "echo nope"},
        ])
        block, failures = _automated_checks(str(ws), ["x.ts"])
        assert "nope" not in block
        assert failures == []

    def test_no_gate_file_means_no_custom_checks(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)  # no spine-gate.yaml here
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "x.php").write_text("<?php\n", encoding="utf-8")
        block, failures = _automated_checks(str(ws), ["x.php"])
        assert block == "" and failures == []


class TestPsr4Evidence:
    """PSR-4 ground truth as judge evidence (probe 9: the judge disputed a
    compliant namespace, demanding an invented one)."""

    def _ws(self, tmp_path):
        import json as _json

        (tmp_path / "composer.json").write_text(_json.dumps({
            "autoload": {"psr-4": {"Database\\Factories\\": "database/factories/"}},
        }), encoding="utf-8")
        d = tmp_path / "database" / "factories"
        d.mkdir(parents=True)
        return d

    def test_compliant_namespace_shows_ok(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)  # no spine-gate.yaml → no custom checks
        d = self._ws(tmp_path)
        (d / "XFactory.php").write_text(
            "<?php\n\nnamespace Database\\Factories;\n\nclass XFactory {}\n",
            encoding="utf-8",
        )
        block, failures = _automated_checks(
            str(tmp_path), ["database/factories/XFactory.php"]
        )
        assert "$ [psr4]" in block
        assert "OK database/factories/XFactory.php" in block
        assert failures == []

    def test_mismatch_is_a_hard_failure(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)
        d = self._ws(tmp_path)
        (d / "YFactory.php").write_text(
            "<?php\n\nnamespace App\\Wrong;\n\nclass YFactory {}\n",
            encoding="utf-8",
        )
        block, failures = _automated_checks(
            str(tmp_path), ["database/factories/YFactory.php"]
        )
        assert "MISMATCH" in block
        assert failures and "PSR-4 namespace mismatch" in failures[0]

    def test_classless_pest_file_is_not_a_mismatch(self, tmp_path, monkeypatch):
        """Procedural Pest files declare no autoloadable type, so a missing
        namespace is NOT a PSR-4 violation (probe 20/8eaa5887: a correct
        test file hard-failed on declared '(none)')."""
        import json as _json

        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)
        (tmp_path / "composer.json").write_text(_json.dumps({
            "autoload-dev": {"psr-4": {"Tests\\": "tests/"}},
        }), encoding="utf-8")
        d = tmp_path / "tests" / "Unit"
        d.mkdir(parents=True)
        (d / "ZTest.php").write_text(
            "<?php\n\ndeclare(strict_types=1);\n\n"
            "it('works', function () { expect(true)->toBeTrue(); });\n",
            encoding="utf-8",
        )
        block, failures = _automated_checks(str(tmp_path), ["tests/Unit/ZTest.php"])
        assert "PSR-4 not applicable" in block
        assert failures == []

    def test_classful_file_missing_namespace_still_fails(self, tmp_path, monkeypatch):
        """A file that DOES declare a class must still declare the namespace."""
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)
        d = self._ws(tmp_path)
        (d / "WFactory.php").write_text(
            "<?php\n\nclass WFactory {}\n", encoding="utf-8",
        )
        block, failures = _automated_checks(
            str(tmp_path), ["database/factories/WFactory.php"]
        )
        assert failures and "PSR-4 namespace mismatch" in failures[0]


class TestFeatureWideTestEvidence:
    """Sibling slices' tests run as ADVISORY evidence (probe 12/ed2c9f85:
    the migration slice verified clean while its TypeError only surfaced in
    the test slice's run — the gap landed on the wrong slice)."""

    def test_sibling_test_failure_is_advisory_not_hard(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)  # no gate file → no custom checks
        ws = tmp_path / "ws"
        (ws / "app").mkdir(parents=True)
        (ws / "tests").mkdir()
        (ws / "app" / "mod.py").write_text("BROKEN = 1/0\n", encoding="utf-8")
        (ws / "tests" / "test_feature.py").write_text(
            "import sys; sys.path.insert(0, 'app')\n"
            "def test_mod():\n    import mod\n    assert mod.BROKEN\n",
            encoding="utf-8",
        )
        # Verifying the NON-test slice (app/mod.py): the sibling test run
        # shows the failure, but only as advisory evidence.
        block, failures = _automated_checks(
            str(ws), ["app/mod.py"],
            feature_test_files=["tests/test_feature.py"],
        )
        assert "feature tests" in block and "ADVISORY" in block
        assert "ZeroDivisionError" in block
        assert not any("feature" in f for f in failures)

    def test_own_test_files_are_not_double_run(self, tmp_path, monkeypatch):
        from spine.workflow.subgraphs.verify_subgraph import _automated_checks

        monkeypatch.chdir(tmp_path)
        ws = tmp_path / "ws"
        (ws / "tests").mkdir(parents=True)
        (ws / "tests" / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        block, failures = _automated_checks(
            str(ws), ["tests/test_ok.py"],
            feature_test_files=["tests/test_ok.py"],
        )
        # Own file runs as the HARD pytest section only — no advisory dup.
        assert block.count("test_ok.py") >= 1
        assert "ADVISORY" not in block
        assert failures == []
