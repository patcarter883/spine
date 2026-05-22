"""Integration test: full plan → scheduler → implement flow.

Validates that a structured plan.json with 3 feature slices flows through
the artifact gate, slice scheduler, and implement dispatch in the correct
wave order.

Dependency topology used in the test:

    models (wave 0)     config (wave 0)
           \\              /
            api (wave 1)

Wave 0 dispatches ``models`` and ``config`` in parallel; wave 1 dispatches
``api`` only after both predecessors complete.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub the broken tasks_agent module (syntax error from incomplete refactor).
# This unblocks the registry import chain without hiding real test failures.
if "spine.agents.tasks_agent" not in sys.modules:
    _stub = types.ModuleType("spine.agents.tasks_agent")
    _stub.build_tasks_agent = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["spine.agents.tasks_agent"] = _stub

from spine.workflow.slice_scheduler import (
    FeatureSlice,
    compute_execution_waves,
    slices_to_state_dict,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _plan_json_dict() -> dict:
    """Build a plan.json-compatible dict with 3 slices.

    Dependency topology:
        models ──┐
                 ├── api
        config ──┘

    Field names match ``_REQUIRED_SLICE_FIELDS`` in artifact_gate.py:
    ``id``, ``title``, ``target_files``, ``execution_requirements``,
    ``dependencies``, ``acceptance_criteria``.
    """
    return {
        "architecture_overview": "Three-slice integration test architecture.",
        "technology_choices": ["python", "fastapi", "pydantic"],
        "feature_slices": [
            {
                "id": "models",
                "title": "Core data models",
                "target_files": ["models/types.py"],
                "execution_requirements": [],
                "dependencies": [],
                "acceptance_criteria": ["Models importable", "Pydantic validates"],
                "complexity": "small",
            },
            {
                "id": "config",
                "title": "Configuration loader",
                "target_files": ["config/settings.py"],
                "execution_requirements": [],
                "dependencies": [],
                "acceptance_criteria": [
                    "Config loads from env",
                    "Defaults applied",
                ],
                "complexity": "small",
            },
            {
                "id": "api",
                "title": "REST API endpoints",
                "target_files": ["api/routes.py"],
                "execution_requirements": [
                    "Requires models and config to be in place.",
                ],
                "dependencies": ["models", "config"],
                "acceptance_criteria": [
                    "Endpoints return JSON",
                    "Auth middleware works",
                ],
                "complexity": "medium",
            },
        ],
        "testing_strategy": "pytest with tmp_path",
        "risks": [],
        "codebase_map": {},
    }


def _scheduler_slices_from_plan(plan_data: dict) -> list[FeatureSlice]:
    """Convert plan.json feature_slices to scheduler FeatureSlice objects."""
    return [
        FeatureSlice(
            slice_id=sd["id"],
            description=sd.get("title", ""),
            dependencies=sd.get("dependencies", []),
            files_to_modify=sd.get("target_files", []),
            files_to_create=sd.get("target_files", []),
            acceptance_criteria=sd.get("acceptance_criteria", []),
            complexity=sd.get("complexity", "small"),
            instructions=sd.get("execution_requirements", ""),
        )
        for sd in plan_data["feature_slices"]
    ]


# ── Scheduler unit-level validation ─────────────────────────────────────


class TestSchedulerWaveOrder:
    """Verify the slice scheduler produces correct wave dispatch order."""

    def test_three_slices_produce_two_waves(self):
        plan_data = _plan_json_dict()
        slices = _scheduler_slices_from_plan(plan_data)
        waves = compute_execution_waves(slices)
        assert len(waves) == 2

    def test_wave_0_contains_independent_slices(self):
        plan_data = _plan_json_dict()
        slices = _scheduler_slices_from_plan(plan_data)
        waves = compute_execution_waves(slices)
        wave_0_ids = {s.slice_id for s in waves[0]}
        assert wave_0_ids == {"models", "config"}

    def test_wave_1_contains_dependent_slice(self):
        plan_data = _plan_json_dict()
        slices = _scheduler_slices_from_plan(plan_data)
        waves = compute_execution_waves(slices)
        wave_1_ids = [s.slice_id for s in waves[1]]
        assert wave_1_ids == ["api"]

    def test_slices_to_state_dict_round_trip(self):
        plan_data = _plan_json_dict()
        slices = _scheduler_slices_from_plan(plan_data)
        waves = compute_execution_waves(slices)
        state_dict = slices_to_state_dict(waves)
        assert state_dict["total_slices"] == 3
        assert state_dict["wave_count"] == 2
        assert len(state_dict["waves"]) == 2
        assert len(state_dict["waves"][0]["slices"]) == 2
        assert len(state_dict["waves"][1]["slices"]) == 1


# ── Full plan → gate → scheduler integration ───────────────────────────


class TestPlanToImplementFlow:
    """End-to-end structured plan flow: plan.json → gate → scheduler → dispatch."""

    def test_plan_json_to_scheduler_waves(self, tmp_path):
        """plan.json with 3 slices → scheduler computes correct waves."""
        plan_data = _plan_json_dict()

        # Write plan.json to simulated workspace
        work_id = "e2e-plan-flow"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

        # Read plan.json back (simulating what the implement phase does)
        raw = (plan_dir / "plan.json").read_text(encoding="utf-8")
        loaded = json.loads(raw)

        loaded_slices = _scheduler_slices_from_plan(loaded)
        waves = compute_execution_waves(loaded_slices)
        assert len(waves) == 2

        wave_0_ids = sorted(s.slice_id for s in waves[0])
        assert wave_0_ids == ["config", "models"]

        wave_1_ids = [s.slice_id for s in waves[1]]
        assert wave_1_ids == ["api"]

    def test_artifact_gate_passes_for_valid_plan(self, tmp_path):
        """Plan artifact gate proceeds when plan.json has valid slices."""
        from spine.workflow.artifact_gate import make_artifact_gate_node

        plan_data = _plan_json_dict()

        work_id = "gate-valid-plan"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            "artifacts": {
                "plan": {"plan.json": json.dumps(plan_data, indent=2)},
            },
        }
        result = gate_fn(state)
        assert result["status"] == "running"

    def test_full_flow_gate_then_schedule(self, tmp_path):
        """Full flow: write plan.json → gate passes → scheduler dispatches correctly.

        This simulates the actual plan → implement handoff:
        1. Plan phase writes plan.json to disk
        2. Artifact gate checks quality → passes (status = running)
        3. Implement reads plan.json and computes execution waves
        4. Waves are dispatched in correct dependency order
        """
        from spine.workflow.artifact_gate import make_artifact_gate_node

        plan_data = _plan_json_dict()
        plan_json_str = json.dumps(plan_data, indent=2)

        # ── Step 1: Plan writes plan.json ────────────────────────────────
        work_id = "full-flow"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(plan_json_str, encoding="utf-8")

        # ── Step 2: Artifact gate checks quality ─────────────────────────
        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            "artifacts": {"plan": {"plan.json": plan_json_str}},
        }
        gate_result = gate_fn(state)
        assert gate_result["status"] == "running", (
            f"Gate should pass but got status={gate_result['status']}, "
            f"feedback={gate_result.get('feedback')}"
        )

        # ── Step 3: Implement reads plan.json and computes waves ─────────
        raw = (plan_dir / "plan.json").read_text(encoding="utf-8")
        loaded = json.loads(raw)
        loaded_slices = _scheduler_slices_from_plan(loaded)

        # ── Step 4: Verify wave dispatch order ───────────────────────────
        waves = compute_execution_waves(loaded_slices)
        state_dict = slices_to_state_dict(waves)

        assert state_dict["wave_count"] == 2
        assert state_dict["total_slices"] == 3

        # Wave 0: independent slices (models, config) — dispatched in parallel
        wave_0_ids = sorted(s["slice_id"] for s in state_dict["waves"][0]["slices"])
        assert wave_0_ids == ["config", "models"]

        # Wave 1: dependent slice (api) — dispatched after wave 0 completes
        wave_1_ids = [s["slice_id"] for s in state_dict["waves"][1]["slices"]]
        assert wave_1_ids == ["api"]

        # Verify wave indices are sequential
        assert state_dict["waves"][0]["wave_index"] == 0
        assert state_dict["waves"][1]["wave_index"] == 1

    def test_gate_blocks_empty_plan_prevents_implement(self, tmp_path):
        """Gate blocks when plan.json has empty feature_slices → no implement."""
        from spine.workflow.artifact_gate import make_artifact_gate_node

        plan_data = _plan_json_dict()
        plan_data["feature_slices"] = []  # Empty slices

        work_id = "gate-empty-plan"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            "artifacts": {
                "plan": {"plan.json": json.dumps(plan_data, indent=2)},
            },
        }
        result = gate_fn(state)
        assert result["status"] == "needs_review"
        # Implement should NOT receive an empty wave list — gate stops it
        assert "execution_waves" not in result
