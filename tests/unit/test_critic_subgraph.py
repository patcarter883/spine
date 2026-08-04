"""Critic subgraph — mechanical literal-fix application."""

from __future__ import annotations

import json

import pytest

class TestLiteralFixes:
    """Critic-supplied exact corrections apply MECHANICALLY when the rework
    leaves the flagged text in place (run 019f8405: an identical two-line
    fix the critic spelled out verbatim went unapplied twice — stagnation
    park on a defect with a fully-specified remedy)."""

    PLAN = json.dumps({
        "architecture_overview": "x",
        "feature_slices": [{
            "id": "api-resources",
            "execution_requirements": (
                "Line 6: 'rainfalls' => RainfallResource::collection("
                "$this->whenLoaded('rainfalls'))"
            ),
            "acceptance_criteria": [
                "'rainfalls' => RainfallResource::collection($this->whenLoaded('rainfalls')) is used",
            ],
        }],
    })

    def test_applies_across_string_leaves_and_counts(self):
        from spine.workflow.subgraphs.critic_subgraph import apply_literal_fixes

        fixes = [{
            "find": "RainfallResource::collection($this->whenLoaded('rainfalls'))",
            "replace": "$this->whenLoaded('rainfalls', fn () => RainfallResource::collection($this->rainfalls))",
        }]
        patched, applied = apply_literal_fixes(self.PLAN, fixes)
        assert len(applied) == 1 and applied[0]["occurrences"] == 2
        doc = json.loads(patched)
        assert "fn () =>" in doc["feature_slices"][0]["execution_requirements"]
        assert "fn () =>" in doc["feature_slices"][0]["acceptance_criteria"][0]

    def test_noop_when_author_already_applied(self):
        from spine.workflow.subgraphs.critic_subgraph import apply_literal_fixes

        fixes = [{"find": "text that is not present anywhere", "replace": "y" * 20}]
        patched, applied = apply_literal_fixes(self.PLAN, fixes)
        assert applied == [] and patched == self.PLAN

    def test_short_find_ignored(self):
        from spine.workflow.subgraphs.critic_subgraph import apply_literal_fixes

        fixes = [{"find": "Line 6", "replace": "Line seven"}]
        patched, applied = apply_literal_fixes(self.PLAN, fixes)
        assert applied == []

    def test_invalid_json_passthrough(self):
        from spine.workflow.subgraphs.critic_subgraph import apply_literal_fixes

        text, applied = apply_literal_fixes("not json {", [
            {"find": "x" * 20, "replace": "y" * 20}])
        assert text == "not json {" and applied == []

    @pytest.mark.asyncio
    async def test_structural_check_applies_prior_round_fixes(self, tmp_path):
        from spine.workflow.subgraphs.critic_subgraph import _structural_check_node

        plan_dir = tmp_path / ".spine" / "artifacts" / "w1" / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(self.PLAN, encoding="utf-8")

        state = {
            "reviewed_phase": "plan",
            "work_id": "w1",
            "workspace_root": str(tmp_path),
            "plan_json": self.PLAN,
            "artifacts": {"plan": {"plan.json": self.PLAN}},
            "last_critic_review": {
                "literal_fixes": [{
                    "find": "RainfallResource::collection($this->whenLoaded('rainfalls'))",
                    "replace": "$this->whenLoaded('rainfalls', fn () => RainfallResource::collection($this->rainfalls))",
                }],
            },
        }
        out = await _structural_check_node(state, None)
        assert out.get("literal_fixes_applied")
        assert "fn () =>" in out["plan_json"]
        assert "fn () =>" in out["artifacts"]["plan"]["plan.json"]
        assert "fn () =>" in (plan_dir / "plan.json").read_text()

    @pytest.mark.asyncio
    async def test_structural_check_noop_without_fixes(self, tmp_path):
        from spine.workflow.subgraphs.critic_subgraph import _structural_check_node

        state = {
            "reviewed_phase": "plan",
            "work_id": "w1",
            "workspace_root": str(tmp_path),
            "plan_json": self.PLAN,
            "artifacts": {"plan": {"plan.json": self.PLAN}},
            "last_critic_review": {"literal_fixes": []},
        }
        out = await _structural_check_node(state, None)
        assert "plan_json" not in out
        assert "literal_fixes_applied" not in out
