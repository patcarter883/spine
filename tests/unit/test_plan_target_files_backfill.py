"""Deterministic recovery of empty plan-slice ``target_files`` (trace 019ec997).

The PLAN synthesizer routinely leaves ``target_files`` empty while naming those
same files in its ``execution_requirements`` prose, which the PLAN critic
rejects — spinning the phase through rework until a human cancels. The backfill
mines the paths the model already wrote so the on-disk plan.json the critic
loads is already populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.workflow.subgraphs.exploration_subgraph import (
    _backfill_target_files,
    _derive_target_files,
)


class TestDeriveTargetFiles:
    def test_extracts_known_findings_path_from_prose(self):
        slice_dict = {
            "id": "api",
            "title": "API methods",
            "execution_requirements": (
                "Add four methods to the UIApi class in spine/ui_api/api.py."
            ),
            "acceptance_criteria": ["methods persist to config"],
        }
        derived = _derive_target_files(slice_dict, ["spine/ui_api/api.py"])
        assert derived == ["spine/ui_api/api.py"]

    def test_extracts_pathish_token_not_in_findings(self):
        # A file the slice will *create* won't be in the research file_map yet.
        slice_dict = {
            "id": "ui",
            "title": "New page",
            "execution_requirements": "Create spine/ui/_pages/new_view.py with render().",
            "acceptance_criteria": ["page renders"],
        }
        derived = _derive_target_files(slice_dict, [])
        assert derived == ["spine/ui/_pages/new_view.py"]

    def test_known_path_preferred_then_pathish_deduped(self):
        slice_dict = {
            "id": "mix",
            "title": "t",
            "execution_requirements": (
                "Edit spine/ui_api/api.py and spine/ui/_pages/config_view.py; "
                "also touch spine/ui_api/api.py again."
            ),
            "acceptance_criteria": ["x"],
        }
        derived = _derive_target_files(slice_dict, ["spine/ui_api/api.py"])
        # Known path first, then the path-shaped token, each exactly once.
        assert derived == ["spine/ui_api/api.py", "spine/ui/_pages/config_view.py"]

    def test_no_paths_returns_empty(self):
        slice_dict = {
            "id": "vague",
            "title": "Refactor",
            "execution_requirements": "Refactor the thing generally for clarity.",
            "acceptance_criteria": ["cleaner"],
        }
        assert _derive_target_files(slice_dict, []) == []

    def test_does_not_grab_sentence_words_or_class_names(self):
        slice_dict = {
            "id": "prose",
            "title": "Update UIApi.set_phase_provider",
            "execution_requirements": "Update the UIApi class method. No file path here.",
            "acceptance_criteria": ["works"],
        }
        # "UIApi.set_phase_provider" has a dot but no dir/ segment → not a path.
        assert _derive_target_files(slice_dict, []) == []


class TestBackfillTargetFiles:
    def test_backfills_only_empty_slices(self):
        plan = {
            "feature_slices": [
                {
                    "id": "a",
                    "title": "A",
                    "execution_requirements": "Methods in spine/ui_api/api.py.",
                    "target_files": [],
                    "acceptance_criteria": ["x"],
                },
                {
                    "id": "b",
                    "title": "B",
                    "execution_requirements": "Edit spine/ui/_pages/config_view.py.",
                    "target_files": ["already/set.py"],
                    "acceptance_criteria": ["y"],
                },
            ]
        }
        n = _backfill_target_files(plan, ["spine/ui_api/api.py"], "wid")
        assert n == 1
        assert plan["feature_slices"][0]["target_files"] == ["spine/ui_api/api.py"]
        # Pre-populated slice is left untouched.
        assert plan["feature_slices"][1]["target_files"] == ["already/set.py"]

    def test_leaves_pathless_slice_empty(self):
        plan = {
            "feature_slices": [
                {
                    "id": "c",
                    "title": "Vague",
                    "execution_requirements": "Do the work.",
                    "target_files": [],
                    "acceptance_criteria": ["z"],
                }
            ]
        }
        n = _backfill_target_files(plan, [], "wid")
        assert n == 0
        assert plan["feature_slices"][0]["target_files"] == []

    def test_tolerates_missing_or_malformed_slices(self):
        assert _backfill_target_files({}, [], "wid") == 0
        assert _backfill_target_files({"feature_slices": "nope"}, [], "wid") == 0
        # Non-dict entries are skipped, dict entries still processed.
        plan = {
            "feature_slices": [
                "garbage",
                {
                    "id": "d",
                    "title": "D",
                    "execution_requirements": "Patch spine/config.py.",
                    "target_files": [],
                    "acceptance_criteria": ["w"],
                },
            ]
        }
        assert _backfill_target_files(plan, [], "wid") == 1
        assert plan["feature_slices"][1]["target_files"] == ["spine/config.py"]
