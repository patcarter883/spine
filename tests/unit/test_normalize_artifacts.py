"""Tests for normalize_artifacts — handles both persisted artifact shapes.

The workflow dispatcher persists ``result["artifacts"]`` as a
``{phase: [names]}`` mapping, while onboarding persists a flat ``[names]``
list. The UI/CLI render paths must tolerate both without crashing
(regression for the ``'list' object has no attribute 'items'`` error on
onboarding work items).
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.ui.utils import normalize_artifacts


class TestNormalizeArtifacts:
    def test_dispatcher_mapping_shape(self):
        """{phase: [names]} → one labelled row per phase, names joined."""
        rows = normalize_artifacts({"specify": ["spec.md"], "plan": ["plan.md", "tasks.md"]})
        assert rows == [("specify", "spec.md"), ("plan", "plan.md, tasks.md")]

    def test_onboarding_flat_list_shape(self):
        """A flat [names] list → a single empty-label row (no .items() crash)."""
        rows = normalize_artifacts(["architecture.md", "conventions.md"])
        assert rows == [("", "architecture.md, conventions.md")]

    def test_mapping_with_scalar_value(self):
        """A non-list phase value is stringified rather than joined."""
        assert normalize_artifacts({"verify": "verification.md"}) == [
            ("verify", "verification.md")
        ]

    def test_unexpected_scalar_does_not_crash(self):
        assert normalize_artifacts("weird") == [("", "weird")]

    def test_empty_shapes(self):
        # Empty containers are falsy and guarded by callers, but must not raise.
        assert normalize_artifacts({}) == []
        assert normalize_artifacts([]) == [("", "")]
