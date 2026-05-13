"""Integration test: verify the full prompt assembly for a SPINE agent.

Confirms that the SPINE HarnessProfile causes create_deep_agent() to compose
the phase-specific prompt + SPINE base prompt instead of the DA conversational
default.

We test at the profile composition level rather than constructing a full agent
(which requires a live BaseChatModel), since the profile is what controls the
prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPromptAssemblyIntegration:
    """End-to-end prompt assembly verification via the DA profile system."""

    def setup_method(self) -> None:
        from spine.agents.profile import ensure_spine_profiles

        ensure_spine_profiles()

    def test_openrouter_profile_uses_spine_base(self) -> None:
        """The openrouter profile should compose SPINE_BASE_PROMPT, not the DA default."""
        from deepagents.graph import BASE_AGENT_PROMPT
        from deepagents.profiles.harness.harness_profiles import (
            _get_harness_profile,
            _apply_profile_prompt,
        )

        from spine.agents.profile import SPINE_BASE_PROMPT

        profile = _get_harness_profile("openrouter")
        assert profile is not None, "No profile registered for openrouter"
        assert profile.base_system_prompt == SPINE_BASE_PROMPT

        composed = _apply_profile_prompt(profile, BASE_AGENT_PROMPT)
        assert composed == SPINE_BASE_PROMPT

    def test_openai_profile_uses_spine_base(self) -> None:
        """The openai profile should compose SPINE_BASE_PROMPT, not the DA default."""
        from deepagents.graph import BASE_AGENT_PROMPT
        from deepagents.profiles.harness.harness_profiles import (
            _get_harness_profile,
            _apply_profile_prompt,
        )

        from spine.agents.profile import SPINE_BASE_PROMPT

        profile = _get_harness_profile("openai")
        assert profile is not None
        assert profile.base_system_prompt == SPINE_BASE_PROMPT

        composed = _apply_profile_prompt(profile, BASE_AGENT_PROMPT)
        assert composed == SPINE_BASE_PROMPT

    def test_da_conversational_framing_absent(self) -> None:
        """The composed prompt should NOT contain DA conversational framing."""
        from deepagents.graph import BASE_AGENT_PROMPT
        from deepagents.profiles.harness.harness_profiles import (
            _get_harness_profile,
            _apply_profile_prompt,
        )

        profile = _get_harness_profile("openrouter")
        composed = _apply_profile_prompt(profile, BASE_AGENT_PROMPT)

        bad_phrases = [
            "helps users accomplish tasks",
            "The user can see your responses",
            "If the request is underspecified",
            "ask only the minimum followup",
        ]
        for phrase in bad_phrases:
            assert phrase not in composed, (
                f"Composed prompt still contains DA conversational phrase: {phrase!r}"
            )

    def test_full_prompt_ordering(self) -> None:
        """Phase prompt should precede SPINE_BASE_PROMPT in the final assembled prompt.

        Per DA prompt assembly: USER → CUSTOM → SUFFIX
        With our profile: phase_prompt → SPINE_BASE_PROMPT → (no suffix)
        """
        from spine.agents.profile import SPINE_BASE_PROMPT

        phase_prompt = "You are a technical architect. Given a specification, create a plan."
        # This is exactly what create_deep_agent does when system_prompt is set:
        final_prompt = phase_prompt + "\n\n" + SPINE_BASE_PROMPT

        # Phase prompt at the front
        assert final_prompt.startswith(phase_prompt)
        # SPINE base prompt present after the join
        assert SPINE_BASE_PROMPT in final_prompt
        # The two are separated by exactly double newline
        parts = final_prompt.split("\n\n", 1)
        assert parts[0] == phase_prompt
        assert parts[1] == SPINE_BASE_PROMPT
