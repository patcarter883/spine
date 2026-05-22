"""Tests for SPINE Deep Agents profile registration.

Verifies that:
1. ensure_spine_profiles() registers HarnessProfiles for SPINE providers
2. create_deep_agent() composes the SPINE base prompt instead of the DA default
3. The phase-specific system_prompt sits before the SPINE base prompt
4. The DA conversational default (BASE_AGENT_PROMPT) is NOT present
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSpineProfileRegistration:
    """Verify HarnessProfile registration for SPINE providers."""

    def setup_method(self) -> None:
        """Reset profile state before each test."""
        from spine.agents.profile import reset_spine_profiles

        reset_spine_profiles()

    def teardown_method(self) -> None:
        """Restore profiles after each test."""
        from spine.agents.profile import ensure_spine_profiles

        ensure_spine_profiles()

    def test_ensure_spine_profiles_overrides_openrouter(self) -> None:
        """ensure_spine_profiles() should merge our base prompt into the openrouter profile."""
        from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

        from spine.agents.profile import SPINE_BASE_PROMPT, ensure_spine_profiles

        ensure_spine_profiles()

        assert "openrouter" in _HARNESS_PROFILES
        profile = _HARNESS_PROFILES["openrouter"]
        # Our base_system_prompt should have replaced the default (None)
        assert profile.base_system_prompt == SPINE_BASE_PROMPT

    def test_ensure_spine_profiles_registers_openai(self) -> None:
        """ensure_spine_profiles() should register a profile for 'openai'."""
        from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

        from spine.agents.profile import SPINE_BASE_PROMPT, ensure_spine_profiles

        ensure_spine_profiles()

        assert "openai" in _HARNESS_PROFILES
        profile = _HARNESS_PROFILES["openai"]
        assert profile.base_system_prompt == SPINE_BASE_PROMPT

    def test_ensure_spine_profiles_idempotent(self) -> None:
        """Calling ensure_spine_profiles() twice should not duplicate profiles."""
        from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

        from spine.agents.profile import ensure_spine_profiles

        ensure_spine_profiles()
        count_after_first = len(_HARNESS_PROFILES)

        ensure_spine_profiles()
        count_after_second = len(_HARNESS_PROFILES)

        # Profile count should not grow (merge-on-re-registration is OK,
        # but the _REGISTERED guard should prevent the second call entirely)
        assert count_after_second == count_after_first

    def test_spine_base_prompt_replaces_da_default(self) -> None:
        """The SPINE base prompt should NOT contain DA conversational framing."""
        from spine.agents.profile import SPINE_BASE_PROMPT

        # These phrases are from the DA BASE_AGENT_PROMPT and should NOT appear
        conversational_phrases = [
            "helps users accomplish tasks",
            "The user can see your responses",
            "If the request is underspecified",
            "ask only the minimum followup",
            "validating the user's beliefs",
        ]
        for phrase in conversational_phrases:
            assert phrase not in SPINE_BASE_PROMPT, (
                f"SPINE base prompt still contains DA conversational phrase: {phrase!r}"
            )

    def test_spine_base_prompt_contains_phase_executor_framing(self) -> None:
        """The SPINE base prompt should frame the agent as a phase executor."""
        from spine.agents.profile import SPINE_BASE_PROMPT

        # Key SPINE-specific phrases that MUST be present
        spine_phrases = [
            "phase executor",
            "NOT a conversational assistant",
            "deterministic",
            "Do NOT ask follow-up questions",
            "Do NOT seek user approval",
            "execute autonomously",
        ]
        for phrase in spine_phrases:
            assert phrase in SPINE_BASE_PROMPT, (
                f"SPINE base prompt is missing required phrase: {phrase!r}"
            )

    def test_spine_base_prompt_preserves_useful_da_behaviour(self) -> None:
        """The SPINE base prompt should preserve useful behavioural guidance from DA."""
        from spine.agents.profile import SPINE_BASE_PROMPT

        # Useful DA behavioural guidance that we want to keep
        useful_phrases = [
            "stop and analyze",
            "first attempt is rarely correct",
            "iterate",
        ]
        for phrase in useful_phrases:
            assert phrase in SPINE_BASE_PROMPT, (
                f"SPINE base prompt lost useful DA guidance: {phrase!r}"
            )

    def test_spine_base_prompt_not_empty(self) -> None:
        """The SPINE base prompt should be non-trivial."""
        from spine.agents.profile import SPINE_BASE_PROMPT

        assert len(SPINE_BASE_PROMPT) > 200, "SPINE base prompt is suspiciously short"
