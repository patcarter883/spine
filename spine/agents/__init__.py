"""SPINE agents package — one agent builder per file."""

from __future__ import annotations

# Activate SPINE HarnessProfiles on first import so that any subsequent
# create_deep_agent() call picks up our base prompt instead of the DA default.
from spine.agents.profile import ensure_spine_profiles

ensure_spine_profiles()
