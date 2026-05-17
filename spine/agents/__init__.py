"""SPINE agents package — one agent builder per file.

Context engineering modules:
- ``context`` — SpineContext dataclass for per-run runtime context
- ``artifacts`` — Materialize artifacts to disk, reference by path
- ``factory`` — Shared build_phase_agent() with memory, skills, context
- ``backend`` — Single backend factory with CompositeBackend + cross-work memory
- ``skills_resolver`` — Locate skill directories for progressive disclosure
- ``profile`` — SPINE HarnessProfile (replaces DA BASE_AGENT_PROMPT)
"""

from __future__ import annotations

# Activate SPINE HarnessProfiles on first import so that any subsequent
# create_deep_agent() call picks up our base prompt instead of the DA default.
from spine.agents.profile import ensure_spine_profiles

ensure_spine_profiles()