"""Shared prompt fragments reused verbatim across multiple agent prompts.

These blocks were previously hand-copied into 2–4 prompt builders each, which
let them drift out of sync. Defining them once here keeps every consumer
consistent and means a wording change lands in one place. Import the constant
and splice it into the prompt rather than re-typing the text.

Keep these fragments self-contained (no leading/trailing blank lines) so
callers control the surrounding spacing.
"""

from __future__ import annotations

# Filesystem path conventions for agents that write under the virtual
# workspace filesystem (researcher, slice-implementer). Absolute paths
# double-nest under the root and create files in the wrong place.
WORKSPACE_PATH_RULES = (
    "Path conventions: all file paths MUST be relative from the project "
    "workspace root.\n"
    "- Correct: `spine/ui/pages.py`, `.spine/artifacts/doc.md`\n"
    "- Correct: `/spine/ui/pages.py` (a leading `/` is workspace-relative)\n"
    "- WRONG: `/home/user/project/spine/ui/pages.py` — absolute paths "
    "double-nest under the virtual filesystem root and resolve to "
    "non-existent files.\n"
    "- WRONG: `../other/file.py` — traversal is blocked by the virtual "
    "filesystem."
)

# Output discipline for the single-write-tool synthesizers (SPECIFY / PLAN):
# the write tool renders the markdown and serializes the JSON itself.
NO_MARKDOWN_WRITE_NOTE = (
    "The tool renders markdown and emits JSON for you — DO NOT author "
    "markdown, DO NOT hand-serialize JSON, DO NOT call write_file."
)

# The scope-exclusion citation rule shared by the critic (PLAN review) and the
# adversarial reviewer: an exclusion objection is only valid if it quotes the
# matching bullet verbatim.
SCOPE_EXCLUSION_CITATION_RULE = (
    "To flag a slice for reaching into an EXCLUDED area, you MUST quote the "
    "matching `scope_exclusions` bullet(s) verbatim in `cited_exclusions`; an "
    "exclusion objection asserted without a matching verbatim `cited_exclusions` "
    "entry is unsupported and is overturned automatically."
)
