# Spine Assistance Requirements

## Assistance Overview

This repository contains 225 files spanning 2989 symbols across 31 modules. The ten largest modules (`spine.agents`, `spine.cli`, `spine.config.py`, `spine.exceptions.py`, `spine.git`, `spine.mcp`, `spine.models`, `spine.persistence`, `spine.phases`, `spine.ui`) each contain only 8 symbols, indicating they are reasonably sized and can be safely read in full when needed.

Key constraints:
- Vector index unavailable: querying falls back to AST-only search
- Context budget should prioritize modules directly related to the task;
- For unrelated areas, rely on indexed references rather than full file reads.

Tech stack: Python and TypeScript.

## Hot Spots & Budget Guidance

**Module Size Limits:** All modules are at 8 symbols (~20–30 lines). The 10 largest (`spine.agents`, `spine.cli`, `spine.config.py`, `spine.exceptions.py`, `spine.git`, `spine.mcp`, `spine.models`, `spine.persistence`, `spine.phases`, `spine.ui`) follow this ceiling. **Guidance

**Context Budget Rule:**   When an assistant needs information from a module, prefer using the codebase index rather than reading the file directly. Due to the vector index being unavailable (AST-only, no summaries), Avoid loading full context for these modules unless absolutely necessary.

**Analysis Notes:** Use the index to locate exported symbols and call sites; only open files for detailed line-level inspection.
