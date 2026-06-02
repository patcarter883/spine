# Spine Assistance Requirements

## Assistance Overview

This repository is moderately 225 files spanning 31 modules with 3,011 total symbols across Python and TypeScript. The codebase is moderate in size, but passes the vector index is unavailable; queries are AST-only and lack summaries—making strategic file selection essential.

The largest modules are evenly balanced at 8 symbols each: `spine.agents`, `spine.cli`, `spine.config.py`, `spine.exceptions.py`, `spine.git`, `spine.mcp`, `spine.models`, `spine.persistence`, `spine.phases`, and `spine.ui`. These constitute 80% of the total module count despite containing minimal symbol density individually. Context budget should prioritize these core modules over less-traveled paths.

With only 6 patterns indexed, few abstractions exist to guide navigation. Rely instead on targeting specific modules aligned with task domains—particularly
"}          "agent behaviors" (spine.agents), "configuration" (spine.config.py), "data flow" (spine.models), or "UI interactions" (spine.ui). Avoid scanning unrelated modules to conserve token allocation.

## Hot Spots & Budget Guidance
|
*Avoid loading the following high-symbol modules in full: spine.agents (8), spine.cli (8), spine.config.py (8), spine.exceptions.py (8), spine.git (8), spine.mcp (8), spine.models (8), spine.persistence (8), spine.phases (8), spine.ui (8).*)
*
*Use the codebase index for navigation; vector index unavailable — AST-only, no summaries available
*
