# Spine Assistance Requirements

## Assistance Overview

This repository contains **224 files** organized into **31 modules**, encompassing approximately **2,964 symbols** and **6 key patterns**, with implementations in **Python** and **TypeScript**. Due to the absence of a vector index (AST-only analysis without summaries), token budget should be allocated strategically: the largest modules — `spine.agents`, `spine.cli`, `spine.config.py`, `spine.exceptions.py`, `spine.git`, `spine.mcp`, `spine.models`, `spine.persistence`, `spine.phases`, and `spine.ui` — each contain 8 symbols and are considered substantial; prioritize referencing these modules through the codebase index rather than reading their contents directly. Context budget is best spent on smaller, targeted inquiries rather than broad module exploration.

## Hot Spots & Budget Guidance

### Largest Modules
 The following 10 modules each contain 8 symbols, making them the most substantial areas in the codebase:
 - `spine.agents`
 - `spine.cli`
 - `spine.config.py`
 - `spine.exceptions.py`
 - `spine.git`
 - `spine.mcp`
 - `spine.models`
 - `spine.persistence`
 - `spine.phases`
 - `spine.ui`

### Analysis Notes
 - **Vector index unavailable**: Analysis performed on AST data only; no content summaries exist. Use the codebase index for targeted queries rather than full file reads. 
 - **Tech stack**: Python and TypeScript — prioritize language-specific tools (e.g., `py` or `ts` filters) when indexing. 
 - **Compiler constraints**: Do not attempt semantic analysis without precomputed summaries. 

### Recommendations
 - Prefer Prefer|→
