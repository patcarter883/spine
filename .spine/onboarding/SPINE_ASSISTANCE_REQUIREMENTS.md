# Spine Assistance Requirements

## Assistance Overview

This repository spans 302 parsed files across 33 top-level modules and 4,774 total symbols (classes, functions, and methods), with `python`, `c`, `cpp`, `php`, and `typescript` all present in the tech stack — though the non-Python languages are almost entirely test fixtures (see `tests.fixtures`), not production code. Of the 4,774 symbols, 1,115 (about 23%) have an enriched natural-language summary available from the vector index; the rest are AST-only (name, type, and file location, no generated description). An assistant should treat the index as the primary way to find things — grep/structural search or the codebase-query tool first — and fall back to opening raw source files only once a specific file or symbol has been identified as relevant.

Module size in this repo is extremely uneven. The two largest modules by symbol count — `tests.unit` (2,989 symbols) and `spine.agents` (651 symbols) — together account for over 75% of the entire codebase's symbol count, while most other modules sit in the tens or low hundreds. This is the single most important sizing fact for budgeting: a search or exploration strategy that treats every module as roughly equal will badly underestimate the cost of touching `tests.unit` or `spine.agents`, and badly overestimate the cost of nearly everything else.

## Hot Spots & Budget Guidance

Ranked by real total symbol count (not by how many example symbols happen to be listed for a module — every module's example list is capped at a small fixed number for token-budget reasons, and that cap is unrelated to the module's actual size):

### tests.unit (2,989 symbols)
By far the largest module in the repository. This is the unit-test suite; before reading broadly here, narrow to the specific test file(s) relevant to the code under discussion rather than exploring the directory.

### spine.agents (651 symbols)
The largest production module — implements every LLM-driven phase agent, subagent, and tool. Avoid reading this module wholesale; identify the specific file (e.g. `factory.py`, `helpers.py`, `decomposer.py`, or a file under `tools/`) that's actually relevant first.

### spine.workflow (287 symbols)
LangGraph orchestration: subgraph state schemas and the phase registry. Second-largest production module.

### spine.work (182 symbols)
Execution/dispatch layer plus the entire repo-onboarding analysis-and-synthesis subsystem (`spine/work/onboarding/`) that produced this document.

### tests.integration (118 symbols)
End-to-end integration tests; same narrowing advice as `tests.unit`.

### spine.ui (97 symbols), spine.ui_api (70 symbols), spine.persistence (65 symbols)
Mid-sized modules — the Streamlit UI layer, its backend facade, and the storage layer, respectively.

**Notes:**
- Ranking is by real symbol count, parsed from each module's own reported role string, not by the length of any example-symbol list shown elsewhere in these docs — those lists are capped at a small number per module regardless of the module's true size, and should never be read as a size signal.
- `tests.unit`'s size is almost entirely test volume, not architectural complexity — it is a hot spot for token budget, not necessarily a place requiring careful design reasoning.
- Everything below the modules listed above is comparatively small (`spine.mcp`, `spine.critic`, `spine.exceptions.py`, etc. — mostly under 30 symbols) and is unlikely to need this kind of caution.
