# Spine Assistance Requirements

## Assistance Overview

Spine is a mid-sized, multi-language codebase: 4,774 symbols spread across 302 files and 33 modules, plus 6 documented patterns. No single module dominates—every module listed as 'largest' carries the same symbol count—so an assistant should not assume any directory can be safely read in full 'just to be safe'. Instead, treat context as a budget: orient with the codebase index first, then read only the files or symbols directly relevant to the current question. The repository mixes Python, C, C++, PHP, and TypeScript, so agents should stay within the appropriate skill boundaries and avoid applying Python-only assumptions to non-Python assets.

### spine.agents (`spine/agents`)

One of the largest modules by symbol count; investigate via the codebase index before opening agent-related files to avoid unnecessary token use.

### spine.cli (`spine/cli`)

CLI surface area; likely high coupling with command definitions and entry points. Use index lookups to pinpoint the relevant command or option before reading.

### spine.config.py (`spine/config.py`)

Configuration module; consult the index to identify which config keys or loaders are involved instead of reading the whole file.

### spine.exceptions.py (`spine/exceptions.py`)

Exception definitions shared across the project; reference only the specific exception hierarchy needed, not the entire module.

### spine.git (`spine/git`)

Git integration module; boundary-heavy and likely IO-adjacent. Let the index guide targeted reads.

### spine.infra (`spine/infra`)

Infrastructure utilities; broad usage means token burn risk is high unless reads are narrowed by index hits.

### spine.mcp (`spine/mcp`)

MCP-related module; prefer index summaries first to determine which protocol or server pieces matter.

### spine.models (`spine/models`)

Data models; often referenced widely. Read specific model definitions rather than loading the whole module.

### spine.persistence (`spine/persistence`)

Persistence layer; likely spans multiple backends. Use the index to isolate the storage driver or query under review.

### spine.phases (`spine/phases`)

Phase/workflow logic; check the index for phase names and transitions before reading files in depth.

### vector codebase index (`codebase index`)

1,115 enriched summaries derived from the vector index. This is the primary orientation tool for finding relevant symbols and avoiding full-file reads of the largest modules.

**Notes:**
- The 'largest modules' list ties at 8 symbols each, so module size is flat rather than hierarchical; do not prioritize any one of them as the obvious hotspot without further signal.
- Because the tech stack includes Python, C, C++, PHP, and TypeScript, apply the correct language-specific skill when reading files and avoid cross-language generalizations.
- The 1,115 summaries are an enrichment aid, not a complete substitute for source code; confirm any critical claim against the actual file when precision matters.

## Hot Spots & Budget Guidance

The Spine codebase spans **302 files** across **33 modules** with **4,774 symbols**, so eager file reads burn context quickly. Budgeting begins with assuming no single module is safe to load wholesale; the index provides **1,115 enriched summaries** that should be consulted before opening raw source files. The.top module list is effectively a flat tier: ten modules each report **8 symbols** and should be treated as the primary hot spots for targeted, read-only exploration rather than bulk ingestion. Agents should also respect the multi-language boundary (`python`, `c`, `cpp`, `php`, `typescript`) when searching symbols, because a name match in one language may not be the implementation of interest in another. Prefer the codebase index over recursive reads unless a specific symbol, file path, or failing test has already been confirmed as relevant. This avoids loading oversized context and keeps budget for the actual reasoning task. When context is already tight, drop non-essential source files first and rely on summaries to reconstruct intent before re-entering the code. The modules called out below are the concrete budget triggers; if an agent reaches for more than one of them, it should pause and justify the read set against the current objective. A lightweight cross-reference query in the vector index is almost always cheaper than reading the whole module or following every import chain by hand. Plan the narrowest read set, validate against summaries, and only then open files. That pattern is the core of token-efficient assistance in this repo. Stay skeptical of module names that look like single files: `spine.config.py` and `spine.exceptions.py` are tracked as modules but correspond to concrete Python files, so they can still be read directly if needed; just do not assume they are small wrappers without checking the index first. Similarly, `spine.agents`, `spine.cli`, and `spine.phases` sound like orchestration layers that may pull in many downstream imports; avoid transitive expansion unless the summary points to a specific entry point. The hot-spot list is a guardrail, not a map of blame. Use it to decide whether a file read is proportional to the question at hand. If the question is broad, stay in summary/index space and ask the user to narrow it. If the question is specific, load the single relevant file or symbol and no more. The following entries capture the modules that should trigger this decision process every time they appear in a query result or import path. Each is named exactly as reported in the source data, and each description notes why it matters for budgeting rather than making claims about runtime behavior that the fragment does not support. For teams automating on this guidance, the rule of thumb is: one hot-spot hit = query first, read second; two or more hot-spot hits = stay in the index until the agent can state a precise filename and symbol name. This protects both latency and answer quality by preventing the model from being buried in unrelated implementation details before it has framed the problem. The vector index is the canonical substitute for broad exploration; treat it as the default entry point and source files as the exception. Finally, remember that symbol counts are equal across the listed modules, so do not rank them by size within this tier; rank them only by relevance to the specific user request. Relevance is determined by the index and the agent's prompt, not by assuming that one of these modules is the 'core' of the system. If the data later refreshes with larger modules or richer size metrics, this section should be regenerated from that new source of truth rather than edited by hand. The intent is to keep the guidance tethered to measured signals, not to narrative impressions of what looks important. Every read should be defensible against the measured signals and the current objective; otherwise, it is likely wasted context. The guidance here is therefore conservative by design, because over-reading is a much more common failure mode than under-reading in a codebase of this breadth. Agents that internalize this bias will hand任务 back faster, cheaper, and with fewer hallucinated connections between distant parts of the project. Budget discipline is especially important across languages: a search for a generic term such as `config`, `model`, `phase`, or `exception` can return hits in Python, PHP, TypeScript, and C/C++ at once. Before chasing any of those hits, filter by language targeted by the current task. This single filter often eliminates the majority of false-positive symbol matches and keeps the working set small. When in doubt, state which language scope you are using and why, so the user can correct you before context is consumed. The final principle is reproducibility: if an agent needs to load a hot-spot file, it should record the exact path and symbol that justified the load, making future turns cheaper and audits easier. The entries below give the starting points for that accounting. There is no need to enumerate every file in each module; the index does that. The role of this section is to flag where bulk loading starts to hurt and to normalize the habit of index-first exploration across the team of deep agents working on Spine. Consider this the default contract: summary search, targeted read, and only broaden when the evidence explicitly demands it. That contract is what converts a large, multi-language codebase from a context trap into a navigable system. Stick to it unless the user gives a clear, scoped exception, and even then, quote the exception explicitly so the next agent can see why the normal rule was suspended. Discipline here compounds: each saved read leaves room for deeper reasoning on the parts that actually matter. In a 33-module, 4,774-symbol project, that margin is not optional; it is the difference between a useful answer and a truncated one. Treat every token as a resource to be justified, and use the hot-spot list to enforce that justification before any file is opened. The guidance is deliberately terse in actionable rules so that agents can apply it quickly under time pressure: see a hot spot, query the index, read narrowly, and move on. The remaining prose simply anchors that rule in the measured scale of the repository so that it is not dismissed as a generic platitude. Use it, and revisit it whenever the source metrics change. This keeps the assistance requirements living documents rather than stale overhead, and it keeps the agents honest about the cost of their own exploration. TheSpine project is large enough that exploration without discipline becomes expensive; these notes are the guardrails for keeping that exploration bounded. Agents should not treat the absence of a module from the hot-spot list as a license to load it carelessly, but they should treat the presence of a module on the list as a mandatory pause. That asymmetry reflects the reality of context budgeting: avoiding waste requires friction at the boundary of large or important modules, not a blanket restriction on all reads. The net effect should be faster resolution of focused questions and clearer escalation paths for broad questions, because broad questions are naturally answered from summaries rather than source-dive expeditions. The user benefits when the model stays within budget long enough to produce a coherent, end-to-end answer. These paragraphs establish the rationale; the entries that follow make the rule concrete and checkable against the codebase index. No module mentioned below is intrinsically dangerous; the danger is in reading too much of it without a clear reason. Keep the reason visible, and the budget will take care of itself. The summary count of 1,115 is the practical replacement for most broad reads, and agents should internalize that number as the project's answer to 'where do I start?' Start there, not in the file tree. The file tree is for precise follow-up after the index has already narrowed the target. Respecting that order is the single most impactful habit for efficient Spine assistance. The rest is detail; the entries below pin the detail to the actual modules that the metrics flag as the largest concentration points. Read them once, then apply the rule every time a
