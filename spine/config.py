"""SPINE configuration — load and validate .spine/config.yaml.

Environment variables are loaded from ``.env`` (project root) on first
import so that ``LANGSMITH_*`` and other runtime vars are available to
LangGraph, Deep Agents, and LangSmith tracing without manual sourcing.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Load .env on import ──
# This ensures LANGSMITH_API_KEY, LANGSMITH_TRACING, OPENROUTER_API_KEY,
# etc. are set before any LangGraph or Deep Agents code reads them.
# It's safe to call multiple times (no-op if already loaded).


def _load_dotenv() -> None:
    """Load .env from the project root if python-dotenv is available.

    Search order:
      1. CWD and its parents (works when launched from the project root)
      2. The directory containing the spine package and its parents
         (works when Streamlit or another runner changes CWD away
         from the project root)

    ``override=False`` ensures manually-set env vars always win.
    """
    try:
        from dotenv import load_dotenv

        # Strategy 1: walk up from CWD (includes CWD itself)
        loaded = False
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / ".env").is_file():
                loaded = load_dotenv(dotenv_path=candidate / ".env", override=False)
                break

        # Strategy 2: walk up from the package directory — this handles
        # Streamlit which may launch with a CWD like $HOME or /tmp.
        if not loaded:
            pkg_dir = Path(__file__).resolve().parent
            for candidate in pkg_dir.parents:
                if (candidate / ".env").is_file():
                    load_dotenv(dotenv_path=candidate / ".env", override=False)
                    break
    except ImportError:
        # python-dotenv not installed — env vars must be set manually
        pass


_load_dotenv()


# ── Disable LangSmith tracing by default ──
# Tracing is opt-in *per work task run* (see spine.observability.work_run_tracing).
# Without this, merely importing/using LangGraph anywhere — codebase indexing,
# the test suite, onboarding repo analysis — emits traces, because .env sets
# LANGSMITH_TRACING=true for the work runs that DO want it.  We force the
# ambient flag off here and let work runs re-enable it on demand via a
# contextvar-scoped tracer.  The API key / endpoint / project vars are left
# intact so those work runs can still reach LangSmith.
#
# Escape hatch: set SPINE_TRACE_ALL=1 to keep tracing on globally (trace
# everything, including indexing and tests) for debugging.


def _disable_global_tracing() -> None:
    """Force LangSmith tracing off unless ``SPINE_TRACE_ALL`` is set."""
    if os.environ.get("SPINE_TRACE_ALL", "").lower() in ("1", "true", "yes"):
        return
    # Both the current (LANGSMITH_*) and legacy (LANGCHAIN_*) flags must be
    # cleared — langchain-core honours either when deciding to auto-trace.
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        os.environ[var] = "false"


_disable_global_tracing()


@dataclass
class ConvergenceConfig:
    """Researcher convergence-steering thresholds.

    The ``researcher_soft`` / ``researcher_hard`` / ``researcher_recursion_limit``
    fields are no-ops for the researcher subagent since the supervisor/worker
    split replaced the free-form tool loop (see
    ``spine/agents/researcher_supervisor.py``). They are retained for YAML
    back-compat and for any future code path that still wants tool-call
    convergence steering.

    The ``researcher_supervisor_max_cycles_*`` fields cap the supervisor↔worker
    loop per phase: SPECIFY (broad architectural mapping) gets a larger
    budget than PLAN (focused change-surface scan).
    """

    researcher_soft: int = 10
    researcher_hard: int = 14
    researcher_recursion_limit: int = 50
    researcher_supervisor_max_cycles_specify: int = 12
    researcher_supervisor_max_cycles_plan: int = 6


@dataclass
class TokenCompactionConfig:
    """Token-budget compaction config for phase agents."""

    enabled: bool = False
    default_threshold: int = 0
    thresholds: dict = field(default_factory=dict)
    keep_recent: int = 6
    preserved_tools: list = field(
        default_factory=lambda: [
            "write_file",
            "edit_file",
            "read_edit_lint",
            "write_specification",
            "write_plan",
            "write_tasks",
            "write_verification_report",
        ]
    )


def _parse_convergence_config(raw: dict) -> ConvergenceConfig:
    """Parse the ``convergence:`` block from .spine/config.yaml."""
    if not isinstance(raw, dict):
        return ConvergenceConfig()
    soft = int(os.getenv("SPINE_RESEARCHER_SOFT", raw.get("researcher_soft", 25)))
    hard = int(os.getenv("SPINE_RESEARCHER_HARD", raw.get("researcher_hard", 40)))
    rlimit = int(
        os.getenv(
            "SPINE_RESEARCHER_RECURSION_LIMIT",
            raw.get("researcher_recursion_limit", 50),
        )
    )
    if hard < soft:
        hard = soft
    max_cycles_specify = int(
        os.getenv(
            "SPINE_RESEARCHER_SUPERVISOR_MAX_CYCLES_SPECIFY",
            raw.get("researcher_supervisor_max_cycles_specify", 12),
        )
    )
    max_cycles_plan = int(
        os.getenv(
            "SPINE_RESEARCHER_SUPERVISOR_MAX_CYCLES_PLAN",
            raw.get("researcher_supervisor_max_cycles_plan", 6),
        )
    )
    return ConvergenceConfig(
        researcher_soft=soft,
        researcher_hard=hard,
        researcher_recursion_limit=rlimit,
        researcher_supervisor_max_cycles_specify=max(1, max_cycles_specify),
        researcher_supervisor_max_cycles_plan=max(1, max_cycles_plan),
    )


def _parse_token_compaction_config(raw: dict) -> TokenCompactionConfig:
    """Parse the ``token_compaction:`` block from .spine/config.yaml."""
    if not isinstance(raw, dict):
        return TokenCompactionConfig()
    enabled_raw = os.getenv(
        "SPINE_TOKEN_COMPACTION", str(raw.get("enabled", False)).lower()
    )
    enabled = enabled_raw.lower() in ("1", "true", "yes")
    thresholds_raw = raw.get("thresholds", {}) or {}
    thresholds = {str(k): int(v) for k, v in thresholds_raw.items()}
    preserved = raw.get("preserved_tools")
    if not isinstance(preserved, list) or not preserved:
        preserved = TokenCompactionConfig().preserved_tools
    return TokenCompactionConfig(
        enabled=enabled,
        default_threshold=int(raw.get("default_threshold", 0)),
        thresholds=thresholds,
        keep_recent=int(raw.get("keep_recent", 6)),
        preserved_tools=list(preserved),
    )


@dataclass
class SpineConfig:
    """Runtime configuration for SPINE.

    Loads from ``.spine/config.yaml`` with sensible defaults for missing keys.
    Environment variables override individual settings when set.
    """

    checkpoint_path: str = ".spine/spine.db"
    artifact_path: str = ".spine/artifacts"
    project_path: str = ".spine/project"
    max_critic_retries: int = 2
    max_adversarial_retries: int = 2
    work_type: str = "task"
    providers: dict = field(default_factory=dict)
    queue_backend: str = "sqlite"
    queue_path: str = ".spine/queue.db"
    workspace_root: str = ""
    interpreter_enabled: bool = False
    tool_schema_validation: bool = True
    # Forgiving multi-format tool-call parsing (see ToolCallNormalizer). Recovers
    # tool calls a weak/quantized model emits as text. Deterministic parsing, so
    # safe for replay; off by default until tuned for the local models in use.
    tool_call_normalize: bool = False
    # LLM error diagnosis on command failure (see ExecuteErrorDiagnoser).
    # Off by default: it makes an extra (cheap) LLM call per failed ``execute``
    # whose text enters a checkpointed ToolMessage, so replay can differ.
    error_diagnosis: bool = False
    phase_timeouts: dict = field(
        default_factory=lambda: {
            "specify": 0,
            "plan": 0,
            "tasks": 0,
            "implement": 0,
            "verify": 0,
            "critic": 0,
        }
    )
    default_timeout: int = 0
    mcp_servers: dict = field(default_factory=dict)
    guided_decoding: bool = False

    # RAG (Retrieval-Augmented Generation) configuration
    embedding_provider: str = "openai-embeddings"
    recall_k: int = 10
    # Index test files into the vector store. Default off: test chunks
    # have verbose docstrings that score high on NL similarity but tell a
    # researcher nothing about production code, and they otherwise swamp
    # the index (~66% of symbols). See tests/recall_eval baseline.
    index_tests: bool = False
    # Reciprocal-rank-fusion channel weights for hybrid recall. Lexical BM25
    # is the workhorse for code-symbol search; the dense vector channel was
    # found to add nothing and even hurt ranking on tests/recall_eval (pure
    # BM25 scored the best MRR, 0.55, across curated, mined, AND deliberately
    # paraphrastic queries — for both Qwen3 and nomic embeddings). The vector
    # channel is kept at a small weight as a hedge for queries unlike the eval
    # set. Env overrides: SPINE_RRF_VECTOR_WEIGHT / SPINE_RRF_BM25_WEIGHT.
    rrf_vector_weight: float = 0.2
    rrf_bm25_weight: float = 1.0
    # Cross-encoder reranking. When ``reranker_provider`` names a provider in
    # providers.reranker[], hybrid retrieves ``rerank_pool`` candidates and a
    # cross-encoder re-orders them to the final ``recall_k``. Empty name =
    # disabled (the fused RRF order is returned as-is). A cross-encoder reads
    # query+candidate jointly, so it can lift ranking even though the
    # bi-encoder vector channel is weak. Off by default — enable + measure on
    # tests/recall_eval before trusting it.
    reranker_provider: str = ""
    rerank_pool: int = 50
    vector_indexing: dict = field(
        default_factory=lambda: {
            "max_concurrent_chunks": 5,
            "batch_size": 100,
        }
    )

    # SPECIFY exploration short-circuit. When classification confidence is
    # high enough AND we retrieved at least N hits, skip the multi-round
    # research_manager loop and synthesize directly from the recalled chunks.
    recall_gate_confidence: float = 0.75
    recall_gate_min_hits: int = 5
    # Trivial-task fast path. A short, high-confidence description with no
    # architectural verbs short-circuits SPECIFY exploration even when the
    # recall index returns zero hits (cold/empty index). Without this, a
    # one-line task like "add a --verbose flag" ran a full 3-round, 6-explore
    # research loop (trace 019e77a7) purely because recall returned 0 chunks.
    recall_gate_trivial_max_chars: int = 150
    # Completion-token cap for the researcher's ResearchFindings structured
    # summarisation calls (summarise / finalize / salvage). A findings JSON is
    # small; without a tight cap a local model can ramble to the global window
    # cap (16K) and 207s before raising LengthFinishReasonError, which is then
    # discarded anyway (trace 019e77fe). Capping fails fast to the sentinel.
    summarise_max_completion_tokens: int = 4096
    # Completion-token cap for the per-topic researcher SUPERVISOR's
    # SupervisorDirective calls (in-loop + off-by-one salvage). A directive is
    # tiny (a 2-4 sentence verdict + a one-sentence next move); without a tight
    # cap the call inherits the global window (e.g. 40K) and a local model that
    # mistakes the near-cap soft-landing nudge for "write the findings now" can
    # ramble for minutes into the free-text analysis field before raising
    # LengthFinishReasonError — which stalled SPECIFY on trace 019e8679
    # (run 019e867a, ~9.5min). A tight cap fails fast and the loop proceeds to
    # synthesis from accumulated evidence. Mirrors summarise_max_completion_tokens.
    researcher_supervisor_max_completion_tokens: int = 1024
    # Completion-token cap for the research manager's ResearchManagerDecision
    # structured call. The decision is tiny (explore/done + 2-4 topic
    # strings); uncapped, the empty-parse retry ("Your previous response was
    # empty. Respond with ONLY the JSON…") sent a thinking model into a
    # multi-minute reasoning burn toward the provider's full completion
    # budget (trace 019eb541, observed live at 300s+ solo on the engine).
    # Raised 2048->8192: GLM ignores suppress_reasoning and reasons past 2048,
    # length-erroring the ResearchManagerDecision -> "defaulting to explore one
    # topic" (trace 019efca3). Instruct models still finish well under it.
    research_manager_max_completion_tokens: int = 8192
    # ── Research breadth (task-aware exploration fan-out) ───────────────
    # Default ceilings on the exploration loop: how many research_manager
    # rounds run, and how many explore branches fan out per round. A high-
    # confidence classification in a well-understood category (e.g. a
    # Frontend/UI field addition) does not need the full breadth — trace
    # 019ec965 spent 52 explore_do calls / ~60% of input tokens researching
    # a config-UI form. When confidence >= ``research_lean_confidence`` AND
    # the category is in ``research_lean_categories``, the leaner
    # ``research_lean_*`` ceilings apply instead.
    research_max_rounds: int = 3
    research_max_parallel_explores: int = 4
    research_lean_confidence: float = 0.85
    research_lean_max_rounds: int = 2
    research_lean_max_parallel_explores: int = 2
    research_lean_categories: list[str] = field(
        default_factory=lambda: ["Frontend/UI"]
    )
    # ── Best-of-N plan synthesis (Graph-of-Thoughts Score + KeepBest) ──────
    # Number of candidate plans the PLAN synthesizer generates, scores
    # (spine.workflow.plan_score), and selects the best of before handing one
    # to the critic. ``1`` (default) is the historical single-shot behaviour
    # with zero added cost. ``>1`` front-loads convergence that the critic
    # loop otherwise grinds out one rework cycle at a time. Variants run
    # SEQUENTIALLY (local providers cap max_concurrent_calls at 1-2, so
    # parallel synthesis buys no wall-clock and would collide on plan.json).
    research_synth_variants: int = 1
    # Scorer for KeepBest. ``"structural"`` = deterministic, LLM-free (the
    # default — cheap, reproducible). ``"hybrid"`` adds an LLM-judge tiebreak
    # among the top structural candidates (not yet wired; reserved).
    research_synth_scorer: str = "structural"
    # ── Findings aggregation (Graph-of-Thoughts Aggregate) ─────────────────
    # When true, an LLM merge pass dedups/reconciles/ranks accumulated
    # findings into a tighter evidence set just before synthesis, instead of
    # feeding the raw append-only list straight to the synthesizer. Reduces
    # synthesizer context (mitigates the read-spiral / context-blowup aborts).
    # Only fires when there are at least ``research_aggregate_min_findings``.
    research_aggregate_merge: bool = False
    research_aggregate_min_findings: int = 8
    # Completion-token cap for the no-tool plan_do (run_plan_node)
    # SubagentDirective calls (e.g. plan_slice_implementer). A directive is
    # a few hundred tokens; without a cap the call inherits the provider's
    # max_completion_tokens and a thinking model can burn for minutes in
    # the reasoning channel before LengthFinishReasonError — trace
    # 019eb502: 450s solo on the engine, serializing the whole implement
    # fan-out behind it. Mirrors researcher_supervisor_max_completion_tokens
    # (larger because a SubagentDirective carries approach + steps + risks).
    plan_do_max_completion_tokens: int = 2048
    # Completion-token cap for the slice-implementer agent loop. Implement
    # turns are tool calls (edit payloads), not essays; without a cap the
    # request inherits the global max_completion_tokens (30K) and a
    # finite-window model 400s once the conversation grows past
    # window - 30K (trace 019eb502: 30,001-token prompt + 30K requested vs
    # a 60K window). Lowered 12K→8K (trace 019ece87): the static reservation
    # is the room subtracted from every turn's prompt budget, so a smaller cap
    # buys ~4K more prompt headroom before overflow; DynamicCompletionCap then
    # lowers it further per-turn as the prompt grows. 8K still covers a
    # full_replace of a ~30KB file, which is larger than any single slice's file.
    # Raised to 32768 to give reasoning-heavy local models (North-Mini-Code,
    # whose CoT leaks into the completion) room for both their thinking and the
    # edit payload. On a 64K-window model this halves the prompt budget;
    # DynamicCompletionCap still shrinks it per-turn as the conversation grows.
    implement_max_completion_tokens: int = 32768
    # Completion-token cap for the structural decomposer's no-tool structured
    # calls (PLAN / FALLBACK / PER_FILE in spine.agents.decomposer) AND the
    # single-file enrich pass. The edit_plan now carries ONE entry per change
    # site / new method (not a single bundled "add all the methods" entry), so a
    # large slice emits ~10 EditHint objects with full-signature intents — the
    # old 4096 truncated that mid-JSON on tight-cap local models, raising
    # LengthFinishReasonError and silently dropping the whole edit_plan (North
    # bench 0624: enrich failed for backend-persistence/config-* → slices ran
    # with no plan, neutralising source pre-loading). 8192 fits the larger plan.
    # Without any cap the bare call inherits the global max_completion_tokens
    # (30K); against a finite local window (e.g. context_window=40K) the server
    # must reserve a 30K generation slot, leaving too little KV cache for the
    # prompt and OOM-crashing the backend — which then drops every in-flight
    # request with "CURL error: Could not connect" and fans the failures out
    # into a fallback-decompose retry storm (trace 019ed360). Per-phase
    # max_completion_tokens overrides still win.
    #
    # 16384: North-Mini-Code (a thinking model served via llama.cpp/lemonade,
    # where suppress_reasoning's vLLM knobs are ignored) leaks chain-of-thought
    # into the completion, exhausting 4096 AND 8192 before the JSON closed —
    # every enrich call dropped its edit_plan (North bench 0624). The larger
    # base, plus the length-escalation retry in decomposer (doubles the cap once
    # on LengthFinishReasonError), gives the structured JSON room to land.
    # Raised to 32768 (global completion bump) for North, whose leaked CoT
    # exhausted 4096/8192/16384 before the edit_plan JSON closed.
    decompose_max_completion_tokens: int = 32768
    # Max chars of the failure traceback embedded in a FALLBACK decompose
    # prompt. The traceback is otherwise unbounded — a large one (plus the
    # verbatim failed-slice JSON) inflates the prompt until the structured call
    # truncates mid-JSON and dies with LengthFinishReasonError, dropping a
    # recoverable slice (trace 019ed3dc). We keep the TAIL (tracebacks are most
    # informative at the end). 0 = unbounded.
    decompose_max_traceback_chars: int = 4000
    # Max depth of the IMPLEMENT fallback-decompose recursion. Each failed
    # slice that is re-sliced increments the depth; at the cap the slice is
    # surfaced as permanently blocked instead of being decomposed again. Depth
    # is a fan-out MULTIPLIER: at depth 2 one stubborn slice can spawn 1 + 3 +
    # 9 = 13 implementer attempts, each re-reading its target file into a fresh
    # context (trace 019ed3dc). For weak local models that fail the original
    # slice, finer micro-slicing rarely succeeds, so default to a single
    # decomposition pass (1) and let stronger setups raise it.
    implement_max_decompose_depth: int = 1
    # Soft ceiling on the number of model turns a single slice-implementer
    # agent may take before the TurnBudgetGuard injects a hard "converge now"
    # directive. Per-call cost is already clamped (implement_max_completion_tokens
    # + DynamicCompletionCap) and the prompt is capped at the compaction
    # threshold, but neither bounds the NUMBER of turns: a slice that keeps
    # re-querying and re-editing grinds dozens of turns at the compaction
    # ceiling, so total input scales linearly with turn count (trace 019ed413:
    # one slice took 69 model turns ≈ 954K input tokens for a config-UI edit).
    # The guard only nudges — tools stay bound — so legitimate long slices can
    # still finish; 0 disables it.
    implement_max_turns: int = 30
    # Soft turn budget for the (tool-using) slice-verifier ReAct fallback — the
    # verify-side analogue of ``implement_max_turns``. When the evidence-then-judge
    # path is off, the verifier reads files + runs checks in a loop; nothing
    # bounded its turn count, so a healthy-sandbox probe loop could grind to the
    # token budget (trace 019f16cf). Past this many turns the TurnBudgetGuard
    # nudges it to stop probing and return its verdict from evidence in hand. A
    # healthy verify is only a handful of turns; this is a backstop. 0 disables it.
    verify_max_turns: int = 20
    # ── Synthesis + placement editor (two pure nodes, no tool loop) ────────
    # When True, ``_route_slices`` sends each pending slice to the synthesis
    # implementer (spine.agents.synthesis_implementer) instead of the tool-using
    # slice_implementer: a no-tool structured call synthesizes complete
    # symbol-anchored edits, then placement applies them deterministically
    # through ReadEditLintTool (lint is the oracle). The editor cannot read, so
    # it cannot survey-spiral (Laguna 79 calls / 0 edits; Qwen3-Coder empty diff).
    # Default off — opt-in until benched against the tool-using path.
    implement_synthesis_placement: bool = False
    # Best-of-N for synthesis (Graph-of-Thoughts Score + KeepBest, the
    # IMPLEMENT-side mirror of research_synth_variants / plan_score). Synthesis
    # is side-effect-free, so sampling N candidates is cheap; placement scores
    # each by how cleanly its edits apply + lint and keeps the best. 1 = single
    # candidate (no extra cost). Only meaningful with implement_synthesis_placement.
    implement_synthesis_variants: int = 1
    # VERIFY: evidence-then-judge slice-verifier. The default verifier is a
    # tool-using ReAct agent (read_file + run_checks); with a 9999 recursion
    # limit and no run_checks call cap it spirals on introspection probes until
    # the per-work token budget trips and CRASHES the slice to NOT_VERIFIED with
    # no verdict (trace 019f16cf: 6 verifiers, 2.75M tokens, zero verdicts — the
    # spiral survives a strong coder model, so it is structural, not model). When
    # this is on, the verify node pre-computes the evidence deterministically
    # (worktree diff + pre-loaded target source + a single py_compile/ruff pass
    # over the changed files) and the verifier becomes a NO-TOOL, schema-bound
    # judge that emits a VerificationResult in one shot. No loop ⇒ no spiral; the
    # schema-skips-tools hazard that forces the ReAct design does not apply when
    # there are no tools. Default off — opt-in until benched against the ReAct path.
    verify_evidence_then_judge: bool = False
    # Hard ceiling on total slice-implementer executions in one IMPLEMENT run, a
    # backstop against the dispatch runaway in trace 019efd92 (3 same-file slices
    # raced to ~687 executions / 1.33M tokens). Each implementer execution ticks
    # ``slice_dispatch_count``; when it reaches this cap ``_route_slices`` aborts
    # the dispatch loop to synthesis and surfaces remaining slices as incomplete.
    # 100 comfortably clears a sane plan (slices + per-file splits + bounded
    # decompose) while catching a runaway early. 0 disables the backstop.
    implement_max_slice_dispatches: int = 100
    # Hard super-step ceiling for phase subgraphs (LangGraph recursion_limit).
    # LangGraph's default is 25 but SPINE raises it implicitly via fan-out; the
    # IMPLEMENT dispatch loop re-dispatches one Send per pending/failed slice and
    # only drains when failed_slices is empty, so a non-converging slice (or a
    # 0-token server-crash spin where the token-budget breaker can't advance —
    # trace 019ece87) loops far past any healthy phase. This caps super-steps far
    # below LangGraph's runaway backstop so a spin aborts in seconds, surfacing as
    # needs_review rather than thousands of dead Sends. 0 = use LangGraph default.
    subgraph_recursion_limit: int = 80
    specify_context_token_budget: int = 30000

    # Token budget for the findings block injected into plan/specify
    # synthesize prompts. Caps the rendered output of _format_findings
    # so an accumulation of long research summaries can't dominate the
    # synthesize prompt (trace 019e6d27: 42K plan-synthesize prompt
    # vs 40K TokenBudgetCompactor threshold). 0 or negative = unbounded.
    synthesize_findings_token_budget: int = 20000

    # Completion-token clamp for the SPECIFY/PLAN synthesizer calls. The
    # structured spec/plan JSON is 2-4K tokens; without a clamp the synth
    # request inherits the global max_completion_tokens (30K) and a
    # finite-window model 400s once prompt + completion budget exceeds the
    # window (trace 019eb3dd: ~33K prompt + 30K requested vs a 60K window).
    # 8K (not 4K) leaves headroom for reasoning-channel tokens on thinking
    # models. Per-phase ``max_completion_tokens`` overrides still win.
    synthesize_max_completion_tokens: int = 8000
    # PLAN-specific override of the synth completion clamp (passed as the
    # ``phase_cap`` to synthesis_completion_cap, exactly as IMPLEMENT passes
    # implement_max_completion_tokens). The plan JSON has grown well past the
    # 2-4K the shared 8K cap assumed: every slice now carries target_files,
    # reference_symbols and an edit_plan with full-signature intents, so a
    # larger plan truncated the forced write_structured_plan tool call mid-
    # arguments at exactly 8K and an identical retry truncated identically
    # (trace 019eb940). GLM-4.7-Flash plans on a 60K window, so 16K leaves
    # ample prompt room; the length-escalation retry (escalated_completion_cap)
    # still doubles from here if a plan legitimately needs more. SPECIFY keeps
    # the tighter shared cap. 0 falls back to synthesize_max_completion_tokens.
    plan_synthesize_max_completion_tokens: int = 16384
    # Safety margin subtracted from the window when computing the synth
    # input budget — covers tool schemas, chat-template framing, and
    # tiktoken-vs-model-tokenizer drift (cl100k underestimates Qwen ~10%).
    synthesize_overhead_tokens: int = 4000
    # Kill switch for map-reduce evidence compression in the synth nodes.
    # When False, oversized findings/recall blocks degrade by truncation
    # only (pre-019eb3dd behaviour).
    evidence_compression_enabled: bool = True

    # Token budget for the prior-phase findings injected into PLAN
    # researcher / manager prompts (SPECIFY's research_log.json findings
    # carried across into PLAN exploration). Tighter than the synthesis
    # budget because the PLAN researcher prompt already carries the spec
    # (~8K). 0 or negative = unbounded.
    prior_phase_findings_token_budget: int = 6000

    # Global default for the model's ``max_completion_tokens`` request
    # field. Per-provider settings still win — this is the fallback used
    # when ``providers.llm[].max_completion_tokens`` is unset. Without a
    # cap, finite-window local providers (vLLM/SGLang) consume the entire
    # remaining context as output budget and 400 once prompt+budget
    # exceeds the model window (trace 019e6e53: 80K-context model
    # rejected 80001-token prompts with "0 output tokens requested"
    # because no per-provider cap was set). 0 disables the global
    # fallback (falls back to provider/library defaults).
    max_completion_tokens: int = 0

    # Per-topic recall lookup (runs between research_manager and the
    # research_router). For each topic emitted by the manager, recall the
    # top-K symbols whose cosine similarity is at least this threshold —
    # those are then attached to the topic that gets sent to the explore
    # subagent.
    topic_lookup_top_k: int = 5
    topic_lookup_min_similarity: float = 0.5

    # Distributed onboarding engine (design Revision 2). The synthesis stage
    # decomposes into a documentation-manager + section-worker hierarchy where
    # no LLM ever sees the whole manifest. ``onboarding_section_token_cap`` is
    # the hard per-fragment ceiling resolve_fragment() enforces (degrade
    # key_symbols→names→truncate so a fragment never exceeds it);
    # ``onboarding_max_sections`` caps call volume by ranking + grouping the
    # module tail in the compact index. Per-phase model overrides resolve via
    # the existing ``providers.phases`` convention under the keys
    # ``onboarding/doc-manager`` and ``onboarding/section-worker`` (both
    # default to the resolved default model when unset).
    onboarding_section_token_cap: int = 6000
    onboarding_max_sections: int = 32
    # Hard cap on completion tokens for the section-worker's single
    # with_structured_output call. Sections are 100-700 tokens of markdown;
    # without a tight cap a local model can run to the global max_completion_tokens
    # window (16K) before raising LengthFinishReasonError, costing 290-450s per
    # affected worker before the retry. 2048 is safe headroom above the largest
    # observed section (2498 chars ≈ 700 tokens). Trace 019e7855.
    onboarding_section_max_completion_tokens: int = 2048
    # ``onboarding_distributed_analysis`` routes analysis through the
    # deterministic map-reduce graph (one explorer Send per module unit); when
    # ``False`` the manager calls ``RepoAnalyzer.analyze`` inline with 0 explorer
    # Sends (the monolithic fallback for tiny repos). ``onboarding_explorer_llm``
    # + ``onboarding_explorer_max_cycles`` gate the opt-in LLM-enriched explorer
    # mode (NOT implemented yet — flags only; the deterministic branch is the
    # only path today).
    onboarding_distributed_analysis: bool = True
    onboarding_explorer_llm: bool = False
    onboarding_explorer_max_cycles: int = 3
    # When True (default), phase agents receive the relevant onboarding document
    # injected into their system prompt (hybrid: the most-relevant doc per phase
    # in full, the rest referenced by path). Set False to disable injection if it
    # regresses small-model behaviour. See spine.agents.skills_resolver.
    onboarding_context_injection: bool = True

    # ── Cross-run distilled experience ──
    # SPINE captures a compact "lesson" from each run's critic/adversarial
    # feedback and injects the relevant lessons into the matching phase's
    # prompt on future runs, so repeat defects are caught earlier. Lessons
    # live under ``experience_path`` (file-backed, deduped, per-phase capped).
    experience_path: str = ".spine/experience"
    # Capture lessons at the end of each run (write path). Default on.
    experience_capture: bool = True
    # Inject lessons into phase prompts (read path). Default on.
    experience_injection: bool = True
    # Run a one-shot LLM pass at capture time that rewrites each run-specific
    # defect into a general, reusable rule (and drops un-generalisable noise).
    # One cheap call per run that produced lessons; fails open to the raw
    # deterministic lessons. Default on.
    experience_generalize: bool = True

    # Researcher convergence steering (see ResearcherConvergenceMiddleware)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    # Token-based phase compaction (see TokenBudgetCompactor); default off
    token_compaction: TokenCompactionConfig = field(default_factory=TokenCompactionConfig)

    # Per-model behavioural profiles, keyed by model string. Each entry supplies
    # provider-level defaults (``context_window``, ``reasoning``,
    # ``guided_decoding``, ``tool_format``, …) that sit UNDER explicit
    # provider/phase settings — see resolve_provider_config(). Lets a model be
    # configured once by name regardless of which phase/provider routes to it.
    model_profiles: dict = field(default_factory=dict)

    # Ephemeral GPU pod(s) for the run (see spine.infra.ephemeral_pod). A run
    # boots only the pods whose ``phases`` intersect the phases it will execute,
    # and tears them down at the end. Two shapes, mutually exclusive:
    #   * ``ephemeral_pods`` — a LIST of named pods, each with a ``phases`` list.
    #     A run boots only the pod(s) its lane needs (e.g. reasoner for
    #     specify/plan, coder for implement/verify).
    #   * ``ephemeral_pod`` — the legacy SINGLE unnamed pod (always boots when
    #     enabled). Kept for back-compat.
    # Each pod backs a ``providers.llm[]`` entry whose ``base_url`` is
    # ``env:<pod url_env>``; which phases route to it stays the existing
    # ``providers.phases`` routing. Stored verbatim; parsed by parse_pods().
    ephemeral_pod: dict = field(default_factory=dict)
    ephemeral_pods: list = field(default_factory=list)

    @staticmethod
    def _find_workspace_root() -> str:
        """Auto-detect workspace root by searching upward for ``.spine/``.

        Search order:
          1. Walk up from CWD (works when launched from the project root)
          2. Walk up from the spine package directory (handles Streamlit,
             systemd, or other runners that change CWD away from the project)
          3. Fall back to CWD if neither search finds ``.spine/``

        This mirrors the ``_load_dotenv`` strategy for ``.env`` discovery.
        """
        # Strategy 1: walk up from CWD
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / ".spine").is_dir():
                return str(candidate)

        # Strategy 2: walk up from the package directory — this handles
        # Streamlit, systemd, or other runners that change CWD away from
        # the project root (e.g. to /root or /tmp).  Without this, the
        # workspace_root resolves to an inaccessible directory like /root,
        # causing LocalShellBackend to fail with Permission denied.
        pkg_dir = Path(__file__).resolve().parent
        for candidate in pkg_dir.parents:
            if (candidate / ".spine").is_dir():
                return str(candidate)

        return str(cwd)

    @classmethod
    def load(cls, path: str = ".spine/config.yaml") -> SpineConfig:
        """Load configuration from a YAML file, falling back to defaults.

        When *path* is relative and doesn't exist relative to CWD, also
        searches upward from the spine package directory for the config
        file.  This ensures Streamlit, systemd, and other runners that
        change CWD away from the project root can still find the config.

        Args:
            path: Path to the configuration YAML file.

        Returns:
            A SpineConfig instance with values from the file or defaults.
        """
        config = {}
        resolved_path = path

        if os.path.exists(path):
            resolved_path = path
        else:
            # Search from the package directory for the config file
            # (same strategy as _find_workspace_root and _load_dotenv).
            pkg_dir = Path(__file__).resolve().parent
            for candidate in pkg_dir.parents:
                candidate_path = candidate / path
                if candidate_path.is_file():
                    resolved_path = str(candidate_path)
                    break

        if os.path.exists(resolved_path):
            try:
                with open(resolved_path) as f:
                    config = yaml.safe_load(f) or {}
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                # If YAML is invalid, fall back to empty config (defaults will be used)
                config = {}

        spine = config.get("spine", {})

        # ── MCP servers ──────────────────────────────────────────────
        mcp_servers: dict[str, dict] = {}
        raw_mcp = config.get("mcp_servers", {})
        for name, server_cfg in raw_mcp.items():
            if not isinstance(server_cfg, dict):
                continue
            mcp_servers[name] = {
                "transport": server_cfg.get("transport", "stdio"),
                "command": server_cfg.get("command", ""),
                "args": server_cfg.get("args", []),
                "env": server_cfg.get("env", {}),
            }
        # Allow env var override (JSON string)
        env_mcp = os.environ.get("SPINE_MCP_SERVERS")
        if env_mcp:
            try:
                mcp_servers.update(json.loads(env_mcp))
            except json.JSONDecodeError:
                pass

        # Resolve workspace_root: use Path.resolve() to get the canonical
        # (case-correct) absolute path.  On case-sensitive Linux, a typo
        # like /home/pat/projects vs /home/pat/Projects would silently
        # point at a different (or non-existent) directory, causing the
        # deep agent to write files to the wrong place.
        #
        # Auto-detect by searching upward for .spine/ when neither the env
        # var nor the config file explicitly set a value.
        raw_root = os.getenv("SPINE_WORKSPACE_ROOT", spine.get("workspace_root", None))
        if raw_root is None:
            raw_root = cls._find_workspace_root()
        resolved_root = str(Path(raw_root).resolve())

        # Sanity check: if workspace_root points to a directory the agent
        # can't write to (e.g. /root when not running as root), log a
        # warning.  This is a common failure mode when CWD is wrong and
        # auto-detection falls back to an inaccessible path.
        root_path = Path(resolved_root)
        if not os.access(resolved_root, os.W_OK):
            import logging

            logging.getLogger(__name__).warning(
                "workspace_root %s is not writable — agents will fail. "
                "Set SPINE_WORKSPACE_ROOT or add 'workspace_root' to "
                ".spine/config.yaml to fix this.",
                resolved_root,
            )
        elif not (root_path / ".spine").is_dir():
            import logging

            logging.getLogger(__name__).warning(
                "workspace_root %s has no .spine/ directory — auto-detection "
                "may have resolved to the wrong path. Consider setting "
                "SPINE_WORKSPACE_ROOT explicitly.",
                resolved_root,
            )

        return cls(
            checkpoint_path=os.getenv(
                "SPINE_CHECKPOINT_PATH", spine.get("checkpoint_path", ".spine/spine.db")
            ),
            artifact_path=os.getenv(
                "SPINE_ARTIFACT_PATH", spine.get("artifact_path", ".spine/artifacts")
            ),
            project_path=os.getenv(
                "SPINE_PROJECT_PATH", spine.get("project_path", ".spine/project")
            ),
            max_critic_retries=int(
                os.getenv("SPINE_MAX_CRITIC_RETRIES", spine.get("max_critic_retries", 2))
            ),
            max_adversarial_retries=int(
                os.getenv(
                    "SPINE_MAX_ADVERSARIAL_RETRIES",
                    spine.get("max_adversarial_retries", 2),
                )
            ),
            work_type=os.getenv("SPINE_WORK_TYPE", spine.get("work_type", "task")),
            providers=config.get("providers", {}),
            queue_backend=os.getenv("SPINE_QUEUE_BACKEND", spine.get("queue_backend", "sqlite")),
            queue_path=os.getenv("SPINE_QUEUE_PATH", spine.get("queue_path", ".spine/queue.db")),
            workspace_root=resolved_root,
            interpreter_enabled=os.getenv(
                "SPINE_INTERPRETER", str(spine.get("interpreter_enabled", False)).lower()
            )
            in ("1", "true", "yes"),
            tool_schema_validation=os.getenv(
                "SPINE_TOOL_SCHEMA_VALIDATION",
                str(spine.get("tool_schema_validation", True)).lower(),
            )
            not in ("0", "false", "no"),
            tool_call_normalize=os.getenv(
                "SPINE_TOOL_CALL_NORMALIZE",
                str(spine.get("tool_call_normalize", False)).lower(),
            )
            in ("1", "true", "yes"),
            error_diagnosis=os.getenv(
                "SPINE_ERROR_DIAGNOSIS",
                str(spine.get("error_diagnosis", False)).lower(),
            )
            in ("1", "true", "yes"),
            phase_timeouts=spine.get(
                "phase_timeouts",
                {
                    "specify": 0,
                    "plan": 0,
                    "tasks": 0,
                    "implement": 0,
                    "verify": 0,
                    "critic": 0,
                },
            ),
            default_timeout=int(spine.get("default_timeout", 0)),
            mcp_servers=mcp_servers,
            guided_decoding=os.getenv(
                "SPINE_GUIDED_DECODING",
                str(spine.get("guided_decoding", False)).lower(),
            )
            in ("1", "true", "yes"),
            embedding_provider=spine.get("embedding_provider", "openai-embeddings"),
            recall_k=int(spine.get("recall_k", 10)),
            index_tests=str(spine.get("index_tests", False)).lower() in ("1", "true", "yes"),
            rrf_vector_weight=float(spine.get("rrf_vector_weight", 0.2)),
            rrf_bm25_weight=float(spine.get("rrf_bm25_weight", 1.0)),
            reranker_provider=spine.get("reranker_provider", ""),
            rerank_pool=int(spine.get("rerank_pool", 50)),
            vector_indexing=spine.get(
                "vector_indexing",
                {
                    "max_concurrent_chunks": 5,
                    "batch_size": 100,
                },
            ),
            recall_gate_confidence=float(spine.get("recall_gate_confidence", 0.75)),
            recall_gate_min_hits=int(spine.get("recall_gate_min_hits", 5)),
            recall_gate_trivial_max_chars=int(
                spine.get("recall_gate_trivial_max_chars", 150)
            ),
            summarise_max_completion_tokens=int(
                spine.get("summarise_max_completion_tokens", 4096)
            ),
            researcher_supervisor_max_completion_tokens=int(
                spine.get("researcher_supervisor_max_completion_tokens", 1024)
            ),
            research_max_rounds=int(spine.get("research_max_rounds", 3)),
            research_max_parallel_explores=int(
                spine.get("research_max_parallel_explores", 4)
            ),
            research_lean_confidence=float(
                spine.get("research_lean_confidence", 0.85)
            ),
            research_lean_max_rounds=int(spine.get("research_lean_max_rounds", 2)),
            research_lean_max_parallel_explores=int(
                spine.get("research_lean_max_parallel_explores", 2)
            ),
            research_lean_categories=list(
                spine.get("research_lean_categories", ["Frontend/UI"])
            ),
            research_synth_variants=int(spine.get("research_synth_variants", 1)),
            research_synth_scorer=str(
                spine.get("research_synth_scorer", "structural")
            ),
            research_aggregate_merge=bool(
                spine.get("research_aggregate_merge", False)
            ),
            research_aggregate_min_findings=int(
                spine.get("research_aggregate_min_findings", 8)
            ),
            specify_context_token_budget=int(
                spine.get("specify_context_token_budget", 30000)
            ),
            synthesize_findings_token_budget=int(
                os.getenv(
                    "SPINE_SYNTHESIZE_FINDINGS_TOKEN_BUDGET",
                    spine.get("synthesize_findings_token_budget", 20000),
                )
            ),
            synthesize_max_completion_tokens=int(
                os.getenv(
                    "SPINE_SYNTHESIZE_MAX_COMPLETION_TOKENS",
                    spine.get("synthesize_max_completion_tokens", 8000),
                )
            ),
            plan_synthesize_max_completion_tokens=int(
                os.getenv(
                    "SPINE_PLAN_SYNTHESIZE_MAX_COMPLETION_TOKENS",
                    spine.get("plan_synthesize_max_completion_tokens", 16384),
                )
            ),
            synthesize_overhead_tokens=int(
                os.getenv(
                    "SPINE_SYNTHESIZE_OVERHEAD_TOKENS",
                    spine.get("synthesize_overhead_tokens", 4000),
                )
            ),
            evidence_compression_enabled=os.getenv(
                "SPINE_EVIDENCE_COMPRESSION",
                str(spine.get("evidence_compression_enabled", True)).lower(),
            )
            not in ("0", "false", "no"),
            prior_phase_findings_token_budget=int(
                os.getenv(
                    "SPINE_PRIOR_PHASE_FINDINGS_TOKEN_BUDGET",
                    spine.get("prior_phase_findings_token_budget", 6000),
                )
            ),
            max_completion_tokens=int(
                os.getenv(
                    "SPINE_MAX_COMPLETION_TOKENS",
                    spine.get("max_completion_tokens", 0),
                )
            ),
            decompose_max_traceback_chars=int(
                os.getenv(
                    "SPINE_DECOMPOSE_MAX_TRACEBACK_CHARS",
                    spine.get("decompose_max_traceback_chars", 4000),
                )
            ),
            decompose_max_completion_tokens=int(
                os.getenv(
                    "SPINE_DECOMPOSE_MAX_COMPLETION_TOKENS",
                    spine.get("decompose_max_completion_tokens", 4096),
                )
            ),
            implement_max_decompose_depth=int(
                os.getenv(
                    "SPINE_IMPLEMENT_MAX_DECOMPOSE_DEPTH",
                    spine.get("implement_max_decompose_depth", 1),
                )
            ),
            implement_max_turns=int(
                os.getenv(
                    "SPINE_IMPLEMENT_MAX_TURNS",
                    spine.get("implement_max_turns", 30),
                )
            ),
            verify_max_turns=int(
                os.getenv(
                    "SPINE_VERIFY_MAX_TURNS",
                    spine.get("verify_max_turns", 20),
                )
            ),
            implement_synthesis_placement=bool(
                spine.get("implement_synthesis_placement", False)
            ),
            implement_synthesis_variants=int(
                spine.get("implement_synthesis_variants", 1)
            ),
            verify_evidence_then_judge=bool(
                spine.get("verify_evidence_then_judge", False)
            ),
            implement_max_slice_dispatches=int(
                os.getenv(
                    "SPINE_IMPLEMENT_MAX_SLICE_DISPATCHES",
                    spine.get("implement_max_slice_dispatches", 100),
                )
            ),
            subgraph_recursion_limit=int(
                os.getenv(
                    "SPINE_SUBGRAPH_RECURSION_LIMIT",
                    spine.get("subgraph_recursion_limit", 80),
                )
            ),
            topic_lookup_top_k=int(spine.get("topic_lookup_top_k", 5)),
            topic_lookup_min_similarity=float(
                spine.get("topic_lookup_min_similarity", 0.5)
            ),
            onboarding_section_token_cap=int(
                os.getenv(
                    "SPINE_ONBOARDING_SECTION_TOKEN_CAP",
                    spine.get("onboarding_section_token_cap", 6000),
                )
            ),
            onboarding_max_sections=int(
                os.getenv(
                    "SPINE_ONBOARDING_MAX_SECTIONS",
                    spine.get("onboarding_max_sections", 32),
                )
            ),
            onboarding_section_max_completion_tokens=int(
                os.getenv(
                    "SPINE_ONBOARDING_SECTION_MAX_COMPLETION_TOKENS",
                    spine.get("onboarding_section_max_completion_tokens", 2048),
                )
            ),
            onboarding_distributed_analysis=os.getenv(
                "SPINE_ONBOARDING_DISTRIBUTED_ANALYSIS",
                str(spine.get("onboarding_distributed_analysis", True)).lower(),
            )
            not in ("0", "false", "no"),
            onboarding_explorer_llm=os.getenv(
                "SPINE_ONBOARDING_EXPLORER_LLM",
                str(spine.get("onboarding_explorer_llm", False)).lower(),
            )
            in ("1", "true", "yes"),
            onboarding_explorer_max_cycles=int(
                os.getenv(
                    "SPINE_ONBOARDING_EXPLORER_MAX_CYCLES",
                    spine.get("onboarding_explorer_max_cycles", 3),
                )
            ),
            experience_path=os.getenv(
                "SPINE_EXPERIENCE_PATH",
                spine.get("experience_path", ".spine/experience"),
            ),
            experience_capture=os.getenv(
                "SPINE_EXPERIENCE_CAPTURE",
                str(spine.get("experience_capture", True)).lower(),
            )
            not in ("0", "false", "no"),
            experience_injection=os.getenv(
                "SPINE_EXPERIENCE_INJECTION",
                str(spine.get("experience_injection", True)).lower(),
            )
            not in ("0", "false", "no"),
            experience_generalize=os.getenv(
                "SPINE_EXPERIENCE_GENERALIZE",
                str(spine.get("experience_generalize", True)).lower(),
            )
            not in ("0", "false", "no"),
            convergence=_parse_convergence_config(spine.get("convergence", {})),
            token_compaction=_parse_token_compaction_config(
                spine.get("token_compaction", {})
            ),
            model_profiles=spine.get("model_profiles", {}) or {},
            ephemeral_pod=config.get("ephemeral_pod", {}) or {},
            ephemeral_pods=config.get("ephemeral_pods", []) or [],
        )

    def resolve_model(self, phase: str | None = None, escalation_level: int = 0) -> str:
        """Resolve the LLM model identifier from provider config.

        Supports per-phase and per-subagent model overrides via the
        ``providers.phases`` section of ``.spine/config.yaml``.  Resolution
        order:

        1. ``providers.phases.<phase>.model`` (explicit model string)
        2. ``providers.phases.<phase>.provider`` → look up the named
           provider in ``providers.llm[]`` and return its ``model``
        3. ``providers.phases.<phase/subagents/name>.model`` or
           ``.provider`` (e.g. ``implement/subagents/slice-implementer``)
        4. First enabled LLM provider's ``model`` field
        5. ``SPINE_MODEL`` env var
        6. ``ValueError`` if none of the above are set

        Path-style keys are resolved by walking prefixes from most-specific to
        least: ``implement/decomposer/fallback`` consults
        ``implement/decomposer/fallback``, then ``implement/decomposer``, then
        ``implement`` — so a subagent/sub-phase override always wins over the
        bare phase default, and an intermediate key (e.g.
        ``implement/decomposer``) covers all of its modes at once.

        When ``escalation_level > 0`` an escalation ladder entry
        (``providers.phases.<phase>.escalation[level-1]``, most-specific key
        wins, clamped to the strongest defined rung) overrides the base model.
        With no ladder defined anywhere, a non-zero level resolves identically
        to level 0 — so escalation degrades gracefully to the static config.

        Args:
            phase: Optional phase or phase/subagent path (e.g. ``"implement"``
                or ``"implement/subagents/slice-implementer"``).  When
                ``None``, only the default provider and env var are consulted.
            escalation_level: Failure-driven escalation rung (0 = base). Driven
                by SPINE's persisted counters (slice ``_decompose_depth`` /
                critic ``retry_count``), not an in-memory failure rate.

        Returns:
            A model string like ``openrouter:z-ai/glm-4.5-air:free``.

        Raises:
            ValueError: If no model is configured anywhere.
        """
        # Escalation overlay: a defined ladder entry overrides the base model.
        if phase and escalation_level > 0:
            entry = self._escalation_entry_for(phase, escalation_level)
            if entry is not None:
                if entry.get("model"):
                    return entry["model"]
                provider_ref = entry.get("provider")
                if provider_ref:
                    named = self._lookup_provider_by_name(provider_ref)
                    if named and named.get("model"):
                        return named["model"]

        # Check phase-specific overrides first (more specific key wins).
        if phase:
            phases = self.providers.get("phases", {})
            # Walk path prefixes from most-specific to least: e.g.
            # 'implement/decomposer/fallback' -> 'implement/decomposer' ->
            # 'implement'. An intermediate key like 'implement/decomposer' can
            # then override all three decomposer modes (plan/fallback/per_file)
            # without enumerating each, while an exact key still wins over it.
            parts = phase.split("/")
            for i in range(len(parts), 0, -1):
                key = "/".join(parts[:i])
                phase_cfg = phases.get(key, {})
                if not isinstance(phase_cfg, dict):
                    continue
                # 1. Explicit model string on the phase config
                if phase_cfg.get("model"):
                    return phase_cfg["model"]
                # 2. Provider reference — look up the named provider
                provider_ref = phase_cfg.get("provider")
                if provider_ref:
                    named = self._lookup_provider_by_name(provider_ref)
                    if named and named.get("model"):
                        return named["model"]

        # Default provider resolution
        provider = self.resolve_active_provider()
        if provider:
            return provider["model"]

        env_model = os.getenv("SPINE_MODEL")
        if env_model:
            return env_model

        raise ValueError(
            "No LLM model configured. Set 'providers.llm[].model' in "
            ".spine/config.yaml or set the SPINE_MODEL environment variable."
        )

    # ── Provider keys that phases can override locally ────────────────
    _PROVIDER_KEYS: tuple[str, ...] = (
        "base_url",
        "api_key",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "request_timeout",
        "stream_chunk_timeout",
        "max_retries",
        "guided_decoding",
        "structured_method",
        "streaming",
        "fallback_provider",
        "degrade_with",
        "max_concurrent_calls",
        "stream_usage",
        "reasoning",
        "context_window",
        "rsa",
        "cam",
    )

    def _resolve_model_profile(
        self, phase: str | None = None, escalation_level: int = 0
    ) -> dict:
        """Return the ``model_profiles`` entry for the phase's effective model.

        Profiles supply provider-level defaults keyed by model string. Matching
        tries the full model spec first (e.g. ``"openai:qwen3.6"``), then the
        normalised name (``"qwen3.6"``) using the same stripping that
        ``helpers._extract_model_name`` applies, so a profile can be keyed either
        way. Returns an empty dict when profiles are unset or none match — so the
        merge in :meth:`resolve_provider_config` is a no-op by default.
        """
        if not self.model_profiles:
            return {}
        try:
            model_spec = self.resolve_model(
                phase=phase, escalation_level=escalation_level
            )
        except ValueError:
            return {}
        if not model_spec:
            return {}
        profile = self.model_profiles.get(model_spec)
        if profile is None:
            from spine.agents.helpers import _extract_model_name

            profile = self.model_profiles.get(_extract_model_name(model_spec))
        return dict(profile) if isinstance(profile, dict) else {}

    def resolve_active_provider(self) -> dict | None:
        """Return the full config dict for the first enabled LLM provider.

        This exposes ``base_url``, ``api_key``, ``temperature``, and other
        provider-specific fields that ``resolve_model()`` alone discards.
        Returns ``None`` when no enabled provider is found.

        Returns:
            The provider config dict, or ``None``.
        """
        llm_providers = self.providers.get("llm", [])
        for provider in llm_providers:
            if provider.get("enabled", True) and provider.get("model"):
                return provider
        return None

    def _lookup_provider_by_name(self, name: str) -> dict | None:
        """Find a named provider in ``providers.llm[]``.

        Args:
            name: The ``"name"`` field of the provider entry to find.

        Returns:
            The full provider config dict, or ``None`` if not found.
        """
        for provider in self.providers.get("llm", []):
            if provider.get("name") == name:
                return provider
        return None

    @staticmethod
    def _escalation_entry(phase_cfg: dict, level: int) -> dict | None:
        """Return the escalation ladder entry for ``level`` on a phase config.

        ``providers.phases.<phase>.escalation`` is a list where index ``i`` is
        escalation *level ``i + 1``* (level 0 is the phase's own base config).
        When ``level`` exceeds the ladder length the request is clamped to the
        last (strongest) entry — so once the ladder is exhausted, escalation
        stays at the top rung rather than silently falling back. Returns
        ``None`` when ``level <= 0`` or no usable ladder is present.
        """
        if level <= 0 or not isinstance(phase_cfg, dict):
            return None
        ladder = phase_cfg.get("escalation")
        if not isinstance(ladder, list) or not ladder:
            return None
        entry = ladder[min(level, len(ladder)) - 1]
        return entry if isinstance(entry, dict) else None

    def _escalation_entry_for(
        self, phase: str | None, level: int
    ) -> dict | None:
        """Most-specific escalation ladder entry across a phase path, or None.

        Walks path prefixes most-specific→least (like :meth:`resolve_model`) and
        returns the first key that defines an ``escalation`` ladder, clamped to
        ``level`` via :meth:`_escalation_entry`. A ladder defined on the bare
        phase (``implement``) therefore covers its subagents
        (``implement/subagents/slice-implementer``) unless one overrides it.
        """
        if not phase or level <= 0:
            return None
        phases = self.providers.get("phases", {})
        parts = phase.split("/")
        for i in range(len(parts), 0, -1):
            key = "/".join(parts[:i])
            entry = self._escalation_entry(phases.get(key, {}), level)
            if entry is not None:
                return entry
        return None

    def resolve_provider_config(
        self, phase: str | None = None, escalation_level: int = 0
    ) -> dict:
        """Resolve provider-level settings for a given phase.

        Unlike :meth:`resolve_model` (which returns only the model string),
        this returns the full provider config dict — ``base_url``,
        ``api_key``, ``temperature``, ``max_tokens``,
        ``max_completion_tokens``, ``request_timeout``, ``max_retries`` —
        after applying any per-phase overrides.

        Resolution order (most specific wins, values are merged):

        1. Phase config's direct provider keys (``base_url``,
           ``temperature``, etc.) — take priority
        2. Phase config's ``provider`` reference — look up
           ``providers.llm[name]`` and inherit its settings
        3. First enabled provider in ``providers.llm[]``

        Args:
            phase: Optional phase or phase/subagent path (e.g.
                ``"implement"`` or
                ``"implement/subagents/slice-implementer"``).  When
                ``None``, only the default provider is consulted.

        Returns:
            A provider config dict containing ``base_url``, ``api_key``,
            and any other provider-level fields.  May be empty if no
            enabled provider is found.

        Example config::

            providers:
              llm:
                - name: vllm-local
                  model: openai:qwen3.6
                  base_url: http://localhost:8000/v1
                  api_key: vllm
                  temperature: 0.7
                  enabled: true
                - name: openrouter-gateway
                  model: openrouter:deepseek/deepseek-v4-pro
                  enabled: true
              phases:
                implement:
                  provider: vllm-local           # inherit vllm-local settings
                  temperature: 0.3               # but override temp
                verify:
                  base_url: http://other:8000/v1  # fully custom
                  api_key: other-key
        """
        # When escalated, an escalation ladder entry overrides the model/provider
        # (and may carry its own provider-level keys). Resolved once here and
        # used both to pick the base provider (its ``provider`` ref) and as the
        # highest-priority override layer (Step 3).
        esc_entry = self._escalation_entry_for(phase, escalation_level)

        # ── Step 1: resolve base provider (from reference or default) ──
        base: dict = {}
        # An escalation entry's ``provider`` ref defines the base provider.
        if esc_entry is not None and esc_entry.get("provider"):
            named = self._lookup_provider_by_name(esc_entry["provider"])
            if named:
                base = dict(named)
        if not base and phase:
            phases = self.providers.get("phases", {})
            for key in (phase, phase.split("/")[0] if "/" in phase else None):
                if key is None:
                    continue
                phase_cfg = phases.get(key, {})
                if not isinstance(phase_cfg, dict):
                    continue
                provider_ref = phase_cfg.get("provider")
                if provider_ref:
                    named = self._lookup_provider_by_name(provider_ref)
                    if named:
                        base = dict(named)
                        break

        if not base:
            default = self.resolve_active_provider()
            if default:
                base = dict(default)

        # ── Step 1.5: layer per-model profile defaults UNDER explicit values ──
        # Behavioural defaults (context_window, reasoning, guided_decoding,
        # tool_format, …) keyed by model string. ``setdefault`` leaves any
        # provider/base value intact, and Step 2's phase overrides still win
        # because they run afterwards — so precedence is phase > provider >
        # profile. No-op when ``model_profiles`` is unset.
        for k, v in self._resolve_model_profile(
            phase=phase, escalation_level=escalation_level
        ).items():
            base.setdefault(k, v)

        # ── Step 2: apply phase-level overrides on top ──
        if phase:
            phases = self.providers.get("phases", {})
            for key in (phase, phase.split("/")[0] if "/" in phase else None):
                if key is None:
                    continue
                phase_cfg = phases.get(key, {})
                if not isinstance(phase_cfg, dict):
                    continue
                for k in self._PROVIDER_KEYS:
                    if k in phase_cfg:
                        base[k] = phase_cfg[k]

        # ── Step 3: escalation entry's own keys win (highest priority) ──
        # The phase's tuning (Step 2) carries onto the escalated model unless the
        # entry overrides it explicitly — so e.g. a frontier escalation can keep
        # the phase temperature but bump its own request_timeout/context_window.
        if esc_entry is not None:
            for k in self._PROVIDER_KEYS:
                if k in esc_entry:
                    base[k] = esc_entry[k]

        return base

    def resolve_embedding_provider(self) -> dict | None:
        """Resolve the embedding provider config.

        Uses the ``embedding_provider`` name to look up the provider in
        ``providers.embedding[]``.

        Returns:
            The embedding provider config dict, or None if not found.
        """
        for provider in self.providers.get("embedding", []):
            if provider.get("name") == self.embedding_provider:
                return provider
        return None

    def resolve_reranker_provider(self) -> dict | None:
        """Resolve the reranker provider config.

        Uses the ``reranker_provider`` name to look it up in
        ``providers.reranker[]``. Returns None when reranking is disabled
        (empty name) or the named provider is absent — callers treat that
        as "no reranking" and fall back to the fused order.
        """
        if not self.reranker_provider:
            return None
        for provider in self.providers.get("reranker", []):
            if provider.get("name") == self.reranker_provider:
                return provider
        return None

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        for p in [self.checkpoint_path, self.artifact_path, self.queue_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(self.project_path).mkdir(parents=True, exist_ok=True)
        Path(self.experience_path).mkdir(parents=True, exist_ok=True)
