#!/usr/bin/env python3
"""trace_audit.py — Telemetry audit for SPINE LangSmith traces.

Reports prompt:completion ratios, prompt-cache utilisation, duplicate
tool calls within the dedupe boundary (a single agent invocation),
duplicate tool calls across invocations (would-be-shared cache misses),
context-management markers, and any tool whose name is not on the SPINE
allowlist.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from langsmith import Client


# Every tool SPINE agents are expected to use. Anything outside this set
# means an agent reached for a non-custom tool — flag it.
KNOWN_TOOL_NAMES: frozenset[str] = frozenset({
    # FilesystemMiddleware
    "read_file", "write_file", "edit_file", "glob", "grep", "ls", "execute",
    # TodoListMiddleware
    "write_todos",
    # SPINE custom tools
    "search_codebase", "ast_extract_symbol",
    "read_edit_lint", "recall",
    # Artifact/orchestrator tools
    "write_structured_plan", "write_specification",
    "read_prior_artifacts", "read_work_context",
    "write_implementation_report", "read_slice_files",
    # CodeInterpreterMiddleware
    "eval",
    # SubAgentMiddleware
    "task",
})

# Tool name prefixes that are always custom (variable-suffix tool families).
KNOWN_TOOL_PREFIXES: tuple[str, ...] = ("mcp_",)

# Tools the deduper (ReadCacheMiddleware) MUST short-circuit.
# Mirrors spine.agents.context_editing._DEDUPED_TOOLS plus the mcp_ prefix.
DEDUPED_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "search_codebase", "ast_extract_symbol", "glob", "grep",
})


def is_known_tool(name: str) -> bool:
    return name in KNOWN_TOOL_NAMES or any(name.startswith(p) for p in KNOWN_TOOL_PREFIXES)


def is_dedupable(name: str) -> bool:
    return name in DEDUPED_TOOL_NAMES or name.startswith("mcp_")


def agent_ns(run) -> str:
    """Dedupe boundary = checkpoint_ns minus the trailing |tools:<id> segment.

    The ReadCacheMiddleware caches in SpineContext.read_cache, which is
    freshly instantiated per agent.ainvoke() and shared across the
    tool-call loop within that one invocation. checkpoint_ns identifies
    the calling agent; the |tools:<id> tail identifies the per-step tool
    executor and must be stripped to recover the agent boundary.
    """
    md = (run.extra or {}).get("metadata", {}) or {}
    ns = md.get("checkpoint_ns") or md.get("langgraph_checkpoint_ns") or ""
    parts = [p for p in ns.split("|") if not p.startswith("tools:")]
    return "|".join(parts) or "(no-ns)"


def _sig(run) -> tuple[str, str]:
    return (run.name, json.dumps(run.inputs, sort_keys=True, default=str)[:400])


def audit_trace(trace_id: str) -> None:
    client = Client()
    print(f"[*] Querying trace: {trace_id} ...")

    try:
        runs = list(client.list_runs(trace_id=trace_id))
    except Exception as exc:
        print(f"[-] Error querying LangSmith: {exc}")
        return

    if not runs:
        print("[-] No runs found for that trace ID.")
        return

    runs.sort(key=lambda r: r.start_time)
    llm_runs = [r for r in runs if r.run_type == "llm" and r.end_time]
    tool_runs = [r for r in runs if r.run_type == "tool"]
    chain_runs = [r for r in runs if r.run_type == "chain"]

    print(f"[+] Loaded {len(runs)} spans  (LLM={len(llm_runs)}  Tool={len(tool_runs)}  Chain={len(chain_runs)})")

    # ── 1. Token efficiency ───────────────────────────────────────────────
    total_prompt = total_completion = total_cache_read = total_reasoning = 0
    for r in llm_runs:
        usage = ((r.extra or {}).get("metadata", {}) or {}).get("usage_metadata") or {}
        total_prompt += usage.get("input_tokens", 0) or 0
        total_completion += usage.get("output_tokens", 0) or 0
        total_cache_read += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        total_reasoning += (usage.get("output_token_details") or {}).get("reasoning", 0) or 0

    print("\n======== Token & Cache Efficiency ========")
    print(f"  Input (prompt) tokens   : {total_prompt:>12,}")
    print(f"  Output tokens           : {total_completion:>12,}")
    print(f"  Cache-read tokens       : {total_cache_read:>12,}")
    print(f"  Reasoning tokens        : {total_reasoning:>12,}")
    if total_completion:
        ratio = total_prompt / total_completion
        flag = "🚨" if ratio > 40 else ("⚠" if ratio > 20 else "✓")
        print(f"  P:C ratio               : {ratio:>12.1f} : 1   [{flag}]")
    if total_prompt + total_cache_read:
        cache_rate = 100 * total_cache_read / (total_prompt + total_cache_read)
        flag = "🚨" if cache_rate < 30 else ("⚠" if cache_rate < 50 else "✓")
        print(f"  Prompt-cache savings    : {cache_rate:>11.1f}%      [{flag}]")
        print("    NOTE: cache_read reporting may be unreliable for non-Anthropic")
        print("    backends (e.g. vLLM/llama.cpp prefix caching) — treat with caution.")

    # ── 2. Tool usage + non-custom flagging ──────────────────────────────
    print("\n======== Tool Usage ========")
    counts = Counter(r.name for r in tool_runs)
    unknown = {n: c for n, c in counts.items() if not is_known_tool(n)}
    for n, c in counts.most_common():
        marker = "  " if is_known_tool(n) else "❌"
        print(f"  {marker} {n}: {c}")
    if unknown:
        print(f"\n  [! WARNING] {sum(unknown.values())} calls to non-SPINE tools:")
        for n, c in unknown.items():
            print(f"      • {n} ×{c}")
    else:
        print("\n  [✓] All tool calls hit the SPINE allowlist.")

    # ── 3. Duplicate analysis (deduper effectiveness) ────────────────────
    # 3a. Duplicates WITHIN a single agent invocation = deduper miss.
    by_agent: dict[str, list] = defaultdict(list)
    for t in tool_runs:
        by_agent[agent_ns(t)].append(t)

    within_dups: list[tuple[int, str, tuple[str, str]]] = []
    within_excess = 0
    within_dedupable_excess = 0
    for ns, ts in by_agent.items():
        c = Counter(_sig(r) for r in ts)
        for sig, n in c.items():
            if n > 1:
                within_dups.append((n, ns, sig))
                within_excess += n - 1
                if is_dedupable(sig[0]):
                    within_dedupable_excess += n - 1
    within_dups.sort(reverse=True)

    # 3b. Duplicates ACROSS invocations = cross-invocation cache miss
    #     (would need a thread-scoped or persisted cache to dedupe).
    global_counts: Counter = Counter(_sig(r) for r in tool_runs)
    across_dups = [(c, sig) for sig, c in global_counts.items() if c > 1]
    across_excess = sum(c - 1 for c, _ in across_dups)
    across_dups.sort(reverse=True)

    print("\n======== Deduper Effectiveness ========")
    print(f"  Agent invocations              : {len(by_agent)}")
    print(f"  Dup tool-call signatures (within agent) : {len(within_dups)}")
    print(f"  Excess calls (within agent)             : {within_excess}  "
          f"({100 * within_excess / max(len(tool_runs), 1):.1f}% of all tool spans)")
    print(f"  ↳ of those, on dedupable tools         : {within_dedupable_excess} "
          "← MUST be zero if ReadCacheMiddleware is wired correctly")
    print(f"  Dup tool-call signatures (across)       : {len(across_dups)}")
    print(f"  Excess calls (across invocations)       : {across_excess}")

    if within_dups:
        print("\n  Top within-invocation duplicate signatures:")
        for n, ns, sig in within_dups[:10]:
            print(f"    x{n}  {sig[0]}   {sig[1][:110]}")
            print(f"          in: {ns[:90]}")
    if across_dups:
        print("\n  Top across-invocation duplicate signatures (cache could carry):")
        for c, sig in across_dups[:10]:
            print(f"    x{c}  {sig[0]}   {sig[1][:110]}")

    # ── 4. Context-management markers ────────────────────────────────────
    print("\n======== Context Management Markers ========")
    cache_markers = ["[cached:"]
    evict_markers = ["[read:", "[exec:", "[grep:", "[written:", "[edited:",
                     "[glob:", "[ls:", "[evicted("]
    trim_markers = ["chars written to", "chars from", "chars →"]

    cache_hits = Counter()
    evict_hits = Counter()
    trim_hits = Counter()

    for r in tool_runs:
        if not r.outputs:
            continue
        s = json.dumps(r.outputs, default=str)
        for m in cache_markers:
            if m in s: cache_hits[m] += 1
        for m in evict_markers:
            if m in s: evict_hits[m] += 1

    for r in llm_runs:
        if not r.inputs or "messages" not in r.inputs:
            continue
        s = json.dumps(r.inputs.get("messages", []), default=str)
        for m in trim_markers:
            if m in s: trim_hits[m] += 1

    print(f"  ReadCacheMiddleware hits ([cached:…])   : {sum(cache_hits.values())}")
    if sum(cache_hits.values()) == 0:
        print("    [✗] No cache hits — confirm context= is passed to every")
        print("        agent.ainvoke() (esp. subagents). RCM bails on ctx=None.")
    print(f"  ToolOutputTrimmer eviction markers      : {sum(evict_hits.values())}")
    if sum(evict_hits.values()) == 0:
        print("    [✗] No eviction markers — ToolOutputTrimmer is not active")
        print("        (currently removed from _build_middleware_stack).")
    print(f"  AI-arg trimming markers in LLM inputs   : {sum(trim_hits.values())}")

    # ── 5. Per-node token roll-up ────────────────────────────────────────
    by_node = defaultdict(lambda: [0, 0, 0, 0])  # n, in, out, cache
    for r in llm_runs:
        md = (r.extra or {}).get("metadata", {}) or {}
        node = md.get("langgraph_node", "?")
        usage = md.get("usage_metadata") or {}
        by_node[node][0] += 1
        by_node[node][1] += usage.get("input_tokens", 0) or 0
        by_node[node][2] += usage.get("output_tokens", 0) or 0
        by_node[node][3] += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0

    print("\n======== Per-node Token Roll-up ========")
    for node, (n, inp, out, cr) in sorted(by_node.items(), key=lambda x: -x[1][1]):
        print(f"  {node:24s}  calls={n:>5}  in={inp:>12,}  out={out:>8,}  "
              f"cache_read={cr:>8,}  avg_in={inp // max(n, 1):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a SPINE LangSmith trace.")
    parser.add_argument("trace_id", help="LangSmith trace UUID")
    audit_trace(parser.parse_args().trace_id)
