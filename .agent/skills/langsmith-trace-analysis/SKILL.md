---
name: langsmith-trace-analysis
description: "Use when auditing agent trace telemetry in LangSmith to assess behavior, token consumption, and context management efficiency. Provides structured instructions for querying trace/thread runs, measuring prompt-to-completion ratios, spotting tool call loops, assessing prompt compliance, and quantifying the token impact of Smart Eviction v2, AI argument trimming, and low-threshold summarization."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MLOps, LangSmith, Telemetry, Trace Analysis, Token Efficiency, Context Compaction, Agent Evaluation, Debugging]
    related_skills: [systematic-debugging, writing-plans, autonomous-ai-agents]
---

# LangSmith Trace Analysis: Telemetry-Driven Behavior & Token Auditing

## Overview
This skill provides a structured methodology for auditing deep AI agent executions (such as SPINE) using LangSmith trace and thread telemetry. Telemetry auditing lets developers move past blind debugging to quantify prompt-to-completion (P:C) imbalances, diagnose costly or redundant tool use, ensure prompt compliance, and measure the token-saving effectiveness of active context management frameworks.

This guide targets the three layers of advanced context control:
1. **Smart Eviction (v2)**: Replacing old tool responses with metadata (rather than vague `[evicted...]` placeholders) to prevent agents from re-reading files.
2. **AI Argument Trimming**: Truncating redundant `tool_calls` arguments (like large written content blocks) from conversation history once their corresponding outputs are evicted.
3. **Earlier Summarization**: Shifting compaction triggers (e.g. from 80K to 60K tokens) to keep KV caches thin and compress history before it balloons late-stage API costs.

---

## When to Use This Skill
Use this skill when:
- **Investigating high API billing or token waste** in multi-turn agent runs.
- **Auditing agent loop behaviors** (e.g. redundant `read_file` or `execute` calls).
- **Evaluating prompt leaks or failures to comply** with system instructions.
- **Validating context management work** to quantify token reductions.
- **Debugging failed/errored runs** by reconstructing parent-child execution graphs.

Do not use this skill for:
- Direct local system performance tracing (use visual profilers like `cProfile` instead).
- General test assertions (use `pytest` instead).

---

## Quick-Reference Metrics

| Metric | Target / Healthy Range | Alarm Range | Root Cause & Diagnostics |
| :--- | :--- | :--- | :--- |
| **Prompt:Completion Ratio** | `< 20:1` | `> 40:1` | **Unbounded prompt growth**: Conversation history not compacted, or redundant large strings are carried over. |
| **Cache Read Save (CSS)** | `> 50%` | `< 30%` | **Poor prompt alignment/frequent context fragmentation**: System prompt changing block-to-block, or tool output eviction breaking prompt prefix matching. |
| **Duplicate Reads** | `0` (Paths read $\le 1$ times) | `> 3` | **Lack of caching awareness**: Agent repeatedly fetching the same source code or configuration files in different turns instead of caching locally. |
| **Tool Stall Loops** | `0` | `> 2` matching calls | **Command spinning**: Agent executing identical commands (e.g., `pytest`, `git status`) repeatedly with no modifications made to code. |

---

## Trace Analysis Playbook

Given a `trace_id` or `thread_id`, execute the auditing workflow in four steps:

### 1. Retrieve & Assemble the Trace Family
Query ALL child runs under the trace ID to rebuild the execution order. If given a `thread_id`, group runs by thread first.

*Using Python SDK:*
```python
from langsmith import Client

client = Client()
# Retrieve runs for a trace ID
runs = list(client.list_runs(trace_id="your-trace-id-uuid"))
# Sort chronologically by start time
runs.sort(key=lambda r: r.start_time)
```

### 2. Audit Token and Cache Consumption
Examine the run usages. Look for token properties at `run.extra['metadata']['usage_metadata']`:

- **Input Tokens (`input_tokens`)**: The weight of context history. Compare late-stage runs in the trace against early runs.
- **Cache Read Saves (`input_token_details.cache_read`)**: Check if OpenRouter/Anthropic prompt-caching is active and working.
- **Reasoning Tokens (`output_token_details.reasoning`)**: Assess how many reasoning tokens (e.g., in reasoning models) are spent vs normal outputs.

### 3. Diagnose Behavioral Inefficiencies & Spin Loops
Examine the sequence of tool execution:
- **Scan tool calls**: Filter runs where `run_type == "tool"`.
- **Compile file counters**: Keep a list of all file paths passed to `read_file`. Calculate how many unique files are read, and identify paths read multiple times.
- **Detect Command Spin**: Find identical command configurations sent to `execute`. If an agent is executing `pytest` repeatedly with no interposed `write_file`/`edit_file` modifications, it is in a local minima loop.

### 4. Evaluate Context Management Effectiveness

Compare traces before and after three-layer context optimization:

#### Smart Eviction Verification
Verify if evicted tool-result content maintains descriptive placeholder summaries.
- **Old (Vague Hints)**: `[evicted:      1\tSystem reminder...]` $\rightarrow$ Causes agents to lose context and re-read the file.
- **New (Smart Eviction)**: `[read: src/main.py (142 lines) — def create_agent, class Worker]` $\rightarrow$ Agent knows exactly what is in the file without re-fetching.

#### AI Message Argument Trimming
Identify AI messages with tool calls such as `write_file` or `edit_file` where the tool output is evicted. Check if the input arguments (like the `content` parameter) have been trimmed.
- **Unoptimized**: `write_file` arguments in the conversation history still hold a 5KB serialized code payload inside `AImessage.tool_calls` for 30 consecutive turns.
- **Optimized**: Truncated to `[5120 chars written to src/main.py]`, instantly freeing several thousand prompt tokens.

#### Early Compaction Triggers
Trace the token volume curve over conversational turns:
- If a summarization pattern is triggered, audit whether it fires *early* (e.g., at 60,000 tokens instead of 80,000). Compressing the history at 60KB costs less in completion tokens and avoids over-filling the KV cache, preventing a "cold" prompt-caching miss.

---

## Trace Analysis CLI Script

Deliver this Python script to run programmatic audits directly of any loaded trace. It outputs a comprehensive performance breakdown, identifies inefficiencies, and scores prompt compliance.

```python
#!/usr/bin/env python3
"""
trace_audit.py - Telemetry Audit Script for AI Agent Traces.
Analyzes agent behavior, token ratios, prompt caching, loops, and context savings.
"""

import sys
import json
import argparse
from collections import Counter
from langsmith import Client

def audit_trace(trace_id, project_name=None):
    client = Client()
    print(f"[*] Querying trace: {trace_id}...")
    
    # Fetch runs
    try:
        runs = list(client.list_runs(trace_id=trace_id))
    except Exception as e:
        print(f"[-] Error querying LangSmith: {e}")
        return
        
    if not runs:
        print("[-] No runs found matching this Trace ID.")
        return
        
    print(f"[+] Loaded {len(runs)} execution spans.")
    
    # Classify spans
    llm_runs = [r for r in runs if r.run_type == "llm"]
    tool_runs = [r for r in runs if r.run_type == "tool"]
    chain_runs = [r for r in runs if r.run_type == "chain"]
    
    print(f"\n======== Span Classification ========")
    print(f"  LLM Spans   : {len(llm_runs)}")
    print(f"  Tool Spans  : {len(tool_runs)}")
    print(f"  Chain Spans : {len(chain_runs)}")
    
    # Calculate token budgets
    total_prompt = 0
    total_completion = 0
    total_cache_read = 0
    total_reasoning = 0
    
    for r in llm_runs:
        meta = r.extra.get("metadata", {}) if r.extra else {}
        usage = meta.get("usage_metadata")
        if not usage:
            # Fallback to direct run usage fields
            usage = {
                "input_tokens": getattr(r, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(r, "completion_tokens", 0) or 0,
            }
            
        total_prompt += usage.get("input_tokens", 0) or 0
        total_completion += usage.get("output_tokens", 0) or 0
        
        # Caching saves (e.g. Claude Prompt Caching or OpenRouter saves)
        input_details = usage.get("input_token_details", {}) or {}
        total_cache_read += input_details.get("cache_read", 0) or 0
        
        # Reasoning output saves
        output_details = usage.get("output_token_details", {}) or {}
        total_reasoning += output_details.get("reasoning", 0) or 0

    print(f"\n======== Token & Cache Efficiency ========")
    print(f"  Total Input (Prompt) Tokens   : {total_prompt:,}")
    print(f"  Total Output (Output) Tokens  : {total_completion:,}")
    print(f"  Total Cached Tokens Saved     : {total_cache_read:,}")
    print(f"  Total Reasoning Tokens Used   : {total_reasoning:,}")
    
    if total_completion > 0:
        ratio = total_prompt / total_completion
        print(f"  Prompt : Completion Ratio     : {ratio:.1f}:1")
        if ratio > 40:
            print("  [! WARNING] High P:C Ratio! History optimization (Compaction/Trimming) required.")
        else:
            print("  [✓] Healthy P:C Ratio.")
    else:
        print("  Prompt:Completion Ratio: N/A (no output completions)")
        
    total_attempted = total_prompt + total_cache_read
    if total_attempted > 0:
        cache_rate = (total_cache_read / total_attempted) * 100
        print(f"  Prompt Cache Savings Rate     : {cache_rate:.1f}%")
        if cache_rate < 30:
            print("  [! WARNING] Low cache rate. History changes are causing frequent cache fragmentation!")
        else:
            print("  [✓] Excellent cache utilization.")

    # Audit tool behaviors
    print(f"\n======== Tool Usage Audit ========")
    tool_counts = Counter([r.name for r in tool_runs])
    for tool_name, count in tool_counts.most_common():
        print(f"  - {tool_name}: {count} calls")
        
    # Spot duplicate reads
    read_paths = []
    executed_commands = []
    for r in tool_runs:
        if r.name == "read_file":
            path = r.inputs.get("file_path") or r.inputs.get("path")
            if path:
                read_paths.append(path)
        elif r.name in ("execute", "terminal"):
            cmd = r.inputs.get("command")
            if cmd:
                executed_commands.append(cmd)
                
    duplicate_reads = [path for path, count in Counter(read_paths).items() if count > 1]
    if duplicate_reads:
        print(f"\n  [! WARNING] Duplicate Reads detected ({len(duplicate_reads)} paths):")
        for path in duplicate_reads:
            occurrences = read_paths.count(path)
            print(f"    - '{path}' read {occurrences} times in trace.")
            if occurrences > 4:
                print(f"      [CRITICAL] Repetitive reads on '{path}'. Code should be cached or smart evicted.")
                
    # Spot command spin loops
    command_counts = Counter(executed_commands)
    spins = [cmd for cmd, count in command_counts.items() if count > 1]
    if spins:
        print(f"\n  [! WARNING] Command Spin / Minimal loops detected:")
        for cmd in spins:
            occurrences = command_counts[cmd]
            print(f"    - Command '{cmd}' executed {occurrences} times.")
            
    # Audit Eviction & Trimming
    print(f"\n======== Context Management Check ========")
    evicted_spans = 0
    trimmed_ai_args = 0
    for r in runs:
        if r.run_type == "tool" and r.outputs and "content" in r.outputs:
            out_content = str(r.outputs.get("content", ""))
            if "[read:" in out_content or "[exec:" in out_content or "[grep:" in out_content:
                evicted_spans += 1
        if r.run_type == "llm" and r.inputs and "messages" in r.inputs:
            for msg in r.inputs.get("messages", []):
                if isinstance(msg, dict) and "tool_calls" in msg:
                    for tc in msg.get("tool_calls", []):
                        args = tc.get("args", {})
                        if args and "content" in args and "written to" in str(args["content"]):
                            trimmed_ai_args += 1
                            
    print(f"  Smart Eviction place holders matched: {evicted_spans}")
    print(f"  AI Argument Trimming blocks matched : {trimmed_ai_args}")
    if evicted_spans > 0:
        print("  [✓] Smart Eviction (v2) is active in this trace.")
    else:
        print("  [-] Smart Eviction is not detected or no files were evicted.")
        
    if trimmed_ai_args > 0:
        print("  [✓] AI Argument Trimming is successfully active.")
    else:
        print("  [-] AI Argument Trimming not active (or write files were not evicted).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit LangSmith Trace Quality & Efficiency.")
    parser.add_argument("trace_id", help="LangSmith Trace UUID")
    args = parser.parse_args()
    audit_trace(args.trace_id)
```

---

## Common Pitfalls

1. **Treating All High-Token Spans as Waste**: Long single completions or complex reasoning are expected in implementation or specification phases. Focus analysis on the *growth* of prompt tokens over multiple turns.
2. **Missing AI Message Args**: Only trimming `ToolMessage` outputs and leaving large content parameters in `AIMessage.tool_calls` causes severe context leakage. Always ensure both are trimmed in lock-step.
3. **Triggering Summarization Too Late**: Compacting at 80%–90% of model window leaves insufficient headroom for the summarizer LLM to run safely and efficiently. Lower triggers (e.g. 60K tokens) result in much cheaper, streamlined runs.
4. **Ignoring Prompt Caching alignment**: When designing user prompt changes, never inject dynamic variables (like live system timestamps, process IDs, or system load) into the middle of system blocks. This breaks the prefix alignment and decimates prompt-caching hit rates.

---

## Verification Checklist

- [ ] Fetch the trace data family using `trace_id` or `thread_id`.
- [ ] Measure the overall Prompt:Completion ratio (Target: `< 20:1`, Warning: `> 40:1`).
- [ ] Verify Cache Read Saves are optimized (`> 50%` efficiency).
- [ ] Audit `read_file` tool behaviors to ensure no files are fetched repeatedly.
- [ ] Scan output files to verify Smart Eviction placeholders are printed instead of generic tags.
- [ ] Confirm AI Message argument trimming has pruned large write parameters.
- [ ] Document trace-level token savings to validate context management work.
