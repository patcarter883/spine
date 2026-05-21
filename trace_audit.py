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
