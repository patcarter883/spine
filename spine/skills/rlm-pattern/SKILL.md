---
name: rlm-pattern
description: RLM pattern — use eval to orchestrate work, batch operations, and keep context lean. MUST use eval before making ≥3 manual tool calls.
phase: specify, tasks, implement, verify
---

# RLM Pattern — Eval-First Strategy

**Rule: Before making ≥3 manual tool calls, ask yourself: "Can I write one eval program that does this?"**

The eval tool is a persistent QuickJS interpreter. Variables survive between
turns. Use it to keep intermediate data OUT of the model context.

## When to use eval

| Situation | Use eval? | Why |
|-----------|-----------|-----|
| Reading ≥3 files | YES | Batch reads in one program, return synthesis |
| Dispatching ≥2 subagents | YES | Promise.all for parallel, keep results in JS |
| Sorting/filtering data | YES | Deterministic — no token cost for logic |
| Single file read | NO | Just use read_file directly |
| Writing a file | NO | Use write_file tool directly |

## Pattern 1: Batch file inspection

Instead of reading files one at a time (5 turns × 34K tokens = 170K tokens),
do this:

```js
// Read tasks artifact and extract slice names
const tasks = await tools.read_file({path: '.spine/artifacts/WORK_ID/tasks/tasks.md'});
const sliceMatches = tasks.match(/slice-\w+/g);
const uniqueSlices = [...new Set(sliceMatches)];
console.log('Slices:', uniqueSlices.join(', '));
```

Then read only the slice files you need — in one eval call.

## Pattern 2: Parallel subagent dispatch (IMPLEMENT/VERIFY)

```js
const subagent = runtime.context?.active_subagent || 'slice-implementer';
const slices = ['slice-queue-pending-reorder', 'slice-work-detail-reorder', 'slice-tests'];

// Wave sort: all slices have no dependencies → one wave
const results = await Promise.allSettled(
  slices.map(name => tools.task({
    description: `Implement the slice defined in .spine/artifacts/${runtime.context?.work_id}/tasks/${name}.md. Read the slice file and codebase-map.md first, then implement the changes described.`,
    subagent_type: subagent,
  }))
);

const succeeded = results.filter(r => r.status === 'fulfilled').map(r => r.value);
const failed = results.filter(r => r.status === 'rejected').map(r => r.reason);
console.log(`Done: ${succeeded.length}/${slices.length}, Failed: ${failed.length}`);
JSON.stringify({succeeded, failed}, null, 2);
```

## Pattern 3: Codebase exploration (TASKS/SPECIFY)

```js
const subagent = 'researcher';
const modules = ['spine/work/ralph_worker.py', 'spine/ui_api/api.py', 'spine/ui/_pages/queue.py'];

const reports = await Promise.all(
  modules.map(path => tools.task({
    description: `Research the module at ${path}. Report: 1) Key classes and functions, 2) Imports and dependencies, 3) Patterns and conventions used.`,
    subagent_type: subagent,
  }))
);

// Process reports in eval — don't dump into conversation
const summaries = reports.map(r => r.substring(0, 200));
console.log(summaries.join('\\n\\n'));
```

## Critical rules

1. **Never dump raw data into conversation.** Process in eval, return synthesis.
2. **Use runtime.context.** Access `runtime.context.work_id`, `runtime.context.active_subagent` etc.
3. **Subagent descriptions must be self-contained.** Include file paths and reference codebase-map.md, not "read the slice I mentioned earlier."
4. **Keep eval output under 4000 chars.** The runtime truncates at max_result_chars.
5. **Variables persist across turns.** Store intermediate results in `window.results = ...`.
