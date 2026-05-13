---
name: rlm-pattern
description: Recursive Language Model (RLM) pattern — QuickJS interpreter workspace for inspecting large inputs, orchestrating subagents via PTC, and transforming structured data. Load when the eval tool is available and you need to handle large codebases or parallel work.
phase: specify, tasks, implement, verify
---

# RLM Pattern — Interpreter Workspace

You have a QuickJS interpreter available via the `eval` tool. The interpreter
is your **orchestration brain** — use it to handle work that would overflow
the model context or is deterministic (loops, sorting, filtering, aggregation).

## Core Capabilities

- **Inspect large inputs** — store codebase content in variables, search and
  filter without loading everything into the model context.
- **Orchestrate subagents** — call `tools.task(...)` from code via programmatic
  tool calling (PTC) for loops, parallel batches, and conditional logic.
- **Transform structured data** — sort, group, validate, score, or aggregate
  results deterministically before returning a compact synthesis.

## Inspecting the codebase

Use filesystem tools to read files and `eval` to process them:

```js
// Read a file with the filesystem tool, then process it in eval
const content = `... pasted from read_file tool ...`;
const lines = content.split('\n');
const relevant = lines.filter(l => l.includes('interface'));
console.log(relevant.join('\n'));
```

## Orchestrating subagents (PTC)

You can call `tools.task(...)` from inside eval to spawn subagents for parallel work:

```js
const topics = ['auth', 'api', 'data-model'];
const reports = await Promise.all(
  topics.map(t => tools.task({
    description: `Research ${t} in this codebase and report findings.`,
    subagent_type: 'general-purpose',
  }))
);
reports.join('\n\n');
```

## Parallel execution by wave

Execute independent work concurrently:

```js
const wave1 = items.filter(s => s.deps.length === 0);
const results1 = await Promise.all(
  wave1.map(item => tools.task({
    description: `Process the ${item.name} item. Files: ${item.files.join(', ')}.`,
    subagent_type: 'general-purpose',
  }))
);
```

## Error handling

Handle failures without consuming model context:

```js
const outcomes = await Promise.allSettled(
  wave.map(item => tools.task({
    description: `Process ${item.name}`,
    subagent_type: 'general-purpose',
  }))
);
const succeeded = outcomes
  .filter((r, i) => r.status === 'fulfilled')
  .map((r, i) => ({ name: wave[i].name, result: r.value }));
const failed = outcomes
  .filter(r => r.status === 'rejected')
  .map((r, i) => ({ name: wave[i].name, error: r.reason }));
console.log(`Completed: ${succeeded.length}, Failed: ${failed.length}`);
```

## Dependency sorting (topological wave sort)

Build the dependency graph and sort into waves in code:

```js
const slices = [
  { name: 'A', deps: [] },
  { name: 'B', deps: ['A'] },
  { name: 'C', deps: ['A'] },
  { name: 'D', deps: ['B', 'C'] },
];
const waves = [];
const completed = new Set();
let remaining = [...slices];
while (remaining.length > 0) {
  const wave = remaining.filter(s => s.deps.every(d => completed.has(d)));
  if (wave.length === 0) break; // cycle
  wave.forEach(s => completed.add(s.name));
  waves.push(wave.map(s => s.name));
  remaining = remaining.filter(s => !completed.has(s.name));
}
console.log('Waves:', JSON.stringify(waves));
```

## Important: Filesystem writes go through backend tools

The interpreter has NO filesystem access. Write files using the `write_file`
tool, run tests via the `execute` tool. The interpreter is for orchestration
and data transformation only.
