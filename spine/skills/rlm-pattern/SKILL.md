---
name: rlm-pattern
description: RLM pattern — use eval to orchestrate work, batch operations, and keep context lean. MUST use eval before making ≥3 manual tool calls.
phase: specify, tasks, implement, verify
---

# RLM Pattern — Eval-First Strategy

**Rule: Before making ≥3 manual tool calls, ask yourself: "Can I write one eval program that does this?"**

The eval tool is a persistent **QuickJS** interpreter — NOT Node.js, NOT a browser.
Variables survive between turns. Use it to keep intermediate data OUT of the model context.

## When to use eval

| Situation | Use eval? | Why |
|-----------|-----------|-----|
| Reading ≥3 files | YES | Batch reads in one program, return synthesis |
| Dispatching ≥2 subagents | YES | Promise.all for parallel, keep results in JS |
| Sorting/filtering data | YES | Deterministic — no token cost for logic |
| Single file read | NO | Just use read_file directly |
| Writing a file | NO | Use write_file tool directly |

## PTC API — How to call tools from eval

Tools are exposed on `globalThis.tools` using **camelCase** names.
Arguments use the original **snake_case** parameter names from the tool schema.

```typescript
// Tool name mapping:
//   read_file  → tools.readFile
//   write_file → tools.writeFile
//   edit_file  → tools.editFile
//   grep       → tools.grep
//   glob       → tools.glob
//   ls         → tools.ls
//   task       → tools.task
```

**Return values are native JS types** — strings are strings, arrays are arrays,
objects are objects. Do NOT access `.content` on the result.

```typescript
// ✅ CORRECT — readFile returns a string
const config = await tools.readFile({ file_path: 'pyproject.toml' });
console.log('Lines:', config.split('\n').length);

// ❌ WRONG — result is not an object with .content
const config = await tools.readFile({ file_path: 'pyproject.toml' });
console.log(config.content);  // undefined!
```

```typescript
// ✅ CORRECT — parameter is file_path (snake_case)
const src = await tools.readFile({ file_path: 'src/main.py' });

// ❌ WRONG — parameter is not path
const src = await tools.readFile({ path: 'src/main.py' });  // undefined!
```

## Pattern 1: Batch file inspection

Instead of reading files one at a time (5 turns × 34K tokens = 170K tokens),
do this:

```js
// Read tasks artifact and extract slice names
const tasks = await tools.readFile({ file_path: '.spine/artifacts/WORK_ID/tasks/tasks.md' });
const sliceMatches = tasks.match(/slice-\w+/g);
const uniqueSlices = [...new Set(sliceMatches)];
console.log('Slices:', uniqueSlices.join(', '));
```

Then read only the slice files you need — in one eval call.

## Pattern 2: Parallel subagent dispatch (IMPLEMENT/VERIFY)

```js
const subagent = globalThis.context?.phase === 'verify' ? 'slice-verifier' : 'slice-implementer';
const slices = ['slice-queue-pending-reorder', 'slice-work-detail-reorder', 'slice-tests'];

// Wave sort: all slices have no dependencies → one wave
const results = await Promise.allSettled(
  slices.map(name => tools.task({
    description: `Implement the slice defined in .spine/artifacts/${globalThis.context?.work_id}/tasks/${name}.md. Read the slice file and codebase-map.md first, then implement the changes described.`,
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

## Filesystem tool access

The PTC allowlist includes filesystem tools for codebase exploration directly from eval:

- **`tools.readFile`** — read file contents (returns **string**)
- **`tools.grep`** — search file contents (regex)
- **`tools.glob`** — find files by pattern
- **`tools.ls`** — list directory contents
- **`tools.writeFile`** — write files to disk
- **`tools.editFile`** — make targeted edits to files

Use these to inspect and transform files without leaving the interpreter:

```js
// Batch-read multiple files from eval
const config = await tools.readFile({ file_path: 'pyproject.toml' });
const readme = await tools.readFile({ file_path: 'README.md' });
console.log('Config length:', config.length);

// Search for patterns across the codebase
const matches = await tools.grep({ pattern: 'PhaseName', path: 'spine/' });
console.log('PhaseName occurrences:', matches);

// Find files by glob pattern
const testFiles = await tools.glob({ pattern: 'tests/**/*.py' });
console.log('Test files:', testFiles);
```

These complement the `task` tool — use filesystem tools for direct inspection and `task` for subagent delegation.

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ReferenceError: 'require' is not defined` | Using `require('...')` — QuickJS has NO module system | Never use `require()`. Use `globalThis.tools` for PTC, store data in `globalThis`. |
| `ReferenceError: 'import' is not defined` | Using `import ... from` — no ES modules in QuickJS | Never use `import`. All code runs in a single persistent global scope. |
| `SyntaxError: redeclaration of 'X'` | `const X = ...` used the same name in a previous eval | Use `let` instead of `const`, or use a different variable name. QuickJS state persists across eval calls. |
| `TypeError: cannot read property 'X' of undefined` | Accessing `.content` or `.data` on a tool result | PTC returns native values. `readFile` returns a string, not an object. Use the result directly. |
| `ReferenceError: window is not defined` | Using `window.*` instead of `globalThis.*` | Use `globalThis` — QuickJS has no `window` object. |
| `<result>null</result>` | Variable evaluated to undefined/null | Make sure your eval code ends with an expression that returns a value, or use `console.log()` and return a summary string. |
| `TypeError: tools.X is not a function` | Using snake_case tool name | Tool names are camelCase: `tools.readFile`, `tools.writeFile`, `tools.editFile`. |

## Critical rules

1. **Never dump raw data into conversation.** Process in eval, return synthesis.
2. **Use globalThis.context.** Access `globalThis.context.work_id`, `globalThis.context.phase` etc. — seeded by the phase prompt on the first turn.
3. **Subagent descriptions must be self-contained.** Include file paths and reference codebase-map.md, not "read the slice I mentioned earlier."
4. **Keep eval output under 4000 chars.** The runtime truncates at max_result_chars.
5. **Variables persist across turns.** Store intermediate results in `globalThis.results = ...`.
6. **Tool names are camelCase, arguments are snake_case.** `tools.readFile({ file_path: '...' })` not `tools.read_file({ path: '...' })`.
7. **Tool results are native JS values.** `readFile` returns a string — no `.content` access needed.
