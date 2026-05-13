---
name: feature-slice-decomposition
description: Break a technical plan into executable feature slices with dependency tracking and DAG ordering. Load during the TASKS phase.
phase: tasks
---

# Feature Slice Decomposition

You are breaking a plan into smaller, executable feature slices with clear
dependencies. The goal is to produce slices that can be implemented
independently (or in parallel when no dependencies exist).

## Per-slice specification

For each feature slice, specify:

1. **Name and description** — what this slice implements
2. **Files to create or modify** — concrete file paths
3. **Dependencies** — which slices must complete first
4. **Acceptance criteria** — how to verify this slice is done
5. **Estimated complexity** — small / medium / large

## Dependency structure

- Group slices by **dependency waves** — slices with no dependencies can run
  in parallel within the same wave.
- Use a DAG structure to show ordering.
- Detect and flag circular dependencies.
- Keep slices as independent as possible — minimize cross-slice coupling.

## Output format

Output the slices in structured markdown with clear dependency annotations:

```markdown
## Wave 1 (no dependencies)
### Slice: auth-middleware
- **Files**: src/auth.ts, src/middleware/auth.ts
- **Deps**: (none)
- **Criteria**: Auth middleware validates JWT tokens
- **Complexity**: medium

## Wave 2 (depends on Wave 1)
### Slice: user-routes
- **Files**: src/routes/users.ts
- **Deps**: auth-middleware
- **Criteria**: CRUD endpoints with auth protection
- **Complexity**: small
```

## Using the interpreter

If the eval tool is available:
- Read the plan with filesystem tools, then store key sections in interpreter
  variables for reference without re-reading.
- Dispatch independent research subagents via `tools.task()` in parallel.
- Build the dependency graph and sort into waves in code (topological sort) —
  this is deterministic work that doesn't need the model.
