---
name: code-review
description: Verification and code review — checking implementation against specifications, plans, and feature slices. Load during the VERIFY phase.
phase: verify
---

# Code Review and Verification

You are verifying that an implementation meets its requirements. Use the
filesystem and shell tools to inspect the actual code — don't rely on
summaries or descriptions.

## Checklist

1. **All feature slices are implemented** — every slice from the tasks phase
   has corresponding code on disk.
2. **Architecture matches the plan** — file structure, module boundaries, and
   interfaces follow the technical plan.
3. **Success criteria are met** — each criterion from the specification is
   satisfied.
4. **Code quality** — no obvious bugs, appropriate error handling, clean code.
5. **Tests pass** — run the test suite via the `execute` tool.

## Verification report format

```markdown
# Verification Report

## Status: VERIFIED / NOT VERIFIED

## Slice-by-slice review
### auth-middleware
- **Status**: VERIFIED
- **Evidence**: Files exist at src/auth.ts, tests pass
- **Issues**: (none)

### user-routes
- **Status**: NOT VERIFIED
- **Evidence**: Missing test coverage
- **Issues**: No integration tests for POST /users

## Summary
- Total slices: 5
- Verified: 4
- Issues: 1
```

## Using the interpreter

If the eval tool is available:
- Dispatch verification subagents in parallel per slice via `tools.task()`.
- Aggregate results deterministically in code — count passes/fails, flag
  issues.
- Run the actual test suite and linters via the shell backend (`execute`),
  not through the interpreter.
