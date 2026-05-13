---
name: spec-writing
description: Technical specification writing — how to produce structured spec documents from work descriptions. Load during the SPECIFY phase.
phase: specify
---

# Specification Writing

You are producing a technical specification document. The specification must be
detailed enough that an architect can design from it and an engineer can build
from the resulting plan.

## Structure

1. **Overview** — summary of what needs to be built
2. **Requirements** — functional and non-functional requirements
3. **Architecture** — high-level design decisions
4. **Interfaces** — API endpoints, data models, contracts
5. **Success criteria** — measurable outcomes

## Guidelines

- Be specific and technical. Avoid vague language.
- Include concrete examples for complex behaviors.
- Define acceptance criteria for each requirement.
- Specify error cases and edge cases, not just happy paths.
- Reference existing project conventions when known (read AGENTS.md if available).
- If the workspace has an existing codebase, use filesystem tools to inspect it
  before writing the spec — don't assume.

## Research

If the eval tool is available, use it with `tools.task()` to dispatch parallel
research subagents that inspect different parts of the codebase, then synthesize
findings in code before writing the spec.
