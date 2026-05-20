---
name: mcp-codebase-index
description: Structural codebase navigation via MCP — find symbols, analyze dependencies, assess change impact, search codebases
phases: [plan, tasks, implement, verify]
---

# MCP Codebase Index

You have access to MCP tools for efficient structural codebase navigation.
These tools are MUCH more token-efficient than reading entire files —
**use them FIRST when exploring or navigating the codebase.**

## Why Use These Tools?

| Without MCP Index | With MCP Index |
|---|---|
| Read entire files to find symbols | `find_symbol("TestCase")` → 67 chars |
| Grep for patterns across project | `search_codebase` → targeted results |
| Manually trace call chains | `get_call_chain` → BFS path in ms |
| Guess impact of changes | `get_change_impact` → all dependents |

The indexer extracts structural metadata (functions, classes, imports,
dependencies) for Python, TypeScript/JS, Go, Rust, C#, and Markdown.
All queries return in sub-millisecond time even on million-line codebases.

## Available Tools

### Discovery

| Tool | Use When |
|------|----------|
| `get_project_summary` | You need a high-level overview of the codebase |
| `list_files` | You need to find files matching a pattern |
| `get_structure_summary` | You need to see what's in a file or project |
| `get_functions` | You need to list all functions (optionally filtered) |
| `get_classes` | You need to list all classes (optionally filtered) |
| `get_imports` | You need to see what a file imports |

### Symbol Lookup

| Tool | Use When |
|------|----------|
| `find_symbol` | You need to locate where a symbol is defined |
| `get_function_source` | You need to see a function's full source code |
| `get_class_source` | You need to see a class definition |

### Dependency Analysis

| Tool | Use When |
|------|----------|
| `get_dependencies` | You need to know what a function/class calls |
| `get_dependents` | You need to know who calls a function/class |
| `get_change_impact` | You're planning a refactor — what breaks? |
| `get_call_chain` | You need to trace execution flow between two symbols |

### File Relationships

| Tool | Use When |
|------|----------|
| `get_file_dependencies` | You need to know which files a file imports from |
| `get_file_dependents` | You need to know which files import from this file |

### Search

| Tool | Use When |
|------|----------|
| `search_codebase` | You need to find all occurrences of a regex pattern |
| `get_usage_stats` | You need session efficiency statistics |

## Usage Rules — MANDATORY

1. **ALWAYS try `find_symbol`** before reading files with glob/grep/read
2. **Use `get_dependencies`/`get_dependents`** to understand relationships
3. **Use `get_change_impact`** before modifying signatures or APIs —
   this tells you EVERYTHING that would break
4. **Use `get_call_chain`** to trace how data flows between components
5. **Use `search_codebase`** for regex searches across the entire project
6. **Fall back to read_file ONLY when** MCP tools genuinely don't have
   what you need (e.g., reading non-code files, config, markdown docs)
7. **Use `get_project_summary`** at the start of new tasks for orientation
8. **IF YOU CATCH YOURSELF REACHING FOR GLOB/GREP/READ** to find or
   understand code, STOP and use codebase-index instead

## Token Savings Example

On CPython (41 million characters, 2,464 files):
- `find_symbol("TestCase")` → 67 characters (vs potentially reading thousands of files)
- `get_dependencies("compile")` → 115 characters
- `get_change_impact("TestCase")` → 16,812 characters (154 direct + 492 transitive dependents)

The indexer's response size scales with the ANSWER, not the codebase.
