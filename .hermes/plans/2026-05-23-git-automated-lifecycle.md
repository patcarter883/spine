# Git-Automated Lifecycle: Implementation Plan

## Overview

Wrap SPINE's workflow execution in a **transactional git sandbox**. After the IMPLEMENT phase produces code changes, the orchestrator validates them through a configurable pipeline, then either atomically merges them into the main branch or performs a hard rollback.

### Operational Flow

1. **Transactional Isolation** — Orchestrator checks for clean working tree, creates a patch branch (+ optional worktree), isolates the environment
2. **Agent Code Execution** — SPINE runs its workflow in the isolated workspace, modifies code files, signals completion
3. **Pipeline Evaluation** — Runner executes language-specific validation gates (lint, type-check, tests) sequentially
4. **Atomic Merge (success)** — All gates pass → stage files, automated commit, fast-forward merge to main, prune sandbox
5. **Hard Rollback (failure)** — Any gate fails → escape to main branch, purge tainted branch, `git reset --hard HEAD`, `git clean -fd`

## Architecture

### Two Sandbox Strategies

| Strategy | Isolation Level | Complexity | Recommended |
|----------|----------------|------------|-------------|
| **Branch-based** (`git checkout -b`) | Shared working tree, separate branch | Low | Prototyping only |
| **Worktree-based** (`git worktree add -b`) | Fully isolated directory | Medium | **Production** |

**Worktree is strongly recommended** for unattended operation:
- Agent file writes happen in an isolated `/tmp/spine-sandbox-XXXX` directory
- Crashes can't corrupt the main working tree
- Orchestrator never calls `os.chdir` (avoids "stranded shell" problem)
- Validation runs against the isolated copy

### Integration Point

```
CLI/UI → SpineGitOrchestrator.execute_transactional_run()
           ├── prepare_sandbox()          # git worktree add -b <branch> <sandbox_dir>
           ├── shift workspace_root       # config.workspace_root = sandbox_dir
           ├── submit_work()              # existing SPINE workflow (unchanged)
           ├── run_validation_pipeline()  # spine-gate.yaml sequential gates
           ├── commit_and_merge()         # success: git add, commit, merge, prune
           └── rollback_workspace()       # failure: worktree remove, branch -D, clean
```

The orchestrator wraps `submit_work()` — **no changes to the LangGraph workflow itself**.

## New Modules

### `spine/git/orchestrator.py` — Core Orchestrator

```python
class SpineGitOrchestrator:
    """
    Transactional git wrapper for SPINE workflow execution.

    Config loaded from spine-gate.yaml at project root.
    """

    def __init__(self, config_path="spine-gate.yaml"):
        """Load gate config, resolve git settings."""

    def _execute_shell(self, cmd, cwd=None, timeout=60):
        """Subprocess wrapper with timeout. Returns (success, stdout, stderr)."""

    def prepare_sandbox(self):
        """
        Strategy 'worktree': git worktree add -b <branch> <sandbox_dir> main
        Strategy 'branch':   git checkout -b <branch>
        Returns True on success.
        """

    def execute_transactional_run(self, description, work_type="task"):
        """
        Full lifecycle with try/finally safety:
        1. prepare_sandbox()
        2. patch workspace_root → sandbox_dir
        3. submit_work(description, work_type, config=sandbox_config)
        4. run_validation_pipeline()
        5a. commit_and_merge()     if all gates passed
        5b. rollback_workspace()   if any gate failed
        finally: restore workspace_root, os.chdir(master_dir)
        """

    def run_validation_pipeline(self):
        """
        Iterate through spine-gate.yaml gates sequentially.
        Returns (True, None) on success.
        Returns (False, diagnostic_dict) on first failure.
        """

    def commit_and_merge(self):
        """
        1. git add . (in sandbox)
        2. git commit -m "spine(auto): verified patch <branch>"
        3. git checkout <main_branch> (in master)
        4. git merge <branch> (fast-forward)
        5. git branch -d <branch>
        6. git worktree remove <sandbox_dir>
        """

    def rollback_workspace(self):
        """
        Nuclear purge:
        1. os.chdir(master_dir)
        2. git worktree remove --force <sandbox_dir>
        3. git branch -D <branch>
        4. shutil.rmtree(sandbox_dir, ignore_errors=True)
        5. git reset --hard HEAD
        6. git clean -fd
        """
```

### `spine/git/workspace_shim.py` — Workspace Root Redirection

```python
class WorkspaceShim:
    """
    Context manager that redirects workspace_root to the sandbox directory.

    The Deep Agent's LocalShellBackend uses workspace_root for all file
    operations. We need to override it without permanently mutating config.

    Usage:
        shim = WorkspaceShim(master_dir, sandbox_dir)
        with shim:
            # config.workspace_root == sandbox_dir
            result = await submit_work(...)
        # config.workspace_root restored to master_dir
    """

    def __init__(self, master_dir: str, sandbox_dir: str):
        ...

    def __enter__(self):
        """Patch SpineConfig.workspace_root to sandbox_dir."""
        ...

    def __exit__(self, *exc):
        """Restore SpineConfig.workspace_root to master_dir."""
        ...
```

### `spine/cli/git_commands.py` — New CLI Commands

| Command | Purpose |
|---------|---------|
| `spine gate run "description" [--type task\|critical_task]` | Full git-gated lifecycle |
| `spine gate status` | Show active sandbox/branch |
| `spine gate rollback` | Manual rollback of failed sandbox |
| `spine gate config` | Show current spine-gate.yaml |

### `spine/exceptions.py` — New Exceptions

```python
class GitOrchestratorError(SpineError):
    """Error in git orchestrator execution."""

class SandboxPreparationError(GitOrchestratorError):
    """Failed to create git worktree or branch."""

class ValidationError(GitOrchestratorError):
    """A validation gate in the pipeline failed."""
    def __init__(self, gate_name: str, command: str, output: str):
        self.gate_name = gate_name
        self.command = command
        self.output = output

class MergeError(GitOrchestratorError):
    """Fast-forward merge failed (conflict)."""
```

### `spine/ui/_pages/gate_run.py` — Streamlit UI Page

- Submit form for gated work descriptions
- Live sandbox status display
- Validation gate results with pass/fail per gate
- Merge/rollback outcome display

### `spine-gate.yaml` — Validation Pipeline Config

```yaml
# spine-gate.yaml — Place in project root
git:
  main_branch: main
  branch_prefix: spine/patch-
  strategy: worktree          # "worktree" or "branch"

validation_pipeline:
  lint:
    command: ".venv/bin/ruff check ."
    timeout_seconds: 60
    failure_message: "Ruff linting failed. Fix style issues."
  typecheck:
    command: ".venv/bin/mypy spine/"
    timeout_seconds: 120
    failure_message: "Type checking failed. Fix type errors."
  test:
    command: ".venv/bin/pytest tests/ -x -q"
    timeout_seconds: 300
    failure_message: "Tests failed. Fix failing tests."
```

## Critical Guardrails

### 1. Virtual Environment Paths

The worktree doesn't copy `.venv`. All validation commands must use absolute paths resolved from the master directory:

```python
def _resolve_validation_command(command: str, master_dir: str) -> str:
    """Replace relative .venv paths with absolute paths."""
    if command.startswith(".venv/"):
        return str(Path(master_dir) / command)
    return command
```

### 2. Agent Workspace Root

The `WorkflowState.workspace_root` must point to the sandbox directory. This is done by creating a modified `SpineConfig`:

```python
sandbox_config = SpineConfig.load()
sandbox_config.workspace_root = sandbox_dir
# Pass to submit_work(config=sandbox_config)
```

### 3. os.chdir Safety

```python
original_dir = os.getcwd()
try:
    os.chdir(sandbox_dir)
    # ... run agent and validation ...
finally:
    os.chdir(original_dir)  # ALWAYS restore, even on crash
```

### 4. Untracked File Cleanup (Nuclear Purge)

```python
def rollback_workspace(self):
    os.chdir(self.master_dir)
    self._execute_shell(f"git worktree remove --force {self.sandbox_dir}")
    self._execute_shell(f"git branch -D {self.patch_branch}")
    if os.path.exists(self.sandbox_dir):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
    # These handle any stray files the agent leaked into master:
    self._execute_shell("git reset --hard HEAD")
    self._execute_shell("git clean -fd")
```

### 5. Validation Gate Timeouts

Each gate has a configurable timeout (default 60s). The subprocess is killed on timeout and the gate is marked as failed.

## File Inventory

### New Files

| File | Purpose | Est. Lines |
|------|---------|------------|
| `spine/git/__init__.py` | Package init | 0 |
| `spine/git/orchestrator.py` | Core orchestrator class | ~300 |
| `spine/git/workspace_shim.py` | Workspace root redirection | ~80 |
| `spine/cli/git_commands.py` | CLI `gate` subcommands | ~150 |
| `spine-gate.yaml` | Validation pipeline config | ~25 |
| `spine/ui/_pages/gate_run.py` | Streamlit gated execution page | ~150 |
| `tests/unit/test_git_orchestrator.py` | Unit tests (mocked git) | ~250 |
| `tests/integration/test_git_lifecycle.py` | Integration tests (real git) | ~200 |

### Modified Files

| File | Change |
|------|--------|
| `spine/cli/__init__.py` | Import and register `gate` subcommand group |
| `spine/exceptions.py` | Add `GitOrchestratorError`, `SandboxPreparationError`, `ValidationError`, `MergeError` |
| `spine/ui_api/api.py` | Add `submit_gated_work()`, `get_gate_status()` methods |

## Implementation Order

Proceed in this order to keep the system working at each step:

1. **`spine/exceptions.py`** — Add new exception classes (no dependencies)
2. **`spine/git/__init__.py`** + **`spine/git/orchestrator.py`** — Core logic, testable in isolation
3. **`spine/git/workspace_shim.py`** — Workspace redirection context manager
4. **`tests/unit/test_git_orchestrator.py`** — Validate orchestrator with mocked git subprocess calls
5. **`spine/cli/git_commands.py`** + modify **`spine/cli/__init__.py`** — CLI integration
6. **`spine-gate.yaml`** — Default config template in project root
7. **`tests/integration/test_git_lifecycle.py`** — End-to-end validation with real git repo
8. **`spine/ui_api/api.py`** + **`spine/ui/_pages/gate_run.py`** — UI integration

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Agent writes files outside sandbox via relative paths | Worktree isolation + `git clean -fd` on rollback |
| `.venv` not available inside worktree | Validation commands use absolute `.venv/bin/` paths from master_dir |
| Orphaned worktrees after crash | `git worktree remove --force` + `git worktree prune` in rollback |
| Merge conflicts on fast-forward | Detect non-zero exit from `git merge`, abort and rollback |
| `os.chdir` stranded on crash | `try/finally` in `execute_transactional_run` always restores |
| Agent panics and drops scrap files | `git clean -fd` + `shutil.rmtree` in rollback |
| Long-running validation gates | Per-gate timeout in YAML, enforced via `subprocess.run(timeout=...)` |
| Agent modifies `.gitignore` or git internals | `git reset --hard HEAD` in rollback restores all tracked files |

## Why git clean -fd Matters

A standard `git reset --hard HEAD` does **not** remove untracked files. If an agent panics and drops `code_backup_v2.py`, `debug.log`, or other scrap files into the source directory, they persist after the reset. `git clean -fd` removes all untracked files and directories, guaranteeing the workspace is pristine after rollback.

## Why Worktrees Over Branches

In a branch-based sandbox, the agent's Deep Agent tools (which use `write_file`, `read_file`, etc.) operate on the **same physical directory** as the orchestrator. A crash mid-write can leave corrupted files, half-written patches, or stale lock files. With worktrees, the agent operates in `/tmp/spine-sandbox-XXXX` — a completely separate directory. The only shared surface is the git index itself (managed exclusively by the orchestrator).
