export const meta = {
  name: 'git-automated-lifecycle',
  description: 'Implement the transactional git-sandbox lifecycle for SPINE workflow execution',
  phases: [
    { title: 'Foundation', detail: 'exceptions + spine/git package + spine-gate.yaml; emit interface contract' },
    { title: 'Integration', detail: 'CLI, UI+ui_api, and tests built against the foundation contract' },
  ],
}

// ── Shared context every agent needs (codebase conventions + gotchas) ──
const CONVENTIONS = `
CODEBASE: /home/pat/projects/spine  (Python 3, package root = spine/)

CONVENTIONS (match exactly):
- Every module starts with a one-line-summary docstring, then \`from __future__ import annotations\`.
- Google-style docstrings with Args/Returns/Raises sections on public funcs/classes.
- Logging via \`logger = logging.getLogger(__name__)\` — never print().
- Full type hints. Prefer \`str | None\` unions.
- Tests run from repo root with .venv/bin/pytest. Async tests need \`@pytest.mark.asyncio\` (pytest-asyncio is in STRICT mode — no auto async).
- Tools: .venv/bin/ruff, .venv/bin/mypy, .venv/bin/pytest all exist.

KEY EXISTING INTERFACES (do not change these; integrate with them):
- spine.config.SpineConfig is a MUTABLE @dataclass with many fields incl:
  checkpoint_path:str(".spine/spine.db"), artifact_path:str(".spine/artifacts"),
  queue_path:str(".spine/queue.db"), workspace_root:str (absolute, auto-resolved),
  max_critic_retries:int. Loaded via SpineConfig.load(path=".spine/config.yaml").
  To make a sandbox-pointed copy, use \`dataclasses.replace(base_config, workspace_root=sandbox_dir)\`
  — do NOT re-list fields. KEEP checkpoint_path/queue_path RELATIVE so the checkpoint DB
  stays in the master tree (orchestrator never chdir's in worktree mode, so CWD stays master).
- spine.work.dispatcher.submit_work is ASYNC:
    async def submit_work(description:str, work_type:str="spec", config:SpineConfig|None=None, ...) -> dict
  It returns ONLY {"work_id", "status", "work_type"} (and "error" on failure).
  The per-phase completion flags (spec_completed, plan_completed, implement_completed,
  verify_completed, etc. — all bool) are NOT in that dict. They live in the LangGraph
  checkpoint state, retrieved via:
    from spine.persistence.checkpoint import CheckpointStore
    state = await CheckpointStore(db_path=config.checkpoint_path).get_state(work_id)  # -> dict | None
  state is a dict of WorkflowState fields incl those *_completed flags.
- Orchestrator methods are SYNC and use subprocess; to call the async submit_work / get_state
  from sync code use \`asyncio.run(...)\`. (Callers — CLI and ui_api executor threads — have no
  running loop, so asyncio.run is safe.)
- spine.exceptions defines: class SpineError(Exception) base. Add new exceptions as subclasses.
`.trim()

phase('Foundation')

const FOUNDATION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['files_written', 'orchestrator_interface', 'exceptions_added', 'config_schema'],
  properties: {
    files_written: { type: 'array', items: { type: 'string' }, description: 'Absolute paths written' },
    orchestrator_interface: {
      type: 'string',
      description: 'Exact public API of SpineGitOrchestrator: __init__ signature, every public method name + signature + one-line behavior, and what execute_transactional_run returns (the result dict keys). Markdown.',
    },
    exceptions_added: { type: 'array', items: { type: 'string' }, description: 'Exception class names added to spine/exceptions.py' },
    config_schema: { type: 'string', description: 'The spine-gate.yaml structure (keys + meaning) so integration code can render/read it.' },
    notes: { type: 'string', description: 'Anything integration agents must know (e.g. how to instantiate, how get_gate_status data is exposed).' },
  },
}

const foundation = await agent(`${CONVENTIONS}

You are implementing the FOUNDATION of a transactional git-sandbox lifecycle for SPINE. Wrap workflow execution so that after the workflow produces code changes, a validation pipeline runs and the changes are either atomically merged to main or hard-rolled-back.

Write these files (create spine/git/ as a new package):

=== 1. spine/exceptions.py — ADD these classes (append after the existing ones; do NOT remove anything) ===
- class GitOrchestratorError(SpineError): base for git orchestrator errors.
- class SandboxPreparationError(GitOrchestratorError): failed to create worktree/branch.
- class ValidationError(GitOrchestratorError): a validation gate failed. __init__(self, gate_name:str, command:str, output:str) storing those on self and building a useful message.
- class MergeError(GitOrchestratorError): fast-forward merge failed (conflict).
Read the current file first to match style.

=== 2. spine/git/__init__.py === empty package init (one-line module docstring only).

=== 3. spine/git/orchestrator.py === class SpineGitOrchestrator with:
  __init__(self, config_path:str="spine-gate.yaml", base_config:SpineConfig|None=None):
    - Load the gate YAML (yaml.safe_load) from config_path; if missing, fall back to sensible defaults
      matching the spine-gate.yaml you write below. Resolve git settings (main_branch, branch_prefix,
      strategy, sandbox_dir base). Store base_config (default SpineConfig.load()). Record self.master_dir
      = os.getcwd() (the repo root). self.patch_branch and self.sandbox_dir start None (set in prepare_sandbox).
      Use logging.getLogger(__name__).
  _execute_shell(self, cmd, cwd=None, timeout=60) -> tuple[bool,str,str]:
    - subprocess.run with shell, capture stdout/stderr text, enforce timeout (catch TimeoutExpired ->
      (False, "", "timeout")). Return (returncode==0, stdout, stderr). Log the command at debug.
  _resolve_validation_command(self, command:str) -> str:
    - If command starts with ".venv/", prefix with absolute master_dir (str(Path(self.master_dir)/command)).
      Else return unchanged. (Worktrees don't copy .venv.)
  prepare_sandbox(self) -> str:
    - Pre-check: working tree must be clean (\`git status --porcelain\` empty) else raise SandboxPreparationError.
    - Generate a unique branch name: branch_prefix + a short token. IMPORTANT: Date.now()/random are fine in
      Python; use uuid4().hex[:8] for uniqueness.
    - strategy "worktree": choose sandbox_dir (configured base + branch token, e.g. /tmp/spine-sandbox-<token>);
      run \`git worktree add -b <branch> <sandbox_dir> <main_branch>\`. strategy "branch": \`git checkout -b <branch>\`
      and sandbox_dir = master_dir.
    - On non-zero exit raise SandboxPreparationError with stderr. Set self.patch_branch/self.sandbox_dir; return sandbox_dir.
  run_validation_pipeline(self) -> dict:
    - Iterate the validation_pipeline gates IN ORDER (dict insertion order). For each: resolve command via
      _resolve_validation_command, run with cwd=self.sandbox_dir and the gate's timeout_seconds (default 60).
      On first failure return {"success": False, "gate": name, "command": cmd, "output": stdout+stderr,
      "failure_message": gate.get("failure_message","")}. If all pass return {"success": True}.
  _check_phase_prerequisites(self, work_id:str, required_phases:list[str]) -> bool:
    - If no required_phases, return True. Read checkpoint state:
      state = asyncio.run(CheckpointStore(db_path=self.base_config.checkpoint_path).get_state(work_id)) or {}.
      For each phase in required_phases, require state.get(f"{phase}_completed") truthy; log a warning + return
      False on the first that isn't. Return True if all present.
      (Note required_phases entries in the YAML are like "implement_completed" — strip a trailing "_completed"
      if present before formatting, so both "implement" and "implement_completed" work.)
  commit_and_merge(self) -> dict:
    - In sandbox_dir: \`git add .\`, \`git commit -m "spine(auto): verified patch <branch>"\` (tolerate empty commit
      -> treat "nothing to commit" as success-with-no-changes). Then in master_dir: \`git checkout <main_branch>\`,
      \`git merge --ff-only <branch>\`; if merge returns non-zero raise MergeError. Then \`git branch -d <branch>\`
      and (worktree strategy) \`git worktree remove <sandbox_dir>\`. Return {"success": True, "branch": branch, "merged": True}.
  rollback_workspace(self) -> dict:
    - Nuclear purge, all best-effort (never raise): in master_dir run
      \`git checkout <main_branch>\` (ignore failure), \`git worktree remove --force <sandbox_dir>\` (worktree mode),
      \`git branch -D <branch>\`, \`git worktree prune\`, then shutil.rmtree(sandbox_dir, ignore_errors=True) if it
      exists and != master_dir, then \`git reset --hard HEAD\` and \`git clean -fd\`. Return {"rolled_back": True}.
  execute_transactional_run(self, description:str, work_type:str="task") -> dict:
    - Full lifecycle with try/finally restoring os.getcwd():
      original_dir = os.getcwd()
      try:
        prepare_sandbox()
        sandbox_config = dataclasses.replace(self.base_config, workspace_root=self.sandbox_dir)
        work_result = asyncio.run(submit_work(description, work_type, sandbox_config))
        if work_result has "error": rollback_workspace(); return {"status":"failed","stage":"workflow", "error":..., **work_result}
        # phase prerequisites
        required = self.gate_config.get("require_successful_phases", [])
        if required and not self._check_phase_prerequisites(work_result["work_id"], required):
            rollback_workspace(); return {"status":"rolled_back","stage":"prerequisites", "work_id":...}
        validation = run_validation_pipeline()
        if not validation["success"]:
            rollback_workspace(); return {"status":"rolled_back","stage":"validation","gate":validation["gate"],"output":validation["output"], "work_id":...}
        if self.gate_config.get("auto_merge_on_success", True):
            merge = commit_and_merge(); return {"status":"merged","work_id":...,"branch":...}
        return {"status":"validated_pending_merge","work_id":...,"branch":...}
      finally:
        os.chdir(original_dir)
    - Be defensive: wrap the body so any unexpected exception triggers rollback_workspace() and returns
      {"status":"error","error":str(e)}.
  status(self) -> dict: return {"active": bool(self.patch_branch), "branch": self.patch_branch, "sandbox_dir": self.sandbox_dir, "strategy": self.strategy}.
  Provide a module-level helper \`def load_gate_config(path:str="spine-gate.yaml") -> dict\` reused by CLI \`gate config\`.

=== 4. spine-gate.yaml === at repo root. Structure:
  git: {main_branch: main, branch_prefix: "spine/patch-", strategy: worktree, sandbox_dir: /tmp/spine-sandbox}
  validation_pipeline: lint/typecheck/test gates each {command, timeout_seconds, failure_message}, using
    .venv/bin/ruff check ., .venv/bin/mypy spine/, .venv/bin/pytest tests/ -x -q.
  artifact_path: .spine/artifacts
  auto_merge_on_success: true
  require_successful_phases: [implement_completed, verify_completed]

After writing, run \`.venv/bin/ruff check spine/git/ spine/exceptions.py\` and fix any issues, and verify
\`.venv/bin/python -c "from spine.git.orchestrator import SpineGitOrchestrator; from spine.exceptions import ValidationError, MergeError, SandboxPreparationError, GitOrchestratorError"\` imports cleanly.

Return the structured interface contract so downstream integration code can build against it precisely.`,
  { label: 'foundation:core', phase: 'Foundation', schema: FOUNDATION_SCHEMA })

log(`Foundation done: ${(foundation?.files_written || []).length} files. Building integration against contract.`)

phase('Integration')

const CONTRACT = `
FOUNDATION CONTRACT (build against this exactly):

SpineGitOrchestrator interface:
${foundation?.orchestrator_interface || '(missing)'}

Exceptions added to spine.exceptions: ${(foundation?.exceptions_added || []).join(', ')}

spine-gate.yaml schema:
${foundation?.config_schema || '(missing)'}

Integration notes: ${foundation?.notes || '(none)'}
`.trim()

const integrationTasks = [
  {
    key: 'cli',
    label: 'integration:cli',
    prompt: `${CONVENTIONS}

${CONTRACT}

TASK: Add a \`gate\` Click command group to the SPINE CLI.

1. Create spine/cli/git_commands.py with a Click group \`gate\` exposing:
   - \`spine gate run "<description>" [--type task|critical_task] [--config spine-gate.yaml]\`:
       instantiate SpineGitOrchestrator(config_path=...) and call execute_transactional_run(description, work_type);
       print the result dict with rich (status-colored Panel like the existing \`run\` command in spine/cli/__init__.py).
       sys.exit(1) when result["status"] in ("failed","error","rolled_back").
   - \`spine gate status\`: instantiate orchestrator, print orchestrator.status().
   - \`spine gate rollback\`: instantiate orchestrator and call rollback_workspace(); print outcome.
   - \`spine gate config\`: load and pretty-print the gate config (use the foundation's load_gate_config helper).
   Use \`from rich.console import Console\`/Panel and a module-level \`console = Console()\` mirroring spine/cli/__init__.py.
   Each command should be \`@gate.command()\`. The group: \`@click.group() def gate(): """Git-gated transactional execution."""\`.

2. Modify spine/cli/__init__.py: import the gate group and register it with \`main.add_command(gate)\` near the bottom
   (after the existing command defs). Read spine/cli/__init__.py first to place the import/registration cleanly.

Validate: \`.venv/bin/ruff check spine/cli/\` and \`.venv/bin/python -c "from spine.cli import main"\` and
\`.venv/bin/spine gate config\` (should print the YAML). Fix any failures. Return a short summary of what you wrote.`,
  },
  {
    key: 'ui',
    label: 'integration:ui',
    prompt: `${CONVENTIONS}

${CONTRACT}

TASK: Add the git-gate UI integration.

1. Modify spine/ui_api/api.py (class UIApi) — read it first to match style. Add two methods:
   - \`submit_gated_work(self, description:str, work_type:str="task") -> dict\`: run the gated lifecycle
     WITHOUT blocking Streamlit. Mirror how restart_work() runs work on the worker's shared executor:
       def _run(): import asyncio; from spine.git.orchestrator import SpineGitOrchestrator;
                   SpineGitOrchestrator(base_config=self._config).execute_transactional_run(description, work_type)
       get_worker(self._config).get_executor().submit(_run)
       return {"status": "running", "work_type": work_type, "action": "gate_run"}
     (get_worker is already imported in api.py.)
   - \`get_gate_status(self) -> dict\`: return SpineGitOrchestrator(base_config=self._config).status().
     Keep it defensive (try/except -> {"active": False} on error).

2. Create spine/ui/_pages/gate_run.py with \`def render(api: UIApi) -> None\` — a Streamlit page modeled on
   spine/ui/_pages/work_submit.py: a title (e.g. "🔒 Git-Gated Execution"), an explanatory markdown blurb,
   a description text_area, a work_type selectbox (task/critical_task), a Submit button that calls
   api.submit_gated_work(...) and st.success/st.json the result. Add a "Sandbox Status" section that calls
   api.get_gate_status() and shows it. Add a short divider + info explaining the lifecycle (isolate → validate →
   merge-or-rollback).

3. Modify spine/ui/app.py to register the page. Read it first. Add a \`def _gate_run() -> None: from
   spine.ui._pages.gate_run import render; render(api)\` and add \`st.Page(_gate_run, title="Git Gate",
   icon="🔒", url_path="git-gate")\` into the "Work" section of the \`pages\` dict.

Validate: \`.venv/bin/ruff check spine/ui_api/ spine/ui/\` and
\`.venv/bin/python -c "import spine.ui._pages.gate_run; from spine.ui_api import UIApi"\`. Fix failures.
Return a short summary.`,
  },
  {
    key: 'tests',
    label: 'integration:tests',
    prompt: `${CONVENTIONS}

${CONTRACT}

TASK: Write tests for the git orchestrator. Read spine/git/orchestrator.py and spine/exceptions.py FIRST to
match the real signatures (the contract above is a guide; the actual code is the source of truth).

1. tests/unit/test_git_orchestrator.py — fast unit tests with git subprocess MOCKED:
   - Patch SpineGitOrchestrator._execute_shell (monkeypatch) to return scripted (success,stdout,stderr) tuples.
   - Test _resolve_validation_command rewrites ".venv/..." to an absolute master path and leaves other commands alone.
   - Test run_validation_pipeline: all-pass returns {"success":True}; first-gate-fail returns success False with the
     failing gate name and stops (later gates not run).
   - Test prepare_sandbox raises SandboxPreparationError when the tree is dirty (mock git status --porcelain nonempty)
     or when worktree add fails.
   - Test rollback_workspace never raises even if every shell call "fails", and returns {"rolled_back": True}.
   - Test execute_transactional_run on a validation failure path: stub submit_work (monkeypatch the symbol the
     orchestrator imports) to return a fake {"work_id":"abc","status":"completed","work_type":"task"}, stub
     _check_phase_prerequisites True, force run_validation_pipeline to fail, assert rollback happened and status
     "rolled_back". Build a SpineConfig via SpineConfig() or a tmp_path config; pass base_config to the constructor.
   Use pytest, monkeypatch, tmp_path. Mark async tests with @pytest.mark.asyncio only if needed (prefer mocking so
   you don't actually run asyncio).

2. tests/integration/test_git_lifecycle.py — end-to-end with a REAL throwaway git repo in tmp_path:
   - A fixture that \`git init\`s tmp_path, configures user.email/user.name, makes an initial commit on \`main\`
     (\`git checkout -b main\` or \`git branch -m main\`), writes a SpineConfig pointed at it.
   - Test prepare_sandbox + commit_and_merge round-trip: monkeypatch the orchestrator so the "workflow" just writes
     a file into the sandbox, then validate pipeline trivially passes (override gate_config with an \`echo\`-based
     gate or empty pipeline), assert the file lands on main after merge and the branch/worktree are gone.
   - Test the rollback path leaves main pristine (\`git status --porcelain\` empty, the stray file absent) after a
     forced validation failure.
   - Skip the whole module if \`git\` is not on PATH (shutil.which("git")) via pytestmark.
   Keep timeouts short. These must not touch the real spine repo — everything in tmp_path.

Validate: \`.venv/bin/ruff check tests/unit/test_git_orchestrator.py tests/integration/test_git_lifecycle.py\` and run
\`.venv/bin/pytest tests/unit/test_git_orchestrator.py tests/integration/test_git_lifecycle.py -q\`. Iterate until green.
Return the pytest summary line (passed/failed counts).`,
  },
]

const results = await parallel(integrationTasks.map((t) => () =>
  agent(t.prompt, { label: t.label, phase: 'Integration' }).then((r) => ({ key: t.key, summary: r }))
))

return {
  foundation_files: foundation?.files_written || [],
  exceptions: foundation?.exceptions_added || [],
  integration: results.filter(Boolean).map((r) => ({ key: r.key, summary: (r.summary || '').slice(0, 400) })),
}
