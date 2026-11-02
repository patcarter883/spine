# RFC 0001 — Spine v2: Go Runtime and Reconciler Control Plane

| Field | Value |
|---|---|
| Status | Draft, for review |
| Date | 2026-11-02 |
| Supersedes | LangGraph `StateGraph` workflow engine (`spine/workflow/`), `RalphLoopWorker` |
| Carries forward | Worktree sandbox lessons (`spine/git/sandbox.py`, `spine/git/orchestrator.py`), gate config (`spine-gate.yaml`), verify ratchet, slice scheduler |

---

## 0. Summary

Spine v2 is one Go binary. It replaces the fixed phase DAG with a **level-triggered reconciler**, the same pattern Kubernetes controllers use. Each task has a declared **Spec** (intent plus acceptance criteria written as gate definitions) and an **Observed Status** that only deterministic tools can write. A controller loop runs one bounded step at a time until the two match.

An LLM **supervisor** picks the next step from a closed, typed set of actions (`RunWorker`, `PatchWorker`, `Rollback`, `Replan`, `Escalate`, …) based on a typed observation. It never reads a conversation, and it cannot write the `Verified` condition. Only the gate engine can set that condition, and only when the gate exits `0`.

The rest of the design follows from three rules:

1. **The DAG is data, not control flow.** Task dependencies are rows in a table. The order of work inside a task is decided again on every reconcile, so an unexpected test break means *observe → decide again*. It does not need a pre-wired fallback edge.
2. **Git plumbing, not porcelain, for anything shared.** Integration uses `git merge-tree --write-tree` and `git update-ref` compare-and-swap. No shared index is touched and no branch is ever checked out in the primary tree.
3. **Every external effect is a lease.** Worktrees, child process groups and git locks are all recorded in SQLite before they are created, and they are reconciled on startup. A `kill -9` of the supervisor at any point leaves state that the next boot can repair.

---

## 1. What we keep from v1 (and why)

v1 was not a mistake. Its trace audits produced hard-won operational rules that v2 must encode structurally:

| v1 lesson (source) | v2 encoding |
|---|---|
| Gates must pass on a clean `main`, or every landing is blocked on pre-existing debt (`spine-gate.yaml` header) | **Baseline-relative gates**: every gate runs once on the base tree; a task gate passes when it adds *no new failures* (§5.4). The gate tool still returns a strict exit code. |
| Rolling back a `needs_review` / `stalled` run destroyed the only reviewable copy of good work (`sandbox.py` `_PRESERVE_STATUSES`) | **Snapshots are refs** (`refs/spine/snap/<task>/<n>`). Rollback never deletes work. Teardown deletes the *worktree*, never the snapshot refs, until GC policy says so. |
| Verify ratchet: on-disk state should be the best-scoring cycle, not the last edit | `Snapshot` after every gate run, tagged with its score; `Rollback(best)` is a first-class action. |
| `git reset --hard` / `clean` on the master tree destroyed operator state (`orchestrator.py` `rollback_workspace`) | The primary checkout is **never** a worktree target. The manager refuses any path equal to the main worktree. |
| Fresh worktrees lack untracked env (`.env`, keys) (`sandbox_setup` hooks) | `Provisioner` hooks run inside the new worktree before the lease turns `Ready`; a hook failure tears down before any tokens are spent. |
| Spine's own `.spine/` output must not count as a dirty tree | Spine state lives under `$GIT_COMMON_DIR/spine/`, outside every working tree. |
| Sync nodes broke asyncio locks; providers not serializable in state | Moot: no serialized graph state, and providers are config, resolved per call. |
| Slice scheduler (topological waves) | Kept as `internal/scheduler`, but it schedules **tasks** (data) and does not wire nodes. |

What we drop: LangGraph checkpoints of whole agent state, critic-as-a-phase, and the four hard-coded work types. Work types become **Spec templates** (§4.1).

---

## 2. Package layout

```
spine/
├── cmd/
│   └── spine/                 # main: cobra/urfave CLI, wires deps, signal handling
│       └── main.go
├── internal/
│   ├── api/                   # Public-ish domain types: Task, Spec, Status, Condition, Action
│   ├── reconciler/            # Controller: workqueue, Reconcile(), supervisor policy, action executor
│   │   ├── controller.go
│   │   ├── reconcile.go
│   │   ├── actions.go         # typed Action set + legality rules
│   │   └── supervisor.go      # LLM decision adapter (structured output only)
│   ├── agent/                 # AgentRunner: spawn headless CLI agents, stream events, budgets
│   │   ├── runner.go
│   │   ├── procgroup_unix.go  # Setpgid, group kill, reaper
│   │   └── adapters/          # claude, codex, opencode, aider, local-harness (stdio JSONL)
│   ├── worktree/              # WorktreeManager: leases, provision, snapshot, rollback, merge
│   │   ├── manager.go
│   │   ├── plumbing.go        # merge-tree, update-ref CAS, temp-index snapshots
│   │   └── locks.go           # per-repo ref mutex, stale lock recovery
│   ├── tools/                 # ToolRegistry + deterministic tools
│   │   ├── registry.go
│   │   ├── cache.go           # content-addressed by git tree id
│   │   ├── build/             # compile/lint/test runners with parsers → Diagnostics
│   │   ├── ast/               # tree-sitter symbol extraction, API-contract diff
│   │   └── diff/              # patch validation, merge pre-flight, scope checks
│   ├── gate/                  # Gate engine: Spec.Acceptance → tool plan → verdict (baseline-relative)
│   ├── scheduler/             # task dependency DAG, concurrency limits, fair share
│   ├── storage/               # SQLite (WAL): tasks, attempts, events, leases, tool cache; migrations
│   ├── events/                # typed event bus (fan-out to storage, UI SSE, logs)
│   ├── config/                # .spine/config.yaml + spine-gate.yaml loader, validation
│   └── exec/                  # hardened os/exec wrapper: timeouts, output caps, env scrubbing
├── pkg/
│   └── spineapi/              # (optional) stable types for external UIs / plugins
└── web/                       # (optional) embedded static UI served by `spine ui` via embed.FS
```

**Boundary rules (enforced by `go vet`-style import checks in CI):**

* `api` imports nothing internal. Everything imports `api`.
* `reconciler` depends on the *interfaces* `worktree.Manager`, `tools.Registry`, `gate.Engine`, `agent.Runner` and `storage.Store`. It never touches `os/exec` or git.
* Only `internal/exec` calls `os/exec`. Only `internal/worktree` runs git commands that mutate state. Tools may run read-only git (`git diff`, `git ls-tree`) through `exec`.
* The supervisor (`reconciler/supervisor.go`) is the **only** code that talks to an LLM *as a decision-maker*. Workers talk to LLMs through their own CLI processes, and Spine sees only their stdio.

---

## 3. Core interfaces

### 3.1 Domain types (`internal/api`)

```go
package api

import "time"

type TaskID string

// Spec is desired state. Immutable once a task leaves Pending, except via a
// Replan action, which creates a new Spec generation (like k8s metadata.generation).
type Spec struct {
	Generation  int64
	Title       string
	Intent      string            // natural-language goal handed to workers
	BaseRef     string            // integration target, e.g. "refs/heads/main"
	Scope       []string          // path globs the task may modify (enforced by diff tool)
	Acceptance  []GateRef         // deterministic acceptance criteria
	DependsOn   []TaskID
	Budget      Budget
	Escalation  EscalationPolicy  // e.g. require human approval before merge
}

type Budget struct {
	MaxAttempts     int           // worker generations (fresh runs)
	MaxPatchRounds  int           // N: feedback iterations per attempt
	MaxWallClock    time.Duration
	MaxTokens       int64         // summed from agent usage events
	PerRunTimeout   time.Duration
}

type GateRef struct {
	Tool   string            // registry name, e.g. "go.test", "ts.api_contract"
	Args   map[string]any    // validated against the tool's input schema
	Policy GatePolicy        // BaselineRelative | Absolute
}

type GatePolicy int

const (
	BaselineRelative GatePolicy = iota // pass iff no new failures vs base tree
	Absolute                           // pass iff exit code 0, full stop
)

type Phase string

const (
	PhasePending     Phase = "Pending"
	PhaseProvisioned Phase = "Provisioned"
	PhaseWorking     Phase = "Working"
	PhaseVerifying   Phase = "Verifying"
	PhaseVerified    Phase = "Verified"    // gate exit 0, not yet integrated
	PhaseIntegrated  Phase = "Integrated"  // terminal success
	PhaseEscalated   Phase = "Escalated"   // terminal until human acts
	PhaseFailed      Phase = "Failed"      // terminal, budget exhausted
)

// Status is observed state. Fields are written by specific owners only (see §4.3).
type Status struct {
	ObservedGeneration int64
	Phase              Phase
	Conditions         []Condition
	Attempt            int
	PatchRound         int
	Worktree           *WorktreeRef
	BestSnapshot       *SnapshotRef
	LastGate           *GateVerdict
	TokensUsed         int64
}

type Condition struct {
	Type               string // "Provisioned", "WorkerSucceeded", "GatePassed", "Mergeable", "Verified"
	Status             bool
	Reason             string // machine-readable CamelCase
	Message            string // bounded, for humans
	ObservedAt         time.Time
	Source             string // owner that wrote it: "gate", "worktree", "agent"
}

type Task struct {
	ID     TaskID
	Spec   Spec
	Status Status
	// Optimistic concurrency token for storage CAS writes.
	ResourceVersion int64
}
```

### 3.2 `WorktreeManager` (`internal/worktree`)

```go
package worktree

import (
	"context"

	"spine/internal/api"
)

// Lease is a durable claim on a worktree. It is recorded in storage BEFORE
// `git worktree add` runs, so a crash between the two is recoverable.
type Lease struct {
	TaskID   api.TaskID
	Attempt  int
	Path     string // $SPINE_WT_ROOT/<repo-hash>/<task>/<attempt>
	Branch   string // refs/heads/spine/<task>/<attempt>
	BaseOID  string // commit the worktree was cut from (pinned, not a moving ref)
	OwnerPID int    // supervisor PID holding it; for orphan detection
	State    LeaseState
}

type LeaseState string

const (
	LeaseRequested LeaseState = "Requested" // row written, git not yet run
	LeaseReady     LeaseState = "Ready"
	LeaseDraining  LeaseState = "Draining"  // teardown in progress
	LeaseReleased  LeaseState = "Released"
)

type Snapshot struct {
	Ref    string // refs/spine/snap/<task>/<seq>
	Commit string
	Tree   string // tree OID: the content address used by the tool cache
	Label  string // "post-worker", "post-patch-2", "best"
	Score  *float64
}

type MergeResult struct {
	Clean       bool
	ResultTree  string   // from merge-tree --write-tree
	Conflicts   []string // paths, when !Clean
	NewHead     string   // commit OID written to BaseRef, when integrated
}

type Manager interface {
	// Create cuts a new worktree at the pinned commit `base` resolves to, runs
	// provisioning hooks, and returns a Ready lease. Idempotent per (task, attempt):
	// a second call returns the existing Ready lease or finishes a Requested one.
	Create(ctx context.Context, task api.TaskID, attempt int, base string) (*Lease, error)

	// Snapshot records the full working-tree state (tracked + untracked, minus
	// ignored) as a commit on a hidden ref. It never touches the worktree's index.
	Snapshot(ctx context.Context, l *Lease, label string) (*Snapshot, error)

	// Rollback restores the worktree to a snapshot: reset tracked files and
	// remove untracked files that are not in the snapshot.
	Rollback(ctx context.Context, l *Lease, to *Snapshot) error

	// Preflight computes the merge of snapshot onto the current tip of base
	// with merge-tree. It has no side effects on any worktree or ref.
	Preflight(ctx context.Context, snap *Snapshot, base string) (*MergeResult, error)

	// Merge integrates via CAS on the base ref: update-ref <base> <new> <expectedOld>.
	// It fails with ErrBaseMoved when another task integrated first; the caller re-runs
	// Preflight plus the gate on the new merge result.
	Merge(ctx context.Context, snap *Snapshot, base string, msg string) (*MergeResult, error)

	// Teardown kills nothing (processes belong to agent.Runner) but refuses to run
	// while a process lease on the path is live. It removes the worktree and the
	// branch, keeps the snapshot refs, and marks the lease Released. Idempotent.
	Teardown(ctx context.Context, l *Lease) error

	// Recover reconciles lease rows against `git worktree list --porcelain` and the
	// filesystem at startup. It finishes or reverses half-done Create and Teardown
	// operations and clears stale locks (§6.2).
	Recover(ctx context.Context) (RecoveryReport, error)
}

var (
	ErrBaseMoved    = errors.New("worktree: base ref moved during merge")
	ErrPrimaryTree  = errors.New("worktree: refusing to operate on the primary worktree")
	ErrLeaseHeld    = errors.New("worktree: lease has live processes")
)
```

### 3.3 `ToolRegistry` and `Tool` (`internal/tools`)

```go
package tools

import (
	"context"
	"encoding/json"
	"time"
)

// Determinism class decides cacheability.
type Determinism int

const (
	Pure        Determinism = iota // output = f(tree, args). Cache by tree OID.
	Hermetic                       // pure but slow/env-sensitive (e.g. tests): cache with env fingerprint
	Effectful                      // never cached (e.g. network, wall-clock)
)

type Spec struct {
	Name        string          // "go.build", "go.test", "ts.symbols", "ts.api_contract", "diff.scope"
	Version     string          // bump to invalidate cache
	InputSchema json.RawMessage // JSON Schema; args validated before Run
	Determinism Determinism
	Timeout     time.Duration
	// ReadOnly tools may run concurrently on the same worktree.
	// Mutating tools (formatters with --write) take the worktree write lock.
	ReadOnly bool
}

type Invocation struct {
	Worktree string          // absolute path
	Tree     string          // tree OID of the snapshot being evaluated (cache key part)
	Args     json.RawMessage
}

// Result is the ONLY thing a tool returns. No free-form chat.
type Result struct {
	Tool        string        `json:"tool"`
	ExitCode    int           `json:"exit_code"` // 0 = pass. The gate keys on this.
	Diagnostics []Diagnostic  `json:"diagnostics"`
	Metrics     map[string]float64 `json:"metrics,omitempty"` // tests_passed, coverage, ...
	Artifacts   []ArtifactRef `json:"artifacts,omitempty"`   // full logs, stored by digest
	Duration    time.Duration `json:"duration_ns"`
	Cached      bool          `json:"cached"`
	Truncated   bool          `json:"truncated"`             // stdout/stderr hit the cap
}

type Diagnostic struct {
	Severity Severity `json:"severity"` // error|warning|info
	Code     string   `json:"code"`     // compiler/linter/test id, e.g. "TestFoo", "SA4006"
	File     string   `json:"file,omitempty"`
	Line     int      `json:"line,omitempty"`
	Col      int      `json:"col,omitempty"`
	Message  string   `json:"message"`  // capped at 2 KiB
	// Fingerprint identifies "the same failure" across trees so baseline-relative
	// gates can subtract pre-existing failures. Normalized: no line numbers, no tmp paths.
	Fingerprint string `json:"fingerprint"`
}

type Tool interface {
	Spec() Spec
	Run(ctx context.Context, inv Invocation) (Result, error) // error = tool infra failure, NOT a failing check
}

type Registry interface {
	Register(t Tool) error
	Lookup(name string) (Tool, bool)
	List() []Spec
	// Invoke validates args, consults the cache, enforces timeouts, and records
	// an event. Concurrency is bounded per-tool and globally.
	Invoke(ctx context.Context, name string, inv Invocation) (Result, error)
}

type Cache interface {
	Get(key CacheKey) (Result, bool)
	Put(key CacheKey, r Result) error
}

// CacheKey = sha256(tool.Name, tool.Version, tree OID, canonical(args), envFingerprint).
// The tree OID comes from the snapshot, so two worktrees with identical content
// share results. The baseline tree is computed once per base commit, for all tasks.
type CacheKey [32]byte
```

Note the split between `error` and `ExitCode`. A failing test is `ExitCode != 0, err == nil`. A tool that could not run (binary missing, OOM) is `err != nil`. The reconciler treats the second case as an **infrastructure fault**: it requeues with backoff and does not burn an agent patch round.

### 3.4 `AgentRunner` (`internal/agent`)

```go
package agent

import (
	"context"
	"time"

	"spine/internal/api"
)

type RunRequest struct {
	Task      api.TaskID
	Attempt   int
	Round     int              // 0 = fresh generation; >0 = patch round
	Worktree  string
	Adapter   string           // "claude", "codex", "local-vllm", ...
	Prompt    Prompt           // rendered from typed inputs, never from transcripts
	Timeout   time.Duration
	MaxTokens int64
	Env       map[string]string // explicit allow-list; parent env is NOT inherited
}

// Prompt is built from structured data. The worker sees typed feedback, never the
// supervisor's reasoning and never another worker's output buffer.
type Prompt struct {
	Intent      string
	Scope       []string
	Acceptance  []string          // human-readable rendering of GateRefs
	Feedback    []FeedbackItem    // empty on round 0
	Constraints []string          // e.g. "do not modify files outside Scope"
}

type FeedbackItem struct {
	Source      string            // tool name
	Diagnostics []tools.Diagnostic // NEW failures only (baseline-subtracted), top-K by severity
	Hint        string            // optional supervisor instruction, max 500 chars, from a typed action
}

type EventKind string

const (
	EvStarted    EventKind = "started"
	EvToolCall   EventKind = "tool_call"   // the worker's own tool calls, parsed from stream-json
	EvUsage      EventKind = "usage"       // tokens in/out, for budget enforcement
	EvStderr     EventKind = "stderr"
	EvHeartbeat  EventKind = "heartbeat"
	EvExited     EventKind = "exited"
)

type Event struct {
	Kind    EventKind
	At      time.Time
	Payload any // typed per kind
}

type Outcome struct {
	ExitCode   int
	Signal     string        // non-empty if killed
	Reason     OutcomeReason // Completed | Timeout | BudgetExceeded | Crashed | Cancelled | Stalled
	TokensUsed int64
	Duration   time.Duration
}

type Runner interface {
	// Run starts the agent in its own process group with cwd=Worktree, streams
	// events on the returned channel (closed on exit), and blocks until Outcome.
	// Cancelling ctx sends SIGTERM to the group, then SIGKILL after a grace period.
	// Run never returns while any process in the group is still alive.
	Run(ctx context.Context, req RunRequest, events chan<- Event) (Outcome, error)
}

// Adapter translates RunRequest to a concrete CLI invocation and parses its stdout.
type Adapter interface {
	Name() string
	Command(req RunRequest) (argv []string, stdin []byte, err error)
	Parse(line []byte) (Event, bool) // stream-json / JSONL; unknown lines → EvStderr
}
```

### 3.5 `Reconciler` and `Supervisor` (`internal/reconciler`)

```go
package reconciler

type Result struct {
	Requeue      bool
	RequeueAfter time.Duration
}

type Reconciler interface {
	// Reconcile performs AT MOST ONE externally visible action and persists the
	// new Status before returning. It must be idempotent: calling it twice with the
	// same stored Task must not run two workers or two merges.
	Reconcile(ctx context.Context, id api.TaskID) (Result, error)
}

// Observation is everything the supervisor may see. All fields are typed.
type Observation struct {
	Spec          api.Spec
	Status        api.Status
	Gate          *api.GateVerdict       // latest
	GateHistory   []api.GateSummary      // per round: new-failure counts, score
	Diff          DiffStats              // files touched, out-of-scope paths, LOC
	Symbols       *SymbolDelta           // public API added/removed/changed (tree-sitter)
	LastOutcome   *agent.Outcome
	Budget        BudgetRemaining
	Legal         []ActionKind           // precomputed: the supervisor may only pick from this
}

type ActionKind string

const (
	ActRunWorker    ActionKind = "RunWorker"    // fresh generation (new attempt, new worktree)
	ActPatchWorker  ActionKind = "PatchWorker"  // same worktree, feed diagnostics back
	ActRollback     ActionKind = "Rollback"     // restore a snapshot, then PatchWorker
	ActReplan       ActionKind = "Replan"       // propose Spec generation+1 (split / narrow scope)
	ActEscalate     ActionKind = "Escalate"     // to human, with a structured reason
	ActAbandon      ActionKind = "Abandon"      // mark Failed
)

type Decision struct {
	Action   ActionKind `json:"action"`
	Snapshot string     `json:"snapshot,omitempty"` // for Rollback
	Hint     string     `json:"hint,omitempty"`     // ≤500 chars, attached to FeedbackItem
	Focus    []string   `json:"focus,omitempty"`    // diagnostic fingerprints to prioritize
	Split    []api.Spec `json:"split,omitempty"`    // for Replan
	Reason   string     `json:"reason"`             // enum-like code: "RegressionLoop", "ScopeCreep", ...
}

// Supervisor maps an Observation to a Decision. The LLM implementation uses
// constrained decoding / JSON-schema output. A deterministic RuleSupervisor
// implements the same interface and is the fallback when the LLM errors or
// returns an illegal action (and is the default in tests).
type Supervisor interface {
	Decide(ctx context.Context, obs Observation) (Decision, error)
}
```

Note what the supervisor cannot do. `Verify`, `Merge` and `MarkComplete` are not in `ActionKind`. Those transitions belong to the controller and fire only on gate conditions (§4.3).

---

## 4. Reconciliation state machine

### 4.1 Specs replace work types

| v1 work type | v2 Spec template |
|---|---|
| `task` | `Acceptance = [build, unit tests (baseline-relative), diff.scope]` |
| `critical_task` | adds `ts.api_contract (Absolute)` and `Escalation.RequireApprovalBeforeMerge` |
| `reviewed_task` | `Escalation.RequireApprovalBeforeWork` after Replan produces the task split |
| SPECIFY / PLAN phases | A **planner task**: its "worktree" output is a set of child Specs, validated by a schema tool (`spec.validate`). Children are inserted with `DependsOn`. Planning is just another reconciled task whose gate is structural. |

### 4.2 Phase transitions

```
            ┌──────────── Replan (gen+1) ────────────┐
            ▼                                         │
 Pending ─► Provisioned ─► Working ─► Verifying ──┬─► Verified ─► Integrated
   ▲            │             ▲           │       │      │
   │            │             │           │       │      └─ ErrBaseMoved → Verifying (re-gate on new base)
   │            │       Patch/Rollback ◄──┘ fail  │
   │            │                                 │
   │            └─ provision hook fail ─► Failed  └─► Escalated (human)
   └─ deps not Integrated: stay Pending
```

### 4.3 Who may write which condition

Field ownership is the core of "zero theater". Storage enforces it: each write carries a `Source`, and `storage.UpdateStatus` rejects a condition type written by a non-owner.

| Condition | Sole writer | Set true when |
|---|---|---|
| `Provisioned` | worktree | lease `Ready` and hooks exited 0 |
| `WorkerSucceeded` | agent | process group exited 0 within budget |
| `GatePassed` | gate | every `Acceptance` gate exit 0 on snapshot tree T |
| `Mergeable` | worktree | `Preflight` clean against current base tip |
| `Verified` | gate | `GatePassed` evaluated on the **merge result tree**, not just the branch tree |
| `Integrated` | worktree | `update-ref` CAS succeeded |

The supervisor writes **only** `Decision` rows (audit log). A supervisor that returns `{"action":"MarkComplete"}` fails JSON-schema validation and falls back to `RuleSupervisor`.

### 4.4 Controller

```go
// controller.go
type Controller struct {
	store  storage.Store
	queue  workqueue.RateLimitingInterface // k8s client-go/util/workqueue, or a 150-line equivalent
	rec    Reconciler
	sched  scheduler.Scheduler
	workers int
}

func (c *Controller) Run(ctx context.Context) error {
	// 1. Crash recovery BEFORE accepting work.
	if err := c.recoverAll(ctx); err != nil {
		return err
	}
	// 2. Level-triggered resync: every non-terminal task is enqueued at boot and
	//    every resyncPeriod, so a missed event can delay work but never lose it.
	go c.resyncLoop(ctx, 30*time.Second)

	g, ctx := errgroup.WithContext(ctx)
	for i := 0; i < c.workers; i++ {
		g.Go(func() error {
			for c.processNext(ctx) {
			}
			return nil
		})
	}
	<-ctx.Done()
	c.queue.ShutDownWithDrain() // finish in-flight reconciles; agents get SIGTERM via ctx
	return g.Wait()
}

func (c *Controller) processNext(ctx context.Context) bool {
	item, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(item)
	id := item.(api.TaskID)

	res, err := c.rec.Reconcile(ctx, id)
	switch {
	case err != nil:
		c.queue.AddRateLimited(id) // exponential backoff, infra faults only
	case res.RequeueAfter > 0:
		c.queue.Forget(id)
		c.queue.AddAfter(id, res.RequeueAfter)
	case res.Requeue:
		c.queue.AddRateLimited(id)
	default:
		c.queue.Forget(id)
	}
	return true
}
```

The workqueue gives us **per-key serialization for free**. A `TaskID` is never processed by two goroutines at once, and a key that is re-added while in flight is coalesced. Tasks run concurrently with each other, bounded by `workers` and by the scheduler's semaphore of worktree slots.

### 4.5 Reconcile

Long-running work (an agent run or a test suite) happens in a goroutine that the reconciler **starts and records**. The reconcile then returns. When the work finishes, it writes its result and enqueues the task again. Reconcile never blocks for minutes, so one slow task does not hold a controller worker hostage.

```go
// reconcile.go
func (r *reconciler) Reconcile(ctx context.Context, id api.TaskID) (Result, error) {
	t, err := r.store.GetTask(ctx, id)
	if err != nil {
		return Result{}, err
	}
	if t.Status.Phase.Terminal() {
		return Result{}, nil
	}
	// An operation (agent run, gate run, merge) is already in flight for this task.
	// Its completion handler re-enqueues us; nothing to do.
	if op, ok := r.inflight.Get(id); ok {
		return Result{RequeueAfter: op.PollAfter()}, nil // also a stall watchdog (§6.4)
	}
	if err := r.checkBudget(t); err != nil {
		return r.transition(ctx, t, api.PhaseFailed, "BudgetExhausted", err.Error())
	}

	switch t.Status.Phase {

	case api.PhasePending:
		if !r.sched.DepsIntegrated(ctx, t) || !r.sched.TryAcquireSlot(id) {
			return Result{RequeueAfter: 10 * time.Second}, nil
		}
		base, err := r.git.ResolveCommit(ctx, t.Spec.BaseRef)
		if err != nil {
			return Result{}, err
		}
		lease, err := r.wt.Create(ctx, id, t.Status.Attempt+1, base) // idempotent
		if err != nil {
			if errors.Is(err, worktree.ErrProvisionHook) {
				return r.transition(ctx, t, api.PhaseFailed, "ProvisionFailed", err.Error())
			}
			return Result{}, err
		}
		r.gate.WarmBaseline(base, t.Spec.Acceptance) // async, cached per base commit
		return r.commit(ctx, t, func(s *api.Status) {
			s.Attempt++
			s.PatchRound = 0
			s.Worktree = lease.Ref()
			s.Phase = api.PhaseProvisioned
			s.SetCondition("Provisioned", true, "LeaseReady", "", "worktree")
		})

	case api.PhaseProvisioned:
		// Fresh generation: round 0, no feedback.
		return r.startAgent(ctx, t, agent.Prompt{
			Intent: t.Spec.Intent, Scope: t.Spec.Scope,
			Acceptance: render(t.Spec.Acceptance),
		})

	case api.PhaseWorking:
		// Reached only after the agent's completion handler stored its Outcome.
		out := t.Status.LastOutcome()
		snap, err := r.wt.Snapshot(ctx, t.Lease(), fmt.Sprintf("a%d-r%d", t.Status.Attempt, t.Status.PatchRound))
		if err != nil {
			return Result{}, err
		}
		if out.Reason == agent.Crashed || out.Reason == agent.Timeout {
			// Do not gate a half-written tree blindly; let the supervisor pick
			// Rollback(best) / PatchWorker / RunWorker. The snapshot is kept for forensics.
			return r.decide(ctx, t, snap)
		}
		return r.startGate(ctx, t, snap) // → PhaseVerifying, async

	case api.PhaseVerifying:
		v := t.Status.LastGate // written by the gate completion handler
		if v.ExitCode == 0 {
			r.maybePromoteBest(ctx, t, v)
			return r.integrate(ctx, t, v.Snapshot)
		}
		r.maybePromoteBest(ctx, t, v) // the ratchet keeps the best-scoring snapshot even on failure
		return r.decide(ctx, t, v.Snapshot)

	case api.PhaseVerified:
		return r.integrate(ctx, t, t.Status.BestSnapshot)
	}
	return Result{}, fmt.Errorf("reconcile: unhandled phase %q", t.Status.Phase)
}
```

### 4.6 Recovery: deciding after a failed gate

```go
func (r *reconciler) decide(ctx context.Context, t *api.Task, snap *worktree.Snapshot) (Result, error) {
	obs := r.observe(ctx, t, snap) // diff stats, symbol delta, gate history; all cached tools
	obs.Legal = legalActions(t)    // e.g. PatchWorker illegal once PatchRound == MaxPatchRounds

	d, err := r.supervisor.Decide(ctx, obs)
	if err != nil || !slices.Contains(obs.Legal, d.Action) {
		d = r.rules.Decide(obs) // deterministic fallback; never errors
	}
	r.store.AppendDecision(ctx, t.ID, d, obs.Digest()) // audit: what it saw, what it chose

	switch d.Action {
	case ActPatchWorker:
		return r.startAgent(ctx, t, r.feedbackPrompt(t, obs, d)) // round++

	case ActRollback:
		to := r.resolveSnapshot(t, d.Snapshot) // must be one of this task's snapshots
		if err := r.wt.Rollback(ctx, t.Lease(), to); err != nil {
			return Result{}, err
		}
		return r.startAgent(ctx, t, r.feedbackPrompt(t, obs, d))

	case ActRunWorker: // throw the worktree away, new attempt from base
		if err := r.teardownWithProcs(ctx, t); err != nil {
			return Result{}, err
		}
		return r.transition(ctx, t, api.PhasePending, "FreshAttempt", d.Reason)

	case ActReplan:
		return r.replan(ctx, t, d.Split) // validated by spec.validate; children inherit BaseRef

	case ActEscalate:
		return r.transition(ctx, t, api.PhaseEscalated, d.Reason, "") // worktree + snapshots preserved

	default: // ActAbandon
		return r.transition(ctx, t, api.PhaseFailed, d.Reason, "")
	}
}

// legalActions encodes the budget as a closed set. The LLM can't argue past it.
func legalActions(t *api.Task) []ActionKind {
	acts := []ActionKind{ActEscalate, ActAbandon}
	b, s := t.Spec.Budget, t.Status
	if s.PatchRound < b.MaxPatchRounds {
		acts = append(acts, ActPatchWorker)
		if len(t.Snapshots()) > 1 {
			acts = append(acts, ActRollback)
		}
	}
	if s.Attempt < b.MaxAttempts {
		acts = append(acts, ActRunWorker)
	}
	if t.Spec.Generation < maxGenerations {
		acts = append(acts, ActReplan)
	}
	return acts
}
```

**`RuleSupervisor` (deterministic fallback, and the baseline to beat):**

1. New-failure count went up from round *k−1* to *k* → `Rollback(best)`, then `PatchWorker` focusing on the regressed fingerprints.
2. The same fingerprint set has appeared for 2 consecutive rounds (no progress) → `RunWorker` if attempts remain, else `Escalate("NoProgress")`.
3. `diff.scope` reports out-of-scope paths → `PatchWorker` with hint "revert changes outside scope".
4. Otherwise → `PatchWorker` with the top-K new diagnostics.
5. No legal patch/attempt left → `Escalate("BudgetExhausted")`.

The LLM supervisor earns its keep on the cases rules handle badly: picking *which* failing tests to focus on, telling an interface-shift cascade (→ `Replan` to split out the interface change first) from a local bug, and writing the short `Hint`.

### 4.7 Integration

```go
func (r *reconciler) integrate(ctx context.Context, t *api.Task, snap *worktree.Snapshot) (Result, error) {
	if t.Spec.Escalation.RequireApprovalBeforeMerge && !t.Status.Approved() {
		return r.transition(ctx, t, api.PhaseEscalated, "AwaitingApproval", "")
	}
	pre, err := r.wt.Preflight(ctx, snap, t.Spec.BaseRef)
	if err != nil {
		return Result{}, err
	}
	if !pre.Clean {
		// Conflicts are observations. Feed them to the worker after rebasing its
		// worktree on the new base (worktree.Rebase on a temp branch), within PatchRound budget.
		return r.decide(ctx, t, snap.WithConflicts(pre.Conflicts))
	}
	// Gate the MERGE RESULT tree if base moved since the snapshot was cut. The tool
	// cache makes this free when base did not move (same tree OID).
	if pre.ResultTree != snap.Tree {
		return r.startGateOnTree(ctx, t, pre.ResultTree) // back to Verifying
	}
	res, err := r.wt.Merge(ctx, snap, t.Spec.BaseRef, commitMsg(t))
	if errors.Is(err, worktree.ErrBaseMoved) {
		return Result{Requeue: true}, nil // someone else landed; re-preflight
	}
	if err != nil {
		return Result{}, err
	}
	r.commit(ctx, t, func(s *api.Status) { s.SetCondition("Integrated", true, "FastForward", res.NewHead, "worktree") })
	return Result{}, r.teardownWithProcs(ctx, t) // keeps snapshots per GC policy
}
```

### 4.8 Sequence: one task, one failed round, then success

```
Controller   Reconciler        Worktree           Agent(proc grp)      Tools/Gate        Supervisor
    │ Get(T1)      │                 │                    │                   │                │
    │─────────────►│ Create(T1,1,b)  │                    │                   │                │
    │              │────────────────►│ lease row→wt add→hooks                 │                │
    │              │◄──── Ready ─────│                    │                   │                │
    │              │ WarmBaseline(b) ─────────────────────────────────────────►│ (cached by tree)│
    │              │ startAgent(r0) ─────────────────────►│ spawn, Setpgid    │                │
    │◄─ return ────│                 │                    │ …stream events…   │                │
    │              │                 │                    │ exit 0            │                │
    │ Add(T1) ◄────────────────────── completion handler ─┘                   │                │
    │─────────────►│ Snapshot(a1-r0) │                    │                   │                │
    │              │────────────────►│ temp-index commit → refs/spine/snap/T1/1                │
    │              │ startGate(tree) ─────────────────────────────────────────►│ build ✓ test ✗ │
    │ Add(T1) ◄─────────────────────────────────────────────────── verdict ───┘ (2 new fails) │
    │─────────────►│ decide()        │                    │                   │                │
    │              │──────────────────────────────── Observation{Legal:[Patch,Run,…]} ─────────►│
    │              │◄──────────────────────────────────────────── {PatchWorker, focus:[fp1]} ──│
    │              │ startAgent(r1, feedback=new diags) ──►│ …                │                │
    │   … same loop: Snapshot(a1-r1) → gate exit 0 → integrate …              │                │
    │              │ Preflight ─────►│ merge-tree --write-tree (no side effects)                │
    │              │ Merge ─────────►│ commit-tree + update-ref main NEW OLD (CAS)              │
    │              │ Teardown ──────►│ worktree remove; branch -D; keep snap refs               │
```

---

## 5. Deterministic tool registry: the gate engine

### 5.1 Built-in tools (v2.0)

| Name | Class | What it returns |
|---|---|---|
| `diff.stats` | Pure | files/LOC changed vs base, binary files, renames |
| `diff.scope` | Pure | exit 1 if any changed path is outside `Spec.Scope`; diagnostics list the paths |
| `diff.applycheck` | Pure | validates a worker-produced patch file (`git apply --check` on a temp index) |
| `git.preflight` | Pure | `merge-tree --write-tree` result + conflict paths |
| `ts.symbols` | Pure | per-file exported symbols (name, kind, signature hash) via tree-sitter queries |
| `ts.api_contract` | Pure | exit 1 when exported symbols were removed or had their signature changed vs base and are not on an allow-list; also detects unresolved new imports |
| `ts.parse` | Pure | exit 1 on any `ERROR`/`MISSING` node in changed files (catches truncated LLM edits fast, ~ms) |
| `cmd.run` | Hermetic | generic: argv + timeout + an output parser (`gotest-json`, `pytest-junit`, `eslint-json`, `tsc`, `ruff-json`, `regex`) → Diagnostics |
| `go.build` / `go.vet` / `go.test` | Hermetic | specializations of `cmd.run` with `-json` parsing |
| `spec.validate` | Pure | JSON-schema + dependency-cycle check for planner output |

Projects declare gates in `spine-gate.yaml`, which keeps v1's format and extends it:

```yaml
gates:
  parse:      { tool: ts.parse, policy: absolute }
  scope:      { tool: diff.scope, policy: absolute }
  build:      { tool: cmd.run, args: { argv: [go, build, ./...], parser: go-build }, timeout: 5m }
  unit:       { tool: cmd.run, args: { argv: [go, test, -json, ./...], parser: gotest-json }, timeout: 10m, policy: baseline }
  api:        { tool: ts.api_contract, args: { languages: [go, python] }, policy: absolute }
order: [parse, scope, build, unit, api]   # cheap → expensive; first hard failure short-circuits
```

### 5.2 Tree-sitter specifics

* Bindings: use `github.com/tree-sitter/go-tree-sitter` (the official org bindings). `smacker/go-tree-sitter` is fine but less actively maintained. Both need **CGO** (§8.1). Grammars are vendored and compiled in, so there is no runtime download.
* Symbol extraction is driven by **per-language `.scm` query files** embedded with `embed.FS` (`@definition.function`, `@definition.class`, `@export`). Adding a language means adding a grammar plus one query file, with no Go code.
* `SymbolDelta = diff(symbols(baseTree), symbols(snapTree))` restricted to changed files. That keeps it O(changed files), and the base side is cached across all tasks sharing a base commit.
* For Go targets, `go/parser` + `go/types` (stdlib, no CGO) gives *type-checked* API diffs. It is the preferred implementation for Go, and tree-sitter is the polyglot fallback.

### 5.3 Execution hardening (`internal/exec`)

* `exec.CommandContext` with `SysProcAttr{Setpgid: true}`. The context cancel func kills `-pgid`, not just the leader.
* stdout/stderr go through a ring buffer with a hard cap (default 4 MiB). The full log is streamed to the artifact store by digest, and `Result.Truncated` is set.
* Env is an explicit allow-list, plus `GIT_*` scrubbing so a tool cannot redirect `GIT_DIR`.
* `HOME` and `TMPDIR` point to a per-run directory inside the lease path (and are torn down with it).
* Optional sandbox: Linux `unshare`/landlock or `bwrap` for workers. This is off by default and configured per adapter.

### 5.4 Baseline-relative verdicts

```go
func verdict(base, cur tools.Result, p api.GatePolicy) api.GateVerdict {
	if p == api.Absolute {
		return api.GateVerdict{ExitCode: cur.ExitCode, New: cur.Diagnostics}
	}
	known := fingerprintSet(base.Diagnostics)
	var fresh []tools.Diagnostic
	for _, d := range cur.Diagnostics {
		if d.Severity == tools.Error && !known[d.Fingerprint] {
			fresh = append(fresh, d)
		}
	}
	code := 0
	if len(fresh) > 0 || (cur.ExitCode != 0 && len(cur.Diagnostics) == 0) {
		code = 1 // unparseable failure is never excused by the baseline
	}
	return api.GateVerdict{ExitCode: code, New: fresh, Fixed: diffSet(known, cur)}
}
```

This keeps the hard rule ("cannot complete unless the gate returns 0") and fixes v1's "red on pre-existing debt" deadlock without disabling gates. `Fixed` feeds the score, so the ratchet rewards paying down debt.

---

## 6. Fault tolerance

### 6.1 Lease-before-effect

Every effect follows the same three steps: write intent → perform → write completion. Crash recovery replays intents.

| Effect | Intent row | Recovery if crash after intent |
|---|---|---|
| worktree create | `leases(state=Requested)` | path exists & registered → finish hooks; else `git worktree prune` + retry or mark Released |
| agent spawn | `procs(task, pgid, start_time)` | pgid alive & start_time matches `/proc/<pid>/stat` → kill group; dead → mark exited `Crashed` |
| gate run | `ops(kind=gate, tree)` | re-run (tools are idempotent; cache hits make it cheap) |
| merge | `ops(kind=merge, expected_old, new)` | read base ref: == new → completed; == old → retry; else ErrBaseMoved |
| teardown | `leases(state=Draining)` | re-run teardown (idempotent) |

`recoverAll` runs before the controller accepts work. Process recovery comes first, then worktrees, then ops, because a live orphan can still be writing into a worktree.

### 6.2 Git lock discipline

Worktrees share `$GIT_COMMON_DIR` (objects, refs, `packed-refs`, `worktrees/`). Rules:

1. **Per-worktree mutex** around every command that touches that worktree's index (`Snapshot`, `Rollback`, mutating tools). Snapshots never use the worktree's index anyway (below), but agents do, so we never run our index commands while an agent is live on the worktree.
2. **Per-repo ref mutex** (a Go `sync.Mutex` plus an `flock` on `$GIT_COMMON_DIR/spine/refs.lock` for multi-process safety) around `worktree add/remove/prune`, `branch -D`, `update-ref` on shared refs, and `pack-refs`. These are short (ms), so serializing them costs nothing measurable even at 32 concurrent tasks.
3. **Stale lock recovery** happens only in `Recover`, and only for `*.lock` files under paths Spine leases, whose mtime is older than this process's boot time, and where no live process from our `procs` table has a cwd under that path. The primary worktree's `index.lock` is **never** removed. We report it instead.
4. `git worktree add --lock --reason "spine:<task>"` stops an operator's `git worktree prune` from deleting a live lease's metadata. Teardown unlocks before it removes.
5. Set `gc.auto=0` for Spine's own git invocations (`-c gc.auto=0`). A scheduled `spine gc` runs `git gc` / `pack-refs` under the repo mutex while no merges are in flight.

**Snapshot without touching the index** (temp index file, plumbing only):

```go
func (m *manager) Snapshot(ctx context.Context, l *Lease, label string) (*Snapshot, error) {
	idx := filepath.Join(l.Path, ".git-spine-snap.idx") // inside the lease, deleted after
	defer os.Remove(idx)
	env := []string{"GIT_INDEX_FILE=" + idx}
	m.git(ctx, l.Path, env, "read-tree", "HEAD")
	m.git(ctx, l.Path, env, "add", "-A", "--", ".", ":(exclude).git-spine-*")
	tree := m.gitOut(ctx, l.Path, env, "write-tree")
	commit := m.gitOut(ctx, l.Path, nil, "commit-tree", tree, "-p", m.headOf(l), "-m", "spine snapshot "+label)
	ref := fmt.Sprintf("refs/spine/snap/%s/%d", l.TaskID, m.nextSeq(l))
	m.withRefLock(func() { m.git(ctx, l.Path, nil, "update-ref", ref, commit, zeroOID) }) // create-only CAS
	return &Snapshot{Ref: ref, Commit: commit, Tree: tree, Label: label}, nil
}
```

(Each `m.git` returns an error. The checks are omitted here for brevity, and the real code aborts on the first error.)

**Merge without any worktree:**

```sh
old=$(git rev-parse refs/heads/main)
tree=$(git merge-tree --write-tree --name-only "$old" "$snap_commit")   # exit 1 + conflict list if unclean
new=$(git commit-tree "$tree" -p "$old" -p "$snap_commit" -m "...")     # or squash: -p "$old" only
git update-ref refs/heads/main "$new" "$old"                             # CAS; fails if main moved
```

This needs git ≥ 2.38 (merge-tree `--write-tree`), and `spine doctor` checks for it. The primary checkout's working tree is not updated. That matches v1's "primary stays on main" rule: the operator's checkout simply sees `main` advance, as with a `git fetch`. If the primary checkout has `main` checked out with a clean tree, we optionally run `git -C <primary> merge --ff-only` afterwards (the one sanctioned primary-checkout op, unchanged from AGENTS.md).

### 6.3 Process lifecycle

```go
func (r *runner) Run(ctx context.Context, req RunRequest, ev chan<- Event) (Outcome, error) {
	defer close(ev)
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir, cmd.Env = req.Worktree, buildEnv(req.Env)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL} // Linux
	if err := cmd.Start(); err != nil {
		return Outcome{Reason: Crashed}, err
	}
	pgid := cmd.Process.Pid
	r.store.RecordProc(req.Task, pgid, procStartTime(pgid)) // lease

	runCtx, cancel := context.WithTimeout(ctx, req.Timeout)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	go r.pump(stdout, ev, req) // parse JSONL; enforce MaxTokens → cancel() on overflow

	var reason OutcomeReason
	select {
	case err := <-done:
		reason = classify(err)
	case <-runCtx.Done():
		syscall.Kill(-pgid, syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			syscall.Kill(-pgid, syscall.SIGKILL)
			<-done
		}
		reason = reasonFrom(runCtx.Err(), r.budgetTripped(req.Task))
	}
	// Grandchildren that called setsid() escape the group: sweep by cgroup when
	// available (systemd-run --scope / cgroup v2 per run), else by /proc cwd under lease path.
	r.sweepEscapees(req.Worktree)
	r.store.ReleaseProc(req.Task, pgid)
	return r.outcome(cmd, reason), nil
}
```

* With **cgroup v2**, each agent run gets its own cgroup (`spine.slice/task-<id>-<attempt>.scope`), with memory and pid limits. `cgroup.kill` is an atomic, escape-proof group kill. That is the preferred path on Linux, and process groups are the portable fallback (macOS dev).
* The supervisor's own SIGTERM cancels the root context. Agents get SIGTERM, the reconcile workers drain, and leases stay in a recoverable state. SIGKILL of the supervisor is covered by `Pdeathsig` (Linux) plus `Recover`.

### 6.4 Stall detection

Each in-flight op has a heartbeat deadline taken from `EvHeartbeat`/`EvUsage` events. No events for `SPINE_STALL_TIMEOUT` → cancel with `Reason=Stalled`. That is an observation for the supervisor, like any other. Following v1's lesson, a stall is an infrastructure event, so the snapshot is preserved and the next action defaults to `PatchWorker` from the same tree, not a rollback.

---

## 7. Storage

SQLite in WAL mode. There is one writer goroutine (a channel-fed batcher) and many readers. That avoids `SQLITE_BUSY` under concurrency and keeps writes ordered for the event log.

```sql
CREATE TABLE tasks (
  id TEXT PRIMARY KEY, parent_id TEXT, generation INTEGER NOT NULL,
  spec_json TEXT NOT NULL, status_json TEXT NOT NULL,
  phase TEXT NOT NULL, resource_version INTEGER NOT NULL,
  created_at INTEGER, updated_at INTEGER
);
CREATE INDEX tasks_phase ON tasks(phase);
CREATE TABLE task_deps   (task_id TEXT, depends_on TEXT, PRIMARY KEY (task_id, depends_on));
CREATE TABLE leases      (task_id TEXT, attempt INTEGER, path TEXT UNIQUE, branch TEXT,
                          base_oid TEXT, state TEXT, owner_pid INTEGER, PRIMARY KEY (task_id, attempt));
CREATE TABLE procs       (task_id TEXT, pgid INTEGER, start_time INTEGER, cgroup TEXT, state TEXT);
CREATE TABLE ops         (id INTEGER PRIMARY KEY, task_id TEXT, kind TEXT, intent_json TEXT, state TEXT);
CREATE TABLE snapshots   (task_id TEXT, seq INTEGER, ref TEXT, commit_oid TEXT, tree_oid TEXT,
                          label TEXT, score REAL, PRIMARY KEY (task_id, seq));
CREATE TABLE decisions   (task_id TEXT, at INTEGER, observation_digest TEXT, decision_json TEXT, source TEXT);
CREATE TABLE events      (seq INTEGER PRIMARY KEY, task_id TEXT, kind TEXT, at INTEGER, payload BLOB);
CREATE TABLE tool_cache  (key BLOB PRIMARY KEY, result_json BLOB, created_at INTEGER);
CREATE TABLE artifacts   (digest TEXT PRIMARY KEY, path TEXT, size INTEGER); -- blobs on disk, by sha256
```

* Status writes use CAS on `resource_version` (`UPDATE … WHERE id=? AND resource_version=?`) plus condition ownership checks.
* Location: `$GIT_COMMON_DIR/spine/spine.db`, shared by all worktrees and invisible to `git status`.
* **Git-backed export (optional):** `spine export` writes `refs/spine/tasks` as a commit of one Markdown + YAML-frontmatter file per task. The ref can be pushed and pulled, which gives the Beads-style "durable, git-native, survives context loss" property for humans and other machines. SQLite stays the source of truth, because a ref-per-write design serializes every status change through git, and that is the wrong cost for a control loop that writes several times per second.
* Vector search is not needed for v2.0. If it is added later (e.g. experience retrieval), use `sqlite-vec` as a loadable extension, which requires a driver that supports extensions (§8.1).

---

## 8. Feasibility notes

### 8.1 CGO decision

Tree-sitter forces CGO, so the "single static binary" goal needs an explicit choice:

| Option | Static binary | Tree-sitter | SQLite | Verdict |
|---|---|---|---|---|
| A. CGO on, `zig cc` / musl static link | yes (per target triple) | native, fast | `mattn/go-sqlite3` (+ `sqlite-vec` extension) | **Recommended.** One cross-compile toolchain in CI; goreleaser supports it. |
| B. CGO off; tree-sitter grammars as WASM via `wazero` | yes, trivially | ~3–10× slower parse, still ms-scale per file | `ncruces/go-sqlite3` (wazero) or `modernc.org/sqlite` | Viable fallback if cross-compiling CGO hurts. Same wazero runtime for both. |
| C. CGO off, no tree-sitter (go/ast only) | yes | Go-only | modernc | Rejected: Spine targets polyglot repos (v1 already serves a Laravel/PHP project). |

Put both SQLite drivers behind `database/sql`, and put tree-sitter behind an `ast.Parser` interface, so we can switch from A to B with build tags.

### 8.2 Concurrency budget

* The controller has `workers = 2×GOMAXPROCS` reconcile goroutines. Reconcile is cheap because long work is async.
* The scheduler owns a `semaphore.Weighted` of **worktree slots** (default 8) and separate weights for expensive gates (e.g. `go test` takes `ceil(cpus/2)`). Agents are mostly I/O-bound waiting on the LLM. Gates are CPU-bound, so they get their own pool and do not starve each other.
* Expected overhead per task outside agent/test time is under 200 ms (worktree add ≈ 50–150 ms on mid-size repos, snapshot ≈ 20 ms, merge-tree ≈ 10 ms). Go's scheduler is not the bottleneck. Disk is (N worktrees × checkout size). Mitigations: `git worktree add --no-checkout` + sparse checkout driven by `Spec.Scope` for large monorepos; `core.untrackedCache`; `index.skipHash`.

### 8.3 Risks

| Risk | Mitigation |
|---|---|
| Supervisor LLM still loops (Patch → regress → Patch) | Budget is a closed action set (§4.6); `RuleSupervisor` progress checks override when the fingerprint set is unchanged for two rounds. |
| Workers edit outside scope or tamper with tests to go green | `diff.scope` is an absolute gate; protected-path globs (tests, CI config, `spine-gate.yaml`) need `Spec.AllowTestEdits`; `ts.api_contract` on test files flags deleted test functions. |
| Baseline flakiness makes "new failures" noisy | Run the baseline twice. A fingerprint that flips between runs goes into a per-repo `flaky` set, which counts as known. A gate fails only on deterministic new failures; flaky ones are reported as warnings. |
| CLI agent output formats drift | Adapters parse defensively: unknown lines → `EvStderr`, and **the gate never trusts agent self-reports**. Exit code + tree diff are the only inputs that matter. |
| Disk exhaustion from many worktrees / snapshot refs | Slot semaphore; GC policy (`snap` refs of Integrated tasks pruned after N days; Escalated kept until resolved); `spine gc`. |

---

## 9. Gas City / Beads vs. in-house

**Scope of the evaluation.** Beads (`bd`) is a Go issue tracker built for agents: dependency-aware issues, a "ready work" query, and storage that travels with the git repo (originally JSONL in-repo with a SQLite cache; recent versions moved toward a Dolt-backed store). Gas Town and its later refactor work (referred to here as "Gas City") are the multi-agent orchestrator built on Beads, with worktree-per-agent "polecats", tmux-hosted sessions, and supervisor agents. Their APIs have changed quickly through 2025–26. **Verify current versions and licenses before acting on the specifics below.** The recommendation depends on the *architectural fit*, not on a particular release.

| Criterion | Import Beads (task store) | Import Gas City (orchestration / worktrees) | In-house minimal |
|---|---|---|---|
| Fit with reconciler model | Partial. Beads models *issues*. We need spec/status split, condition ownership, CAS resource versions, leases and ops intents. Those would be bolted on through custom fields. | Poor. Its control model is conversational supervision through agent sessions, which is exactly what this RFC removes. | Exact. |
| Durability properties | Strong: git-native, merge-friendly, survives context loss. | Inherits Beads. | Gets the same property through `spine export` to `refs/spine/tasks` (§7). |
| Fault tolerance needs (lock discipline, process leases, crash replay) | Not provided (out of scope for a tracker). | Partial; built around long-lived tmux sessions and not hardened for our crash semantics. | We must write ~1.5–2.5k LOC (worktree + locks + recovery), which is the heart of the system and should be owned. |
| Dependency / API stability | Fast-moving; storage backend changed at least once. A dep in the hot write path. | Fast-moving, opinionated runtime layout (towns/rigs). | Stable; tied only to git plumbing (stable for 15+ years) and SQLite. |
| Binary/deps | Pulls a Dolt dependency in newer versions (large; contradicts lean binary). | Same, plus tmux at runtime. | stdlib + sqlite driver + tree-sitter. |
| Interop value | High: humans and other agents already use `bd ready`. | Low for us. | — |

**Recommendation: build in-house, and interoperate with Beads at the edges.**

1. **Task persistence: in-house** (§7). The control loop needs CAS status writes, field ownership and intent logs. A tracker's data model is the wrong primitive, and putting a fast-moving external store in the hot path is a fault-tolerance liability.
2. **Worktree management: in-house** (§3.2, §6.2). This is where the "no dangling lockfiles / orphaned processes / corrupt indices" criterion is won or lost, so it must be ours to test and harden. It is small: about 600 LOC of plumbing plus about 800 LOC of recovery and locks.
3. **Beads interop: optional `internal/bridge/beads`** (read-mostly). Import `bd ready` issues as Spine `Spec`s (`spine import --from beads`), and write status back as Beads comments/state when a task reaches a terminal phase. This uses the `bd` CLI's JSON output, so we do not take a library dependency. It is a v2.1 feature behind a flag.
4. **Borrow ideas, not code:** Beads' hash-based short IDs (collision-resistant across machines), its "ready = all deps closed" query semantics (the same as our `DepsIntegrated`), and Gas Town's practice of committing agent work-in-progress frequently (our snapshot refs).

---

## 10. Migration plan

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| M0 (1 wk) | `cmd/spine doctor`, `internal/exec`, `internal/worktree` + chaos tests | 1,000 randomized `kill -9` points during create/snapshot/merge/teardown → `Recover` leaves `git fsck` clean, zero leftover `*.lock`, zero orphan pids |
| M1 (1–2 wk) | `tools` registry, `cmd.run` parsers, `diff.*`, `ts.parse`/`ts.symbols`, cache, baseline verdicts | Gate verdicts on Spine v1's own repo match `pytest` ground truth; cache hit on identical trees |
| M2 (2 wk) | storage, controller, reconcile, `RuleSupervisor`, one adapter (headless Claude Code / codex stream-json) | 8 parallel tasks on a fixture repo, all integrated via CAS, no index races; zero LLM supervisor yet |
| M3 (1–2 wk) | LLM supervisor (structured output), planner task + `spec.validate`, Replan | A/B vs RuleSupervisor on a fixed benchmark of ~30 historical v1 work items: ≥ equal pass rate, lower tokens/task |
| M4 (1 wk) | `spine ui` (embedded, SSE from event bus), v1 config import, Beads bridge (flag) | Feature parity for CLI verbs: `run`, `status`, `list`, `resume` (= approve Escalated), `restart` (= new attempt), `worker` (= controller) |

During M2–M4, v1 stays the production path. v2 runs in **shadow mode** on the same work items in separate worktrees, with merge disabled, so we can compare gate verdicts and cost per item before cutover.

### CLI verb mapping

| v1 | v2 |
|---|---|
| `spine run "…" --type task` | `spine run "…" --template task` (creates Spec; planner task if `--plan`) |
| `spine status/list` | same; reads `tasks` + `conditions` |
| `spine resume <id>` | `spine approve <id>` / `spine resume <id>` → clears `Escalated`, re-enqueues |
| `spine restart <id>` | `spine retry <id> [--from-snapshot N]` → new attempt or Rollback |
| `spine worker` | `spine controller` (alias `worker`) |
| `spine ui` | `spine ui` (embedded web, no Streamlit) |

---

## 11. Open questions

1. **Squash vs merge commits** on integration. The proposal is squash by default (`commit-tree -p old`), with snapshot refs keeping the full history for forensics.
2. **Supervisor model placement.** It needs a small context and structured output, so a local model is plausible (in keeping with v1's local-first goal). That needs an M3 benchmark.
3. **Cross-task interface changes.** Should `Replan` be allowed to *edit sibling* Specs (e.g. "T2 must wait for the new interface from T1a"), or only its own children? The proposal is children only, plus adding `DependsOn` edges to siblings, which is monotonic and safe.
4. **Windows support.** Process groups → Job Objects; `flock` → `LockFileEx`. Out of scope for v2.0 and kept compilable behind build tags.
