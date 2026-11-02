# RFC 0001 — Spine v2: Go Runtime and Reconciler Control Plane

| Field | Value |
|---|---|
| Status | Draft, for review |
| Date | 2026-11-02 |
| Scope | **Greenfield rebuild.** No code, data, config or CLI compatibility with the Python prototype is promised; there is no migration path. |
| Prior art studied | Spine v1 prototype (this repo, Python/LangGraph): operational lessons only. Gas City `gastownhall/gascity` @ `4ebacb1` and Beads `gastownhall/beads` (module `github.com/steveyegge/beads`) @ `ce907a9`, both read at source on 2026-11-02 (§9). |

---

## 0. Summary

Spine v2 is one Go binary. It replaces the fixed phase DAG with a **level-triggered reconciler**, the same pattern Kubernetes controllers use. Each task has a declared **Spec** (intent plus acceptance criteria written as gate definitions) and an **Observed Status** that only deterministic tools can write. A controller loop runs one bounded step at a time until the two match.

An LLM **supervisor** picks the next step from a closed, typed set of actions (`RunWorker`, `PatchWorker`, `Rollback`, `Replan`, `Escalate`, …) based on a typed observation. It never reads a conversation, and it cannot write the `Verified` condition. Only the gate engine can set that condition, and only when the gate exits `0`.

The rest of the design follows from four rules:

1. **The DAG is data, not control flow.** Task dependencies are rows in a table. The order of work inside a task is decided again on every reconcile, so an unexpected test break means *observe → decide again*. It does not need a pre-wired fallback edge.
2. **Git plumbing, not porcelain, for anything shared.** Integration uses `git merge-tree --write-tree` and `git update-ref` compare-and-swap. No shared index is touched and no branch is ever checked out in the primary tree.
3. **Build vs. reuse (§9, from a source-level read of both projects).** Build the control plane and storage in-house, because Gas City reconciles *agent sessions* rather than task acceptance, and by design keeps judgment out of Go. Vendor Gas City's MIT-licensed worktree, git and process-group primitives (about 4.1k LOC) as a one-time fork and harden them. Use Beads only through an optional `bd` CLI bridge.
4. **Every external effect is a lease.** Worktrees, child process groups and git locks are all recorded in SQLite before they are created, and they are reconciled on startup. A `kill -9` of the supervisor at any point leaves state that the next boot can repair.

---

## 1. Prior art: failure modes the design must rule out

v2 is a rebuild from scratch, and nothing from the Python prototype is ported. The prototype's trace audits are still the best record of how this kind of system fails in practice. Each lesson below becomes a structural rule in the new design, not an inherited mechanism. Gas City's lessons are covered in §9.

| Observed failure (source) | Design response |
|---|---|
| Gates must pass on a clean `main`, or every landing is blocked on pre-existing debt (`spine-gate.yaml` header) | **Baseline-relative gates**: every gate runs once on the base tree; a task gate passes when it adds *no new failures* (§5.4). The gate tool still returns a strict exit code. |
| Rolling back a `needs_review` / `stalled` run destroyed the only reviewable copy of good work (`sandbox.py` `_PRESERVE_STATUSES`) | **Snapshots are refs** (`refs/spine/snap/<task>/<n>`). Rollback never deletes work. Teardown deletes the *worktree*, never the snapshot refs, until GC policy says so. |
| Verify ratchet: on-disk state should be the best-scoring cycle, not the last edit | `Snapshot` after every gate run, tagged with its score; `Rollback(best)` is a first-class action. |
| `git reset --hard` / `clean` on the master tree destroyed operator state (`orchestrator.py` `rollback_workspace`) | The primary checkout is **never** a worktree target. The manager refuses any path equal to the main worktree. |
| Fresh worktrees lack untracked env (`.env`, keys) (`sandbox_setup` hooks) | `Provisioner` hooks run inside the new worktree before the lease turns `Ready`; a hook failure tears down before any tokens are spent. |
| Spine's own `.spine/` output must not count as a dirty tree | Spine state lives under `$GIT_COMMON_DIR/spine/`, outside every working tree. |
| A pre-wired phase DAG deadlocked or needed combinatorial fallback edges whenever an interface shifted mid-task | No phase graph at all. Task *dependencies* are a DAG of data (`internal/scheduler`), and order *within* a task is decided again on every reconcile (§4). |
| An LLM critic phase graded LLM output and was gamed or looped | No LLM can write a completion condition (§4.3). The only critic is a gate exit code. |

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
	// AttemptID is a random 128-bit token, also written to a provenance file in the
	// worktree's private git dir (fsynced temp file + rename). Teardown acts only
	// when they match, so a late teardown can never remove a re-provisioned worktree.
	AttemptID string
	State     LeaseState
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

	// Teardown kills nothing (processes belong to agent.Runner). It fails closed,
	// in the order Gas City's worktree.Cleanup uses (internal/worktree/worktree.go:409):
	//  - a process lease on the path is still live → ErrLeaseHeld
	//  - the provenance AttemptID does not match l → ErrStaleAttempt
	//  - HEAD has commits not reachable from a branch, tag or refs/spine/snap/* → ErrUnreachable
	// Spine snapshots before every teardown, so the reachability check is a guard
	// that should never trip, not a common outcome. Removal uses `git worktree
	// remove` (no --force) after the snapshot. The branch is deleted, the snapshot
	// refs are kept, and the lease is marked Released. Idempotent.
	Teardown(ctx context.Context, l *Lease) error

	// Recover reconciles lease rows against `git worktree list --porcelain` and the
	// filesystem at startup. It finishes or reverses half-done Create and Teardown
	// operations and clears stale locks (§6.2).
	Recover(ctx context.Context) (RecoveryReport, error)
}

var (
	ErrStaleAttempt = errors.New("worktree: provenance attempt id mismatch")
	ErrUnreachable  = errors.New("worktree: HEAD has commits reachable from no ref")
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

### 4.1 Spec templates (no fixed work types)

| Template | Spec |
|---|---|
| `change` | `Acceptance = [ts.parse, diff.scope, build, unit tests (baseline-relative)]` |
| `interface-change` | adds `ts.api_contract (Absolute)` and `Escalation.RequireApprovalBeforeMerge` |
| `reviewed` | `Escalation.RequireApprovalBeforeWork` once Replan has produced the task split |
| `plan` | A **planner task**. Its output is a set of child Specs, validated by a schema tool (`spec.validate`), and the children are inserted with `DependsOn`. Planning is just another reconciled task whose gate is structural. |

**Human approval is additive, never a substitute for the gate.** An approval can *hold* a Verified task, but it cannot turn a failing task into a passing one. Gas City's convergence loops do allow that: `manual` and `hybrid` gates park in `waiting_manual`, and `ApproveHandler` (`internal/convergence/manual.go:19`) terminates the loop as `approved` without running a gate. Spine's only override is for a human to *edit the Spec*, for example to drop or relax a gate. That creates a new generation, which is audited and re-reconciled, so the tree still has to pass whatever gates remain.

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

Projects declare gates in `.spine/gates.yaml`. This is a new format; nothing from the prototype is read:

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
* Env is an explicit allow-list, plus `GIT_*` scrubbing so a tool cannot redirect `GIT_DIR`. For Spine's own git calls, use the scrub list and hermetic config from Gas City `internal/git/git.go` (`SanitizedEnv` L439 and `HermeticEnv` L451: `GIT_CONFIG_NOSYSTEM=1`, `GIT_CONFIG_GLOBAL=/dev/null`), plus its leading-dash argument validation (`worktree_ops.go:14`).
* Git subprocesses on a deadline are killed **as a process group**. Gas City's `runCtx` (`git.go:529`) kills only the git process and relies on `WaitDelay`, so a descendant such as a hook, credential helper or `ssh` can outlive it.
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

0. **Per-worktree path lock** (adapted from Gas City `internal/worktree/lock.go`). This is an `flock(LOCK_EX)` on `$GIT_COMMON_DIR/spine/wt-locks/<sha256(canonical path)>.lock`, held across every check-then-act on that path. The path is canonicalized through symlinks, including ancestors that don't exist yet, and lock files are never unlinked, because unlinking would let one process delete a file that another is blocked on. The kernel releases the lock when its holder dies, so a crashed holder never leaves a stale lock. Unlike Gas City's lock, which blocks with no timeout (`lock_unix.go:109`), Spine polls `LOCK_NB` with a deadline, so a hung holder turns into an infrastructure fault rather than a stalled controller.
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

This needs git ≥ 2.38 (merge-tree `--write-tree`), and `spine doctor` checks for it. The primary checkout's working tree is not updated. The primary checkout is never switched, staged or reset. The operator simply sees `main` advance, as after a `git fetch`. If the primary checkout has `main` checked out with a clean tree, Spine optionally runs `git -C <primary> merge --ff-only` afterwards. That is the only operation Spine ever performs on the primary checkout.

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

* Recovery checks PID liveness with start-time matching (`/proc/<pid>/stat` field 22), adapted from Gas City `internal/pidutil` (`AliveWithStartTime`, `pidutil.go:173`), so a recycled PID is never signalled.
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
* **Git-backed export (optional):** `spine export` writes `refs/spine/tasks` as a commit of one Markdown + YAML-frontmatter file per task. The ref can be pushed and pulled, This makes task state durable, git-native and able to survive context loss, for humans and for other machines. Beads itself now gets this through `bd dolt push/pull` against a ref on the git remote, and treats its `issues.jsonl` as "an export for viewers and interchange, not the source of truth" (Beads `README.md:137`). SQLite stays the source of truth, because a ref-per-write design serializes every status change through git, and that is the wrong cost for a control loop that writes several times per second.
* Vector search is not needed for v2.0. If it is added later (e.g. experience retrieval), use `sqlite-vec` as a loadable extension, which requires a driver that supports extensions (§8.1).

---

## 8. Feasibility notes

### 8.1 CGO decision

Tree-sitter forces CGO, so the "single static binary" goal needs an explicit choice:

| Option | Static binary | Tree-sitter | SQLite | Verdict |
|---|---|---|---|---|
| A. CGO on, `zig cc` / musl static link | yes (per target triple) | native, fast | `mattn/go-sqlite3` (+ `sqlite-vec` extension) | **Recommended.** One cross-compile toolchain in CI; goreleaser supports it. |
| B. CGO off; tree-sitter grammars as WASM via `wazero` | yes, trivially | ~3–10× slower parse, still ms-scale per file | `ncruces/go-sqlite3` (wazero) or `modernc.org/sqlite` | Viable fallback if cross-compiling CGO hurts. Same wazero runtime for both. |
| C. CGO off, no tree-sitter (go/ast only) | yes | Go-only | modernc | Rejected: target repos are polyglot (PHP/Laravel, Python, TS, Go are all in the expected workload). |

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

### 9.0 Method

Both projects were cloned at HEAD on 2026-11-02 and read at source:

| Repo | Commit | Module / license |
|---|---|---|
| `gastownhall/gascity` | `4ebacb1` | `github.com/gastownhall/gascity`, Go 1.27.3, MIT |
| `gastownhall/beads` | `ce907a9` | `github.com/steveyegge/beads` (Gas City pins `v1.3.2`), Go 1.27.3, MIT |

All `path:line` references below are at those commits. Both projects change quickly. Gas City's own architecture docs already lag the code: `engdocs/architecture/reconciler-v2.md` describes an `internal/workqueue` package that no longer exists, and `cmd/gc/reconcile_keys_test.go:60` is a test that *bans* reintroducing it. So the claims below cite code, not docs. Both clones were shallow, so history-based claims were not checked.

### 9.1 What Gas City actually is

Gas City is "an orchestration-builder SDK for multi-agent systems" that "extracts the reusable infrastructure from Gas Town" (`README.md`). Its features include "a controller/supervisor loop that reconciles desired state to running state". That is close to the pattern this RFC proposes, so the comparison deserves care.

**What it reconciles: sessions, not work.** The controller compares the *running agent processes* with the *desired agent set*: `city.toml` agents, with pool sizes taken from a user `scale_check` command clamped to `[min,max]`. The legacy pass has four outcomes per session: not alive → start, healthy → skip, orphan → drain, config drift → drain and restart (`cmd/gc/session_reconciler.go:1896`, `engdocs/architecture/health-patrol.md:130-150`). The v2 core is a single-goroutine planner that "solely owns the reconciler's mutable state, so none of that state takes a lock" (`cmd/gc/reconcile_planner.go:19-22`). It runs a resync backstop and fences store writes with revision-conditional updates. This is a sound controller, but its desired state is "these agents are alive", **not** "this task's acceptance criteria hold". Work reaches agents through beads, and the model is Erlang/OTP: "agents die, hooks persist, fresh sessions resume the work" (`health-patrol.md:16-18`).

**Its explicit design rule is the opposite of ours.** `engdocs/architecture/nine-concepts.md:56-57`, "Primitive Test" condition 3: "*Judgment out of Go* — does Go handle transport only, with no judgment calls? If Go makes decisions, it's a violation." Roles (mayor, deacon, witness, refinery, polecat) exist only in pack TOML and prompts. That is a coherent position for a *general* orchestration SDK. Spine's thesis is the inverse: the decisions that matter most (is this done? may this merge?) are made by deterministic Go code over typed tool results, and the LLM handles only the residual choice among legal actions.

**Where Gas City has deterministic gates (`internal/convergence`).** A convergence loop repeatedly runs a formula iteration ("wisp") and then evaluates a gate:

```go
// internal/convergence/gate.go:17
type GateConfig struct {
	Mode          string        // "manual", "condition", "hybrid"
	Condition     string        // path to gate condition script
	Timeout       time.Duration // default 5m
	TimeoutAction string        // "iterate", "retry", "manual", "terminate"
}
```

`RunCondition` (`condition.go:457`) runs the script with a sanitized env, and exit 0 means pass. Exit 75 (EX_TEMPFAIL) and some stderr markers are reclassified as *infrastructure* failures that do not use up an iteration (`gate_infra.go:115`, with a separate budget of `maxCheckInfraRetries = 20` in `internal/dispatch/ralph.go:46`). Spine keeps both ideas: the `Result.ExitCode` vs `error` split (§3.3) and the separate infrastructure retry budget.

It differs from this RFC in three ways:
1. **Gates can be bypassed by humans.** `manual` mode, `hybrid` mode without a condition, and `TimeoutAction=manual` all park in `waiting_manual`, where `ApproveHandler` (`manual.go:19`) terminates the loop as `approved` without running a gate. §4.1 forbids this.
2. **Gates are opaque scripts.** A gate returns pass/fail, stdout/stderr and an exit code, and "no Go code inspects what the gate was checking" (`gate_infra.go`). Spine's tools return typed `Diagnostic`s with fingerprints. Those are what make baseline-relative verdicts, regression detection and targeted feedback possible (§5.4, §4.6).
3. **The budget is iteration count only** (`handler.go` ~L365, `TerminalNoConvergence`). There is no notion of rollback to a best snapshot, and no progress check over failure fingerprints.

**Agent runtime: mostly terminal-shaped.** The provider interface (`internal/runtime/runtime.go:471`) is `Start` / `Nudge` / `Peek` / `SendKeys` / `Attach`. `tmux` is the default, and it is required even when other backends are used (`README.md`). Under tmux, nudges are `send-keys` and observation is `capture-pane` scraping, with heuristics for startup dialogs (`tmux/adapter.go:819`, `tmux/interaction.go:935`). The `k8s` provider runs tmux inside a pod (`k8s/provider.go:171`). The headless `subprocess` provider sends stdout/stderr to `/dev/null`, and its `Peek` returns `""` (`subprocess/subprocess.go:197-204`). The exception is the **ACP provider**: JSON-RPC 2.0 over stdio, with structured transcripts (`runtime/acp/protocol.go:1-10`). There are also `internal/sessionlog/*_reader.go` parsers for each CLI's on-disk JSONL transcripts (claude, codex, gemini, opencode). Both are directly relevant prior art for `agent.Adapter` (§3.4).

**Integration (merge) is an agent role.** The "refinery" is an agent driven by the `mol-refinery-patrol` formula, which lives in the external `gascity-packs` repo and was **not read** for this review. The in-repo scripted example (`examples/lifecycle/.../lifecycle-refinery-merge.yaml:77-146`) runs `checkout -B base origin/base`, then `git merge --no-edit`, with `merge --abort` on conflict. It describes itself as "happy-path merge only", with no test gate in the script. Nothing in `internal/` or `cmd/` uses `merge-tree`, `update-ref` CAS, or `index.lock` detection or recovery. The archived worktree roadmap still lists "Merge-back flow" as unchecked (`engdocs/archive/backlogs/worktree-roadmap.md:87-101`).

**Worktrees, git and processes: high quality, and the best reuse candidate.**

* `internal/worktree` (1,212 non-test LOC) is a "transactional owner for agent workspace worktrees":
  - `Ensure` / `Verify` / `Cleanup` / `RollbackAttempt`, with an AttemptID recorded in a provenance file.
  - A per-path `flock` in the common git dir; lock files are never unlinked.
  - Fail-closed cleanup: coded errors for ambiguous registration, ownership, stale attempt, dirty tree, `unreachable_commits` and `unmerged_commits`. No `--force` anywhere, and branches are deleted with `-d` only.
  - Tests for concurrent `Ensure`, stale attempts, and squash-merged remote branches.
* `internal/git` (1,395 LOC): env sanitization, hermetic config, hardened clone, and argument-injection validation. Its timeout path kills the git process only, not its process group.
* `internal/processgroup` (245), `internal/execgrace` (406) and `internal/pidutil` (835):
  - Group SIGTERM, then SIGKILL; the group kill refuses pgid ≤ 1 and its own group.
  - CommandContext cancellation replaced with a *graceful* group signal, because an untrappable SIGKILL means "a setup script killed mid-flight can never restore state".
  - An idle/ceiling output monitor.
  - `AliveWithStartTime` PID-reuse protection, and subreaper detection.
* `internal/opjournal` (13.3k LOC) is an intent-before-effect lease journal in concept. In practice it is tied to conversations, sessions and beads (the lease lives in a bead's 255-rune assignee field). It describes itself as a shadow store that "decides and executes nothing", with "no production caller until OJ-12" (`doc.go:1-10`). It is **not** reusable.

**Importability.** `pkg/` exports only `pkg/eventexport`. Everything above is under `internal/`, and the controller is in `cmd/gc` (`package main`, about 2,000 files), so none of it can be imported. Reuse means vendoring or forking. Dependency weight: `k8s.io/client-go`, the Dolt / go-mysql-server stack (indirect), the OpenTelemetry SDK, cloud SDKs, and 205 indirect modules. Runtime dependencies are `tmux`, `git`, `jq`, `pgrep`, `lsof`, and either `bd` + `dolt` + `flock` for the default store or `GC_BEADS=file`. Release binaries build with `CGO_ENABLED=0` (`.goreleaser.yml:7`).

### 9.2 What Beads actually is

Beads is "a distributed graph issue tracker for AI agents, powered by Dolt". Its public Go API is real: `beads.Open` / `OpenFromConfig` / `OpenGated` / `OpenBestAvailable` return a `Storage` (`beads.go:412-472`), with capability interfaces reached by type assertion.

Strengths that line up with this RFC:

* **CAS exists.** `UpdateIssueOptions{ExpectedVersion, ExpectedAssignee, ExpectedStatus}` checks the preconditions and writes in one transaction, and returns `ErrVersionMismatch` and siblings on a mismatch (`internal/storage/storage.go:428-452, 690-741`). `issueops.MetadataCAS.CompareAndSetKey` gives per-key JSON-metadata compare-and-set (`issueops/metadatacas.go:19,148`).
* **Atomic claim.** `ClaimIssueInTx` runs `UPDATE … WHERE id=? AND row_lock=? AND status IN (…)`, and `ClaimReadyIssue` checks readiness and claims in one transaction. Claims have leases with heartbeats (`internal/storage/issueops/claim.go`).
* **Dependency DAG and ready work.** It has typed blocking edges (`blocks`, `parent-child`, `conditional-blocks`, `waits-for`) plus many non-blocking ones, and rejects cycles. Ready work is `status ∈ active ∧ ¬is_blocked ∧ defer_until ≤ now` (`issueops/ready_work.go:102`, `sqlbuild/ready.go:103-144`).

Mismatches:

* **`RowVersion` is not a whole-object resource version.** It is a random `row_lock` that changes only on status, assignee, `started_at` and generic-update writes, and "NOT on direct-UPDATE paths". Label, dependency and `is_blocked` changes leave it unchanged (`internal/types/types.go:76-99`, `storage.go:696-704`). Spine's §4.3 condition ownership needs CAS over the whole status document.
* **Concurrency model.**
  - Embedded mode (the default) is "single writer" (`README.md:132`), and "embedded transactions serialize" (`internal/storage/embeddeddolt/store.go:450-454`).
  - Multiple writers need an *external* `dolt sql-server` process (`internal/doltserver` runs `exec.LookPath("dolt")`).
  - Spine needs dozens of goroutines writing several times a second from one process, with no extra daemon.
* **Build.**
  - Embedded Dolt requires CGO. Without CGO, `beads_nocgo.go:84` returns "embedded Dolt requires CGO; use server mode".
  - Even with CGO, it needs the `gms_pure_go` tag to avoid ICU (Makefile:113-118).
  - The SQLite backend was removed "to consolidate storage on a single engine" (`backend_messages.go`), and the pluggable-backends proposal is marked "superseded" (`PROPOSAL-pluggable-storage-backends.md`).
  - Heavy dependencies: Dolt, go-mysql-server, vitess.
* **No write-throughput evidence.** `BENCHMARKS.md` reports 18 ms per update and about 19 ms per write sequentially on an M2 Pro, but that table appears to predate Dolt because it references SQLite bench files. There are no concurrent-writer numbers.

Gas City itself does not rely on Beads alone. Its `internal/beads` defines its own `Store` with `BdStore` (shells out to the `bd` CLI), `NativeDoltStore` (behind the `gascity_native_beads` build tag), `SQLiteStore` ("`CGO_ENABLED=0` safe", using `modernc.org/sqlite`), `MemStore` and `FileStore`. The `FileStore` rewrites a whole JSON file on every write ("fine for Tutorial 01 volumes", `filestore.go:141-144`). This is a useful data point: the most mature Beads consumer also keeps a pure-Go SQLite path.

### 9.3 Fit matrix

| Requirement (this RFC) | Gas City | Beads | In-house |
|---|---|---|---|
| Reconcile **task acceptance**, not process liveness | ✗ reconciles sessions | n/a | ✓ (§4) |
| Only a deterministic gate completes work | ◐ exit-0 gates exist, but manual approval bypasses them | n/a | ✓ (§4.3) |
| Typed diagnostics, baseline-relative verdicts | ✗ opaque script | n/a | ✓ (§5) |
| Structured worker I/O, no terminal scraping | ◐ ACP only; tmux default | n/a | ✓ (§3.4); Gas City ACP is a reference |
| Worktree lifecycle that fails closed | ✓ strong | n/a | adapt Gas City |
| Conflict pre-flight + CAS integration | ✗ agent role, porcelain merge | n/a | ✓ (§6.2) |
| Git lockfile / index recovery | ✗ none | n/a | ✓ (§6.2) |
| Process-group supervision, PID-reuse safety | ✓ strong (git timeouts excepted) | n/a | adapt Gas City |
| Intent-before-effect journal | ◐ shadow-only, tied to beads | ◐ `journalops` | ✓ SQLite `ops` table (§6.1) |
| Whole-document CAS status writes | via Beads | ◐ partial `RowVersion` | ✓ |
| Many in-process writers, no daemon | via Beads/Dolt | ✗ embedded is single-writer | ✓ SQLite WAL + one writer goroutine |
| Dependency DAG + ready query | via Beads | ✓ strong | ✓ (simple; Beads semantics as the reference) |
| Importable as a library | ✗ all `internal/` / `main` | ✓ public API | — |
| Lean single binary | ✗ heavy deps + tmux/bd/dolt at runtime | ✗ Dolt stack, CGO for embedded | ✓ |

### 9.4 Recommendation

**Build the control plane and storage in-house. Vendor Gas City's worktree, git and process packages as the starting point for `internal/worktree` and `internal/exec`. Use Beads only as an optional interop surface.**

1. **Do not adopt Gas City's controller.** It is well engineered but solves a different problem: keeping a declared fleet of agent sessions alive. Its "judgment out of Go" principle puts acceptance and merging into prompts and pack scripts, and that is the exact property this RFC removes. Forking `cmd/gc` (`package main`, about 2,000 files) to change what is reconciled would cost more than writing the about 1.5k-LOC reconciler in §4.

2. **Vendor, then harden, Gas City's isolation primitives.** These are MIT-licensed, have no in-repo dependencies beyond `pathutil` (and `gitcred` for clone), and total about 4.1k non-test LOC across `internal/worktree`, `internal/git`, `internal/processgroup`, `internal/execgrace` and `internal/pidutil`. They encode lessons we would otherwise relearn: attempt-ID provenance, symlink-canonical flock keys, never-unlink locks, reachability-gated cleanup, `-d`-only branch deletion, graceful group cancellation, and PID start-time checks. Copy them into `third_party/gascity/` with the upstream commit pinned and the licence kept, and then add:
   - `merge-tree --write-tree` pre-flight and `update-ref` CAS integration (§6.2)
   - temp-index snapshots (§6.2)
   - stale `*.lock` recovery for Spine-leased paths (§6.2)
   - a deadline on the path flock (§6.2, rule 0)
   - process-group kill for git subprocess timeouts (§5.3)
   - cgroup v2 `cgroup.kill` where available (§6.3)

   Treat it as a one-time fork, not a tracked dependency. Upstream moves too fast, and the docs drift, so we diff upstream once per release and cherry-pick fixes deliberately.

3. **Do not use Beads as the task store.** Its CAS and claim semantics are good, and §7's schema borrows them: `ExpectedVersion`-style preconditions, a ready-work predicate that includes `defer_until`, and typed blocking vs. non-blocking edges. But a single-writer embedded engine that needs CGO, or an external `dolt sql-server`, conflicts with the concurrency and binary-size criteria, and the partial `RowVersion` cannot express §4.3 ownership. Gas City's own `SQLiteStore` (`modernc.org/sqlite`, `CGO_ENABLED=0`) supports SQLite as the right default.

4. **Beads interop, optional, post-v2.0.** Add `internal/bridge/beads`, which imports ready beads as Spine Specs and writes terminal status back. It goes through the `bd` CLI's JSON output, the same boundary Gas City's `BdStore` uses, so Spine never links Dolt. Use Beads' hash-ID scheme (`internal/idgen/hash.go`) for Spine task IDs, so that IDs map cleanly.

5. **Study before writing `agent.Adapter`:** Gas City's ACP provider (`internal/runtime/acp/`) and the per-CLI transcript readers (`internal/sessionlog/`). They are the most current record of each agent CLI's machine-readable output. Re-implement them rather than vendoring, because they are coupled to `internal/worker`.

**Open verification items:**
- the `mol-refinery-patrol` formula in `gascity-packs` (could change the merge assessment if it adds a deterministic gate)
- how `reviewquorum` is used in production (it has no in-repo production importer)
- Beads server-mode multi-writer throughput
- Gas City `FileStore` fsync behaviour

---

## 10. Build plan (greenfield)

There is no migration, shadow mode or cutover. v2 is a new binary in a new module, validated against a fixed benchmark corpus of repos and tasks with known-good outcomes. The prototype is not used as a reference implementation.

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| M0 (1–2 wk) | `spine doctor`, `internal/exec`, `internal/worktree` (vendored Gas City primitives + §6.2 additions), chaos harness | 1,000 randomized `kill -9` points during create, snapshot, merge and teardown. Afterwards `Recover` must leave a clean `git fsck`, no leftover `*.lock`, no orphan PIDs, and no lost snapshot. Gas City's own worktree tests must pass on the vendored copy. |
| M1 (1–2 wk) | `tools` registry, `cmd.run` parsers, `diff.*`, `ts.parse`/`ts.symbols`/`ts.api_contract`, content-addressed cache, baseline verdicts | Verdicts match ground truth on the corpus (Go, Python, TS and PHP repos), and identical trees hit the cache. |
| M2 (2 wk) | storage, controller, reconcile, `RuleSupervisor`, one structured adapter (stream-JSON or ACP) | 8 parallel tasks on a fixture repo, all integrated through CAS, with no index races and no LLM supervisor. |
| M3 (1–2 wk) | LLM supervisor (structured output), planner task + `spec.validate`, Replan | A/B against `RuleSupervisor` on the corpus: pass rate at least equal, fewer tokens per task. |
| M4 (1 wk) | `spine ui` (embedded, SSE), CLI (`run`, `status`, `list`, `approve`, `retry`, `controller`, `gc`), optional Beads bridge | End-to-end on the corpus with a human approval step. |

---

## 11. Open questions

1. **Squash vs merge commits** on integration. The proposal is squash by default (`commit-tree -p old`), with snapshot refs keeping the full history for forensics.
2. **Supervisor model placement.** It needs a small context and structured output, so a local model is plausible (in keeping with v1's local-first goal). That needs an M3 benchmark.
3. **Cross-task interface changes.** Should `Replan` be allowed to *edit sibling* Specs (e.g. "T2 must wait for the new interface from T1a"), or only its own children? The proposal is children only, plus adding `DependsOn` edges to siblings, which is monotonic and safe.
4. **Windows support.** Process groups → Job Objects; `flock` → `LockFileEx`. Out of scope for v2.0 and kept compilable behind build tags.
