# SPINE × memory-organ (CAM) serving integration — feature plan

Status: plan only (2026-07-08). No spine code changed yet.

## 1. What the serving work actually exposes

The memory-organ research (`~/code/memory-organ`, CAM: product-key store + layer-24 gated tap +
gate router + base-uncertainty write gate) has been carried into the **minisgl-rdna4 serve engine**
across two branches that are ahead of the published docs:

- **`cam-serve-optionB`** (`~/code/minisgl-rdna4-camB`) — the four-phase port is complete and
  GPU-validated: tap staged through the scheduler, decode tap inside the CUDA graph (~20% TPOT),
  per-token banks, stats, snapshot/restore. `python/minisgl/cam/memory.py::CAMMemory` is the
  self-contained runtime (no memory-organ import needed at serve time).
- **`cam-production`** (`~/code/minisgl-rdna4-prod`) — the production hardening on top:
  - **Transparent CAM on the OpenAI routes** — `/v1/completions` and `/v1/chat/completions` get
    ambient memory **read** (`_cam_auto_augment`, subject extraction + injection, no-op unless
    enabled) and gated ambient **write** (`_cam_auto_write`, per-request `cam_write: bool` body
    field or `MINISGL_CAM_AUTO_WRITE=1`), `api_server.py:484-486, 582-584`.
  - **Edit plane `/cam/*`** (`server/cam_api.py`, mounted when `MINISGL_CAM=1`):
    `POST /cam/remember {subject, prompt, object} → {stored, base_p}` (write gate: stores only if
    the base can't recall it), `POST /cam/ask {prompt, subject} → {text}` (router-gated seed-once
    generation), `GET /cam/facts`, `DELETE /cam/facts/{subject}`, `GET /cam/stats`,
    `POST /cam/freeze`, `POST /cam/save`, `POST /cam/undo`, `POST /cam/rebuild`, `GET /cam/audit`.
  - **Multi-tenant namespaces** via `X-CAM-Namespace` header (per-session/tenant store isolation).
  - **Auth** via `MINISGL_CAM_API_TOKEN` → `Authorization: Bearer` on all `/cam/*`.
  - **Write policy**: no-clobber default with `protect_tau` (0.70), freeze for curated stores,
    audit/undo/erase, capacity/eviction, persistence (save/restore across restarts).

**Issue tracker** (the serving milestone lives on `patcarter883/minisglang-rdna4`, label
`cam-serving`, opened 2026-07-08):

| # | P | scope | status |
|---|---|---|---|
| #6 | P0 | multi-tenant/per-namespace isolation | implemented, live-validated 5/5, PR pending |
| #7 | P0 | persistence across restart (load-on-boot, debounced autosave, `/cam/save`) | implemented, validated with real restart, unmerged |
| #8 | P1 | bearer auth on `/cam/*` (`MINISGL_CAM_API_TOKEN`) | implemented, validated, unmerged |
| #9 | P1 | capacity + **LRU eviction** (`MINISGL_CAM_MAX_FACTS`, per-namespace; evictions in `/cam/stats`) | implemented, validated, unmerged |
| #10 | P1 | transparent extraction/retrieval quality + semantic subject key | **open — the unfinished one** |
| #11 | P2 | DP-scale store (today: pinned to one replica via `MINISGL_CAM_DP_RANK`) | open |
| #12 | P2 | audit log, undo, true erase (`/cam/audit`, `/cam/undo`, `/cam/rebuild`) | implemented, validated, unmerged |

Mechanism accuracy is gated further upstream by memory-organ #1 (real-knowledge validation), #2
(multi-token transfer parity), #4 (N-scale beyond ~128 facts) — all open research.

Honest limits that shape everything below (from `memory-organ/docs/serving/online_api.md` and the
CAM README):

- **Fact-shaped knowledge only**: `subject → object` associations; the stored value is the object's
  *first-token embedding* by default (`obj_latent` stores a pooled phrase latent). Not a place for
  prose lessons — that's what the existing experience store is for.
- **One live value per subject**; a re-write is an update.
- **Capacity budget ≈ 128 comfortable facts per store** (32 banks × ~4 subjects before crowding);
  overflow does not error — **delivery silently degrades**, so stats polling is mandatory.
- **Coupled to one base**: the tap is trained per-base (Qwen3.5-4B, layer 24). The whole feature is
  meaningful only when spine's active provider is that minisgl server. Everything must fail open.

## 2. Integration thesis

Spine already has a *lesson* memory (cross-run distilled experience → prompt injection). The memory
organ adds a complementary, sharper capability: **durable, editable, subject-keyed project facts
that the served model itself answers with** — no prompt tokens spent, no retrieval step, delivered
inside the forward pass, and *filtered by the write gate so spine only stores what the base doesn't
already know*.

The mapping:

| spine concept | memory-organ concept |
|---|---|
| project (repo) | CAM **namespace** (`X-CAM-Namespace: <project-slug>`) |
| work item run | optional ephemeral namespace (`work-<id>`), promoted on approve |
| distilled lesson (prose) | stays in `ExperienceStore` (unchanged) |
| distilled **fact** (subject→object) | `/cam/remember` write (gate decides) |
| experience prompt block | transparent read on `/v1/chat` **or** `<known_facts>` block |
| `.spine/experience/lessons.jsonl` | `.spine/experience/facts.jsonl` side index (intent log) |
| run finalization capture | fact distillation + `/cam/save` |
| resume/rework | `/cam/rebuild` replay from facts.jsonl if server state lost |

## 3. Features, phased

### Phase 0 — plumbing: config + client (no behavior change)

**F0.1 Provider config surface.** New provider-config key `cam` (mirroring the `rsa` precedent):

```yaml
providers:
  - name: minisgl
    base_url: http://127.0.0.1:1919/v1
    cam:
      enabled: true
      api_token: ${MINISGL_CAM_API_TOKEN}   # optional
      namespace: auto          # auto = project slug; or explicit string
      write: distill           # off | distill | ambient
      read: transparent        # off | transparent | facts_block | both
      capacity_alert: 100      # warn threshold (store knee ≈128)
```

Touch points: add `cam` to `_PROVIDER_KEYS` (`spine/config.py:1103-1118`, next to `rsa` at
`:1117`) so it participates in per-phase override merging in `resolve_provider_config()`
(`config.py:1220`); document in `.spine/config.reference.yaml` beside the `rsa` docs (~line 416).

**F0.2 CAM client.** `spine/services/cam_client.py` — thin async `httpx` wrapper over `/cam/*`:
`remember/ask/facts/delete/stats/freeze/save/undo/rebuild/audit`, carrying the Bearer token and
`X-CAM-Namespace`. Reuse the pooled clients in `spine/agents/http_clients.py` and the transient
classifier from `spine/agents/retry.py:265` (`_is_transient_error`) for its own small retry loop —
do **not** route through `invoke_with_retry` (that's model-invocation-specific, token budgets etc.).
Every method is **fail-open**: 503 ("CAM not loaded"), connection refused, or `enabled: false` all
degrade to no-op + one debug log, exactly like `capture_run_experience` never raises.

Acceptance: with CAM absent/disabled, zero behavior change and zero extra latency on the agent path.

### Phase 1 — read side: served model answers from project memory

**F1.1 Transparent read for agent traffic.** When `cam.read` is `transparent`/`both` and the active
provider is the CAM server, spine only needs to send the namespace header. Thread it in
`_build_local_model()` (`spine/agents/helpers.py:975`): `ChatOpenAI(default_headers={"X-CAM-Namespace": ns})`
resolved per work item at agent-build time (`build_phase_agent`, `spine/agents/factory.py`).
Ambient read is server-side and no-op when the store is empty — safe default.

**F1.2 `cam_write` passthrough (default OFF for agents).** Mirror the RSA `extra_body` block at
`helpers.py:1081-1098`: `extra_body["cam_write"] = False` whenever CAM is configured, unless
`cam.write: ambient`. Rationale: ambient auto-write extraction (LLM + regex fallback) is tuned for
conversational turns; spine's agent prompts are enormous multi-part system prompts — letting the
server auto-extract "facts" from them would pollute the store. Spine writes explicitly (Phase 2).
Unlike RSA, do not force non-streaming.

**F1.3 `<known_facts>` block (deterministic alternative).** `cam.read: facts_block` fetches
`GET /cam/facts` for the namespace once per phase-agent build and renders a compact
`<known_facts>` block injected in `build_phase_agent()`'s prompt assembly
(`factory.py:516-528`), immediately after the experience block. This trades prompt tokens for
determinism and works even when transparent augmentation is off; `both` uses the block *and* the
in-forward delivery.

Acceptance: a fact written via `/cam/remember` under the project namespace observably changes a
phase agent's answer to the matching subject question, with `cam.read: transparent` and no prompt
change.

### Phase 2 — write side: durable project facts from runs

**F2.1 Fact distillation at run finalization.** Alongside `distill_run_experience()`
(`spine/agents/experience.py:133`), add `distill_run_facts()`: an LLM pass over the same run
material that emits **fact-shaped triples** `{subject, probe_prompt, object}` with short objects
(≤ a few tokens) — e.g. `("spine default branch", "The default branch of spine is", "main")`,
pinned tool names, decision outcomes, config values. Hook it at the same four dispatcher capture
points (`spine/work/dispatcher.py:809, 1253, 1608, 2288`), best-effort.

**F2.2 Gate-filtered write + side index.** Each candidate goes to `POST /cam/remember`; the server's
base-uncertainty gate decides (`stored: bool, base_p: float`). Record every *attempt* (stored or
skipped, with `base_p`) in `.spine/experience/facts.jsonl`, `flock`-guarded, modeled on
`ExperienceStore` (`spine/persistence/experience_store.py:46`) with a new `ProjectFact` type beside
`ExperienceLesson` (`spine/models/types.py:308`). The side index is the authoritative intent log —
it is what makes rebuild, capacity planning, and audit reconciliation possible (the bank tensor
cannot be enumerated).

**F2.3 Store lifecycle.** After a successful write batch: `POST /cam/save` (persist across server
restarts). Respect no-clobber: conflicting subject → surface in run summary rather than
force-overwrite. Optional `cam.freeze_after_seed: true` for curated stores.

**F2.4 Capacity discipline.** Before writing, `GET /cam/stats`; if `total_facts ≥ capacity_alert`
(default 100 of the ~128 knee), skip writes and emit a warning observation. Never write past the
knee — crowding degrades *other* facts silently. **New hazard from #9:** the server now LRU-evicts
past `MINISGL_CAM_MAX_FACTS` — a durable project fact can be *silently evicted* by later writes.
Mitigations: keep spine's write volume well under the cap, prefer `freeze` for curated stores, and
reconcile `facts.jsonl` against `/cam/facts` + the `evicted` counter in stats at run start
(re-`remember` anything evicted that is still wanted).

Acceptance: after a run that established a fact the base didn't know, the fact is in `/cam/facts`,
`facts.jsonl` has the record with its `base_p`, and a fresh session answers it correctly; a fact
the base already knew is recorded as `skipped` and not written.

### Phase 3 — operator surface: CLI, UI, observability

**F3.1 CLI.** `spine facts` group mirroring the `experience` group (`spine/cli/__init__.py:936`):
`list` (server facts ⟂ side index reconciliation), `add` (manual `/cam/remember`), `delete`
(tombstone via `DELETE /cam/facts/{subject}` + side-index update), `stats`, `audit`, `freeze`,
`save`, `undo`, `rebuild` (replay `facts.jsonl` through the write path — the exact-erase /
re-shard path).

**F3.2 UI + observability.** A "Project facts" panel beside the experience page
(`spine/ui/_pages/experience.py` pattern, API in `spine/ui_api/api.py:531-561` pattern): fact list
with per-fact `base_p`, store occupancy/crowding from `/cam/stats`, and an alert badge past the
capacity threshold. Emit a `cam_stats` observation at run start via `spine/observability.py`.

**F3.3 Write verification.** After each accepted write, optionally `POST /cam/ask` the fact's own
probe prompt and check the object appears (the only ground-truth readback). Record
`verified: bool` in `facts.jsonl`. Cheap (one short generation per new fact) and catches silent
crowding immediately.

### Phase 4 — run-scoped memory + resume (experimental)

**F4.1 Ephemeral work namespaces.** At work start, namespace `work-<id>`; facts discovered
mid-run (IMPLEMENT/VERIFY) are written there so later phases of the *same run* benefit without
touching the durable project store. On approve/finalize, promote surviving facts to the project
namespace (re-`remember` there); the ephemeral namespace is abandoned. This maps cleanly onto
spine's approve-continues-same-work-item resume semantics.

**F4.2 Resume/rebuild tie-in.** On `resume_work`, verify the server store agrees with
`facts.jsonl` (count + spot probe); on mismatch (server restarted without `/cam/save`, or
namespace evicted) replay the side index via `rebuild`. Model on the fail-open
`verify_snapshot.py` ratchet (`snapshot_best`/`restore_best`), keyed by work_id alongside
`CheckpointStore.get_state` (`spine/persistence/checkpoint.py:71`).

## 4. Risks / non-goals

1. **Base coupling.** Facts live in a store bound to one tap/base checkpoint. Switching the
   provider model orphans the served memory — but `facts.jsonl` is tokenizer-agnostic strings, so
   the store is always rebuildable against a new checkpoint. The side index is the source of
   truth; the bank is a cache of it.
2. **Capacity is small (~128/namespace).** This is a *pinned-facts* store, not a knowledge base.
   Distillation must be very selective (cap per run, e.g. ≤5 candidates), and eviction pressure
   surfaces in `/cam/stats` — F2.4 is not optional.
3. **Ambient auto-write on agent prompts is off by default** (F1.2). Spine's prompts are not
   conversational turns; explicit distillation writes are the clean path. Server issue #10
   confirms this from the other side: transparent read matches capitalised proper-noun spans
   only, auto-write leans on a regex fallback, and the pooled-embedding subject key confuses
   paraphrases with name collisions (tau overlap 0.58 vs 0.62). Until #10's semantic-key work
   lands, prefer `cam.read: facts_block` (deterministic) over `transparent` for correctness-
   sensitive phases.
4. **DP deployments (#11):** CAM ops are pinned to one scheduler replica
   (`MINISGL_CAM_DP_RANK`); spine's client needs no change, but expect edit-plane calls to be a
   single-replica bottleneck at DP>1.
5. **Research-preview provenance.** CAM is days old, synthetic-task-validated, single-base. All
   spine features are additive, provider-gated, and fail-open; nothing in the workflow may ever
   block on the CAM plane.
6. **Not a replacement for the experience store.** Lessons (prose guidance, per-phase) and facts
   (subject→object, base-injected) are disjoint; F2.1 routes each run finding to exactly one.

## 5. Suggested build order

Phase 0 + F1.1/F1.2 (one PR: config key, client, namespace header, cam_write=false) →
F2.1–F2.4 (fact distillation + side index) → F3.1/F3.3 (CLI + verify) → F1.3, F3.2, Phase 4.

**Implementation status (2026-07-08):**
- ✅ Phase 0 (F0.1 `cam` provider key; F0.2 `spine/services/cam_client.py`)
- ✅ F1.1/F1.2 (namespace header + `cam_write` pinning in `_build_local_model`)
- ✅ F1.3 (`<known_facts>` block — rendered from the local side index, not a live
  fetch, so the agent-build path never touches the network)
- ✅ F2.1–F2.4 (`spine/agents/facts.py`, `ProjectFact`,
  `spine/persistence/facts_store.py`, wired at all four dispatcher finalize sites)
- ✅ F3.1 (`spine facts` CLI: list/--server drift check, add, delete, sync,
  stats, audit, freeze) — `sync` is the eviction-recovery replay
- ✅ F3.3 (post-write `/cam/ask` readback probe recorded as `verified`)
- ✅ F3.2 UI panel ("Project Facts" page: side-index metrics, probe-failure
  alert, live `/cam/stats` expander, per-fact inspector) + `UIApi`
  list/stats/delete/server-stats methods. *Skipped:* the run-start `cam_stats`
  observation — `spine/observability.py` is LangSmith tracing only (no event
  API); the capacity guard and readback-probe warnings at write time cover the
  alerting need.
- ⬜ Phase 4 (ephemeral work namespaces, resume reconciliation)
- ⬜ End-to-end validation against a live CAM serve stack (`MINISGL_CAM=1` +
  trained checkpoint) — required before relying on it in real runs

Prereq on the serving side: `cam-production` running with `MINISGL_CAM=1`, a trained checkpoint
dir, and namespaces enabled — plus the parity spike (`memory-organ/docs/serving/parity_spike.md`)
green if the tap checkpoint predates the current engine build.
