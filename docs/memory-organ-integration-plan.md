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
- ✅ End-to-end validated against the live CAM serve stack (2026-07-08):
  Qwen3.5-4B on minisgl `cam-production` via the compose overlay
  `minisgl-rdna4-prod/docker-compose.cam.yml` (launch:
  `MINISGL_MODEL=Qwen/Qwen3.5-4B MINISGL_TP=1 MINISGL_ATTN_BACKEND=hip
  MINISGL_EXTRA_ARGS="--max-running-requests 16" gpu-lease -n 1 --detach
  --name cam-serve -- docker compose -f docker-compose.yml -f
  docker-compose.cam.yml --profile serve up -d`). Verified: gated write via
  `spine facts add` (novel "capital of Zorbia→Flumevale" **stored**, base-known
  "capital of France→Paris" **skipped**), `/cam/ask` readback delivers
  Flumevale, transparent read injects the fact into a plain `/v1/chat` request
  under the namespace header, store survives a server restart (load-on-boot),
  and `spine facts list --server` drift-checks correctly per namespace.
  **Workflow-level validation (2026-07-09, spine-sandbox):** a real `spine run`
  exercised the finalize hook end-to-end — `capture_run_facts` distilled
  exactly the durable facts from the run ("project codename → Brindlewharf",
  "codename marker filename → CODENAME.md"), both passed the server write gate
  into the `wf-e2e` namespace, and a `failed`-status run correctly skipped
  capture. Caveats from the exercise: (a) the CAM base (Qwen3.5-4B) cannot
  itself drive spine's structured phases — production shape is hybrid routing
  (capable model for phases, CAM hooks keyed off the active provider);
  (b) `spine run --config` is not honored by code paths that call
  `SpineConfig.load()` bare — the provider must be in `.spine/config.yaml`;
  (c) a full green run was blocked by a **pre-existing, CAM-unrelated**
  OpenRouter failure: agent-stack requests 404 "No endpoints found that can
  handle the requested parameters" on glm/deepseek/kimi alike under
  `require_parameters=true`, while every standalone repro (structured bind,
  tools, tools+response_format+tool_choice=required, streamed) succeeds —
  the offending parameter is added inside the deepagents middleware stack;
  needs its own investigation.
  **Server env findings** (now baked into the overlay): the full server needs
  `MINISGL_CAM_FRONTEND=1` (routes `/cam/*` to the backend store; without it
  the API process tries to load its own HF base and 503s),
  `MINISGL_CAM_WRITE_GATE=1` (the base-uncertainty gate is OPT-IN in frontend
  mode — default `/cam/remember` force-writes), and `MINISGL_CAM_AUTO=1` for
  transparent read. In frontend mode `base_p` is not computed (always 0.0) —
  the gate is a "does the base emit the object" generation probe, so treat the
  recorded `base_p` as informational only on this deployment mode. The 4B on a
  16 GB card also needs `--max-running-requests 16` (the default 256 GDN
  recurrent-state slots reserve 12.7 GiB and starve the KV pool).

Prereq on the serving side: `cam-production` running with `MINISGL_CAM=1`, a trained checkpoint
dir, and namespaces enabled — plus the parity spike (`memory-organ/docs/serving/parity_spike.md`)
green if the tap checkpoint predates the current engine build.

---

## 6. Addendum (2026-07-16): upstream reassessment — pointer pivot + frozen-base scorecard

Reassessment of this plan against everything memory-organ / minisgl-rdna4 landed after the
2026-07-08 snapshot above. **Nothing spine shipped breaks** — the live serving surface
(`cam-production`) is unchanged since Jul 8, and the side-index-first design (facts.jsonl is the
source of truth, the bank is a rebuildable cache, everything fail-open) is exactly what makes
spine indifferent to the mechanism pivot described below.

### 6.1 What changed upstream

1. **memory-organ #100 solved (Jul 8, merged).** The **pointer id-bank** delivers genuine
   multi-token objects end-to-end, 4/4 span-exact, with the base continuing fluently.
   "Remaining is serving integration only."
2. **A/B/C bake-off (Jul 9, `bakeoff-abc`).** Pointer architecture (GTE-whitened subject keys +
   exact object token-ids) wins delivery outright: paraphrase addressing **0.543 vs 0.06** for
   the incumbent input-embedding keys, 5% false delivery, lossless multi-token (46% of natural
   facts are multi-token), zero per-base training, base-agnostic. Whitening is mandatory (raw
   GTE delivers every distractor).
3. **Hybrid serving design spec (Jul 9, minisgl-rdna4 `cam-hybrid-spec`,
   `docs/zaya-port/CAM_HYBRID_DESIGN.md` — design only, not implemented).** Pointer becomes the
   default delivery mode; the tap survives as an opt-in "editor" mode over one store. API
   deltas when it lands: `mode: "pointer"|"tap"|"both"` on `/cam/remember`, `mode` +
   `X-CAM-Mode` on ask with a `mode_served` echo, GTE key + whitening artifacts shipped in the
   checkpoint (this is the real fix for serving issue #10), taus re-tuned for the GTE key space.
4. **Frozen-base vs Titans scorecard (merged Jul 13, memory-organ PR #103:
   `docs/research/frozen-base-titans-scorecard.md`, RESULTS §8, DIARY Phase 17, ROADMAP).**
   The strategic result: **injection cannot integrate an edit into multi-hop reasoning** —
   every activation-write variant at every depth sits below the in-context (RAG) ceiling
   (~0.5); only distilling the base's own in-context behavior (a cartridge) clears the wall.
   The structural win of the frozen approach is **continual/no-forget via a routed bank of
   isolated memories** (general knowledge preserved 1.00 vs 0.17 for naive accumulation; zero
   cross-fact interference by construction). ROADMAP now codifies the product shape as
   "a reasoning model plus a routed bank of cheap, editable, non-interfering test-time
   memories."

Upstream research gates cited in §1: **two of three are now substantially resolved** — #100
multi-token delivery (pointer, solved) and #2 cross-base transfer (translator lifts 0.393 →
0.812, "largely closed"). Only N-scaling beyond ~128 facts remains genuinely open, and the
pointer store likely reshapes that problem (see 6.3).

### 6.2 Plan changes to adopt now (no upstream dependency)

- **`cam.read: facts_block` is the principled default, not a fallback.** The scorecard proves
  in-forward injection delivers single-hop recall at best and cannot participate in multi-hop
  reasoning — the prompt block IS the quality ceiling. Transparent read is a token
  optimization for recall-shaped queries only. Reframe F1.3/risk-3 accordingly.
- **Subject canonicalization in F2.1 distillation.** The routed bank's accuracy bottleneck is
  router retrieval (0.70), and its misses are same-structure aliases. Spine controls the
  subject strings it writes: the `distill_run_facts` prompt should enforce one canonical
  subject naming convention per entity (no near-alias subjects for the same thing).
- **Fact-block selectivity is a delivery-quality lever, not just capacity discipline.**
  BABILong shows even RAG degrades under distractor-heavy context (0.75 → 0.45 as filler
  grows). The ≤5-candidates-per-run cap and a small curated `<known_facts>` block keep the
  in-context mechanism in its high-accuracy regime; do not let the block grow into a dump.

### 6.3 Changes that activate when the hybrid serving lands

- **`cam_client.py`:** send `mode: "pointer"` on remember; surface `mode_served` from ask.
  Spine should never opt facts into tap mode — its facts are exactly what pointer is for.
- **F2.1 "objects ≤ a few tokens" relaxes** — pointer delivery is lossless multi-token.
- **Risk 1 (base coupling) mostly dissolves** — pointer facts are base-agnostic; switching the
  provider model orphans only opt-in tap facts.
- **Risk 3 (paraphrase/collision, issue #10)** is fixed by GTE-whitened keys (0.06 → 0.54
  paraphrase addressing), making transparent read trustworthy for recall.
- **F2.4 capacity knee (~128) is a product-key-bank constraint** — under the hybrid, the bank
  write only happens for tap-mode facts, so pointer facts shouldn't crowd. LRU eviction (#9)
  still applies: keep the reconcile-and-replay discipline.

### 6.4 Watch item — Phase 5 candidate (does not change current phases)

The scorecard's cartridge result sketches a per-project **distilled "codebase cartridge"**: a
build-once, query-many context (a codebase is the canonical case) distilled into a persistent
~38× smaller prefix at in-context quality, organized as a routed bank per project. That is a
new serving surface that does not exist yet — track it; do not build against it.

### 6.5 Corrections to the 2026-07-08 text

- The worktree paths in §1 and the launch recipe (`~/code/minisgl-rdna4-camB`,
  `~/code/minisgl-rdna4-prod`) have been pruned. The branches live in `~/code/minisgl-rdna4`
  proper (`cam-serve-optionB`, `cam-production`, plus `cam-hybrid-spec` for the design spec).
- `cam-production` gained one commit after the snapshot (182c330, Jul 8): a #10
  subject-canonicalization lever + retrieval eval harness, and #11 DP-scale reload/design.
  Issues #6–#12 all remain open on the tracker; the "implemented, validated" branches are
  still unmerged and PR-less.

---

## 7. Live deployment state (2026-07-16, evening) — pointer is the deployed reality

Validated against the production serve (10.50.1.51:1919, `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
on mini-sglang, CAM rev `02f0231`) during the first spine↔minisgl coordination loop.

### 7.1 What the deployed build actually is

- **Pointer id-bank is the default AND only path** — the 4B-trained tap checkpoint's hidden
  dim (2560) ≠ the served 35B's (2048), so CAM auto-falls-back to pointer/retrieve-only and
  the residual tap is OFF. The hybrid `mode: pointer|tap|both` request field is design-only;
  `mode_served` reads `pointer` everywhere. Spine sets `cam.mode: pointer` (config, live).
- §6.3's predictions held: multi-token objects and code identifiers deliver span-exact;
  the 4-word object cap is relaxed to 12 under pointer.
- **Write gate works and reports `gate_reason`** ("novel" / base-recalls), which replaces
  `base_p` (always 0.0 in frontend mode) as the meaningful verdict. Recorded per fact in
  the side index.
- **`/cam/ask` returns `delivered` + exact `object` + `mode_served`** — spine's readback
  verification is now exact, not substring. **`/cam/lookup?subject=|text=`** pre-checks
  delivery; **`GET/DELETE /cam/namespaces`** manage stores; deletes are durable across
  restarts (rev 02f0231).
- **Transparent read delivers on phrase subjects** (lexical n-gram windows, tau-gated):
  a stored `capital of zorbia` answers "What is the capital of Zorbia?" in plain chat.
  The GTE semantic key remains the deeper #10 fix. `facts_block` stays spine's read
  default per §6.2 (multi-hop ceiling argument is unchanged); `both` is now a credible
  option for recall-heavy phases.

### 7.2 Spine-side state

- Store seeded: 18 verified facts in namespace `spine` via `spine facts seed`
  (onboarding docs → chunked distillation → gate → readback). Seed hardening landed:
  per-chunk progress + distillation-lane visibility, deterministic near-alias dedupe
  (same object + overlapping subject tokens), `--dry-run` candidate caching + `--from`
  replay. Distillation pinned to the CAM provider via the `experience` phase override.
- Operational findings from the first full-run exercise: (a) the serve can wedge under
  concurrent large-prompt load — generation dead while `/cam/*` answers; minisgl tracks
  hardening + a timed-generation health gate; (b) spine's landing/rollback left `main`
  checked out in the master dir, silently reverting live branch-committed config —
  fixed on `fix/master-branch-restore` (operator branch recorded + restored; master-tree
  scrub limited to non-worktree strategies).

### 7.3 Coordination protocol

Unattended iteration runs through the note board at `http://10.50.1.51:7071`
(`/serve`, `/notes`, `/note`, `/board`): spine tests and files findings; minisgl lands
one server change per tick, health-gated with auto-rollback. Ledger of the first day:
5 API asks + phrase-match + durable-delete requested, landed, and live-verified within
three revs (a009317 → 3e8c1b3 → 02f0231).
