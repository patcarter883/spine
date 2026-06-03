# vLLM Prefill Tuning — Spine `implement` / `index` Workloads

**Server:** `10.50.1.51:8000` (the `pat` provider in `.spine/config.yaml`)
**Model:** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` — Qwen3 MoE, A3B active, AWQ 4-bit, int8 KV, 80k ctx
**Engine:** vLLM `0.22.1rc1` (V1)
**Data source:** Prometheus at `http://10.50.1.51:9090` (read-only, the TSDB behind the auth-walled Grafana on `:3000`)
**Analysis window:** 7 days, 2026-05-27 → 06-03

> This is the LLM backend for the spine agent framework. The workload shape is
> textbook agentic: large prompts in, small edits out, with a high prefix-cache
> hit rate from repeated context. Tuning here targets the **prefill / TTFT** path.

---

## TL;DR

The server is idle ~90% of the time, then takes bursts. Decode is healthy
throughout (inter-token latency p50 22 ms). **All the pain is on prefill**: during
the heavy `implement` burst, time-to-first-token p95 hit **68–78 s**.

Three compounding causes, in priority order:

1. **No moderate prefill chunk cap.** `max-num-batched-tokens` is effectively
   **≥16,384** today — single big prompts prefill in one giant engine step that
   freezes everything else.
2. **KV cache is tiny.** `num_gpu_blocks=302` × `block_size=16` ≈ **4,832 token
   slots**, at `gpu_memory_utilization=0.96`. The largest prompts (p99 48k) exceed
   the whole cache; only an 82.6% prefix-cache hit rate makes them fit. Requests
   stall on `reason="capacity"` (peaked at 11).
3. **Spine sends 6 concurrent** (`max_concurrent_calls: 6`) — 6 simultaneous
   48k-token prefills oversubscribe a 4.8k-slot KV.

**Target: balanced** — cut the worst of the TTFT tail while keeping most throughput.

---

## Workload profiles

### `implement` spike — 06-02 11:00–18:00 UTC (reconstructed)
Read-heavy agent loop: big code context in, small edits out.

| hour (UTC) | gen tok/s | prompt tok/s | running | waiting | KV % | TTFT p95 |
|---|---|---|---|---|---|---|
| 11 | 89 | 1759 | 6 | 0 | 23.6 | 12.1 s |
| 12 | 57 | 1449 | 6 | 1 | 34.2 | 15.8 s |
| 13 | 173 | 712 | 6 | 2 | 39.9 | **68.0 s** |
| 14 | 37 | 1259 | 6 | 1 | 23.6 | 17.4 s |
| 15 | 14 | 1149 | 5 | 2 | 27.9 | 38.9 s |
| 16 | 17 | 841 | 5 | 2 | 29.6 | **66.8 s** |
| 17 | 14 | 758 | 5 | 1 | 34.9 | **77.6 s** |
| 18 | 8 | 1514 | 5 | 1 | 40.9 | 64.8 s |

`running` pinned at the `max_concurrent_calls: 6` client cap the whole time. E2E p99 reached 134 s.

### `index` task — live (decode-bound, the opposite shape)
Small prompts (p50 **529**), large generations (p50 **1,582**), KV only **10%**,
TTFT p50 **0.19 s**. Comfortable — prefill tuning must **not regress** it. A
moderate chunk cap won't hurt it because its prompts are tiny.

### 7-day aggregate
24,247 requests, **0 errors / 0 aborts / 0 preemptions**. 195.6M prompt tokens,
4.87M generation tokens (~40:1 read:write). Prefix-cache hit rate **82.6%**.
Latency: TTFT p50 0.26 / p99 30.1 s · ITL p50 22 ms / p99 141 ms · E2E p50 1.5 / p99 134 s.

---

## Evidence for the chunk-cap finding

The only vLLM restart in 7 days was **06-02 10:10 UTC**, so the current instance
is one continuous config. In that window `vllm:iteration_tokens_total` recorded
**314 engine steps above 16,384 tokens** plus 371 in the 8,192–16,384 band. Since
the per-step token budget *is* `max-num-batched-tokens`, steps that large prove
the cap is ≥16k right now — it is **not** set to 2048/4096 on this server.

> Note: the `localhost:8000` provider in `.spine/config.yaml` is a *different*
> model (`Qwen3.6-35B-A3B-MTP-GGUF`, `max_concurrent_calls: 1`) and may run a
> 4096 cap — but that is not the `pat` / `10.50.1.51` server.

`cache_config_info` (current): `num_gpu_blocks=302`, `block_size=16`,
`cache_dtype=int8_per_token_head`, `enable_prefix_caching=True`,
`gpu_memory_utilization=0.96`, `num_cpu_blocks=None` (no CPU swap).

---

## Part A — vLLM launch flags (apply on host `10.50.1.51`)

Primary prefill levers. Add/adjust on the vLLM serve command.

| Flag | Value | Why |
|---|---|---|
| `--max-num-batched-tokens` | **8192** | *Primary fix.* Caps each step at 8k so a 48k prompt chunks into ~6 steps that interleave with the other requests → ends the >16k giant-step head-of-line blocking. Balanced pick (4096 = more tail reduction, lower prefill efficiency; ~16k ≈ today). |
| `--long-prefill-token-threshold` | **4096** | Marks prompts >4k as "long" so V1 chunks them and lets short `index`-style requests slip in between chunks — fairness across the concurrent `implement` requests. |
| `--swap-space` | **16** | Attacks the `capacity` waits. `num_cpu_blocks` is currently `None`; 16 GiB CPU swap gives the scheduler somewhere to hold cached/preempted KV instead of recomputing or refusing admission. |
| `--max-model-len` | *(optional)* **57344** | Down from 80000. Prompts reach 48k, so keep ≥ ~52k. Frees GPU blocks for more live KV. Only if 80k context is genuinely unused — verify first. |

**Do not change:** `--enable-prefix-caching` (load-bearing, 82.6% hit), the int8
KV dtype (doubles capacity), or `gpu_memory_utilization` (already 0.96 — pushing
higher risks OOM).

**Expected effect:** TTFT p95 during `implement` bursts 68–78 s → **< 20 s**;
`capacity` waits → ~0; ITL unchanged (~22 ms); peak throughput within ~10–15% of
today's 232 gen tok/s.

---

## Part B — spine config (`.spine/config.yaml`, optional)

The `pat` provider (→ `10.50.1.51:8000`) sets `max_concurrent_calls: 6`. At 6 the
engine was prefill-serialized anyway; **lowering to 4** lets each large prefill
get more engine + KV share, cutting the TTFT tail with minimal throughput loss.
Best applied *after* the Part A chunk cap is in place, then re-measured.

*Optional follow-up:* if spine grows per-task provider routing, give `index`
(decode-bound, KV only 10%) its own profile at `max_concurrent_calls: 8` while
keeping `implement`/`pat` at 4.

---

## Index context-window sizing

How small can `max-model-len` go for a dedicated **index / summarisation**
instance? The constraint is **request admission**, not actual usage: vLLM 400s any
request where `prompt + max_tokens > max-model-len` (see the comment at
`spine/agents/helpers.py:576-585`). Index per-request data (713 reqs, 3 h):

| Per request | p50 | p99 | max observed |
|---|---|---|---|
| Prompt tokens | 495 | 6,759 | ≤10,000 |
| Generation tokens (actual) | 1,594 | 4,885 | one outlier in 5k–20k band |
| `max_tokens` **requested** | — | — | **40,000 (flat)** |

> The requested `max_tokens` is a flat **40,000**, set by the global
> `spine.max_completion_tokens` (`.spine/config.yaml:14`) and applied to the `pat`
> provider via the fallback in `_build_local_model` (`spine/agents/helpers.py:581-585`)
> because the `summarization` phase route (`.spine/config.yaml:206-207`) has no
> override. Index requests only ever *generate* ~1,600 (p99 4,885), so the 40k
> budget is ~8× over-provisioned. (An earlier read of the coarse Prometheus
> `request_params_max_tokens` histogram showed p50 33.6k / p99 49.7k — those were
> bucket-interpolation artifacts; the true value is 40,000.)

**Sizing, two cases:**

1. **Change nothing else → floor ≈ 56k.** Worst-case admission = max prompt (~10k)
   + requested `max_tokens` (40k) = ~50k. Below ~56k, vLLM starts rejecting the
   largest index requests. Barely under today's 80k — not worth it alone.

2. **Cap the summarisation `max_tokens` first → 32k is safe.** Add a per-phase
   override mirroring the existing `verify` phase (`.spine/config.yaml:200`,
   `max_completion_tokens: 8192`):

   ```yaml
   providers:
     phases:
       summarization:
         provider: pat
         max_completion_tokens: 16384   # 3× the 4,885 p99; covers the rare long summary
   ```

   Then `prompt (≤10k) + 16,384 = ~26k` fits comfortably in **`--max-model-len 32768`**.
   (8,192 would match `verify` and cover the p99, but risks clipping the lone
   >5k-token summary; 16,384 is the safe default.)

**Absolute floor on observed real traffic ≈ 24k** (max prompt 10k + the ~20k gen
outlier, if they coincided) — but 32k is the right round number with headroom.

**Caveats:**
- Shrinking `max-model-len` does **not** by itself grow the KV pool
  (`num_gpu_blocks=302` is fixed by VRAM profiling at `gpu_memory_utilization=0.96`).
  The win is a smaller per-sequence ceiling → safely raise `max-num-seqs` /
  concurrency for the decode-bound index workload without KV over-commit, and it
  stops the 40k `max_tokens` over-reservation.
- **Index-only.** The `implement` workload has prompts to 48k (p99) and needs
  ≥ ~56k context — never apply 32k to a shared instance.

---

## Verification

Re-run during a real `implement` burst against `http://10.50.1.51:9090`:

```promql
# 1. TTFT tail — target < ~20s (was 68–78s)
histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket[15m])))

# 2. Capacity waits — target 0
max_over_time(vllm:num_requests_waiting_by_reason{reason="capacity"}[1h])

# 3. Chunk cap active — top filled bucket should now be <= 8192
vllm:iteration_tokens_total_bucket

# 4. Decode not regressed — still ~0.022s
histogram_quantile(0.5, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket[15m])))

# 5. Throughput within ~15% of 232 gen tok/s peak
max_over_time((sum(rate(vllm:generation_tokens_total[5m])))[1h:5m])

# 6. index task TTFT p50 still ~0.19s (no regression)
histogram_quantile(0.5, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket[15m])))
```
