## Code Review — commit 3167cd195ea03a156b83a30b4c181f1f405123f3 (2026-05-01)

Scope: SPEEDUP_PLAN.md — vectorize env hot paths, multi-trace cache, Numba JIT.
Reviewed: octopus/env.py, octopus/data.py, scripts/train_rl.py, scripts/evaluate.py,
tests/test_new_reward_obs.py, tests/test_augmentation.py

---

### Phase 1: Repo Overview

**Purpose:** RL agent (SAC via SB3) that learns to allocate VM memory across CXL MPDs.
Each episode replays ~14 days of Azure VM trace for a randomly selected pod.

**Directory structure:**
- `octopus/` — core library: env, data, baselines, augmentation, topology
- `scripts/` — train_rl.py (training), evaluate.py (pooling_simulation), eval_rl.py
- `tests/` — pytest suite; no CI; run locally with `pytest tests/`
- `data/traces/` — 10 Azure VM trace pickles (not in git)
- `data/topologies/` — 5 topology CSVs

**Entry points:**
- `scripts/train_rl.py main()` — training; SubprocVecEnv for n_envs>1
- `scripts/evaluate.py main()` — eval; `pooling_simulation()` is the core loop

**Key data flow:**
```
load_trace() → to_arrays() → TraceArrays
TraceArrays → OctopusMemPoolEnv.__init__(trace_arrays, M)
reset(): _generate_pod(seed) → _build_events() → [augmentation] → sim state init
step(action): softmax → alloc → reward → _process_departures_through() → _get_obs()
```

**Key data structures:**
- `TraceArrays` — numpy CSR representation of trace (vm_start, vm_end, vm_mem, node_ids, node_dram, node_offsets, vm_ptrs)
- `events: list[tuple(tick, pod_id, mem_gb, dealloc_tick)]` — sorted VM arrivals per episode
- `mpd_vm_allocs: list[list[tuple(dealloc_tick, mem_gb)]]` — per-MPD active VMs (BOTTLENECK)

---

### Phase 2: File-by-File

**octopus/env.py**

Single responsibility: Gymnasium env. Input: TraceArrays + M. Output: obs, reward per step.

Key functions:
- `__init__(trace_arrays, M, seed, ...)`: accepts TraceArrays only (no raw dicts).
- `reset()`: pod selection → event building → augmentation → sim state init. Returns first obs.
- `step(action)`: softmax allocation → reward → advance event pointer → call `_get_obs()`.
- `_compute_D_j(tick)`: O(num_mhd × active_VMs) **nested Python loops** — documented bottleneck.
- `_get_obs()` variant A/B: **second independent nested Python loop** over mpd_vm_allocs — duplicates D_j/S_j work done in step(). `_cached_D_j` field is set to None in reset() and never written anywhere — dead code.
- `_process_departures_through(tick)`: **third Python loop** — list-comp per MPD to purge expired VMs.

Surprises / risks:
- `_compute_D_j` is called in `step()` as `D_j_pre`, but `_get_obs()` does its own independent D_j/S_j loop. The two computations are at different stages (pre-alloc vs post-advance) so they can't trivially share, but the `_cached_D_j` field was presumably intended to bridge this — it's just never populated.
- `mpd_vm_allocs` grows unboundedly per episode; no capacity limit. This is fine for correctness but means numpy conversion in Chunk 1 requires dynamic sizing.
- `_switch_trace()` swaps `trace_arrays` but does NOT swap `_precomputed_events` — this means multi-trace training never hits the cache. Documented as Chunk 2 fix.

**octopus/data.py**

Responsibility: trace loading + precompute cache.

- `TraceArrays` dataclass — numpy-safe representation.
- `to_arrays(trace_data)`: converts raw dict-based trace to TraceArrays.
- `precompute_pod_events(trace_data, M, seeds)`: uses raw VM-object API (datetimes, dicts) — diverged from `_build_events()` which uses int-second arithmetic. Slight tick-rounding may differ for traces where start_time is not minute-aligned. The newer `precompute_pod_events_arrays(TraceArrays, M, seeds)` function (Chunk 2 target) does not yet exist.

**scripts/train_rl.py**

Responsibility: SB3 training with callbacks. All env construction goes through `_make_env()`.

- Uses `to_arrays()` before forking — good for CoW safety.
- Multi-trace pool: loads all traces into `trace_pool` (list of TraceArrays) but does NOT build per-trace precomputed event caches. Chunk 2 adds this.
- Async eval worker: works correctly; spawns once at training_start.

**scripts/evaluate.py**

Responsibility: `pooling_simulation()` — the eval core used by the async worker.

- Still uses raw VM-object API (all_vms dict, datetime objects). This is intentional — evaluation uses the full pooling_simulation path, not the env.
- `make_rl_alloc_cb()`: bypasses SB3's model.predict() in favor of `policy._predict()` + pre-allocated GPU tensor — good perf. BUT requires a real SB3-style model with `.policy` attribute.

**tests/test_new_reward_obs.py**

All 20 tests fail. Root cause: `_make_minimal_env()` calls `OctopusMemPoolEnv(all_vms=..., node_to_vms=..., ...)` — old constructor API. Current constructor signature is `__init__(self, trace_arrays, M, ...)` with no dict kwargs.

`test_Q_j_asymmetric_topology` and `test_Q_j_values` / `test_recompute_topology_derived_after_link_failure` also build `OctopusMemPoolEnv` directly with old API.

`test_obs_sync_initial_step`: after fixing constructor, will still fail because `_CapturingModel` doesn't provide `.policy` (required by `make_rl_alloc_cb`). Needs a mock policy that exposes `set_training_mode()`, `parameters()`, and `_predict()`.

**tests/test_augmentation.py**

3 tests fail (`test_augmented_episode_runs`, `test_no_augmentation_matches_baseline`, `test_skip_hotfix_includes_more_events`) — all use old positional constructor: `OctopusMemPoolEnv(all_vms, node_to_vms, node_to_machine, machine_sz, M, ...)`. Fix: call `to_arrays(trace_data)` first, pass `trace_arrays`.

---

### Phase 3: Cross-Cutting Issues

1. **API divergence (main risk for Chunk 0):** `OctopusMemPoolEnv` moved to TraceArrays but tests still use old dict API. `precompute_pod_events` in data.py also still uses old VM-object API — this is intentional for now (Chunk 2 adds `precompute_pod_events_arrays`).

2. **Double D_j computation:** `step()` calls `_compute_D_j()` (Python loops); `_get_obs()` has its own identical Python loop. `_cached_D_j` was the intended bridge but is never populated. After Chunk 1 vectorizes both, the redundancy is less costly, but for Chunk 3 (Numba), only one JIT'd version should exist.

3. **Multi-trace cache miss:** `_switch_trace()` swaps trace_arrays but not `_precomputed_events`. Every reset in multi-trace mode falls through to `_build_events()` (~120 ms). Chunk 2 fix.

4. **Most likely bug:** `make_rl_alloc_cb` test (`test_obs_sync_initial_step`) will fail with AttributeError after constructor fix — requires a second fix to the mock.

5. **New engineer warning:** The `_cached_D_j` field looks like live code but is dead. Don't try to use or populate it before Chunk 1 removes it.

---

### Cross-reference: Plan vs Codebase

| Plan item | Status |
|---|---|
| Chunk 0: fix 23 failing tests | NOT DONE — all TypeError from old constructor API |
| Chunk 1: flat `_mpd_dt/_mpd_mem/_mpd_n` arrays | NOT DONE — `mpd_vm_allocs` is still list-of-lists |
| Chunk 1: vectorize `_compute_D_j` | NOT DONE — still nested Python loops |
| Chunk 1: vectorize `_get_obs` A/B | NOT DONE — still nested Python loops |
| Chunk 1: vectorize `_process_departures_through` | NOT DONE — still list-comp per MPD |
| Chunk 1: remove `_cached_D_j` | NOT DONE — field still declared in reset() |
| Chunk 2: `precompute_pod_events_arrays(TraceArrays, ...)` | NOT DONE — function doesn't exist |
| Chunk 2: `_switch_trace` swaps cache | NOT DONE — only swaps trace_arrays |
| Chunk 2: train_rl.py builds per-trace cache | NOT DONE |
| Chunk 3: Numba JIT | NOT DONE |
| ARCH.md | Already documents future state (forward-looking) |

No ambiguities that require a stop — all open questions in SPEEDUP_PLAN §12 are resolved.
