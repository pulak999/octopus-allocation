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

---

## Code Review — commit 34a41ec506b9281bafbfcd919ecddbe4c0aeb012 (2026-05-01) [CORRECTED]

Scope: JAX plan (docs/plans/v4/jax-plan.md) — full codebase re-read.
Reviewed: octopus/env.py, octopus/data.py, octopus/kernels.py, scripts/train_rl.py,
pyproject.toml, TODO.md, all test files. 145 tests confirmed passing.

---

### Phase 1: Repo Overview

**Purpose:** RL agent (SAC via SB3) allocating VM memory across CXL MPDs to minimize peak
load. Trains on replayed Azure traces via Gymnasium env.

**Structure:**
- `octopus/` — env.py, data.py, kernels.py (Numba JIT), augmentation.py, baselines.py, topology.py
- `scripts/` — train_rl.py, evaluate.py, eval_rl.py, various analysis scripts
- `tests/` — 145 pytest tests, all passing; no CI (local only)
- `data/traces/` — 10 Azure trace pickles (not in git)
- `data/topologies/` — 5 topology CSVs
- `lcpo/` — unrelated external code, not part of the RL system

**Entry points:** `scripts/train_rl.py main()` for training; `scripts/evaluate.py pooling_simulation()` for eval.

**Key data flow:**
```
load_trace() → to_arrays() → TraceArrays (numpy CSR)
OctopusMemPoolEnv(trace_arrays, M) → reset(seed) → step(action) loop
```

**Key data structures:**
- `TraceArrays`: numpy CSR (vm_start int64, vm_end int64, vm_mem float32, node_ids int32, node_dram float32, node_offsets int32, vm_ptrs int32)
- `events: list[tuple(tick:int, pod_id:int, vm_mem:float, dealloc_tick:int)]`
- `_mpd_dt (num_mhd, capacity) float64` — dealloc ticks with inf sentinel
- `_mpd_mem (num_mhd, capacity) float64` — mem GB per slot
- `_mpd_n (num_mhd,) int32` — valid VM count per MPD after compaction

---

### Phase 2: File-by-File

**octopus/env.py**

Single responsibility: Gymnasium env. TraceArrays + M in, (obs, reward) per step out.

Key functions:
- `reset(seed)`: pod → events → augmentation → zero state → process departures to first tick
- `step(action)`: softmax alloc → reward → advance event_idx → call `_process_departures_through` → `_get_obs()`
- `_process_departures_through(tick)`: iterates ticks one-by-one via Python `range()`, applies numpy row-subtract, then compacts `_mpd_dt/_mpd_mem` via boolean mask per MPD
- `_compute_D_j(tick)`: delegates to `_nb_compute_D_j` Numba kernel — vectorized, fast
- `_get_obs()` A/B variant: calls `_nb_compute_D_S_j` Numba kernel — vectorized, fast
- `_ensure_mpd_capacity(j)`: doubles capacity of all MPD arrays when MPD j overflows

Surprises:
- **`_mpd_dt` uses float64 + inf sentinel** (not int32). `_mpd_n` is the *valid* count after compaction — compaction runs in `_process_departures_through`. JAX plan specifies int32 and append-only (no compaction) — this is a **semantic difference requiring care in parity tests**.
- `dealloc_events` shape is `(pod_dur, num_mhd)` — episode-dynamic. JAX must use static `(MAX_TICKS=2304, num_mhd)`.
- `host_dealloc_events` shape is `(pod_dur, pod_size)` — episode-dynamic. JAX uses static `(MAX_TICKS, pod_size)`.
- `dealloc_tick < self.pod_dur` check before scheduling dealloc: VMs with dealloc_tick ≥ pod_dur are not scheduled. JAX with fixed 2304-tick buffer must replicate this correctly.

SPEEDUP_PLAN status: **ALL CHUNKS COMPLETE** (verified by reading the code):
- Chunk 0: All 145 tests pass — constructor API fixed.
- Chunk 1: `_mpd_dt/_mpd_mem/_mpd_n` implemented; Numba kernels in kernels.py.
- Chunk 2: `precompute_pod_events_arrays` in data.py.
- Chunk 3: `@njit(cache=True)` on `compute_D_j` and `compute_D_S_j`.

**octopus/data.py**

- `TraceArrays`: All fields except `vm_mem` and `node_dram` use int64/int32. `vm_mem` is float32.
- `precompute_pod_events_arrays(trace_arrays, M, seeds)`: implemented — instantiates a throw-away env and calls `_generate_pod` + `_build_events` for each seed.
- `_generate_pod` uses `_random.seed(seed); _random.shuffle(indices)` — Python stdlib RNG. NOT reproducible via JAX PRNG. JAX plan correctly handles this by calling these on host.

**octopus/kernels.py**

- `compute_D_j`: `mpd_dt` is float64 (not int32). Loop condition: `dt <= t_plus_W` (float compare with inf sentinel naturally excludes empty slots since inf > any tick+W).
- `compute_D_S_j`: same semantics.
- JAX parity: JAX will use `valid = arange(256) < _mpd_n` + separate `_mpd_dt > tick` mask. SB3 relies on compaction making all slots 0..n-1 valid and none expired. These produce identical D_j/S_j IF parity tests run with the same episode state — but the intermediate state representation differs.

**pyproject.toml**

No JAX dependencies yet. Need to add `jax`, `jaxlib`, `flax` or `equinox`, `optax`, `chex`
as optional extras `[jax]`.

---

### Phase 3: Cross-Cutting Issues

1. **Append-only vs. compacting `_mpd_n`**: SB3 `_mpd_n` = valid count AFTER compaction. JAX plan `_mpd_n` = append pointer BEFORE compaction. These produce the same D_j/S_j over a trajectory only if the masking is correct. Risk: if a parity test feeds the same actions but uses JAX's append-only state, the `valid` mask includes expired slots that SB3 compacted away. This is fine because `_mpd_dt > tick` excludes expired slots regardless — but the `_mpd_n` values will diverge after the first departure event.

2. **MAX_ACTIVE_VMS=256 overflow risk (append-only)**: Plan sets this to 256, citing "empirical max 131 simultaneous". But with append-only and no compaction, `_mpd_n[j]` = total VMs ever allocated to MPD j during the entire episode, not just simultaneous. Over a 2304-tick episode with continuous arrivals and departures, this could easily exceed 256. This is the **single highest-risk design decision** in the JAX plan and needs verification against trace data before Phase 1.

3. **`dealloc_tick >= pod_dur` handling**: SB3 skips scheduling dealloc for these VMs. In JAX with a fixed-size dealloc_buf, an out-of-bounds write would be silently ignored by JAX (index clipping) or cause an error. The plan needs explicit handling (clip dealloc_tick to MAX_TICKS-1 or use `jnp.where(dealloc_tick < MAX_TICKS, ...)`).

4. **No octopus/jax/ module exists**: Must be created from scratch. No existing JAX infrastructure.

5. **Most likely parity failure**: float32/float64 mismatch. SB3 uses float64 throughout. JAX defaults to float32. Parity tests must use `jax_enable_x64=True`. Production training uses float32 separately.

---

### Cross-reference: JAX Plan vs Codebase (updated 2026-05-05)

| JAX Plan item | Codebase status |
|---|---|
| Prerequisites: Chunks 0-1 merged | **DONE** — all 145 tests pass, `_mpd_dt/_mpd_mem/_mpd_n` in env.py |
| Chunk 2 (`precompute_pod_events_arrays`) | **DONE** — in data.py |
| `octopus/jax/` module | Does NOT exist — create from scratch |
| JAX `[jax]` optional extras in pyproject.toml | **DONE** — jax==0.6.2, jaxlib==0.6.2, flax==0.10.7, optax==0.2.8, chex==0.1.90 installed |
| `requirements-jax.txt` pinned versions | NOT created yet |
| `_mpd_dt/_mpd_mem/_mpd_n` in JAX OctopusState | **RESOLVED (2026-05-05)** — these are NOT in JAX state. D_j/S_j computed from `dealloc_buf` via einsum. MAX_ACTIVE_VMS=256 would overflow (max 2007 total allocs per MPD per episode observed). |
| D_j/S_j JAX formula | `jnp.einsum('t,tj->j', weight, dealloc_buf)` where `weight[t] = max(0, 1-(t-tick)/W)` for `t > last_depart_tick` and `t <= tick+W`. Mathematically identical to Numba kernel since `dealloc_buf[t,j]` = total mem of VMs departing at t on MPD j. |
| `dealloc_tick ≥ MAX_TICKS` handling | Clip to MAX_TICKS-1 with `jnp.clip(dealloc_tick, 0, MAX_TICKS-1)` before scatter |
| `host_load`/`host_dealloc_buf` in JAX state | Correct — SB3 has these |
| Float64 parity mode | Correct approach — `jax_enable_x64` for tests |

---

## Code Review — commit 34a41ec506b9281bafbfcd919ecddbe4c0aeb012 (2026-05-06)

Scope: `docs/plans/v4/reward-ablation-plan.md` — reward ablation (R1–R5), A→R2/B→R3 rename,
pipeline correctness tests.
Reviewed: octopus/env.py, octopus/baselines.py, scripts/train_rl.py, scripts/eval_rl.py,
scripts/evaluate.py, tests/test_new_reward_obs.py. Baseline: 162 tests passing.

---

### Phase 1: Repo Overview

Same structure as prior review (all SPEEDUP_PLAN and JAX plan chunks complete). The SB3
training pipeline is the target here. Key invariants to maintain:

- `reward_variant` propagates from CLI → `_make_env()` → `OctopusMemPoolEnv.__init__` → assert
- `obs_variant` propagates from CLI → `PoolingSavingsCallback` → async worker → `make_rl_alloc_cb`
- A model trained with variant X must be evaluated with the exact same obs construction

---

### Phase 2: File-by-File (reward ablation scope)

**octopus/env.py**

- Line 94: `assert reward_variant in ("current", "A", "B")` — will reject R1/R4/R5 BEFORE alias normalisation runs if normalisation is added after the assert. Must add alias normalisation BEFORE the assert.
- Lines 101–104: obs-dim branch uses `reward_variant != "current"` → gives rich obs for "A","B". After rename, R1 must also use simple obs. Condition must become `reward_variant in {"current", "R1"}`.
- Lines 246–249: `D_j_pre` computed only if `reward_variant != "current"`. R1 does not need D_j. Condition must become `reward_variant not in {"current", "R1"}`.
- Lines 275–303: Reward block. R1/R4/R5 branches need adding here.
- Lines 321–328: `pooling_savings` computed inline in `if done:` block. R4 needs this pre-computed before the reward block — must hoist above reward.
- `reset()` line 207: `self.max_peak = 0.0` — R5 needs `_sub_peak`, `_sub_step`, `_prev_potential` added here.
- `__init__` line 71: no `sub_episode_len` or `pbrs_gamma` params — must add for R5.

**scripts/train_rl.py**

- Line 515: `choices=["current", "A", "B"]` — must add R1–R5.
- Line 688: `if args.reward_variant != "current":` for mhd_to_hosts/Q_j precompute — must become `not in ("current", "R1", "A")` to skip precompute for R1 and the "A" deprecated alias.
- Line 69 (async worker): `_obs_dim = max_degree * 6 + 2 if obs_variant != "current"` — must handle R1 as simple obs.
- Line 73 (async worker): `track = obs_variant != "current"` — must handle R1 (R1 should NOT track_vm_allocs).
- Lines 83–98 (async worker): `if obs_variant == "current":` … `else:` obs path — R1 must also take the simple-obs path.
- Lines 763: `obs_variant=args.reward_variant` passed to `PoolingSavingsCallback` — fine; worker handles the logic.
- No `--sub-episode-len` or `--pbrs-gamma` args — must add for R5.

**scripts/eval_rl.py**

- Line 122: `choices=["current", "A", "B"]` — must add R1–R5.
- Line 190: `if obs_variant != "current":` for mhd_to_hosts/Q_j — must become `not in {"current", "R1"}` to skip precompute for R1.
- Calls `make_rl_alloc_cb(obs_variant=obs_variant)` — the callback needs to handle R1 as simple obs.

**scripts/evaluate.py** ← **PLAN GAP**

- Line 347: `_obs_dim = max_degree * 6 + 2 if obs_variant != "current" else max_degree * 2 + 4` — R1 must use `max_degree * 2 + 4` but this condition won't catch "R1".
- Line 356: `if obs_variant == "current":` … `else:` obs construction — R1 must take the simple-obs path.
- Line 599: `track_vm_allocs=(reward_variant != "current")` in `evaluate.py main()` — R1 should not track (uses simple obs).

**The plan (Chunk 0) only mentions `env.py`, `train_rl.py`, and `eval_rl.py`. It does NOT mention `evaluate.py`. But `evaluate.py:make_rl_alloc_cb` has the same `obs_variant != "current"` pattern and will produce wrong obs for R1.**

**tests/test_new_reward_obs.py**

- Tests currently use `reward_variant="A"` in several places — these will continue to work via alias.
- `test_reward_variant_invalid` passes `"invalid"` and expects `AssertionError` — will still pass after the assert is extended.
- `test_obs_space_shape_new_variants` checks `("A", "B")` — after rename, "A" should still work via alias.

---

### Phase 3: Cross-Cutting Issues

1. **Plan gap — evaluate.py not updated for R1**: The `make_rl_alloc_cb` in `evaluate.py` uses `obs_variant != "current"` to pick obs dimension and construction path. If a model trained with R1 is evaluated via `eval_rl.py`, `make_rl_alloc_cb` will build a 50-dim obs for a model expecting 20 dims → shape mismatch → crash. The fix is identical to what the plan does in `eval_rl.py`: use `obs_variant not in {"current", "R1"}` for the rich obs branch.

2. **Async worker in train_rl.py not covered by plan**: Lines 69, 73, and 83–98 in the async worker all use `obs_variant != "current"` and will break for R1. Must add `_SIMPLE_OBS = {"current", "R1"}` sentinel or equivalent inline check.

3. **R5: `_sub_step` resets are episode-relative**: `_sub_step` counts steps within the current sub-episode, but the plan resets `_sub_peak = 0.0` at boundaries while MPD loads carry over. The boundary condition `self._sub_step % self.sub_episode_len == 0` will fire at tick 0 (first step) since `0 % 576 == 0`. The plan handles this correctly by computing `sub_savings` even at boundary 0, but an episode of fewer than `sub_episode_len` steps will never hit the boundary — only the `done` branch fires. This is correct behavior.

4. **R4 info block**: After hoisting `_pooling_ratio/_pooling_savings`, the original inline formula in the `if done:` block must be replaced with the pre-computed values. Forgetting this creates a discrepancy between info["pooling_savings"] and the actual terminal reward.

5. **Existing tests under rename**: `test_reward_A_in_range` and `test_reward_B_in_range` in `test_new_reward_obs.py` use "A" and "B". These will still pass via alias normalization. But the obs-shape tests (`test_obs_space_shape_new_variants`) iterate `("A", "B")` — these will also still pass since aliases map to the same obs dim as R2/R3.

---

### Cross-reference: Reward-Ablation Plan vs Codebase

| Plan item | Codebase status | Notes |
|---|---|---|
| Chunk 0: alias normalisation in `env.py` | NOT DONE | Must come BEFORE assert |
| Chunk 0: extend assert to R1–R5 | NOT DONE | |
| Chunk 0: rename A→R2, B→R3 in reward block | NOT DONE | |
| Chunk 0: obs-dim `_SIMPLE_OBS = {"current","R1"}` | NOT DONE | |
| Chunk 0: `train_rl.py` choices + precompute condition | NOT DONE | |
| Chunk 0: `eval_rl.py` choices + obs-path condition | NOT DONE | |
| Chunk 0: **`evaluate.py` obs condition** | NOT IN PLAN → **gap** | Must fix `make_rl_alloc_cb` |
| Chunk 0: **async worker in `train_rl.py`** | NOT IN PLAN → **gap** | Lines 69/73/83-98 |
| Chunk 1: R1 reward branch | NOT DONE | |
| Chunk 2: hoist `_pooling_savings` | NOT DONE | |
| Chunk 2: R4 reward branch | NOT DONE | |
| Chunk 3: R5 `__init__` params | NOT DONE | |
| Chunk 3: R5 reset state | NOT DONE | |
| Chunk 3: R5 reward branch | NOT DONE | |
| Chunk 3: `train_rl.py` R5 args | NOT DONE | |
| Chunk 4: `tests/test_pipeline_correctness.py` | NOT DONE | 10 tests |
