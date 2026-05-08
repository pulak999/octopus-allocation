# JAX Plan — Chunk Progress Log

Source plan: `docs/plans/v4/jax-plan.md`
Implementation order: A → B → C → D → E (throughput-first; parity gate after env, before SAC)

---

## Chunk Map

| Chunk | Contents | Status | Tests |
|-------|----------|--------|-------|
| A | `octopus/jax/__init__.py` + `octopus/jax/env.py` skeleton: `OctopusState`, `StaticConfig`, `_compute_D_S_j` helper | **DONE** ✓ 149 tests pass | `test_jax_env_parity.py` 4 D_j/S_j unit tests |
| B | `reset_fn` + `step_fn` (all 3 reward variants; smoke tests) | **DONE** ✓ 155 tests pass | `test_jax_env_parity.py` 6 smoke tests |
| C | `lax.scan` rollout + `jax.vmap` over n_envs; env-only benchmark vs SB3 2,757 fps | **DONE** ✓ scan smoke + benchmark | `test_jax_env_parity.py` scan tests + `scripts/benchmark_jax_env.py` |
| D | 5 parity tests (float64 mode) — gate before SAC | **DONE** ✓ reset + step obs/reward parity | `test_jax_env_parity.py` parity tests |
| E | JAX SAC (`octopus/jax/sac.py`); `scripts/train_jax.py`; measure new SB3 training baseline first, then compare JAX training fps | **DONE** ✓ 50k smoke + matched SB3 compare | `scripts/train_jax.py` + run artifacts |

---

## Key Decisions (for fresh sessions)

- **No `_mpd_dt/_mpd_mem/_mpd_n` in OctopusState**: MAX_ACTIVE_VMS=256 overflows (max 2007 total VM allocs/MPD/episode in LVL01 trace). D_j/S_j computed from `dealloc_buf` via `jnp.einsum`. Mathematically identical to Numba kernel.
- **D_j formula**: `weight[t] = max(0, 1-(t-tick)/W)` for `t ∈ (last_depart_tick, tick+W]`; `D_j = einsum('t,tj->j', weight, dealloc_buf)`. S_j = sum over `t > tick+W`.
- **`dealloc_tick` clipping**: clip to `MAX_TICKS-1` before scatter (VMs ending beyond trace window).
- **MAX_TICKS = 2304**, **MAX_EVENTS = 4096** (empirical max 2407 events/episode over 50 seeds).
- **JAX GPU now enabled** (`jax-cuda12-plugin` installed in venv). Training runs use `backend=gpu` on TITAN RTX.
- **Parity tests use `jax_enable_x64=True`**. Production training uses float32.
- **Pod selection on host** (Python stdlib `random.shuffle`) — not JAX PRNG. JAX PRNG key in state is for future use.
- **SB3 baselines**: env-only 2,757 fps (n_envs=32); full training fps not yet measured for new setup (53 fps was pre-SPEEDUP_PLAN with n_envs=1). Chunk E measures new SB3 training baseline before implementing SAC.

---

## Chunk A — `OctopusState` + `StaticConfig` + `_compute_D_S_j`

**Status: COMPLETE (2026-05-05) — 149 tests pass**

### Files to create
- `octopus/jax/__init__.py` — empty
- `octopus/jax/env.py` — skeleton

### Files to create (tests)
- `tests/test_jax_env_parity.py` — starts with D_j/S_j unit tests; parity tests added in Chunk D

### Constants
- `MAX_TICKS = 2304`
- `MAX_EVENTS = 4096`
- `_REWARD_CURRENT = 0`, `_REWARD_A = 1`, `_REWARD_B = 2`

### OctopusState fields
```
mpd_load          (num_mhd,) float
dealloc_buf       (MAX_TICKS, num_mhd) float
host_load         (pod_size,) float
host_dealloc_buf  (MAX_TICKS, pod_size) float
max_peak          scalar float
event_idx         scalar int32
last_depart_tick  scalar int32
key               (2,) uint32
```

### StaticConfig fields
```
events          (MAX_EVENTS, 4) float32  [tick, pod_id, vm_mem, dealloc_tick]
host_to_mhds    (pod_size, max_degree) int32, pad=-1
n_accessible    (pod_size,) int32
mhd_to_hosts    (num_mhd, max_conn) int32, pad=-1
n_connected     (num_mhd,) int32
Q_j             (num_mhd,) float32
pod_dram        float
num_events      int
reward_variant  int
lookahead_window int
reward_lambda   float
pod_size        int
num_mhd         int
max_degree      int
max_conn        int
base_time_sec   int  (seconds since _EPOCH; for hour-of-day obs feature)
pod_start_ts    int
```

### Test plan (Chunk A)
`tests/test_jax_env_parity.py`:
- `test_dsj_empty` — zeros dealloc_buf → D_j = S_j = 0
- `test_dsj_matches_numba` — build dealloc_buf from known _mpd_dt/_mpd_mem state; compare D_j/S_j against Numba kernel
- `test_dsj_weight_at_edge` — VM at t=tick+W → weight=0, not in S_j
- `test_dsj_excludes_past` — entries at t ≤ last_depart_tick excluded

---

## Chunk B — `reset_fn` + `step_fn`

**Status: COMPLETE (2026-05-05) — 155 tests pass**

### Files changed
- `octopus/jax/env.py` — all additions (no new files)

### New functions

**Host-side (pure Python/NumPy):**
- `_host_generate_pod(seed, trace_arrays, pod_size)` — stdlib random.seed+shuffle, identical to SB3 `_generate_pod`
- `_host_build_events(trace_arrays, pod_indices, pod_size, skip_hotfix)` — mirrors SB3 `_build_events`
- `build_static_config(events, pod_dram, base_time, pod_start_ts, M, ...)` — pads arrays and builds `StaticConfig`
- `make_episode(seed, trace_arrays, M, *, ...)` — top-level: calls host builders + `build_static_config`

**JAX-side (JIT-compilable):**
- `_process_departures(state, next_tick)` — einsum batch-subtract over `(last_depart_tick, next_tick]`; updates loads + pointer
- `_get_obs_fn(state, static_config)` — both "current" (2·md+4) and "A/B" (6·md+2) obs; returns zeros at done
- `reset_fn(static_config)` — zero-init state, process departures to first tick, return `(OctopusState, obs)`
- `step_fn(state, action, static_config)` — softmax alloc → reward (all 3 variants) → advance → departures → obs

### StaticConfig changes vs Chunk A
- Added `variance_lambda: float` (default 0.5; "current" reward)
- Added `__hash__` / `__eq__` via `id(self)` — enables `jax.jit(step_fn, static_argnums=2)`

### Key implementation notes
- **OOB dealloc_tick**: VMs with `dealloc_tick_raw >= MAX_TICKS` contribute 0 to `dealloc_buf` (matching SB3 which skips them in `dealloc_events`). Their `mpd_load` contribution stays allocated forever.
- **Reward B reachable mask**: built via `jnp.any((accessible[:,None] == mpd_ids[None,:]) & valid_mask[:,None], axis=0)` — avoids the spurious zeroing bug from `.set()` when invalid pad slots map to index 0.
- **Static if/elif on reward_variant**: Python control flow resolved at JIT trace time since `reward_variant` is a static scalar in `StaticConfig`.

### Tests added (6 new smoke tests in `test_jax_env_parity.py`)
- `test_make_episode_builds_valid_config`
- `test_reset_returns_correct_shapes`
- `test_step_returns_correct_shapes_and_terminates`
- `test_step_reward_variant_A`
- `test_step_reward_variant_B`
- `test_jit_step_compiles_and_runs`

---

## Chunk C — `lax.scan` rollout + benchmark

**Status: DONE (2026-05-05)**

### Benchmark run (env-only, random actions; CPU-only JAX)
- Script: `scripts/benchmark_jax_env.py`
- Command:
  `python scripts/benchmark_jax_env.py --n-envs 32 --horizon 256 --n-iters 10`
- Defaults used: trace=`AMS20PrdApp19-tround.sqlite`, topology=`AG16x6_expander_quads_r5_sym_fixed.csv`,
  reward_variant=`current`, lookahead_window=`200`, reward_lambda=`0.2`, seed=`0`, skip_hotfix=`False`
- Device: `cpu` (CUDA-enabled jaxlib not installed)

**Throughput:**
- mean: `4,152 fps`
- median: `4,225 fps`
- SB3 baseline (from SPEEDUP_PLAN): `2,757 fps` at `n_envs=32` random
- ratio: `1.53x` (JAX beats SB3)

---

## Chunk D — 5 parity tests

**Status: DONE (2026-05-05)**

### Parity verification
- Float64 mode: `jax_enable_x64=True`
- Gymnasium env: `octopus.env.OctopusMemPoolEnv`
- Compared (synthetic trace) for:
  - reset obs determinism (`seed` → first obs)
  - step obs + reward + done for `reward_variant in {current, A, B}` (up to 200 steps)

### Test status
- `pytest tests/test_jax_env_parity.py` → **17 passed**

*(Fill in after Chunk C is complete)*

---

## Chunk E — JAX SAC + training benchmark

**Status: DONE (2026-05-05)**

### Implemented
- `octopus/jax/sac.py`:
  - MLP actor + twin critics + learned `log_alpha`
  - host-side ring replay buffer
  - JIT-compiled SAC update (`critic`, `actor`, `alpha`, soft target update)
  - gradient-norm metrics (`critic_grad_norm`, `actor_grad_norm`, `alpha_grad_norm`)
- `scripts/train_jax.py`:
  - CLI subset: `--run-id --seed --n-envs --total-timesteps --reward-variant --lookahead-window --reward-lambda --trace`
  - checkpointing via pickle to `output/checkpoints/<run_id>/step_*.pkl`
  - `config.json` includes `trainer: "jax"` and Phase 4 parity fields (`batch_size`, `buffer_size`, `gamma`, `tau`, LRs, `effective_env_steps_per_update`, `learning_starts`)
  - single-trace v1 path (`reward_variant=current` currently enabled)

### Phase 3 gate evidence (50k smoke)
- Run: `jax-phase3-gate-50k`
- Command:
  `python scripts/train_jax.py --run-id jax-phase3-gate-50k --total-timesteps 50000 --n-envs 32 --learning-starts 2000 --train-freq 32 --gradient-steps 1 --batch-size 256`
- Result: `steps=50016`, `elapsed=72.8s`, end-to-end `~687 fps`
- Stability: finite losses and **nonzero grad norms** through training, e.g.
  - step 20k: `gN(c/a/alpha)=1.794/2.217/11.448`
  - step 40k: `gN(c/a/alpha)=4.046/1.905/9.659`

### Phase 4 matched comparison snapshot (50k, same reward/settings)
- JAX run (`jax-phase3-gate-50k`): `50016 / 72.8s ≈ 687 fps`
- SB3 run (`sb3-phase4-50k`): `50176 / 58s ≈ 865 fps` (from `train_rl.py` wall-time log)
- Comparison at this checkpoint: SB3 is `~1.26x` faster end-to-end for this 50k setup.
