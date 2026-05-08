# Reward Ablation — Tasks (docs/plans/v4/reward-ablation-plan.md)

Active plan. All chunks below are sequentially dependent.

## Chunk 0 — Rename A→R2, B→R3 + add R1–R5 to env/train/eval

### octopus/env.py
- [ ] Add `_VARIANT_ALIASES = {"A": "R2", "B": "R3"}` normalisation BEFORE assert
- [ ] Extend assert to `("current", "R1", "R2", "R3", "R4", "R5")`
- [ ] Replace `reward_variant == "A"` → `== "R2"`, `== "B"` → `== "R3"` in reward block
- [ ] Update obs-dim branch: `_SIMPLE_OBS = {"current", "R1"}`; simple if in _SIMPLE_OBS
- [ ] Update D_j_pre condition: compute only if `reward_variant not in {"current", "R1"}`

### scripts/train_rl.py
- [ ] Extend `choices` for `--reward-variant` to include R1–R5 (keep A, B deprecated)
- [ ] Update precompute condition: `not in ("current", "R1", "A")`
- [ ] Update async worker obs-dim and track_vm_allocs: use `_SIMPLE_OBS = {"current", "R1"}`
- [ ] Update async worker obs-construction: R1 takes simple-obs path

### scripts/eval_rl.py
- [ ] Extend `choices` for `--reward-variant` to include R1–R5
- [ ] Update mhd_to_hosts/Q_j precompute condition: `not in {"current", "R1"}`

### scripts/evaluate.py  ← plan gap, must fix too
- [ ] Update `make_rl_alloc_cb` `_obs_dim`: use `not in {"current", "R1"}` for rich obs
- [ ] Update `make_rl_alloc_cb` obs-construction branch: R1 takes simple-obs path

## Chunk 1 — Add R1

### octopus/env.py
- [ ] Add `elif self.reward_variant == "R1": reward = -new_peak / norm` in reward block

## Chunk 2 — Add R4 (sparse terminal)

### octopus/env.py
- [ ] Hoist `_pooling_ratio/_pooling_savings` computation before reward block
- [ ] Add `elif self.reward_variant == "R4": reward = _pooling_savings if done else 0.0`
- [ ] Replace inline formula in `if done:` info block with pre-computed vars

## Chunk 3 — Add R5 (R4 + PBRS + Sub-episodes)

### octopus/env.py
- [ ] Add `sub_episode_len: int = 576` and `pbrs_gamma: float = 0.99` to `__init__`
- [ ] Add `self._sub_peak = 0.0`, `self._sub_step = 0`, `self._prev_potential = 0.0` to `reset()`
- [ ] Implement R5 reward block in `step()`

### scripts/train_rl.py
- [ ] Add `--sub-episode-len INT` default 576
- [ ] Add `--pbrs-gamma FLOAT` default 0.99
- [ ] Pass both through `_make_env` lambda → `OctopusMemPoolEnv`

## Chunk 4 — Pipeline correctness tests

### tests/test_pipeline_correctness.py (new file)
- [ ] Test 1: Mass conservation — `sum(cur_cxl_mem_vec) == sum(live allocations)`
- [ ] Test 2: Deallocation clears memory exactly
- [ ] Test 3: `pooling_ratio` formula verification
- [ ] Test 4: Eval pipeline vs env parity (highest priority)
- [ ] Test 5: R4 intermediate rewards are zero
- [ ] Test 6: R5 sub-peak resets, MPD loads do not
- [ ] Test 7: R5 PBRS telescoping identity
- [ ] Test 8: Obs reflects actual env state
- [ ] Test 9: `greedy_alloc` mutation regression
- [ ] Test 10: R1 reward non-positive and bounded

---

# JAX SAC Trainer — Tasks (docs/plans/v4/jax-plan.md)

**Gate met:** SPEEDUP_PLAN Chunks 0-3 are all complete (all 145 tests pass, 2026-05-01).

## Phase 0 — Dependencies

- [x] Add `[jax]` optional extras to `pyproject.toml`: `jax>=0.4.25`, `jaxlib>=0.4.25`, `flax>=0.8.0`, `optax>=0.2.2`, `chex>=0.1.86`
- [x] Confirm JAX CPU-mode: installed jax==0.6.2, jaxlib==0.6.2; `jax.devices()` → CpuDevice(id=0)
- [x] Create `requirements-jax.txt` with pinned versions (jax==0.6.2, jaxlib==0.6.2, flax==0.10.7, optax==0.2.8, chex==0.1.90, orbax-checkpoint==0.11.37)

## Phase 1 — JAX environment contract (pure functions)

**1.0 Pre-implementation: verify MAX_ACTIVE_VMS safety for append-only semantics** — RESOLVED
- [x] Analysis complete: max total VM allocations per MPD per episode = **2007** (LVL01 trace, seed=2).
  MAX_ACTIVE_VMS=256 would overflow. **Resolution: Option A — compute D_j/S_j from dealloc_buf
  directly. `_mpd_dt/_mpd_mem/_mpd_n` are NOT in JAX state. D_j/S_j computed via
  `jnp.einsum('t,tj->j', weight, state.dealloc_buf)` — mathematically identical to Numba kernel.**

**1.1 OctopusState dataclass + StaticConfig**
- [ ] Create `octopus/jax/__init__.py` (empty)
- [ ] Create `octopus/jax/env.py` with:
  - `OctopusState` as `@chex.dataclass` with fields:
    `mpd_load (num_mhd,) float32`, `dealloc_buf (MAX_TICKS, num_mhd) float32`,
    `host_load (pod_size,) float32`, `host_dealloc_buf (MAX_TICKS, pod_size) float32`,
    `max_peak scalar float32`, `event_idx scalar int32`,
    `last_depart_tick scalar int32`, `key (JAX PRNG)`
  - `StaticConfig` frozen dataclass: `events (MAX_EVENTS, 4) int32`,
    `host_to_mhds (pod_size, max_degree) int32`, `n_accessible (pod_size,) int32`,
    `mhd_to_hosts (num_mhd, max_conn) int32`, `n_connected (num_mhd,) int32`,
    `Q_j (num_mhd,) float32`, `pod_dram float32`, `reward_variant int32`,
    `lookahead_window int32`, `reward_lambda float32`,
    `base_time_sec int64`, `pod_start_ts int32`, `num_events int32`
  - `_compute_D_S_j(dealloc_buf, tick, W)` helper using einsum over dealloc_buf

**1.2 reset_fn**
- [ ] Implement `reset_fn(key, static_config) → (OctopusState, obs)`
- Host path: call `_generate_pod(seed)` + `_build_events()` on CPU, pad to `(MAX_EVENTS, 4)`, `jax.device_put`
- Device path: zero state, call `step_fn` departure-processing up to first tick

**1.3 step_fn**
- [ ] Implement `step_fn(state, action, static) → (OctopusState, obs, reward, done)`
- Softmax over accessible MPDs with mask (mask inaccessible logits)
- Scatter alloc into `dealloc_buf.at[clip(dealloc_tick, 0, MAX_TICKS-1), mhd_list].add(alloc_gb)`
- Update `mpd_load`, `host_load`, `host_dealloc_buf`
- Batch-subtract departures: `delta = jnp.einsum('t,tj->j', tick_mask, dealloc_buf)`
- D_j/S_j via einsum over `dealloc_buf` (see CLAUDE.md JAX Plan Conventions for formula)
- Separate JIT-compiled kernel per reward variant (no speculative compute)
- Advance `event_idx`; set `done = event_idx >= num_events`

**1.4 _get_obs_fn**
- [ ] Implement observation function matching SB3 `_get_obs()` layout exactly
- `current`: `2*max_degree + 4` — loads norm, mask, vm_norm, peak_norm, hour_sin, hour_cos
- `A/B`: `6*max_degree + 2` — per-MPD (c_j, D_j, S_j, mask, P_j, Q_j) + global

**1.5 Parity tests** (gate for Phase 2)
- [x] Create `tests/test_jax_env_parity.py`
- [x] `test_jax_step_matches_gym_current`: 200 steps, obs allclose atol=1e-5, reward atol=1e-6
- [x] `test_jax_step_matches_gym_A`: same for variant A
- [x] `test_jax_step_matches_gym_B`: same for variant B
- [x] `test_reset_deterministic`: same seed → same first obs
- [x] `test_jit_step_shapes` (via existing `test_jit_step_compiles_and_runs` + parity fixtures): no recompilation / shapes consistent
- [x] All run under `jax.config.update("jax_enable_x64", True)`

## Phase 2 — Vectorized rollout (no SAC yet)

- [x] Implement rollout using `jax.lax.scan` over horizon H (`rollout_fn` / `make_rollout_fn`)
- [x] `jax.vmap` over `n_envs` for parallel collectors (`make_rollout_fn`)
- [x] Benchmark env-only steps/sec vs SubprocVecEnv at matched n_envs (`scripts/benchmark_jax_env.py`)
- [x] Gate: no Python in hot per-step path (verify with `jax.make_jaxpr` in tests)

## Phase 3 — JAX SAC

- [x] MLP actor, twin critics, learned log α in `octopus/jax/sac.py`
- [x] Host-side circular NumPy replay buffer
- [x] JIT-compiled critic + actor + α losses; soft target update
- [x] `scripts/train_jax.py` with CLI flags: `--run-id`, `--seed`, `--n-envs`, `--total-timesteps`, `--reward-variant`, `--lookahead-window`, `--reward-lambda`, `--trace`
- [x] Checkpoints via Orbax or pickle; `config.json` includes `trainer: "jax"` (pickle path implemented)
- [x] Gate: 50k-step smoke with stable losses, nonzero grad norms

## Phase 4 — Fair comparison

- [x] Aligned config.json fields documented in jax-plan.md §4
- [x] Wall time, env samples/sec comparison vs SB3 (50k matched snapshot logged in `docs/plans/v4/throughput_log.md`)

## Phase 5 — Deferred backlog

- [ ] Multi-trace training (two options documented in jax-plan.md §5)
- [ ] Augmentation on device
- [ ] Async eval worker with JAX weights

---

# Training Throughput Speedup — COMPLETE (SPEEDUP_PLAN.md)

All chunks verified complete as of 2026-05-01 (145 tests pass).

- [x] Chunk 0: Fix 23 broken tests (constructor API drift)
- [x] Chunk 1: `_mpd_dt/_mpd_mem/_mpd_n` flat arrays + Numba D_j/S_j kernels
- [x] Chunk 2: `precompute_pod_events_arrays(TraceArrays, M, seeds)`; `_switch_trace` swaps cache
- [x] Chunk 3: `@njit(cache=True)` on `compute_D_j`, `compute_D_S_j`

---

# Async Eval Worker — Tasks (async-plan)

Source: `docs/plans/v3/async-plan.md`

## In Progress

- [x] Add `_eval_worker_main` top-level function to `scripts/train_rl.py`
- [x] Add async fields to `PoolingSavingsCallback.__init__` (`_worker`, `_cmd_q`, `_res_q`, `_pending_step`, `_eval_device`)
- [x] Add `eval_device=None` parameter to `PoolingSavingsCallback.__init__`
- [x] Implement `PoolingSavingsCallback.on_training_start()`
- [x] Replace `PoolingSavingsCallback._on_step()` with async version
- [x] Implement `PoolingSavingsCallback.on_training_end()`
- [x] Delete `PoolingSavingsCallback._run_eval()`
- [x] Add `--eval-device` CLI arg to `main()` (default `cuda:1`)
- [x] Add `mp.set_start_method("spawn", force=True)` at top of `main()`
- [x] Wire `args.eval_device or None` → `PoolingSavingsCallback(eval_device=...)`
- [x] Write `tests/test_async_eval.py` (worker lifecycle + result roundtrip)

---

# Training, Ablations & Evaluation — Tasks (plan-v2)

Source: `docs/plans/v3/plan-v2.md`

## Completed

- [x] Task 1: Trace characterization + train/val/test split assignment
- [x] Task 2: W&B integration (wandb.init, WandbCallback, SACDiagnosticsCallback, AugmentationLogCallback, EnvMetricsCallback)
- [x] Task 3: Pre-training sanity checks (--no-train random baseline, overfit test, reward scale, aug smoke)
- [x] `--skip-hotfix` flag in train_rl.py and eval_rl.py
- [x] `--n-envs` + SubprocVecEnv in train_rl.py
- [x] `--traces` (multi-trace per-env) in train_rl.py

## Task 4 — New State Space & Rewards (next up)

**4a — Per-MPD VM tracking + host CXL load tracking (octopus/env.py)**
- [ ] Add `mpd_vm_allocs: list[list[tuple[int, float]]]` — per-MPD list of (dealloc_tick, mem_gb)
      Init in `reset()`. Update in `step()` on alloc. Purge in `_process_departures_through()`.
- [ ] Add `cur_host_cxl_load = np.zeros(pod_size)` and `host_dealloc_events = np.zeros((pod_dur, pod_size))`
      Reset in `reset()`. Update in `step()`. Drain in `_process_departures_through()`.

**4b — Static topology precomputation (octopus/env.py)**
- [ ] Add `mhd_to_hosts: dict[int, list[int]]` — inverse of host_to_mhds
- [ ] Add `Q_j: np.ndarray` — neighbor scarcity: `Q_j = mean(1/deg(h) for h in mhd_to_hosts[j])`
- [ ] Extract `_recompute_topology_derived()` — builds mhd_to_hosts + Q_j
      Call from `__init__` and `_apply_augmentation` (so Q_j is correct after link failures)

**4c — New reward functions (octopus/env.py step())**
- [ ] Add constructor params: `reward_variant="current"`, `lookahead_window=200`, `reward_lambda=0.2`
- [ ] Implement D_j(t, W) computation — iterate mpd_vm_allocs[j] for VMs ending within W steps
      Cache D_j array for reuse in `_get_obs()`
- [ ] Dispatch on reward_variant:
      - "current": unchanged `-(Δpeak)/fair_share - λ·Var(loads)`
      - "A": `-max(ĉ_j+(t) for j ∈ N(i))` where `ĉ_j = (c_j - D_j) / D_pod`
      - "B": `R_A - λ · max(ĉ_j(t) for j ∉ N(i))`

**4d — New 50-dim observation space (octopus/env.py _get_obs())**
- [ ] When reward_variant != "current": build 6·d_max + 2 obs
      Per-MPD slot k: [c_j/D_pod, D_j/D_pod, S_j/D_pod, mask, P_j/D_pod, Q_j]
      Global: [global_peak/D_pod, vm_mem/D_pod]
- [ ] Update `observation_space` shape in __init__ when reward_variant != "current"

**4e — CLI flags (scripts/train_rl.py)**
- [x] Add `--reward-variant {current,A,B}` default "current"
- [x] Add `--lookahead-window` type=int default=200
- [x] Add `--reward-lambda` type=float default=0.2
- [x] Pass all three to `_make_env()` → `OctopusMemPoolEnv()`
- [x] Update PoolingSavingsCallback to pass obs_variant + mhd_to_hosts + Q_j to make_rl_alloc_cb

**4f — Eval script updates**
- [x] `scripts/evaluate.py` `make_rl_alloc_cb()`: accept `obs_variant`, `mhd_to_hosts`, `Q_j`
      Build 50-dim obs when variant != "current" using ctx-injected state
- [x] `scripts/evaluate.py` `pooling_simulation()`: add `track_vm_allocs=False` param
      When True: maintain `mpd_vm_allocs` + `host_cxl_load` in sim loop; inject into ctx dict
- [x] `scripts/eval_rl.py`: accept `--reward-variant` flag, compute mhd_to_hosts + Q_j from M,
      pass to make_rl_alloc_cb + pooling_simulation

**4g — Tests (tests/test_new_reward_obs.py)**
- [x] D_j and S_j computation with known VM sets (unit test, no trace needed)
- [x] Obs shape = 50 for new variants, 20 for "current"
- [x] Reward A ∈ (-1, 0], Reward B ∈ (-(1+λ), 0]
- [x] `reward_variant="current"` produces identical trajectories to old code (regression)
- [x] Augmentation correctly recomputes Q_j and mhd_to_hosts after link failures

## Task 5 — Pre-Ablation Timing (human-run after Task 4)

- [ ] Run timing tests for reward variants current / A / B (10k steps, n_envs=32)
- [ ] Compute max_timesteps budget for 6-hour overnight run

## Task 6 — Overnight Experiments (human-run)

- [ ] GPU 0: exp_baseline_v4 (reward=current, 20-dim obs)
- [ ] GPU 1: exp_rewardA_v4 (reward=A, 50-dim obs)
- [ ] GPU 2: exp_rewardB_v4 (reward=B, 50-dim obs)

## Task 7 — Post-Overnight Evaluation (human-run)

- [ ] Eval all 3 models on val trace (50 iterations)
- [ ] Comparison table: pooling_ratio, savings, SAC stability metrics
- [ ] Decision point: extend winner, tune W/λ, or ablate further

## Completed (plan-v1)

- [x] AugmentationConfig dataclass
- [x] 6 transform functions in augmentation.py
- [x] apply_augmentation() pipeline
- [x] Env integration (_apply_augmentation, _switch_trace)
- [x] Augmentation CLI args in train_rl.py
- [x] AugmentationLogCallback
- [x] Multi-trace pool loading
- [x] 49 augmentation tests
