# Async Eval Worker — Tasks (async-plan)

Source: `docs/plans/v3/async-plan.md`

## In Progress

- [ ] Add `_eval_worker_main` top-level function to `scripts/train_rl.py`
- [ ] Add async fields to `PoolingSavingsCallback.__init__` (`_worker`, `_cmd_q`, `_res_q`, `_pending_step`, `_eval_device`)
- [ ] Add `eval_device=None` parameter to `PoolingSavingsCallback.__init__`
- [ ] Implement `PoolingSavingsCallback.on_training_start()`
- [ ] Replace `PoolingSavingsCallback._on_step()` with async version
- [ ] Implement `PoolingSavingsCallback.on_training_end()`
- [ ] Delete `PoolingSavingsCallback._run_eval()`
- [ ] Add `--eval-device` CLI arg to `main()` (default `cuda:1`)
- [ ] Add `mp.set_start_method("spawn", force=True)` at top of `main()`
- [ ] Wire `args.eval_device or None` → `PoolingSavingsCallback(eval_device=...)`
- [ ] Write `tests/test_async_eval.py` (worker lifecycle + result roundtrip)

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
