# Adversarial Reproducibility Review (Must-Fix Now)

Scope:
- `octopus-allocation/plot_demand.py`
- `octopus-allocation/plot_memory_pooling_sweep.py`
- `octopus-allocation/train.py`

## Must-fix issues (top 3)

### A) Wrong activity gate in optimal evaluation
- **File + lines:** `plot_memory_pooling_sweep.py` (`optimal_required_capacity`, around `409`-`415`)
- **Bug/risk:** Active-load detection uses `np.max(diff[ts, :])` (instantaneous change), not `node_cxl` (current load).
- **Why it causes inconsistency/wrongness:** Ticks with sustained load but no event delta are skipped, which can underestimate required capacity and create timing-sensitive results.
- **Concrete fix:** Gate on `node_cxl` (or evaluate each window tick):
  - Replace `if float(np.max(diff[ts, :])) <= 0:` with `if float(np.max(node_cxl)) <= 0:`.

### B) Incomplete deterministic training setup
- **File + lines:** `train.py` (`68`, `72`-`75`, `147`; deterministic backend config missing)
- **Bug/risk:** Seed is passed to SB3, but explicit deterministic PyTorch/CUDA settings are missing.
- **Why it causes inconsistency:** CuDNN/kernel selection can still vary across runs and environments even with a fixed seed.
- **Concrete fix:** Before env/model creation, add:
  - `random.seed(args.seed)`, `np.random.seed(args.seed)`, `torch.manual_seed(args.seed)`, `torch.cuda.manual_seed_all(args.seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - Optional strict mode: `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`.

### C) RL checkpoint selection is timing-dependent
- **File + lines:** `plot_memory_pooling_sweep.py` (`599`-`620`, `602`-`616`)
- **Bug/risk:** Script picks whichever candidate model path exists first (with optional waiting).
- **Why it causes inconsistency:** Same command can evaluate different checkpoints depending on filesystem timing/state.
- **Concrete fix:** Require one explicit `--rl-model` path and fail fast if not present; log resolved absolute path + model hash.

## Priorities

1. **P0:** Fix `optimal_required_capacity` gating (`node_cxl` vs `diff`).
2. **P1:** Add deterministic seeding/backend config in `train.py`.
3. **P1:** Make RL model selection explicit and deterministic in sweep script.

---

## Code Review — commit 24216f8a (2026-03-24)

### Phase 1: Repo Overview

**Purpose:** RL-based CXL memory allocation for the Octopus architecture. Agent learns to
distribute VM memory across Memory Pool Devices (MPDs) connected via CXL to minimize peak
load and load variance.

**Directory structure:**
- `octopus/` — core library (env, data loading, baselines, topology, optimal solver)
- `scripts/` — training, evaluation, plotting entry points
- `tests/` — one test file (`test_greedy_alloc.py`) plus data exploration notebooks
- `data/traces/` — Azure VM trace pickles (not in git)
- `data/topologies/` — CSV adjacency matrices (5 topologies)
- `docs/` — plans, data analysis docs
- `output/` — checkpoints, logs, eval results (gitignored)

**Entry points:**
- `scripts/train_rl.py` — train SAC/PPO with SB3
- `scripts/evaluate.py` — evaluate greedy/RL/PID policies
- `scripts/eval_rl.py` — evaluate RL checkpoints across traces
- `scripts/train.py` — older training script

**Key data flow:**
Trace pickle → `load_trace()` → `(all_vms, node_to_vms, ...)` → `OctopusMemPoolEnv.__init__()`.
Each `reset()`: pick random pod of hosts, build event timeline, apply HOTFIX filter.
Each `step()`: agent outputs logits → softmax → allocation proportions → update MPD loads.

**Architecture pattern:** Episode-replay simulation. No learned dynamics — the env replays
real VM arrival/departure events from Azure traces.

### Phase 2: File-by-File

**`octopus/env.py` (402 lines)**
- Core Gymnasium environment. Clean separation: `reset()` builds events, `step()` processes one allocation.
- `M` stored as nested list; `host_to_mhds` precomputed from M in `__init__`. **`step()` and `_get_obs()` use `host_to_mhds`, NOT `self.M` directly.**
- `_build_events()` includes HOTFIX filter (sequential per-node capacity check).
- `_process_departures_through()` subtracts deallocations range-wise.
- `precomputed_events` cache bypasses `_build_events()` for speed.
- No augmentation support currently.

**`octopus/data.py` (239 lines)**
- `VM` class, `load_trace()`, `precompute_pod_events()`, `load_topology()`.
- `precompute_pod_events()` duplicates `_generate_pod()` + `_build_events()` logic offline.
- `load_topology()` returns `(matrix, num_hosts, num_pools)` where matrix is nested list.

**`octopus/baselines.py` (171 lines)**
- `greedy_alloc_ref()` (reference), `greedy_alloc()` (vectorized water-fill), `pid_alloc()`.
- All take `(cxl_mem, mhd_list, cur_cxl_mem_vec)` and modify `cur_cxl_mem_vec` in-place.

**`octopus/topology.py` (116 lines)**
- `generate_pod_to_nodes()`, `expand_M_to_all_nodes()`, `remove_ones()`.
- `remove_ones()` already implements link failure with row-safety guarantee (keeps ≥1 link per host).

**`octopus/optimal.py` (268 lines)**
- Dinic max-flow solver for optimal per-MPD peak (binary search). Used for eval baselines.

**`scripts/train_rl.py` (277 lines)**
- `DummyVecEnv` with 1 env (no parallelism). `PoolingSavingsCallback` runs eval periodically.
- Deterministic seeding already present (lines 154-159).
- `_make_env()` constructs `OctopusMemPoolEnv` from trace data.

**`scripts/evaluate.py` (418 lines)**
- `pooling_simulation()` — generic evaluator. `make_rl_alloc_cb()` wraps RL model for eval.

**`scripts/eval_rl.py` (200 lines)**
- Evaluates saved checkpoints across all 10 traces. Writes CSV results.

**`tests/test_greedy_alloc.py` (21 lines)**
- Parametrized test: fast greedy matches reference implementation across 20 random seeds.

### Phase 3: Cross-Cutting Issues

1. **`host_to_mhds` vs `self.M`:** `step()` and `_get_obs()` use `self.host_to_mhds` (dict of lists), not `self.M`. Any topology perturbation must recompute `host_to_mhds`, not just replace `self.M`.

2. **Existing `remove_ones()` in topology.py:** Already implements link-failure injection with safety. The plan's `inject_link_failures()` duplicates this with numpy arrays instead of nested lists. Consider reusing `remove_ones()` or ensuring consistency.

3. **Constructor signature mismatch:** Plan's test code (Task 8e) uses `trace_name="..."` — the actual constructor takes raw data objects, not trace names.

4. **DummyVecEnv(N=1):** `train_rl.py` uses a single env. Plan and design-decisions reference N=32 SubprocVecEnv, but that's not the current state.

5. **No CI:** No `.github/workflows/` directory. All testing is local.

### Plan Cross-Reference

| Plan section | Codebase state | Notes |
|---|---|---|
| Task 1 (AugmentationConfig) | `octopus/augmentation.py` does not exist | Clean creation |
| Task 2 (transforms) | No augmentation code exists | Clean creation |
| Task 3 (env.py integration) | `self.M` is nested list, not ndarray. `host_to_mhds` is the actual accessor. | **Plan says "replace self.M in step/obs" but self.M isn't used there. Must recompute `host_to_mhds` from augmented topology instead.** |
| Task 4 (precompute compat) | Precompute exists in data.py | Plan correctly says no changes needed |
| Task 5 (multi-trace) | Env takes raw data, not trace name | Need to pass loaded trace data tuples, not names |
| Task 6 (CLI args) | train_rl.py uses DummyVecEnv(1 env) | Works as-is, aug_config passed through `_make_env` |
| Task 7 (logging callback) | PoolingSavingsCallback exists as a pattern | Follow same style |
| Task 8 (tests) | Only 1 test file exists. Test code references `trace_name=` kwarg that doesn't exist in constructor | **Must fix test code to use `load_trace()` + actual constructor** |
| Task 9 (training runs) | Deferred to plan-v2 | Will not implement |

**Critical conflict:** The plan's `active_M` property and "replace self.M references" approach is wrong — `self.M` is not referenced in `step()`/`_get_obs()`. The correct approach is to recompute `self.host_to_mhds` from the augmented topology during `reset()`.

---

### Plan-v2 Cross-Reference (Training, Ablations & Evaluation)

| Plan section | Codebase state | Gap / Action |
|---|---|---|
| **Task 1a:** `--skip-hotfix` flag | Does not exist | Add to `train_rl.py` and `eval_rl.py`; guard HOTFIX code behind flag |
| **Task 1b:** Trace characterization | No characterization script exists | New script needed |
| **Task 1c:** Split assignment | `data/splits/` directory does not exist | Create dir + JSON files |
| **Task 2a:** W&B basic integration | `wandb` not in `requirements.txt`, not imported anywhere | Add dep + `wandb.init()` |
| **Task 2b:** EvalCallback (unaugmented val) | `EvalCallback` exists; eval env already unaugmented | Mostly in place; need `n_eval_episodes=20` and val-only trace |
| **Task 2c:** SAC diagnostics callback | Does not exist | New callback class |
| **Task 2d:** Augmentation logging | `AugmentationLogCallback` exists (scale stats only) | Extend: add trace_id histogram, link_failures_applied count |
| **Task 2e:** Env-specific metrics | `info` dict has `pooling_ratio`, `peak`, `max_peak` | Add peak utilization, MPD load variance, event count logging |
| **Task 2f:** W&B dashboard template | Does not exist | W&B API or manual setup |
| **Task 2g:** W&B alerts | Does not exist | W&B alert config |
| **Task 3a:** `--no-train` flag | Does not exist | Add flag; skip `model.learn()`, run random eval loop |
| **Task 3–4:** `--n-envs` flag + SubprocVecEnv | Does not exist (hardcoded DummyVecEnv N=1) | Add flag + SubprocVecEnv support |
| **Task 4:** `--wandb` flag | Does not exist | Add flag; conditional `wandb.init()` + `WandbCallback` |
| **Task 4:** `--trace <train_traces>` (multi) | `--trace` accepts single string | **Ambiguity: multi-trace coupled with `--augmentation`** |
| **Task 5:** Augmentation CLI args | All exist from plan-v1 | Plan says `--aug-memory-noise` but code has `--aug-noise-sigma` |
| **Task 5g/10:** `pooling_ratio` as primary metric | `env.py` computes it in `info`; `eval_rl.py` reports `savings` | Add explicit `pooling_ratio` column to eval output |
| **Task 6:** W&B Sweeps | Not set up | Need sweep YAML config + agent launch |
| **Task 9:** Reward variants | Single formula in `env.py:214` | Need parameterized reward selection |
| **Infra:** CI | No `.github/workflows/` | Not needed — all testing local (per CLAUDE.md) |

#### Already implemented (from plan-v1, no work needed)

- `AugmentationConfig` dataclass with all knobs
- All 6 transform functions in `augmentation.py`
- `apply_augmentation()` pipeline
- Augmentation integration in `env.py` (`_apply_augmentation`, `_switch_trace`)
- Augmentation CLI args in `train_rl.py` (10 args)
- `AugmentationLogCallback` (basic version)
- Multi-trace pool loading
- Deterministic seeding (random, numpy, torch, cudnn)
- `PoolingSavingsCallback` for eval-time pooling ratio
- 49 augmentation tests + 20 greedy regression tests

#### Codebase dependencies / preconditions

1. **SubprocVecEnv pickling**: `OctopusMemPoolEnv` uses `self.all_vms` (dict of VM objects with datetime fields). Must verify pickle compatibility for SubprocVecEnv multiprocessing.
2. **W&B install**: `wandb` must be added to `requirements.txt` and installed in venv.
3. **`data/splits/` sealing**: Plan requires test traces sealed in JSON, not read until Task 10. This is an honor-system constraint for the executing agent.
4. **Reward parameterization (Task 9)**: Currently hardcoded in `step()`. Needs refactoring to support multiple reward formulations via CLI flag.

---

## Code Review — commit 400e0a49 (2026-03-26)

### Phase 1: Repo Overview

Unchanged from previous review: RL agent for CXL memory allocation. Augmentation system (plan-v1) is complete and integrated. Plan-v2 adds new reward formulations (A, B) and a 50-dim observation space.

**What's new since 24216f8a:**
- W&B, `--skip-hotfix`, `--n-envs` / SubprocVecEnv, SACDiagnosticsCallback, EnvMetricsCallback all present in train_rl.py (plan-v2 Tasks 2–3 complete)
- `--skip-hotfix` present in eval_rl.py

**What plan-v2 Task 4 still needs to add:**
- env.py: mpd_vm_allocs, mhd_to_hosts, Q_j, cur_host_cxl_load, host_dealloc_events, reward_variant, new obs
- train_rl.py: --reward-variant, --lookahead-window, --reward-lambda CLI flags
- evaluate.py: obs_variant in make_rl_alloc_cb, track_vm_allocs in pooling_simulation
- eval_rl.py: pass reward variant config through

### Phase 2: File-by-File Drill Down

**octopus/env.py** (OctopusMemPoolEnv)
- Obs: 2*max_degree + 4 = 20-dim (loads, mask, vm_norm, peak_norm, hour_sin, hour_cos)
- Reward: -(delta_peak)/fair_share - λ·Var(loads) — "current" variant
- No per-MPD VM tracking (needed for D_j, S_j)
- No mhd_to_hosts inverse (needed for P_j, Q_j)
- No cur_host_cxl_load (needed for P_j)
- _apply_augmentation already recomputes host_to_mhds — augmentation hook for _recompute_topology_derived() is ready to use
- _process_departures_through loops through dealloc_events — needs to also drain host loads

**scripts/train_rl.py**
- W&B init, SACDiagnosticsCallback, EnvMetricsCallback, AugmentationLogCallback — complete
- --n-envs + SubprocVecEnv — complete
- Missing: --reward-variant / --lookahead-window / --reward-lambda passed to _make_env() → OctopusMemPoolEnv()
- eval_cb frequency formula uses args.eval_freq // n_envs — correct for timesteps
- PoolingSavingsCallback calls make_rl_alloc_cb(model, max_degree) — needs obs_variant + mhd_to_hosts + Q_j added for new variants

**scripts/evaluate.py**
- make_rl_alloc_cb hardcodes 20-dim obs layout. The layout replicates _get_obs() exactly.
  This is the #1 sync risk — new obs layout must be mirrored here.
- pooling_simulation has no VM tracking state — no mpd_vm_allocs, no per-host loads.
  For new reward variants these must be tracked in the sim loop and injected into ctx.
- ctx dict currently: {tick, pod_start_ts, base_time, num_mhd, pod_rss_mem}
  Needed additions: {mpd_vm_allocs, host_cxl_load, mhd_to_hosts, Q_j} when track_vm_allocs=True

**scripts/eval_rl.py**
- Calls make_rl_alloc_cb(model, max_deg) — needs same extension as train_rl.py
- No reward variant config threaded through yet

**tests/**
- test_greedy_alloc.py + test_augmentation.py: 49 tests passing
- No tests for new reward/obs variants (need for Task 4g)

### Phase 3: Cross-Cutting Issues

1. **Sync invariant: _get_obs() ↔ make_rl_alloc_cb()** — already flagged in LOG.md. Any change to obs layout in env.py must be mirrored in evaluate.py. The plan's 50-dim obs removes time features (hour_sin/cos) and adds 4 new per-MPD features. Test 4g should explicitly verify obs produced by both paths are equal on a known input.

2. **mpd_vm_allocs purging** — the plan says purge in _process_departures_through(). This requires iterating each MPD's active VM list and removing entries where dealloc_tick <= tick. If MPD lists can be large (500+ VMs), list comprehension per tick could be O(N) per tick per MPD. Acceptable for training; for evaluate.py's sim loop (no Python overhead per event), use a deque or sorted list for efficiency.

3. **observation_space shape mismatch at model load** — if a model trained with 20-dim obs is loaded via SAC.load() and then used with a 50-dim env, SB3 will error on the first predict() call. The reward variant must be recorded in config.json at training time and restored at eval time. eval_rl.py needs a way to detect which obs variant a checkpoint used.

4. **Q_j vs. topology augmentation** — Q_j is precomputed from the base topology. If link failures are enabled, host degrees change, so Q_j changes each episode. The plan says _recompute_topology_derived() should be called from _apply_augmentation. This is correct but needs care: Q_j must be recomputed from aug_M, not self.M.

5. **P_j requires current host loads, not MPD loads** — P_j(t) = sum of ℓ_h(t) for h in H(j). In the env, ℓ_h(t) = cur_host_cxl_load[h]. But in pooling_simulation, there's no host load tracking. Need to add host_cxl_load array to pooling_simulation's sweep loop.

### Cross-Reference: Plan vs Codebase

| Plan item | Status |
|-----------|--------|
| Task 1 trace characterization | ✅ already done |
| Task 2 W&B integration | ✅ already done |
| Task 3 sanity checks | ✅ already done |
| --skip-hotfix in train/eval | ✅ already done |
| Task 4a mpd_vm_allocs in env.py | ❌ not started |
| Task 4b mhd_to_hosts, Q_j, _recompute_topology_derived | ❌ not started |
| Task 4c reward A/B in step() | ❌ not started |
| Task 4d new 50-dim obs in _get_obs() | ❌ not started |
| Task 4e CLI flags in train_rl.py | ❌ not started |
| Task 4f evaluate.py/eval_rl.py obs variant support | ❌ not started |
| Task 4g tests | ❌ not started |
| Tasks 5–7 timing/training/eval runs | human-run, code not needed |

---

## Code Review — commit f7cd8f803d3bfaa05c90eab4377466ba88729202 (2026-04-09)

Scope: async eval plan implementation (`scripts/train_rl.py`, `scripts/evaluate.py`, `octopus/`)

---

### Phase 1: Repo Overview

**Purpose:** RL agent (SB3 SAC) for CXL memory pooling. Learns to distribute VM memory across MPDs to minimise peak load. Trains on replayed Azure VM traces via Gymnasium env.

**Directory map:**
- `octopus/` — core env, data loading, topology, baselines, augmentation
- `scripts/train_rl.py` — entry point: SAC training loop + callbacks
- `scripts/evaluate.py` — `pooling_simulation()` + allocation callbacks (greedy, PID, RL)
- `tests/` — pytest suite (67 passing, 23 failing pre-existing from Task 4 not yet implemented)
- `data/` — trace pickles + topology CSVs (not in git)
- No CI workflows (.github/ does not exist)

**Key data flow:**
`load_trace()` → `(all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz)` →
`OctopusMemPoolEnv` → SB3 SAC → `PoolingSavingsCallback._run_eval()` → `pooling_simulation()`

**Architectural pattern:** callback-driven evaluation; `PoolingSavingsCallback._on_step()` fires every `eval_freq` steps and blocks training while running `n_iter` pooling simulations.

---

### Phase 2: File-by-file Drill Down

#### `scripts/train_rl.py`

- `PoolingSavingsCallback` (lines 51–131): current synchronous impl. `_on_step()` calls `_run_eval()` which blocks for ~133s per trigger.
- `_run_eval()` (lines 77–131): imports `pooling_simulation` and `make_rl_alloc_cb` from `scripts.evaluate` inside the method. Runs `n_iter` simulations, logs mean/std.
- `PoolingSavingsCallback.__init__` (lines 55–68): takes `eval_trace_data, M, max_degree, eval_freq, n_iter, obs_variant, mhd_to_hosts, Q_j, lookahead_window, verbose`. No async fields, no `eval_device`.
- `main()`: has `--device` but not `--eval-device`. `PoolingSavingsCallback` is instantiated with `eval_freq=args.eval_freq` (raw, not the n_envs-scaled `eff_eval_freq`). This is intentional — the callback compares against `self.num_timesteps` which is already the global step count.
- `BaseCallback` provides `self.num_timesteps` automatically from SB3.
- `_REPO_ROOT` is set at module level (lines 19–21) — any spawned child that imports this module gets `sys.path` set correctly.

**Bug note:** `except Exception: pass` in the planned `_on_step` catches ALL exceptions including `AttributeError` if `_res_q` is `None`. Should use `queue.Empty` specifically or guard with `if self._res_q is not None`.

#### `scripts/evaluate.py`

- `pooling_simulation()` (lines 65–304): pure function, takes topology + trace data + `alloc_fn` callback. Picklable, no GPU state.
- `make_rl_alloc_cb(model, max_degree, ...)` (lines 326–428): extracts `policy = model.policy`, pre-allocates `_obs_buf` on GPU, returns a closure. Worker can replicate this logic with its own policy directly (no SB3 model dependency needed).
- `_rl_alloc_cb` closure (lines 350–427): builds obs based on `obs_variant`, calls `policy._predict(_obs_buf, deterministic=True)`. Worker will inline this same closure.
- The `make_rl_alloc_cb` logic is the template for the worker's inference closure.

**Key finding:** The "current" obs variant uses `import datetime as _dt` via `ctx["base_time"]`. Worker must also import `datetime`.

#### `octopus/topology.py`

- `generate_pod_to_nodes()` and `expand_M_to_all_nodes()` are used inside `_run_eval()` per iteration. Worker must import and use these too.

---

### Phase 3: Cross-Cutting Issues

1. **`except Exception: pass` is too broad.** Using bare `Exception` to catch `queue.Empty` also silences `AttributeError`, `KeyError`, and worker crashes. Use `import queue; except queue.Empty: pass` specifically. The `_worker.exitcode` check should be done separately.

2. **`eval_freq` in callback vs `eff_eval_freq` for SB3 callbacks.** `PoolingSavingsCallback` uses `args.eval_freq` directly (not scaled by n_envs). This is correct because `self.num_timesteps` reflects total environment steps including all parallel envs — consistent with what SB3 reports.

3. **Worker process pickling.** `_eval_worker_main` must be defined at module top level (not inside a class or function) for `spawn` pickling to work. Confirmed: plan places it before `PoolingSavingsCallback`.

4. **`spawn` start method and `SubprocVecEnv`.** SB3's `SubprocVecEnv` uses `multiprocessing.Process` without setting context explicitly — it inherits whatever is set globally. Setting to `spawn` globally is safe and correct on this codebase.

5. **23 pre-existing test failures** from Task 4 (new reward/obs variants not yet implemented in `octopus/env.py`). These are NOT caused by this plan and must not regress further.

---

### Cross-Reference: Plan vs Codebase

| Plan item | Status | Notes |
|-----------|--------|-------|
| `_eval_worker_main` top-level function | NOT EXISTS | Needs to be added before `PoolingSavingsCallback` |
| `PoolingSavingsCallback` async fields | NOT EXISTS | `_worker`, `_cmd_q`, `_res_q`, `_pending_step`, `_eval_device` |
| `on_training_start()` in callback | NOT EXISTS | BaseCallback has it but PSCallback doesn't override |
| `on_training_end()` in callback | NOT EXISTS | Same |
| New async `_on_step()` | NOT EXISTS | Current sync version exists |
| `_run_eval()` | EXISTS | Will be DELETED |
| `--eval-device` CLI arg | NOT EXISTS | |
| `mp.set_start_method("spawn")` in main | NOT EXISTS | |
| `eval_device` wired to callback | NOT EXISTS | |

**Preconditions already satisfied:**
- `_REPO_ROOT` / `sys.path` setup runs automatically on worker spawn (module-level code)
- `eval_trace_raw` (raw 5-tuple from `load_trace()`) is passed to callback — picklable, suitable for worker
- `make_rl_alloc_cb`'s inference closure logic is in `evaluate.py` — will be inlined in worker

**Key implementation note:** Worker will inline the obs-building + inference logic from `make_rl_alloc_cb` rather than calling `make_rl_alloc_cb(model, ...)`. The worker has its own `policy` object; it doesn't need the SB3 `model` wrapper.
