## [2026-03-26] New State Space & Rewards — Task 4 (plan-v2)

### Features Implemented
- **Task 4a**: Per-MPD VM tracking (`mpd_vm_allocs`) and per-host CXL load tracking (`cur_host_cxl_load`, `host_dealloc_events`) in `OctopusMemPoolEnv`
- **Task 4b**: Static topology precomputation — `mhd_to_hosts` (inverse of `host_to_mhds`) and `Q_j` (neighbour scarcity) via `_recompute_topology_derived()`; called from `__init__` and `_apply_augmentation` so Q_j stays correct after link failures
- **Task 4c**: Two new reward variants — "A" (`-max(ĉ_j+)` over accessible MPDs) and "B" (A − λ·max global stress); "current" path unchanged; dispatch via `reward_variant` constructor param; `_compute_D_j(tick)` computes time-weighted departure relief
- **Task 4d**: New 50-dim observation space (`6·d_max + 2`) for variants A/B — per-MPD slots `[c_j, D_j, S_j, mask, P_j, Q_j]/D_pod` plus global `[peak, vm_mem]/D_pod`; `observation_space` updated accordingly
- **Task 4e**: CLI flags `--reward-variant {current,A,B}`, `--lookahead-window`, `--reward-lambda` in `train_rl.py`; `PoolingSavingsCallback` updated to precompute `mhd_to_hosts`/`Q_j` and pass to `make_rl_alloc_cb`
- **Task 4f**: `pooling_simulation` in `evaluate.py` gains `track_vm_allocs=False` param; `make_rl_alloc_cb` gains `obs_variant`, `mhd_to_hosts`, `Q_j`, `lookahead_window` params; `eval_rl.py` gains `--reward-variant`, `--lookahead-window` flags with auto-detect from `config.json`
- **Task 4g**: 20 tests in `tests/test_new_reward_obs.py` covering topology precomputation, VM tracking, obs shape, D_j formula, reward ranges, current-variant regression, mask/global feature correctness, and obs-sync between env and evaluate.py

### Files Changed
| File | What changed |
|------|-------------|
| `octopus/env.py` | New params (`reward_variant`, `lookahead_window`, `reward_lambda`), `mpd_vm_allocs`/`cur_host_cxl_load`/`host_dealloc_events` in reset, `_compute_D_j()`, `_recompute_topology_derived()`, updated `_get_obs()` and `_apply_augmentation()`, reward dispatch in `step()` |
| `scripts/evaluate.py` | `pooling_simulation` + `track_vm_allocs` param, `make_rl_alloc_cb` new obs_variant/mhd_to_hosts/Q_j/lookahead_window params, 50-dim obs construction |
| `scripts/train_rl.py` | Three new CLI flags, `_make_env` passes new params, `PoolingSavingsCallback` updated |
| `scripts/eval_rl.py` | `--reward-variant`/`--lookahead-window` flags, config.json auto-detect, mhd_to_hosts/Q_j precomputation, `_eval_model_on_trace` updated |
| `tests/test_new_reward_obs.py` | New file — 20 tests |
| `TODO.md` | Tasks 4a–4g marked complete |
| `ARCH.md` | Reward variants, observation space, CRITICAL sync note for make_rl_alloc_cb |

### Functions Written
| Function | File | Description |
|----------|------|-------------|
| `_compute_D_j` | `octopus/env.py` | Time-weighted departure relief per MPD within lookahead window |
| `_recompute_topology_derived` | `octopus/env.py` | Builds mhd_to_hosts and Q_j from host_to_mhds |
| `pooling_simulation` (updated) | `scripts/evaluate.py` | Added track_vm_allocs path for per-VM state |
| `make_rl_alloc_cb` (updated) | `scripts/evaluate.py` | Builds 50-dim obs for variants A/B |
| `_eval_model_on_trace` (updated) | `scripts/eval_rl.py` | Threads obs_variant/mhd_to_hosts/Q_j |
| `test_obs_sync_initial_step` | `tests/test_new_reward_obs.py` | Verifies evaluate.py and env.py obs are identical at first event |

### Data Structures Created
| Name | File | Description |
|------|------|-------------|
| `mpd_vm_allocs` | `octopus/env.py` | `list[list[tuple[int, float]]]` — per-MPD list of (dealloc_tick, mem_gb) for active VMs |
| `cur_host_cxl_load` | `octopus/env.py` | `np.ndarray(pod_size,)` — current CXL memory in-use per host |
| `host_dealloc_events` | `octopus/env.py` | `np.ndarray(pod_dur, pod_size)` — scheduled per-host dealloc amounts by tick |
| `mhd_to_hosts` | `octopus/env.py` | `dict[int, list[int]]` — inverse of host_to_mhds |
| `Q_j` | `octopus/env.py` | `np.ndarray(num_mhd,)` — neighbour scarcity: mean(1/deg(h)) per MPD |

### Notes
- D_j is computed **pre-allocation** in `step()` so it reflects existing VMs only (not the just-arriving VM)
- "current" variant obs/reward is unchanged — existing trained models remain compatible
- The 50-dim obs formula matches between `_get_obs()` (env) and `make_rl_alloc_cb` (evaluate.py) — verified by `test_obs_sync_initial_step`
- Tasks 5-7 are human-run (timing tests, overnight training, post-overnight evaluation)
- No CI; all 90 tests pass locally

## [2026-03-24] Data Augmentation Implementation (plan-v1)

### Features Implemented
- Episode-level data augmentation pipeline with 6 knobs: memory scaling, per-VM noise, arrival jitter, lifetime perturbation, link failures, multi-trace sampling
- Full integration into OctopusMemPoolEnv (reset-time transforms, no step/reward changes)
- CLI args in train_rl.py for all augmentation knobs
- Augmentation logging callback for TensorBoard
- 49 unit + integration tests (all passing)

### Files Changed
| File | What changed |
|------|-------------|
| octopus/augmentation.py | NEW — AugmentationConfig, 6 transform functions, apply_augmentation pipeline |
| octopus/env.py | Added aug_config, trace_pool params; _apply_augmentation() recomputes host_to_mhds; _switch_trace() for multi-trace; aug_params in info dict |
| scripts/train_rl.py | Added augmentation CLI arg group (10 args), AugmentationLogCallback, multi-trace pool loading, conditional callback registration |
| tests/test_augmentation.py | NEW — 49 tests covering all transforms, config sampling, pipeline, and end-to-end env integration |
| code.md | 3-phase code review at commit 24216f8a |
| CLAUDE.md | NEW — project overview, build instructions, conventions |
| ARCH.md | NEW — architecture overview, data flow, augmentation pipeline |
| TODO.md | Updated — all plan-v1 tasks marked complete |

### Functions Written
| Function | File | Description |
|----------|------|-------------|
| scale_memory | octopus/augmentation.py | Uniform episode-level memory scaling |
| add_memory_noise | octopus/augmentation.py | Per-VM multiplicative noise (sigma-bounded) |
| jitter_arrivals | octopus/augmentation.py | Bounded tick shifts preserving non-negative ticks |
| perturb_lifetimes | octopus/augmentation.py | Fractional lifetime noise (positive VMs only) |
| inject_link_failures | octopus/augmentation.py | Random CXL link removal with safety guarantee |
| apply_augmentation | octopus/augmentation.py | Master pipeline: all transforms + re-sort |
| sample_augmentation_params | octopus/augmentation.py | Sample concrete params from config ranges |
| _apply_augmentation | octopus/env.py | Reset-time augmentation with retry logic |
| _switch_trace | octopus/env.py | Swap trace data for multi-trace support |
| AugmentationLogCallback._on_step | scripts/train_rl.py | Log scale distribution stats to TensorBoard |

### Data Structures Created
| Name | File | Description |
|------|------|-------------|
| AugmentationConfig | octopus/augmentation.py | Dataclass: scale_range, memory_noise_sigma, arrival_jitter_ticks, lifetime_noise_frac, link_failure_ratio, multi_trace, min_events, max_resample_attempts, enabled |

### Notes
- Plan's `active_M` property approach was wrong — `step()`/`_get_obs()` use `host_to_mhds`, not `self.M`. Fixed by recomputing `host_to_mhds` from augmented topology in `_apply_augmentation()`.
- Plan's test code used nonexistent `trace_name=` kwarg — fixed to use `load_trace()` + actual constructor.
- Added `memory_noise_sigma` knob (Knob 7 from design-decisions.md) not in original plan.
- Task 9 (training runs) deferred to plan-v2 per user decision.
- `_M_np` (numpy copy of M) added to env for augmentation; base `self.M` (nested list) preserved for compatibility.

---

## [2026-04-09] Async Eval Worker (async-plan)

### Features Implemented
- **Async eval worker**: `PoolingSavingsCallback` now runs pooling_simulation in a persistent background process, overlapping eval with training. Training never blocks on eval.
- **GPU split**: eval worker defaults to `cuda:1`, training stays on `cuda:0`. Controlled by new `--eval-device` CLI arg.
- **Graceful lifecycle**: `on_training_start` spawns worker; `_on_step` does non-blocking collect + dispatch; `on_training_end` sends stop, joins, drains final result.
- **Crash detection**: `_on_step` checks `_worker.exitcode` and logs a warning if worker dies unexpectedly.

### Files Changed
| File | What changed |
|------|-------------|
| `scripts/train_rl.py` | Added `_eval_worker_main` top-level function; replaced `_run_eval()` with async `on_training_start`, `_on_step`, `on_training_end` in `PoolingSavingsCallback`; added `--eval-device` arg; added `mp.set_start_method("spawn")` in `main()` |
| `tests/test_async_eval.py` | New file: 8 tests covering worker roundtrip, multi-eval sequence, callback lifecycle, result drain on shutdown |
| `CLAUDE.md` | Added async eval architecture, GPU split table |
| `ARCH.md` | Added async eval architecture section with sequence diagram |
| `TODO.md` | Added async-plan task list (all complete) |

### Functions Written
| Function | File | Description |
|----------|------|-------------|
| `_eval_worker_main` | `scripts/train_rl.py` | Persistent worker: receives eval commands, runs pooling_simulation on eval GPU, puts results on res_q |
| `PoolingSavingsCallback.on_training_start` | `scripts/train_rl.py` | Spawns persistent worker process via `mp.get_context("spawn")` |
| `PoolingSavingsCallback._on_step` (rewritten) | `scripts/train_rl.py` | Non-blocking result collect + conditional eval dispatch |
| `PoolingSavingsCallback.on_training_end` | `scripts/train_rl.py` | Sends stop command, joins worker, drains final result |

### Notes
- `except queue.Empty` used throughout (not bare `except Exception`) — catches only the expected empty-queue case.
- `_FakePolicy` and `octopus.data.VM` must be module-level for spawn pickling; locally-defined classes silently fail to pickle.
- 23 pre-existing test failures (Task 4 reward/obs variants not yet in env.py) unchanged.
- Testing procedure per plan: run two episodes, compare per-trigger eval wall time (sync ~133s vs async <1s) and `nvidia-smi` GPU utilisation split.
