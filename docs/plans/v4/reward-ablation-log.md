# Reward Ablation Implementation Log

## [2026-05-06] Chunks 0–3 complete

### Features Implemented

- **Chunk 0**: Alias normalisation (`"A"` → `"R2"`, `"B"` → `"R3"`) in `OctopusMemPoolEnv.__init__` before assert; assert extended to accept `current/R1/R2/R3/R4/R5`; reward block internal references updated from `"A"`/`"B"` to `"R2"`/`"R3"`; obs-dim branch switched to `_SIMPLE_OBS = {"current", "R1"}`; `D_j_pre` computation gated on `not in _SIMPLE_OBS`; `_get_obs` obs-path likewise. All four affected scripts updated: `train_rl.py` (choices, precompute condition, async worker obs-dim/track/obs-path), `eval_rl.py` (choices, precompute condition, track), `evaluate.py` (plan gap: `make_rl_alloc_cb` obs-dim and obs-path).
- **Chunk 1**: R1 reward branch: `reward = -new_peak / norm` (simple obs, no departure awareness).
- **Chunk 2**: Hoisted `done_flag`, `_pooling_ratio`, `_pooling_savings` to before reward block; R4 sparse terminal: `reward = _pooling_savings if done_flag else 0.0`; `info["pooling_ratio/savings"]` now use pre-computed vars.
- **Chunk 3**: Added `sub_episode_len=576` and `pbrs_gamma=0.99` constructor params and instance attrs; added `_sub_peak`, `_sub_step`, `_prev_potential` to `reset()`; R5 reward block with PBRS + sub-episode boundary detection; `train_rl.py` wired `--sub-episode-len` / `--pbrs-gamma` args through `_make_env` to both training and eval envs.

### Files Changed

| File | What changed |
|------|-------------|
| `octopus/env.py` | Alias normalisation, extended assert, R1/R4/R5 reward branches, hoisted pooling metrics, R5 state in reset/init |
| `scripts/train_rl.py` | Choices extended, precompute condition, async worker obs logic, `_make_env` signature + 2 call sites, `--sub-episode-len`/`--pbrs-gamma` args |
| `scripts/eval_rl.py` | Choices extended, precompute condition, track_vm_allocs |
| `scripts/evaluate.py` | `make_rl_alloc_cb` obs-dim and obs-path fixed for R1 (plan gap) |

### Functions Written

| Function | File | Description |
|----------|------|-------------|
| R5 reward block | `octopus/env.py:step()` | PBRS shaping + sub-episode savings; resets `_sub_peak` at boundary |

### Data Structures Created

| Name | File | Description |
|------|------|-------------|
| `_sub_peak` | `octopus/env.py` | Max peak within current sub-episode window; reset at boundaries |
| `_sub_step` | `octopus/env.py` | Step counter within current sub-episode |
| `_prev_potential` | `octopus/env.py` | Φ(s_{t-1}) for PBRS; `Φ(s) = -new_peak / D_pod` |

### Test Status

- 162/162 tests pass after each chunk (no regressions).
- No CI — local pytest only.

### Notes

- **Plan gap fixed**: `evaluate.py:make_rl_alloc_cb` was not in the original plan but needed `_SIMPLE_OBS` fix for R1 evaluation correctness.
- **`done_flag`** is pre-computed as `(event_idx + 1) >= len(events)` before advancing event_idx — algebraically equivalent to `done` after advancing.
- **`max_peak` update** moved before reward block (was after in original). No semantic change for `current`/R2/R3; needed for R4/R5 to have correct `_pooling_savings` at reward time.

---

## [2026-05-06] Chunk 4 complete

### Features Implemented

- **Chunk 4**: Created `tests/test_pipeline_correctness.py` with 10 correctness tests.

### Files Changed

| File | What changed |
|------|-------------|
| `tests/test_pipeline_correctness.py` | New file — 10 pipeline correctness tests |

### Functions Written

| Function | File | Description |
|----------|------|-------------|
| `_make_trace_arrays` | `tests/test_pipeline_correctness.py` | Helper: converts synthetic VM dicts to TraceArrays |
| `_make_minimal_env` | `tests/test_pipeline_correctness.py` | Helper: 2-host/3-MPD env with 2 synthetic VMs |
| `_make_long_env_r5` | `tests/test_pipeline_correctness.py` | Helper: R5 env with n_vms events over consecutive ticks |

### Test Status

- 172/172 tests pass (162 existing + 10 new).
- No CI — local pytest only.

### Tests Written

| Test | What it checks |
|------|---------------|
| `test_mass_conservation` | `sum(cur_cxl_mem_vec) == sum(_mpd_mem entries)` after every step |
| `test_deallocation_clears` | After `_process_departures_through(pod_dur-1)`, remaining `_mpd_dt` entries all exceed `_last_depart_tick` |
| `test_pooling_ratio_formula` | `info["pooling_ratio"] == max_peak * num_mhd / pod_dram` at episode end |
| `test_eval_env_parity` | Uniform allocation: env pooling_ratio matches `pooling_simulation` to 1e-4 |
| `test_R4_intermediate_rewards_zero` | All non-terminal R4 rewards are 0.0; terminal == `pooling_savings` |
| `test_R5_sub_peak_resets_mpd_loads_persist` | At sub-episode boundary: `_sub_peak == 0`, `cur_cxl_mem_vec.sum() > 0` |
| `test_R5_pbrs_telescoping` | With γ=1: Σ(Φ(s')−Φ(s)) = Φ(s_T)−Φ(s_0) — verifies `_prev_potential` tracking |
| `test_obs_reflects_env_state` | Rich obs: `obs[k*6]`, `obs[-2]`, `obs[-1]` match env state at reset and after step |
| `test_greedy_mutation_regression` | `greedy_alloc` does not mutate its `cur_cxl_mem_vec` argument |
| `test_R1_bounds` | Every R1 reward is in `[-1.0, 0.0]` |

### Notes

- **Test 2 invariant**: The last VM to end has `dealloc_tick == pod_dur` (by construction of `_build_events`), so it never enters `dealloc_events` and is never cleared by `_process_departures_through(pod_dur-1)`. The test checks the weaker but correct invariant: remaining entries have `dt > last_depart_tick`.
- **Test 7 (PBRS)**: Used γ=1.0 so the sum is a clean Σ(Φ(s')−Φ(s)) = Φ(s_T)−Φ(s_0) telescoping. With γ≠1, cross-terms don't cancel and a different formula is needed — noted in the test comment.
- **Plan complete**: All 4 chunks implemented and tested.
