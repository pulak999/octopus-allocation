# Octopus CXL Memory Pooling — RL Agent

## Overview

RL agent that learns to distribute VM memory across CXL-connected Memory Pool Devices (MPDs)
to minimize peak load and load variance. Trains on replayed Azure VM traces via Gymnasium env
with SB3 (SAC).

## Build / Run

```bash
# Activate venv
source venv/bin/activate

# Train (fast smoke test)
python scripts/train_rl.py --run-id test --fast --total-timesteps 5000

# Evaluate
python scripts/eval_rl.py --run-id test --n-iter 5 --traces AMS20PrdApp19-tround

# Tests
python -m pytest tests/ -v
```

## Key Conventions

- All paths relative to repo root `/home/pm3371/gitrepos/octopus-allocation/`
- Topology matrix `M` is stored as nested `list[list[int]]`, not numpy array
- `host_to_mhds` (dict of lists) is the runtime accessor for topology — `self.M` is only
  used in `__init__` to build it
- Events are tuples: `(tick, pod_id, vm_mem_gb, dealloc_tick)`
- `mem_idx = 1` indexes into `VM.rss` and `machine_sz` for memory (GB)
- Training uses `DummyVecEnv` with 1 env currently (plan-v2 adds `--n-envs` + SubprocVecEnv)
- Episode = one pod of hosts replaying VM arrivals over full trace window (~14 days)
- `pooling_ratio = peak * num_mhd / pod_dram` — primary eval metric (lower = better)

## Data

- **Traces:** 10 Azure VM trace pickles in `data/traces/` (not in git)
- **Topologies:** 5 CSV adjacency matrices in `data/topologies/`
- **Default topology:** `AG16x6_expander_quads_r5_sym_fixed.csv` (16 hosts, 6 MPDs/host)

## Design Decisions

- Episode-level augmentation (not per-VM) to preserve SKU discreteness
- HOTFIX filter removes VMs exceeding per-node DRAM; runs sequentially in `_build_events()`
- Precomputed event cache (`precompute_pod_events()`) bypasses `_build_events()` for speed;
  augmentation runs on top of cached base events
- No CI — all testing is local with pytest
- HOTFIX filter should be skippable (`--skip-hotfix`) for CXL pooling training (plan-v2 Task 1a)
- W&B (`wandb`) used for experiment tracking (plan-v2 Task 2)
- **Async eval (async-plan):** `PoolingSavingsCallback` runs eval in a persistent background
  worker process (spawned at `on_training_start`). Training never blocks on eval.
  Worker runs on `cuda:1` by default; training stays on `cuda:0`. One-step metric lag is acceptable.
  `mp.set_start_method("spawn")` is set globally in `main()` — required for CUDA safety.

## Hardware

3× NVIDIA TITAN RTX (24 GB each), CUDA 12.5, kernel 5.15. SB3 SAC is single-GPU —
pin via `CUDA_VISIBLE_DEVICES`.

**GPU split (async eval):**
- `cuda:0`: SAC training (rollout + update)
- `cuda:1`: async eval worker (inference in `_eval_worker_main`)
- `cuda:2`: free — parallel run or ablation

## Current Plan

Active: `docs/plans/v4/reward-ablation-plan.md` — reward ablation R1–R5, A→R2/B→R3 rename,
pipeline correctness tests. Fresh runs use `--run-id ablation_R1_v1`, `ablation_R2_v1`, etc.
JAX plan (`docs/plans/v4/jax-plan.md`) — parallel JAX SAC trainer — is complete (Phases 0-4 done as of 2026-05-05).
Previous: `SPEEDUP_PLAN.md` (Chunks 0-3 all complete, 2026-05-01). See `docs/plans/v3/async-plan.md`
(async eval worker) and `docs/plans/v3/plan-v2.md` (training / rewards / eval) for prior work.

## JAX Plan Conventions

- JAX module lives in `octopus/jax/` — SB3 imports stay untouched.
- Static shape only: `MAX_TICKS=2304`. No `MAX_ACTIVE_VMS` — see below.
- `OctopusState` is a `@chex.dataclass`. Fields: `mpd_load (num_mhd,)`, `dealloc_buf (MAX_TICKS, num_mhd)`, `host_load (pod_size,)`, `host_dealloc_buf (MAX_TICKS, pod_size)`, `max_peak`, `event_idx`, `last_depart_tick`, `key`.
- **`_mpd_dt/_mpd_mem/_mpd_n` are NOT in JAX state.** MAX_ACTIVE_VMS=256 would overflow (max 2007 total VM allocations per MPD per episode observed in LVL01 trace). D_j and S_j are computed from `dealloc_buf` via `jnp.einsum` — mathematically identical to Numba kernel since `dealloc_buf[t,j]` = sum of memory from VMs departing at tick t on MPD j.
- D_j formula: `weight[t] = max(0, 1-(t-tick)/W)` for `t > last_depart_tick` and `t ≤ tick+W`; then `D_j = einsum('t,tj->j', weight, dealloc_buf)`. S_j = sum over `t > tick+W`.
- `reset_fn` runs pod selection on host (Python stdlib random — not JAX PRNG) then `jax.device_put`.
- `step_fn` departure processing uses `jnp.einsum` batch-subtract over static tick range — no Python loop.
- `dealloc_tick` is clipped to `MAX_TICKS-1` before scatter to handle VMs ending beyond the trace window.
- Parity tests run under `jax.config.update("jax_enable_x64", True)`. Production training uses float32.
- Optional deps: `pip install -e ".[jax]"` — adds jax==0.6.2, jaxlib==0.6.2, flax==0.10.7, optax==0.2.8, chex==0.1.90.

## Perf Conventions (SPEEDUP_PLAN — complete)

- Per-MPD active-VM state: `_mpd_dt (num_mhd, MAX_SLOTS) float64`, `_mpd_mem (num_mhd, MAX_SLOTS) float64`, `_mpd_n (num_mhd,) int32` (valid count after compaction). Numba kernels operate on these.
- `MAX_SLOTS` grows 2× automatically on overflow via `_ensure_mpd_capacity(j)`; VMs are never dropped.
- Precompute cache is keyed per trace — multi-trace training carries `(trace_arrays, precomputed_events)` in `_switch_trace`.
- `precompute_pod_events_arrays(TraceArrays, M, seeds)` is the cache builder. Legacy `precompute_pod_events(raw_tuple, ...)` is kept for compatibility.
