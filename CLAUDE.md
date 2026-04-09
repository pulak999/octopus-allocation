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

Active: `docs/plans/v3/async-plan.md` (async eval worker).
Previous: `docs/plans/v3/plan-v2.md` (training, ablations, evaluation).
Prerequisite plan-v1 (augmentation system) is complete — see `docs/plans/v3/LOG.md`.
