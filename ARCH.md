# Architecture

## Components

```
octopus/
  env.py          — OctopusMemPoolEnv (Gymnasium): reset/step/obs/reward
  data.py         — VM class, load_trace(), precompute_pod_events(), load_topology()
  baselines.py    — greedy_alloc(), pid_alloc() allocation policies
  topology.py     — Pod generation, matrix expansion, link failure injection
  optimal.py      — Dinic max-flow solver for optimal per-MPD peak
  augmentation.py — [NEW] AugmentationConfig, transform functions, apply_augmentation()

scripts/
  train_rl.py     — SB3 SAC/PPO training with callbacks
  evaluate.py     — pooling_simulation() + allocation callbacks
  eval_rl.py      — Multi-trace RL checkpoint evaluation
  eval_baselines.py, plot_*.py, diagnose.py

tests/
  test_greedy_alloc.py     — greedy fast vs ref regression
  test_augmentation.py     — [NEW] augmentation unit + integration tests
```

## Data Flow

```
Trace pickle
  → load_trace() → (all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz)
  → OctopusMemPoolEnv.__init__(all_vms, node_to_vms, ..., M, aug_config)

Episode lifecycle:
  reset()
    → _generate_pod(seed)           # random host subset
    → _build_events()               # VM arrivals + HOTFIX filter
    → apply_augmentation()          # [NEW] scale/jitter/lifetime/links
    → init simulation state

  step(action)
    → softmax(action) → allocation proportions
    → update MPD loads, schedule deallocation
    → reward = -(delta_peak)/fair_share - lambda*var(loads)
    → advance to next event
```

## Topology

`M` is a `list[list[int]]` adjacency matrix (hosts x MPDs). `host_to_mhds` dict maps each
host to its list of accessible MPD indices — this is the runtime accessor used in `step()`
and `_get_obs()`.

For topology perturbation (link failures), `host_to_mhds` must be recomputed from the
augmented topology each episode.

## Augmentation Pipeline

Applied in `reset()` after event construction, before simulation init:

1. `scale_memory(events, factor)` — uniform factor on all VM memory
2. `jitter_arrivals(events, max_shift, rng)` — ±tick shifts
3. `perturb_lifetimes(events, frac, rng)` — ±% lifetime noise
4. `add_memory_noise(events, sigma, rng)` — per-VM multiplicative noise
5. Re-sort events by (tick, host, -mem)
6. `inject_link_failures(M, ratio, rng)` — random CXL link removal
7. Recompute `host_to_mhds` from augmented topology
8. Validity guard: reject if events < min_events, resample

Augmentation does NOT touch step logic, reward, observation space, or action space.

## Reward Function

Current (single formula in `env.py` `step()`):
```
reward = -(new_peak - old_peak) / fair_share - λ * Var(mpd_loads)
```
- `fair_share = pod_dram / num_mhd`
- `variance_lambda` (λ) is a CLI arg, default 0.5

Plan-v2 Task 9 will add alternative reward formulations (peak-only, post-alloc peak,
CV-based, sparse). These will require parameterized reward selection in `step()`.

## Evaluation Metric

Primary metric: `pooling_ratio = max_peak * num_mhd / pod_dram` (lower = better).
`savings = 1.0 - pooling_ratio`. Both are computed in `step()` info dict at episode end
and in `eval_rl.py`.

## Training Pipeline (plan-v2 additions)

```
scripts/train_rl.py
  → DummyVecEnv (N=1) or SubprocVecEnv (N=--n-envs)     [plan-v2]
  → SAC or PPO model
  → Callbacks: CheckpointCallback, EvalCallback, PoolingSavingsCallback,
    AugmentationLogCallback (if aug), WandbCallback (if --wandb)     [plan-v2]
  → W&B logging: SAC diagnostics, env metrics, aug params             [plan-v2]
```
