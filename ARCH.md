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

Three variants selected via `reward_variant` constructor param:

- **"current"** (default): `-(Δpeak)/fair_share - λ·Var(loads)` — 20-dim obs
- **"A"**: `R = -max(ĉ_j+(t) for j ∈ N(i))` — 50-dim obs
- **"B"**: `R_A - λ · max(ĉ_j(t) for j ∉ N(i))` — 50-dim obs

Where `ĉ_j(t) = (c_j(t) - D_j(t, W)) / D_pod` is the departure-adjusted projected load.
`D_j(t, W)` is the time-weighted sum of memory from VMs departing within W steps.

Reward A lies in (-1, 0]. Reward B lies in (-(1+λ), 0].

## Evaluation Metric

Primary metric: `pooling_ratio = max_peak * num_mhd / pod_dram` (lower = better).
`savings = 1.0 - pooling_ratio`. Both are computed in `step()` info dict at episode end
and in `eval_rl.py`.

## Observation Space

**"current" variant (20-dim):**
`[loads(d_max), mask(d_max), vm_norm, peak_norm, hour_sin, hour_cos]`

**"A" / "B" variants (50-dim for AG16x6, d_max=8):**
Per-MPD slot k (×d_max): `[c_j/D_pod, D_j/D_pod, S_j/D_pod, mask, P_j/D_pod, Q_j]`
Global: `[global_peak/D_pod, vm_mem/D_pod]`

New state tracked in env.py for new variants:
- `mpd_vm_allocs`: per-MPD list of `(dealloc_tick, mem_gb)` for active VMs
- `cur_host_cxl_load`: per-host current CXL load (GB)
- `host_dealloc_events`: per-host scheduled deallocations
- `mhd_to_hosts`: inverse of host_to_mhds — MPD → list of connected hosts
- `Q_j`: precomputed neighbor scarcity per MPD (static per topology, recomputed on link failure)

**CRITICAL:** `make_rl_alloc_cb` in evaluate.py manually mirrors `_get_obs()`. Any obs layout
change must be reflected in both.

## Training Pipeline

```
scripts/train_rl.py
  → DummyVecEnv (N=1) or SubprocVecEnv (N=--n-envs)
  → SAC model (--reward-variant current|A|B)
  → Callbacks: CheckpointCallback, EvalCallback, PoolingSavingsCallback,
    AugmentationLogCallback (if aug), WandbCallback (if --wandb)
  → W&B logging: SAC diagnostics, env metrics, aug params
```

## Async Eval Architecture

`PoolingSavingsCallback` spawns a persistent worker process at `on_training_start()`.
Training never blocks on eval; the worker runs concurrently on a separate GPU.

```
Main process (cuda:0, training)         Worker process (cuda:1, eval)
──────────────────────────────          ──────────────────────────────
on_training_start()
  spawn worker with policy_cpu,
  eval_trace_data, topology, config ──→ imports octopus + scripts.evaluate
                                        moves policy to cuda:1
                                        enters cmd_q.get() wait loop

every eval_freq steps (_on_step):
  res_q.get_nowait() → log if ready ←── put (mean, std, snap_step) on res_q
  deepcopy state_dict → CPU
  cmd_q.put(('eval', sd, step))     ──→ load state_dict onto cuda:1 policy
  return True (training resumes)        run n_iter × pooling_simulation
                                        put result on res_q

on_training_end():
  cmd_q.put(('stop',))              ──→ exit
  worker.join(timeout=300)
  drain res_q, log final result
```

**Key functions:**
- `_eval_worker_main(cmd_q, res_q, policy_cpu, eval_trace_data, M, ...)` — top-level worker
- `PoolingSavingsCallback.on_training_start()` — spawns worker
- `PoolingSavingsCallback._on_step()` — non-blocking collect + async dispatch
- `PoolingSavingsCallback.on_training_end()` — graceful shutdown + drain

**Spawn is mandatory** (`mp.set_start_method("spawn", force=True)` in `main()`).
Fork + CUDA = silent corruption.
