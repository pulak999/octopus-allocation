# Architecture

## Components

```
octopus/
  env.py          — OctopusMemPoolEnv (Gymnasium): reset/step/obs/reward
  data.py         — VM class, load_trace(), precompute_pod_events_arrays(), load_topology()
  kernels.py      — Numba @njit kernels: compute_D_j, compute_D_S_j
  baselines.py    — greedy_alloc(), pid_alloc() allocation policies
  topology.py     — Pod generation, matrix expansion, link failure injection
  optimal.py      — Dinic max-flow solver for optimal per-MPD peak
  augmentation.py — AugmentationConfig, transform functions, apply_augmentation()
  jax/            — [PLANNED] Pure-JAX SAC trainer (see docs/plans/v4/jax-plan.md)
    env.py        —   reset_fn / step_fn (pure JAX, static shapes)
    sac.py        —   MLP actor, twin critics, α, replay buffer, update step

scripts/
  train_rl.py     — SB3 SAC/PPO training with callbacks
  train_jax.py    — [PLANNED] JAX SAC training entrypoint
  evaluate.py     — pooling_simulation() + allocation callbacks
  eval_rl.py      — Multi-trace RL checkpoint evaluation
  eval_baselines.py, plot_*.py, diagnose.py

tests/
  test_greedy_alloc.py        — greedy fast vs ref regression
  test_augmentation.py        — augmentation unit + integration tests
  test_env_vectorization.py   — flat MPD array correctness
  test_multi_trace_cache.py   — precompute cache vs. live event equivalence
  test_async_eval.py          — async eval worker lifecycle
  test_jax_env_parity.py      — [PLANNED] JAX env parity vs SB3 (float64 mode)
```

## Data Flow

**SB3 path (existing):**
```
Trace pickle
  → load_trace() → to_arrays() → TraceArrays (numpy CSR)
  → OctopusMemPoolEnv(trace_arrays, M, aug_config)

Episode lifecycle:
  reset(seed)
    → _generate_pod(seed)                  # Python stdlib random.shuffle
    → _build_events()                      # VM arrivals + HOTFIX filter
    → [_apply_augmentation()]              # scale/jitter/lifetime/links if enabled
    → zero simulation state
    → _process_departures_through(first_tick)

  step(action)
    → softmax(action[:n_accessible_mpds])
    → update _mpd_dt/_mpd_mem/_mpd_n, dealloc_events, host_dealloc_events
    → compute reward (variant-specific)
    → advance event_idx → _process_departures_through(next_tick)
    → _get_obs() via Numba kernels
```

**JAX path (planned — octopus/jax/):**
```
OctopusState (@chex.dataclass):
  mpd_load (num_mhd,) float32       — current CXL load per MPD
  dealloc_buf (MAX_TICKS, num_mhd)  — scheduled MPD deallocations per tick
  host_load (pod_size,) float32     — current CXL load per host
  host_dealloc_buf (MAX_TICKS, pod_size) float32
  max_peak, event_idx, last_depart_tick, key
  NOTE: _mpd_dt/_mpd_mem/_mpd_n are NOT in state (overflow at 2007 total allocs/MPD)

D_j/S_j from dealloc_buf (no separate VM tracking needed):
  weight[t] = max(0, 1 - (t - tick) / W) for t ∈ (last_depart_tick, tick+W]
  D_j = einsum('t,tj->j', weight, dealloc_buf)   # mathematically = Numba kernel
  S_j = einsum('t,tj->j', (t > tick+W), dealloc_buf)

Static config (compiled per run):
  events (MAX_EVENTS, 4), host_to_mhds mask, mhd_to_hosts mask, Q_j, pod_dram,
  reward_variant, lookahead_window, reward_lambda

reset_fn(key, static_config):
  host: _generate_pod(seed) + _build_events() + pad → device_put
  device: zero OctopusState → process departures to first tick → _get_obs_fn

step_fn(state, action, static):
  → softmax over accessible MPDs
  → scatter alloc into dealloc_buf at clip(dealloc_tick, 0, MAX_TICKS-1)
  → batch-subtract departures via einsum (no inner scan, no Python loop)
  → D_j/S_j via einsum over dealloc_buf (no _mpd_dt/_mpd_mem arrays)
  → reward dispatch to JIT-compiled variant kernel
  → return (new_state, obs, reward, done)
```

## Topology

`M` is a `list[list[int]]` adjacency matrix (hosts x MPDs). `host_to_mhds` dict maps each
host to its list of accessible MPD indices — this is the runtime accessor used in `step()`
and `_get_obs()`.

For topology perturbation (link failures), `host_to_mhds` must be recomputed from the
augmented topology each episode.

## Augmentation Pipeline

## Precompute Cache (Multi-Trace)

`precompute_pod_events_arrays(trace_arrays, M, seeds)` produces `{seed: (events, pod_dur, pod_dram, base_time, pod_start_ts)}` from a `TraceArrays` (not the raw pickle tuple). When `multi_trace=True`, `train_rl.py` builds one cache per trace in the pool, and `env._switch_trace(trace_arrays, precomputed_events)` swaps both together — so multi-trace resets actually hit the cache (fixed vs. pre-SPEEDUP_PLAN state).

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

Six variants selected via `reward_variant` constructor param. "A" and "B" are deprecated aliases
for "R2" and "R3" (normalised in `__init__` via `_VARIANT_ALIASES`). Canonical names:

| Variant | Paper | Obs | Description |
|---|---|---|---|
| `"current"` | Run-6 | 20-dim | `-(Δpeak)/fair_share - λ·Var(loads)` |
| `"R1"` | R1 | 20-dim | `-max_j(c_j) / D_pod` — worst MPD, no departure awareness |
| `"R2"` (alias `"A"`) | R2 | 50-dim | `-max(ĉ_j+(t) for j ∈ N(i))` — localized, departure-aware |
| `"R3"` (alias `"B"`) | R3 | 50-dim | R2 `- λ·max(ĉ_j for j ∉ N(i))` — adds global stress term |
| `"R4"` | R4 | 50-dim | Sparse terminal: `pooling_savings` at done, 0 otherwise |
| `"R5"` | R4+PBRS+Sub | 50-dim | R4 + PBRS shaping (`γΦ(s')-Φ(s)`) + sub-episode resets |

**Obs-dim rule**: `_SIMPLE_OBS = {"current", "R1"}` → `2*max_degree+4`. All others → `6*max_degree+2`.

Where `ĉ_j(t) = (c_j(t) - D_j(t, W)) / D_pod` is the departure-adjusted projected load.
`D_j(t, W)` is the time-weighted sum of memory from VMs departing within W steps.

R5 adds `sub_episode_len=576` and `pbrs_gamma=0.99` constructor params. `Φ(s) = -max_j(c_j)/D_pod`.
R6 (optimal-gap shaping via Dinic max-flow) is **gated** until R1–R5 sweep completes.

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
- `_mpd_dt`: `(num_mhd, MAX_SLOTS) int32` — dealloc ticks of active VMs, flat array
- `_mpd_mem`: `(num_mhd, MAX_SLOTS) float64` — memory of those VMs
- `_mpd_n`: `(num_mhd,) int32` — valid VM count per MPD row
- `cur_host_cxl_load`: per-host current CXL load (GB)
- `host_dealloc_events`: per-host scheduled deallocations
- `mhd_to_hosts`: inverse of host_to_mhds — MPD → list of connected hosts
- `Q_j`: precomputed neighbor scarcity per MPD (static per topology, recomputed on link failure)

The flat `_mpd_dt` / `_mpd_mem` arrays replace the per-MPD Python list of tuples. All hot
paths (`_compute_D_j`, `_get_obs` variant A/B, `_process_departures_through`) operate on these
arrays via masked numpy ops — never iterate in Python. `MAX_SLOTS` grows 2× on overflow.

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
