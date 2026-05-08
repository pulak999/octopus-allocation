# Training Throughput Speedup Plan

**Goal:** Cut 2M-step training wall time from ~10.5 hr → ≤3 hr so 6 experiments fit in one day.

**Constraint:** No algorithmic changes. Learning curves must remain comparable to `run7` / `aug_v4_rewardA_v2` at matched seeds so any policy-quality regression is attributable to reward/state work, not throughput work.

---

## 1. Current State (measured)

From `run7.txt` and `output/checkpoints/aug_v4_rewardA_v2/config.json`:

| Metric | Value |
|---|---|
| Config | `reward_variant=A`, `n_envs=32`, `train_freq=8`, `gradient_steps=1`, `batch_size=256`, `multi_trace=true` (10 traces), `augmentation=true`, `fast=false` |
| Total timesteps | 2,000,000 |
| Wall time (SB3 `time_elapsed`) | 37,717 s ≈ **10.47 hr** |
| Steady-state `fps` | 50–56 (≈ **53 steps/sec**) |
| GPU util (training) | <10% (per `DEBUG_SLOW_TRAINING.md`) |
| Hardware | 3× TITAN RTX; train on cuda:0, async eval on cuda:1 |
| Reset time (multi-trace, no cache) | ~120 ms (per `DEBUG_SLOW_TRAINING.md` Q1b) |

**Gap vs. benchmark claim** (napkin: 1221 fps at N=32) — **~23×**. Bottleneck is CPU Python in the variant-A env hot path, not GPU kernels.

---

## 2. Target

- 2M steps in ≤ 3 hr → **≥ 185 fps**
- Minimum speedup required: **~3.5×**
- With headroom (to absorb augmentation variance), aim for **5×** → ~265 fps → ~2.1 hr per 2M run.

---

## 3. Scope

### In scope (this plan)

1. Vectorize variant-A hot loops in `octopus/env.py`.
2. Extend the `precompute_pod_events` cache to multi-trace.
3. Numba-JIT the remaining Python-hot inner kernels in the env and eval sim.

### Explicitly out of scope

- **Raising `train_freq` / `gradient_steps` / `batch_size` (item 3 from earlier discussion).** Changes algorithm behavior; defer until after the current reward-quality investigation completes. Captured in §9 as a future lever.
- **Moving env to GPU (Isaac-Gym style).** Event-driven, sparse, branchy; wrong shape for SM work. GPU util is <10%, so this solves a bottleneck we don't have. Defer indefinitely.
- **SM pinning / memcpy optimizations.** Only helps when GPU-bound. We're CPU-bound.

---

## 4. Pre-work: Profile to Confirm Diagnosis

**Before touching code**, produce a py-spy flamegraph to validate which functions are actually hot. If top-3 functions don't include `_compute_D_j`, `_get_obs` (variant A branch), or `_process_departures_through`, the plan below is wrong and we stop.

```bash
# Terminal 1
source venv/bin/activate
python scripts/train_rl.py --run-id speedprobe \
  --reward-variant A --multi-trace --augmentation \
  --n-envs 32 --total-timesteps 50000 --no-wandb

# Terminal 2 (after training_rl has passed learning_starts)
pip install py-spy
sudo py-spy record -o speedprobe.svg \
  --pid <train_rl_main_pid> --duration 90 --subprocesses
```

**Acceptance:** At least one of `_compute_D_j` / `_get_obs` / `_process_departures_through` / `_build_events` appears in the top-5 self-time functions across subprocess workers.

---

## 5. Change (1): Vectorize Variant-A Hot Paths

**File:** `octopus/env.py`

### Current hot pattern

Three places each run `O(num_mhd · |active VMs|)` Python loops **every `step()`**:

- `_compute_D_j` (called once per step in variant A/B for the reward)
- `_get_obs` variant-A branch (called once per step)
- `_process_departures_through` (list-comp rebuild of `mpd_vm_allocs[j]` for every MPD, every tick gap)

All operate on `self.mpd_vm_allocs: list[list[tuple[int, float]]]`. With 192 MPDs and variable per-MPD VM lists, this is the documented bottleneck.

### Proposed change

Replace the list-of-lists-of-tuples with two flat numpy arrays and a count pointer:

```python
# Pre-allocated once per episode in reset()
self._mpd_dt   = np.zeros((self.num_mhd, MAX_SLOTS), dtype=np.int32)   # dealloc ticks
self._mpd_mem  = np.zeros((self.num_mhd, MAX_SLOTS), dtype=np.float32) # mem GB
self._mpd_n    = np.zeros(self.num_mhd, dtype=np.int32)                # valid count per row
```

`MAX_SLOTS` sized from the max per-MPD concurrent VMs observed in the trace (conservative; fall back to dynamic doubling if exceeded).

Each hot function collapses to vectorized numpy:

- **`_compute_D_j`** → one `np.where` mask + `np.einsum` / `np.sum` across axis=1. No Python loop over `j`.
- **`_get_obs`** variant-A → same D_j/S_j computation reused; P_j becomes a single indexed gather.
- **`_process_departures_through`** → numpy boolean mask per-MPD in one broadcast, no per-MPD list-comp.

### Expected speedup

**3–6× on `step()`** (the dominant time sink). Net effect on wall time: ~3–4×.

### Correctness strategy

- Add a unit test `tests/test_env_vectorization.py` that:
  - Seeds a `reward_variant="A"` env with fixed seed.
  - Runs N=200 steps collecting `(obs, reward)` after each.
  - Compares to the current (list-based) implementation's trace at the same seed.
  - Asserts `np.allclose(obs_new, obs_old, atol=1e-6)` and `reward_new == reward_old` elementwise.
- Before merging, run a 50k-step smoke training and diff the first 5 `eval/pooling_savings_mean` values against run7's early values at matched seeds.

### Risks

- **`MAX_SLOTS` overflow** on atypical pods. Mitigation: track observed max during training; if we ever saturate, grow arrays 2×.
- **Dtype drift** — old code uses `float` (≡ float64); switching to float32 in the MPD arrays could shift rewards by ~1e-6. Test will catch; switch to float64 if needed.

---

## 6. Change (2): Multi-Trace Precompute Cache

**Files:** `octopus/data.py`, `octopus/env.py`

### Current behavior

`precompute_pod_events` returns `{seed → events}` keyed by pod seed only. When `multi_trace=True`, `reset()` swaps traces via `_switch_trace` but the cache (`self._precomputed_events`) is still from the original trace, so it's never hit. Every reset falls through to the full `_build_events` path (~120 ms / reset per Q1b in `DEBUG_SLOW_TRAINING.md`).

### Proposed change

- Change cache key to `(trace_idx, seed)`.
- In `train_rl.py`, when building `trace_pool`, also precompute events for each trace: `trace_pool_events = [precompute_pod_events(td, M, N_SEEDS) for td in trace_pool]`. N_SEEDS sized to cover the episode count expected per worker (e.g. 2048).
- Pass both the trace and its matching precomputed cache into the env.
- `_switch_trace` now swaps both `self.trace_arrays` and `self._precomputed_events` together.

### Expected speedup

Per Q1c: 120 ms → 7 ms per reset = **16.6× on reset**. Reset cost is a smaller fraction of step cost, so overall training speedup is **1.3–1.8×**.

### Correctness strategy

- Add `tests/test_multi_trace_cache.py`:
  - Reset env with `multi_trace=True`, cache enabled, 10 seeds × 10 trace swaps.
  - Same setup with cache disabled.
  - Assert `events` lists are identical (same ticks, same memory values, same VM ids).

### Risks

- **Memory footprint.** 10 traces × 2048 pod seeds × ~5k events × ~40 bytes ≈ 4 GB RAM in main process. Mitigation: precompute fewer seeds (512); envs with `seed > 512` fall through to the slow path (rare).
- **Fork duplication across 32 workers.** Since `mp.set_start_method("spawn")` is already forced, precomputed caches are serialized once per subprocess at spawn. Adds ~30–60 s to startup; acceptable vs. 10-hr training.

---

## 7. Change (3): Numba JIT

**File:** `octopus/env.py` (decorators only; no API change)

Committed step. Run after (1)+(2) land so the JIT'd helpers operate on the vectorized flat-array state.

### Targets

- Vectorized `_compute_D_j` kernel from (1).
- `_build_events` HOTFIX inner loop in `env.py` lines ~404–432 (per-node cumulative-load sweep).
- `pooling_simulation` inner sweep in `scripts/evaluate.py` (for the eval worker, not training).

Add `@njit(cache=True)` to pure-numpy helpers extracted from these functions.

### Expected speedup

**2–5× on the JIT'd function**, translating to ~1.3–2× overall depending on remaining hot-path share.

### Risks

- Adds `numba>=0.58` to `requirements.txt`.
- JIT compile cost on first invocation (~1–3 s). Cache via `@njit(cache=True)` so later runs skip it.
- Numba doesn't support Python objects / dicts in JIT'd code — any extraction needs pure numpy inputs. Low risk since (1) already moves us that way.

---

## 8. Validation & Rollout

### 8.1 Per-change acceptance gates

Run in order; do not merge change N+1 until N is green.

1. **After (1):**
   - Unit test `tests/test_env_vectorization.py` passes.
   - 50k-step smoke training: `eval/pooling_savings_mean` at step 20k/40k within 15% of run7's values at matched seeds.
   - Throughput ≥ 150 fps (3× baseline).

2. **After (2):**
   - Unit test `tests/test_multi_trace_cache.py` passes.
   - Throughput ≥ 200 fps combined.

3. **After (3):**
   - All existing tests still pass (`pytest tests/`).
   - Throughput ≥ 250 fps combined.

### 8.2 Full-run calibration (before the 6-experiment sprint)

Run **one** 500k-step calibration with:

- Same config as run7 (`reward_variant=A`, `multi_trace=true`, `n_envs=32`, unchanged SAC kwargs).
- Seed = 42 (matches run7).

**Pass criteria:**

- Wall time ≤ 45 min (extrapolates to ≤ 3 hr for 2M).
- `eval/mean_reward` curve overlaps run7's within ±1 σ at steps 100k / 200k / 500k.
- `eval/pooling_savings_mean` at step 500k within 0.005 absolute of run7's.

If calibration fails, bisect: revert (3), then (2), then (1). Each revert is one file.

### 8.3 The 6-experiment sprint

Only launch after calibration passes. Use a small runner script that spawns runs sequentially (not in parallel — GPU is shared between train + async eval worker, so two trainings on the same card will contend).

---

## 9. Out-of-Scope Future Lever (for reference, not this plan)

If after this plan 3 hr is still tight, the next lever is **SAC hyperparameter change**: `train_freq=32, gradient_steps=8, batch_size=1024, learning_starts=10000`. Doubles UTD, gives another ~1.3–1.5× throughput, but changes learning dynamics. Run a dedicated calibration before adopting. Not in this plan.

---

## 10. Deliverables

New / modified files:

| File | Change |
|---|---|
| `octopus/env.py` | Vectorize `_compute_D_j`, `_get_obs` (variant A), `_process_departures_through`. Flat-array `mpd_vm_allocs`. |
| `octopus/data.py` | `precompute_pod_events` returns keyed-by-trace cache; new helper to precompute for a pool. |
| `scripts/train_rl.py` | Build per-trace precompute caches when `multi_trace=True`; pass to env. |
| `tests/test_env_vectorization.py` | NEW — regression vs. list-based impl. |
| `tests/test_multi_trace_cache.py` | NEW — cache-hit correctness. |
| `requirements.txt` | `numba>=0.58`. |
| `DEBUG_SLOW_TRAINING.md` | Append a §6 with post-change numbers and new baseline fps. |

---

## 11. Estimated Effort

| Task | Wall time |
|---|---|
| Profile (§4) | 30 min |
| Change (1) implement + test | 3–4 hr |
| Change (2) implement + test | 1–2 hr |
| Change (3) implement + test | 1–2 hr |
| Calibration run (§8.2) | 45 min wall, 15 min attention |
| **Total before sprint** | **~5–8 hr active work** |

---

## 12. Open Questions for You

1. **`MAX_SLOTS` sizing for the flat `mpd_vm_allocs` array in Change (1)** — OK to dynamically grow 2× on overflow, or do you want a hard cap (memory-safer, risks dropping VMs)? — *Resolved: dynamic grow 2×.*
2. **Cache seed budget in Change (2)** — 512 pod-seeds × 10 traces is ~2 GB main-process RAM. OK to allocate? — *Resolved: yes.*
3. **Calibration run acceptance threshold (§8.2)** — ±1 σ on `mean_reward`. — *Resolved: ±1 σ at 100k / 200k / 500k.*
4. **Go/no-go on Change (3) (Numba)** — *Resolved: committed step, runs after (1)+(2).*
