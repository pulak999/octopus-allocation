# Phase 0 + 0.5 Implementation Plan (v2)

Agent instructions for completing Phase 0 and Phase 0.5 of the Octopus CXL RL project.
All paths relative to `/home/pm3371/gitrepos/octopus-allocation/`. Run with venv active.

**What is already done (do not redo):**
- `octopus/env.py`: variance_lambda reward shaping, constructor arg
- `scripts/train.py` / `scripts/train_rl.py`: deterministic seeding, `--variance-lambda`, `PoolingSavingsCallback`
- `scripts/evaluate.py`: `pid` policy, `make_pid_alloc_cb`, fail-fast RL model check + md5 log
- `octopus/baselines.py`: `pid_alloc`
- `scripts/diagnose.py`: 4 diagnostic plots (weight dist, MPD loads, variance, ratio)
- `scripts/eval_baselines.py`: idempotent CSV sweep, tqdm, LaTeX table printer
- `scripts/eval_rl.py`: per-run-id results + episode_detail CSVs
- `scripts/plot_results.py`: 5 figures from saved CSVs
- `plot_memory_pooling_sweep.py` Task 1a bug: already fixed

---

## Task A — SAC entropy + Q-value diagnostics in `diagnose.py`

**File:** `scripts/diagnose.py`

Add two new plots to the existing script (keep the existing 4 plots, add 5 and 6).

### Plot 5 — SAC entropy proxy (`entropy_proxy.png`)

SAC doesn't expose H(π) directly at inference time, but we can approximate it from the
action distribution standard deviation logged during a rollout.

At each step of the RL episode (already run in `_run_rl_episode`):
- Call `model.actor.get_distribution(obs_tensor)` to get the action distribution.
- Record `dist.distribution.entropy().mean().item()` (mean entropy across action dims).
- Plot: x = step index, y = mean action entropy.
- Add a horizontal dashed line at `entropy = 0` for reference.
- Goal: near-zero entropy = policy has collapsed to near-deterministic; healthy = nonzero
  throughout.

Implementation notes:
- Wrap obs in `torch.tensor(...).unsqueeze(0)` and move to `model.device`.
- Call inside the existing episode loop in `_run_rl_episode` — add `entropies` list
  alongside `weights_per_step`.
- Return `entropies` from `_run_rl_episode`; pass to plotting code.
- Only runs if `model.actor` has `get_distribution` (SAC-specific; skip gracefully for
  other model types with a warning).

### Plot 6 — Q-value vs Monte Carlo return (`qvalue_vs_return.png`)

Run one full episode, record at each step:
- Predicted Q-value: `min(model.critic(obs_tensor, action_tensor))` — use the min of the
  two critic heads (conservative Q estimate, matches SAC training target).
- Monte Carlo return: compute offline after the episode as
  `G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}` where `gamma=0.999` (match training).

Plot: scatter of predicted Q (x-axis) vs MC return (y-axis), one point per step.
Overlay `y = x` identity line. Points above the line = overestimation (reward hacking
signal). Points below = underestimation (safe but pessimistic).

Implementation notes:
- Record `(obs, action, reward)` at each step inside the existing episode loop.
- After episode ends, compute MC returns in reverse with a single numpy cumsum pass:
  `G = np.zeros(T); G[-1] = rewards[-1]; for t in range(T-2,-1,-1): G[t] = rewards[t] + 0.999 * G[t+1]`
- For Q-values: batch all obs+actions into tensors, call `model.critic(obs_batch,
  act_batch)` once (faster than per-step calls).

---

## Task B — Reward smoothness verification

**File:** `scripts/diagnose.py`

Add a CLI flag `--check-smoothness` (default off). When set, after loading the model:

1. Sample 100 random observations from the episode (use the existing episode run).
2. For each obs, compute the gradient of `reward` w.r.t. the raw action logits via
   finite differences: perturb each logit dimension by ±ε=1e-4, recompute the softmax
   allocation, recompute the peak-change reward component, record the finite-difference
   gradient.
3. Report: `mean |∂R/∂a_i|` per action dimension, and flag any dimension where the
   gradient is identically zero across all 100 samples (indicates dead reward signal for
   that MPD slot).
4. Print a summary table to stdout; no plot needed.

This is a one-time sanity check, not a training loop diagnostic.

---

## Task C — Speed up `pooling_simulation` (`scripts/evaluate.py`)

This is the hottest function — called `n_iter × n_combos` times in the baseline sweep.

### C1. Precompute `mhd_list` per node

In the tick-sweep loop (step 4), `mhd_list` is currently recomputed from scratch for
every VM arrival:
```python
# BEFORE (inner loop, called once per VM)
mhd_list = [mhd for mhd in range(num_mhd) if M[node_in_pod_id][mhd] != 0]
```

Replace with a precomputed dict built once before the tick sweep:
```python
# AFTER (built once, O(pod_size * num_mhd))
node_mhd_lists = {
    node_id: [mhd for mhd in range(num_mhd) if M[node_id][mhd] != 0]
    for node_id in range(len(M))
}
```
Then in the inner loop: `mhd_list = node_mhd_lists[node_in_pod_id]`

### C2. Cache VM memory once per VM, convert ticks upfront

In both the HOTFIX pass (step 2) and event-build pass (step 3), each VM's `rss` is
converted via `np.asarray(vm.rss, dtype=float)` and `to_tick` is called separately.

Consolidate into a single preprocessing pass at the start of the function:

```python
# At the top of pooling_simulation, after computing base/pod_start_ts:
base_sec = base.timestamp()  # float seconds since epoch (computed once)

vm_cache = {}  # vmkey -> (vm_tick_start, vm_tick_end, mem_gb)
for cur_node in node_to_M.keys():
    for cur_vmkey in node_to_vms.get(cur_node, []):
        if cur_vmkey in vm_cache:
            continue
        cur_vm = all_vms[cur_vmkey]
        vm_tick_start = int((cur_vm.start_time.timestamp() - base_sec) // 300) - pod_start_ts
        vm_tick_end   = int((cur_vm.end_time.timestamp()   - base_sec) // 300) - pod_start_ts
        mem = float(cur_vm.rss[MEM_IDX])
        vm_cache[cur_vmkey] = (vm_tick_start, vm_tick_end, mem)
```

Then replace all `to_tick(cur_vm.start_time)` and `np.asarray(cur_vm.rss)` calls in
steps 2 and 3 with lookups into `vm_cache`.

Note: `datetime.timestamp()` is faster than `(t - base).total_seconds()` because it
avoids constructing a `timedelta` object.

### C3. Replace list-of-lists with flat sorted event list

`alloc_events_sim` is currently a list of length `pod_dur` (≈ 4032 ticks for a 14-day
trace), most slots empty. The tick sweep iterates all `pod_dur` slots even when most
have no events.

Replace with a flat sorted list + pointer:
```python
# Build flat event list (after HOTFIX filtering):
flat_events = []  # (tick, node_in_pod_id, dealloc_tick, mem)
for cur_node, node_in_pod_id in node_to_M.items():
    for cur_vmkey in node_to_vms.get(cur_node, []):
        if cur_vmkey in vmkey_to_skip:
            continue
        vm_s, vm_e, mem = vm_cache[cur_vmkey]
        if mem <= 0:
            continue
        flat_events.append((vm_s, node_in_pod_id, vm_e + 1, mem))
flat_events.sort()

# Sweep only non-empty ticks:
ev_idx = 0
n_ev = len(flat_events)
prev_tick = -1

while ev_idx < n_ev:
    tick = flat_events[ev_idx][0]

    # Apply all deallocs from prev_tick+1 to tick (vectorised slice sum)
    if tick > prev_tick + 1:
        cur_cxl_mem_vec -= dealloc_events[prev_tick+1:tick+1, :].sum(axis=0)
    else:
        cur_cxl_mem_vec -= dealloc_events[tick, :]
    cur_cxl_mem_vec = np.maximum(cur_cxl_mem_vec, 0.0)
    prev_tick = tick

    # Process all events at this tick
    while ev_idx < n_ev and flat_events[ev_idx][0] == tick:
        _, node_in_pod_id, dealloc_time, mem = flat_events[ev_idx]
        ev_idx += 1
        mhd_list = node_mhd_lists[node_in_pod_id]
        # ... alloc_fn call, dealloc scheduling (unchanged)

    max_cxl_mem_vec = np.maximum(max_cxl_mem_vec, cur_cxl_mem_vec)
```

### C4. Remove per-VM ctx dict allocation

`ctx` is a dict with 5 compile-time-constant fields, allocated fresh for every VM.
For greedy and PID, only `num_mhd` and `pod_rss_mem` are actually read.

Extract these as local variables before the tick sweep and pass them directly:
```python
# Before tick sweep:
_num_mhd = num_mhd
_pod_rss_mem = float(pod_rss[MEM_IDX])
_pod_start_ts = pod_start_ts
_base = base

# In inner loop, replace dict construction with:
ctx = {
    "tick": tick,
    "pod_start_ts": _pod_start_ts,
    "base_time": _base,
    "num_mhd": _num_mhd,
    "pod_rss_mem": _pod_rss_mem,
}
```

This keeps the ctx interface unchanged (RL callback still needs `base_time`, `tick`) but
avoids reconstructing the constant fields. The net saving is small but measurable at
scale.

### Verification for Task C

Add a timing wrapper at the top of `_run_combo` in `eval_baselines.py`:
```python
import time
t0 = time.perf_counter()
ratio = pooling_simulation(...)
wall = time.perf_counter() - t0
```

Run the smoke test before and after C1–C4 and print speedup:
```bash
# Before (revert temporarily or keep old function as pooling_simulation_ref):
python scripts/eval_baselines.py --policies greedy --n-iter 5 \
  --traces AMS20PrdApp19-tround \
  --topologies data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv

# Target: wall_sec/iter should drop by ≥3× vs baseline
```

---

## Task D — Speed up `greedy_alloc` (`octopus/baselines.py`)

### D1. Write regression test first

Create `tests/test_greedy_alloc.py`:

```python
import numpy as np
import pytest
from octopus.baselines import greedy_alloc

def _random_input(seed, n_mhd=6, n_acc=4):
    rng = np.random.default_rng(seed)
    cur = rng.uniform(0, 100, size=n_mhd)
    mhd_list = sorted(rng.choice(n_mhd, size=n_acc, replace=False).tolist())
    cxl_mem = float(rng.uniform(1, 50))
    return cxl_mem, mhd_list, cur

@pytest.mark.parametrize("seed", range(20))
def test_greedy_fast_matches_ref(seed):
    from octopus.baselines import greedy_alloc_ref, greedy_alloc
    cxl_mem, mhd_list, cur = _random_input(seed)
    ref = greedy_alloc_ref(cxl_mem, mhd_list, cur.copy())
    fast = greedy_alloc(cxl_mem, mhd_list, cur.copy())
    assert np.allclose(ref, fast, atol=1e-9), f"seed={seed}: ref={ref} fast={fast}"
```

### D2. Implement fast vectorised `greedy_alloc`

Keep the existing implementation as `greedy_alloc_ref` (rename, don't delete).
Replace `greedy_alloc` with:

```python
def greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec):
    mhd_arr = np.asarray(mhd_list)
    loads = cur_cxl_mem_vec[mhd_arr].copy()
    n = len(mhd_arr)
    alloc = np.zeros(len(cur_cxl_mem_vec))

    # Sort by load ascending; fill levels from lowest up
    order = np.argsort(loads, kind="stable")
    sorted_loads = loads[order]

    remaining = float(cxl_mem)
    for i in range(n):
        # How many MPDs share the current minimum level?
        level = sorted_loads[i]
        # Next level (or +inf if last)
        next_level = sorted_loads[i + 1] if i + 1 < n else np.inf
        count = n - i  # all MPDs at or above this level can receive fill

        # Room to fill up to next_level across all `count` slots
        gap = (next_level - level) * count
        if remaining <= gap:
            each = remaining / count
            alloc[mhd_arr[order[i:]]] += each
            break
        else:
            each = next_level - level
            alloc[mhd_arr[order[i:]]] += each
            remaining -= gap
            # Advance i: next iteration starts at next distinct level

    return alloc
```

Note: the loop runs at most `degree` iterations (≤ 8), so it is O(degree log degree)
dominated by `argsort`. The Python loop body is now free of Python list scans.

### D3. Run regression test, then replace

```bash
python -m pytest tests/test_greedy_alloc.py -v   # must pass all 20 seeds
```

Only after all tests pass: remove `greedy_alloc_ref` and mark D complete.

---

## Task E — Parallel workers in `eval_baselines.py`

### E1. Add `--n-workers` argument

```python
ap.add_argument("--n-workers", type=int, default=os.cpu_count(),
                help="Parallel workers for combo evaluation (1 = serial)")
```

### E2. Worker function

Extract `_run_combo` (already exists) into a top-level function (not a closure) so it
can be pickled by `multiprocessing`. Add `wall_sec` to its return dict:

```python
def _run_combo(args_tuple):
    policy, topology_path, trace_name, n_iter, kp, ki, kd = args_tuple
    import time
    t0 = time.perf_counter()
    # ... existing body ...
    wall = time.perf_counter() - t0
    result["wall_sec"] = round(wall, 2)
    return result
```

### E3. Pool execution with atomic CSV append

```python
from multiprocessing import Pool
import tempfile, shutil

todo_args = [
    (policy, topology_path, trace_name, args.n_iter, args.kp, args.ki, args.kd)
    for policy, topology_path, trace_name in combos_todo
]

with Pool(processes=args.n_workers) as pool:
    for result in tqdm(pool.imap_unordered(_run_combo, todo_args), total=len(todo_args)):
        # Atomic append: write to temp file then rename
        tmp = args.out + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerow(result)
        # Append tmp to main CSV
        with open(args.out, "a") as main, open(tmp) as t:
            next(t)  # skip header
            shutil.copyfileobj(t, main)
        os.remove(tmp)
        print(f"  [timing] {result['policy']} / {result['trace'][:20]} : "
              f"{result['wall_sec']:.1f}s")
```

### E4. Add `wall_sec` to CSV schema

Add `"wall_sec"` to `CSV_FIELDNAMES` in `eval_baselines.py`. Existing rows without this
column will have it as empty string (csv.DictReader returns `""` for missing fields —
tolerate this in downstream code).

### E5. Final timing summary

After the pool completes, print:
```python
total_wall = sum(r["wall_sec"] for r in all_results)
print(f"\n[timing] TOTAL: {total_wall:.0f}s compute across {len(all_results)} combos, "
      f"{args.n_workers} workers, wall={elapsed:.0f}s")
```

---

## Task F — Run the full baseline sweep

After Tasks C–E are implemented and smoke-tested:

```bash
# AG16x6 only first (validate numbers, ~2h serial → ~15min with 16 workers)
python scripts/eval_baselines.py \
  --policies greedy pid \
  --n-iter 50 \
  --topologies data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
  --n-workers 16 \
  2>&1 | tee output/baselines/sweep_ag16x6.log

# Full sweep all 4 topologies (only after AG16x6 validates)
python scripts/eval_baselines.py \
  --policies greedy pid \
  --n-iter 50 \
  --n-workers 16 \
  2>&1 | tee output/baselines/sweep_full.log
```

Expected output:
- `output/baselines/results.csv`: 20 rows after AG16x6 sweep (2 policies × 10 traces)
- `output/baselines/results.csv`: 80 rows after full sweep (2 × 10 × 4)
- Re-running either command adds 0 new rows (idempotent check).
- LaTeX table printed to stdout — copy into `docs/v1/v1.tex` as `tab:greedy_vs_pid`.

---

## Task G — Evaluate existing checkpoints with `eval_rl.py`

Run against the existing slow and fast checkpoints to establish a pre-variance-shaping
baseline RL number:

```bash
# Fast checkpoint
python scripts/eval_rl.py \
  --model-path output/checkpoints_fast/best_model \
  --traces AMS20PrdApp19-tround LON23PrdApp01-troundgrt5m \
  --n-iter 50 \
  --out-dir output/rl_evals/fast_baseline/

# Slow checkpoint
python scripts/eval_rl.py \
  --model-path output/checkpoints/best_model \
  --traces AMS20PrdApp19-tround LON23PrdApp01-troundgrt5m \
  --n-iter 50 \
  --out-dir output/rl_evals/slow_baseline/
```

Then run `plot_results.py` to generate comparison figures:
```bash
python scripts/plot_results.py \
  --rl-dirs output/rl_evals/fast_baseline output/rl_evals/slow_baseline \
  --out-dir output/figures/baseline_comparison/
```

---

## Execution order

```
C (pooling_simulation speedup)
D (greedy_alloc speedup)       ← both independent, do in parallel
      ↓
E (parallel workers in eval_baselines)
      ↓
A (entropy + Q-value plots in diagnose.py)   ← independent of C/D/E
B (reward smoothness check)                  ← independent of C/D/E
      ↓
F (run full baseline sweep)
      ↓
G (eval existing RL checkpoints)
      ↓
plot_results.py (figures for paper)
```

---

## Verification checklist

- **Task A**: `python scripts/diagnose.py --model output/checkpoints/best_model --trace LON23PrdApp01-troundgrt5m` produces 6 PNGs (was 4) in `output/diagnostics/`.
- **Task B**: `python scripts/diagnose.py --model ... --trace ... --check-smoothness` prints gradient table; no zero-gradient dimensions flagged for a healthy policy.
- **Task C**: smoke `eval_baselines.py --n-iter 5` runs ≥3× faster than before (compare `wall_sec` in CSV).
- **Task D**: `python -m pytest tests/test_greedy_alloc.py -v` passes all 20 seeds.
- **Task E**: `eval_baselines.py --n-workers 8` uses 8 cores (verify with `htop`); CSV is valid after run; re-run adds 0 rows.
- **Task F**: `output/baselines/results.csv` has 20 rows after AG16x6 sweep; 80 rows after full sweep.
- **Task G**: `output/rl_evals/fast_baseline/results.csv` and `slow_baseline/results.csv` exist with 2 rows each; `output/figures/baseline_comparison/savings_by_trace.png` shows greedy, pid, rl:fast_baseline, rl:slow_baseline bars side by side.
