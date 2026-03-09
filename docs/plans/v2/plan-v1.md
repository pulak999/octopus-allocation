# Phase 0 Implementation Plan

Agent instructions for implementing Phase 0 of the Octopus CXL RL project.
All paths relative to `/home/pm3371/gitrepos/octopus-allocation/`. Run with venv active.

---

## Context

The codebase trains a SAC agent (`scripts/train.py`) on `OctopusMemPoolEnv` (`octopus/env.py`)
and evaluates it against a greedy baseline (`octopus/baselines.py`) using `scripts/evaluate.py`.

**Current reward** in `env.py` (around line 240):
```python
reward = -(new_peak - old_peak) / fair_share
```
where `fair_share = pod_dram / num_mhd`.

**Current observation** (size = `2*max_degree + 4`):
- `[0:max_degree]` — normalized MPD loads (padded)
- `[max_degree:2*max_degree]` — accessibility mask
- `[2*max_degree]` — VM memory request (normalized)
- `[2*max_degree+1]` — global peak MPD load (normalized)
- `[2*max_degree+2:+4]` — sin/cos hour-of-day

---

## Task 1 — Fix known bugs (do this first)

These are correctness bugs that will corrupt all downstream results if left unfixed.

### 1a. Fix optimal activity gate (`plot_memory_pooling_sweep.py`)

Find the function `optimal_required_capacity` (around line 409).
Replace the activity gate:
```python
# BEFORE (wrong — skips ticks with sustained load but no new arrivals)
if float(np.max(diff[ts, :])) <= 0:

# AFTER
if float(np.max(node_cxl)) <= 0:
```

### 1b. Fix deterministic seeding (`scripts/train.py`)

At the top of `main()`, after `args = parser.parse_args()` and before any env/model creation, add:
```python
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Confirm `import random` is at the top of the file.

### 1c. Fix RL model selection (`scripts/evaluate.py`)

The current code picks whichever checkpoint path exists first.
- Add `--model` argument (already exists per codebase scan — confirm it is required, not optional with a glob fallback).
- If there is a glob/wait fallback, remove it. Fail fast with a clear error if `--model` is not provided when `--policy rl` is used.
- At the start of RL evaluation, log: resolved absolute path + `md5sum` of the model file.

---

## Task 2 — Add variance shaping to reward

**File:** `octopus/env.py`

In the `step()` method, after computing `new_peak`, replace the reward line with:

```python
mpd_loads = self.cur_cxl_mem_vec / self.mhd_cap  # normalized loads, shape (num_mhd,)
load_variance = float(np.var(mpd_loads))
LAMBDA = 0.5  # start here; tune in Task 3
reward = -(new_peak - old_peak) / fair_share - LAMBDA * load_variance
```

Make `LAMBDA` a constructor argument `variance_lambda=0.5` so it can be swept without editing the file:
```python
def __init__(self, ..., variance_lambda=0.5):
    ...
    self.variance_lambda = variance_lambda
```

In `scripts/train.py`, expose `--variance-lambda` CLI argument (default `0.5`) and pass it through to env construction.

---

## Task 3 — Diagnostics script

Create `scripts/diagnose.py`. This script loads a trained checkpoint and a trace, runs one full episode, and produces diagnostic plots. It does **not** train anything.

### Arguments
```
--model PATH        path to SB3 checkpoint (required)
--trace NAME        cluster name stem, e.g. LON23PrdApp01-troundgrt5m (required)
--topology PATH     default: data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv
--seed INT          default: 42
--out-dir PATH      default: output/diagnostics/
```

### Plots to produce

**Plot 1 — Softmax weight distribution** (`weight_distribution.png`)
- Run one full episode deterministically.
- At each step, record the softmax allocation weights (the action after softmax, not raw logits).
- Plot: x = step index, y = weight per MPD (one line per accessible MPD, use the mask).
- Goal: detect collapse (near-uniform) or degeneracy (always same MPD).

**Plot 2 — Per-MPD load time series** (`mpd_loads.png`)
- At each step, record `cur_cxl_mem_vec / mhd_cap` for all MPDs.
- Plot: x = step, y = normalized load, one line per MPD.
- Overlay: greedy policy on same episode (same seed, same pod mapping).
- Goal: visual comparison of load spreading.

**Plot 3 — Load variance over episode** (`load_variance.png`)
- Plot Var(MPD loads) at each step for both RL and greedy.
- Goal: confirm variance penalty is having an effect.

**Plot 4 — Max/min MPD load ratio** (`load_ratio.png`)
- At each step: `ratio = max(mpd_loads) / (min(mpd_loads) + 1e-9)`.
- Plot for RL and greedy. Target: RL ratio should approach greedy's ~2.3×.

All plots: use `matplotlib`, save to `--out-dir`, print path on completion.

---

## Task 4 — Pooling savings curve during training

**File:** `scripts/train.py`

After every `eval_freq` timesteps (same cadence as EvalCallback), run a full
`pooling_simulation` on the eval trace (LON23 by default) and log:
- `eval/pooling_savings_mean`
- `eval/pooling_savings_std`
- `eval/mpd_load_variance_mean`

Use SB3's `EvalCallback` subclass or a custom `BaseCallback`. Log via `self.logger.record()`.

This is the primary convergence signal — training reward alone is insufficient.

---

## Task 5 — PID baseline

**File:** `octopus/baselines.py`

Add the following function:

```python
def pid_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, pid_state,
              kp=1.0, ki=0.01, kd=0.1):
    """
    PID allocation: distribute cxl_mem across accessible MPDs in mhd_list,
    proportional to inverse load with integral tracking cumulative imbalance.

    pid_state: dict with keys 'integral' (np.array len num_mhd), 'prev_error' (np.array).
               Mutated in-place each call.
    Returns: allocation vector (length = num_mhd, zero for inaccessible MPDs).
    """
```

Algorithm:
1. Compute target load = mean load across accessible MPDs.
2. Error per MPD = target_load - current_load (positive = underloaded, wants more).
3. Update integral: `pid_state['integral'][mhd_list] += error * dt` (use `dt=1`).
4. Derivative: `d_error = error - pid_state['prev_error'][mhd_list]`.
5. PID weight per MPD: `w = kp*error + ki*integral + kd*d_error`, clip to `[0, inf]`.
6. Normalize weights to sum to 1; allocate `cxl_mem * w[j]` to MPD j.
7. Update `pid_state['prev_error']`.

**File:** `scripts/evaluate.py`

- Add `"pid"` to `--policy` choices.
- Add `make_pid_alloc_cb()` that initializes `pid_state` and wraps `pid_alloc`.
- Reset `pid_state` at the start of each pod mapping (not shared across mappings).
- Pass `kp`, `ki`, `kd` as CLI args with the defaults above.

---

## The evaluation pipeline

The key design principle: baselines are run **once**, results saved to disk, and RL is iterated
independently. Plotting always reads from saved results files — it never re-runs simulations.

```
scripts/eval_baselines.py   →   output/baselines/results.csv
scripts/train_rl.py         →   output/checkpoints/<run_id>/
scripts/eval_rl.py          →   output/rl_evals/<run_id>/results.csv
scripts/plot_results.py     ←   reads both CSVs, produces all figures
```

`eval_baselines.py` is run once per topology/trace combination and never again (unless
baselines change). `eval_rl.py` is run after every training run. `plot_results.py` reads
whatever results exist and produces a consistent set of figures.

---

## Task 6 — `scripts/eval_baselines.py`

Replaces the old `evaluate.py` baseline path. Runs greedy, PID, and optional optimal on
all specified traces × topologies and saves results to a single CSV. Run once; re-run only
if baselines change.

### Arguments
```
--policies {greedy, pid, optimal} [...]   default: greedy pid
--n-iter INT                              pod mappings per combo (default: 50)
--out CSV                                 default: output/baselines/results.csv
--topologies [paths...]                   default: all 4 in data/topologies/
--traces [names...]                       default: all 10
--kp FLOAT                                PID proportional gain (default: 1.0)
--ki FLOAT                                PID integral gain (default: 0.01)
--kd FLOAT                                PID derivative gain (default: 0.1)
```

For optimal: recommend `--n-iter 10` (slow, O(flow) per tick).

### Output CSV schema
```
policy, topology, trace, savings_mean, savings_min, savings_max, savings_std,
mpd_load_variance_mean, n_iter, timestamp
```

- Append to existing CSV rather than overwrite — allows partial re-runs.
- Skip any `(policy, topology, trace)` combo already present in the CSV (idempotent).
- Print a progress bar (tqdm) across combos.
- On completion, print the `tab:greedy_vs_opt` table to stdout in LaTeX tabular format
  (topology × trace grid, cells = `mean ± std`).

### Verification
```bash
python scripts/eval_baselines.py --policies greedy --n-iter 5 \
  --traces AMS20PrdApp19-tround --topologies data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv
```
Produces 1 row in `output/baselines/results.csv`. Re-running adds no new rows (idempotent).

---

## Task 7 — `scripts/train_rl.py`

Rename/replace `scripts/train.py`. Identical functionality but with a mandatory `--run-id`
argument that namespaces all output artifacts. This makes it easy to compare multiple RL runs.

### Changes from current `train.py`
- Add `--run-id STR` (required). Suggested format: `v0_fast`, `v0_slow`, `v1_variance_lambda05`.
- Save checkpoints to `output/checkpoints/<run-id>/` instead of `output/checkpoints/`.
- Save TensorBoard logs to `output/logs/<run-id>/` instead of `output/logs/`.
- Save a `output/checkpoints/<run-id>/config.json` with all CLI args + timestamp at start of run.
- All other logic (SAC hyperparams, EvalCallback, pooling savings callback from Task 4) unchanged.

### Verification
```bash
python scripts/train_rl.py --run-id smoke_test --fast --total-timesteps 5000
```
Creates `output/checkpoints/smoke_test/` with checkpoint files and `config.json`.

---

## Task 8 — `scripts/eval_rl.py`

Evaluates one or more RL checkpoints using `pooling_simulation` (same simulator as baselines)
and saves results to a per-run CSV. Run after every training run.

### Arguments
```
--run-id STR [...]     one or more run IDs to evaluate (looks up output/checkpoints/<run-id>/)
--model-path PATH      explicit model path (alternative to --run-id; for one-off evals)
--traces [names...]    default: all 10
--topology PATH        default: AG16x6_expander_quads_r5_sym_fixed.csv
--n-iter INT           default: 50
--out-dir PATH         default: output/rl_evals/
```

### Output
- One CSV per run-id: `output/rl_evals/<run-id>/results.csv`
- Same schema as baseline CSV: `policy, topology, trace, savings_mean, savings_min, savings_max, savings_std, mpd_load_variance_mean, n_iter, timestamp`
- Also writes `output/rl_evals/<run-id>/episode_detail.csv`: per-episode savings for each
  (trace, mapping_seed) — needed for significance testing later.
- Prints summary table to stdout on completion.

### Verification
```bash
python scripts/eval_rl.py --run-id smoke_test --n-iter 5 \
  --traces AMS20PrdApp19-tround
```
Produces `output/rl_evals/smoke_test/results.csv` with 1 row.

---

## Task 9 — `scripts/plot_results.py`

Reads saved CSVs from `eval_baselines.py` and `eval_rl.py`. Produces all figures.
Never runs any simulation itself.

### Arguments
```
--baselines-csv PATH    default: output/baselines/results.csv
--rl-dirs [paths...]    default: all subdirs of output/rl_evals/
--out-dir PATH          default: output/figures/
--traces [names...]     filter to subset of traces (default: all present in CSVs)
--topology STR          filter to one topology (default: AG16x6_expander_quads_r5_sym_fixed)
```

### Figures to produce

**Figure 1 — Savings comparison bar chart** (`savings_by_trace.png`)
- One grouped bar per trace, one bar per policy (greedy, pid, optimal if present, one bar per RL run-id).
- Error bars = std across pod mappings.
- Sorted by greedy savings descending.

**Figure 2 — Training curves** (`training_curves.png`)
- For each run-id that has a TensorBoard `evaluations.npz`: plot `results.mean(axis=1)` vs `timesteps`.
- Overlay `eval/pooling_savings_mean` from TensorBoard logs if available.
- One line per run-id.

**Figure 3 — Per-MPD load time series** (`timeseries_<trace>_<run-id>.png`)
- Load episode detail from `eval_rl.py` output.
- For a single representative episode: greedy loads vs RL loads, stacked subplots.
- One figure per (trace, run-id) combination.

**Figure 4 — Load variance comparison** (`load_variance_by_trace.png`)
- `mpd_load_variance_mean` for each policy × trace combo.
- Same grouped bar format as Figure 1.

**Figure 5 — LaTeX table** (`tab_greedy_vs_opt.tex`)
- Topology × trace grid, cells = `greedy_mean ± greedy_std`.
- Gap column: `optimal_mean - greedy_mean` (omit if optimal not present).
- Written as a `.tex` snippet, ready for `\input{}` in the paper.

### Verification
```bash
python scripts/plot_results.py
```
Produces at minimum Figure 1 and Figure 5 if baselines CSV exists. Remaining figures
appear as RL eval results are added.

---

## Execution order

```
1 (bugs) → 2 (reward) → 5 (PID impl)
                      ↓
          6 (eval_baselines) ─────────────────────────────────→ 9 (plot)
                      ↓                                              ↑
          7 (train_rl) → 4 (savings callback) → 8 (eval_rl) ────────┘
                      ↑
          3 (diagnose) — runs any time after 2
```

- Tasks 1+2 are prerequisites for everything.
- Task 6 (baselines) and Task 7+8 (RL pipeline) are independent after Task 5.
- Task 9 (plotting) can be run incrementally — it plots whatever data exists.
- Task 3 (diagnose) is a standalone debugging tool; run it any time.

---

## Verification checklist

- **Task 1:** `git diff` shows only the three targeted lines changed. Run `eval_baselines.py --policies greedy --n-iter 2` and confirm it completes without error.
- **Task 2:** 1000-step training run; TensorBoard shows `train/reward` more negative than before. `--variance-lambda 0.0` reproduces old reward exactly.
- **Task 3:** `python scripts/diagnose.py --model output/checkpoints/smoke_test/best_model --trace LON23PrdApp01-troundgrt5m` produces 4 PNGs in `output/diagnostics/`.
- **Task 4:** Training run with `--total-timesteps 50000` logs `eval/pooling_savings_mean` in TensorBoard.
- **Task 5:** `python scripts/eval_baselines.py --policies greedy pid --n-iter 5 --traces AMS20PrdApp19-tround` completes and prints stats for both policies.
- **Task 6:** `output/baselines/results.csv` has 40 rows after full greedy sweep. Re-running adds 0 rows.
- **Task 7:** `python scripts/train_rl.py --run-id smoke_test --fast --total-timesteps 5000` creates `output/checkpoints/smoke_test/config.json`.
- **Task 8:** `output/rl_evals/smoke_test/results.csv` exists after running `eval_rl.py --run-id smoke_test`.
- **Task 9:** `python scripts/plot_results.py` produces `output/figures/savings_by_trace.png` with at least greedy bars visible.
