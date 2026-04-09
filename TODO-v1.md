# Octopus CXL RL — Master TODO

All commands run from `/home/pm3371/gitrepos/octopus-allocation/` with venv active.

**Topologies available:**
- `AG16x6_expander_quads_r5_sym_fixed.csv` — 16 hosts, 192 MPDs, degree=6 (main)
- `AG16x4_latin3_r5_sym.csv` — 16 hosts, degree=4
- `random_L64_X8_N4_Y128.csv` — 64 hosts, 128 MPDs, degree=8
- `random_L96_X8_N4_Y192.csv` — 96 hosts, 192 MPDs, degree=8

**Traces available (10 total):**
- Training: `AMS20PrdApp19-tround.sqlite.pkl`
- Held-out: `LON23PrdApp01-troundgrt5m.sqlite.pkl`, `LVL01PrdApp05-troundgrt5m.sqlite.pkl`,
  `BLAPrdApp19-troundgrt5m.sqlite.pkl`, `BN9PrdApp18-troundgrt5m.sqlite.pkl`,
  `DSM08PrdApp05-troundgrt5m.sqlite.pkl`, `DUB24PrdApp09-troundgrt5m.sqlite.pkl`,
  `SG2PrdApp35-troundgrt5m.sqlite.pkl`, `SYD21PrdApp07-troundgrt5m.sqlite.pkl`,
  `YTO21PrdApp05-troundgrt5m.sqlite.pkl`

---

## Phase 0 — Quick Wins (existing SB3 codebase)

### Diagnostics
- [x] **Weight vector distribution** — `scripts/diagnose.py` Plot 1: softmax weights per accessible MPD over a held-out episode.
- [x] **Pooling savings curve** — `PoolingSavingsCallback` in `train.py`/`train_rl.py` logs `eval/pooling_savings_mean` every `eval_freq` steps to TensorBoard.
- [x] **MPD load variance** Var(**c**_t) — `scripts/diagnose.py` Plot 3: load variance over episode for RL vs greedy.
- [ ] **SAC entropy H(π)** — plot from TensorBoard (`output/logs/`). Healthy SAC maintains nonzero entropy; collapse to near-zero = loss of exploration before convergence.
- [ ] **Q-value vs actual return** — compare predicted Q to Monte Carlo returns on held-out episodes. Systematic overestimation = reward hacking.
- [x] **Behavioral side-by-side** — `scripts/diagnose.py` Plot 2: per-MPD load time series, RL vs greedy on same episode.

### Reward
- [x] Add variance shaping term: `R -= λ * var(mpd_load)`. `variance_lambda` constructor arg in `OctopusMemPoolEnv`; `--variance-lambda` CLI arg in `train.py`/`train_rl.py`.
- [ ] Verify agent stops concentrating load — max/min MHD ratio should drop from ~20× toward greedy's ~2.3×. (`scripts/diagnose.py` Plot 4 will show this once a model is trained with variance shaping.)
- [ ] Verify reward is smooth and differentiable w.r.t. continuous action logits at every step. Non-smooth rewards (e.g. involving `argmax`) prevent SAC from learning a good policy gradient.

### Baselines
- [x] Implement `pid_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, pid_state)` in `octopus/baselines.py`.
- [x] Add `"pid"` to `--policy` choices in `scripts/evaluate.py`; added `make_pid_alloc_cb()` with fresh state per pod mapping.
- [ ] Run greedy and optimal baselines on all 10 traces × all 4 topologies (50 pod mappings each) — use `scripts/eval_baselines.py` (implemented); sweep not yet run.
- [ ] Implement `make_optimal_alloc_cb(M)` in `scripts/evaluate.py` — builds tick-level demand accumulator and calls `find_optimal` at each step. Add `"optimal"` to `--policy` choices. Test on small trace first (optimal is O(flow) per tick).
- [ ] Fill `tab:greedy_vs_opt` in `docs/v1/v1.tex` — depends on baseline sweep above.

### Model comparison (slow vs. fast)
- [ ] Evaluate slow and fast checkpoints on AMS20 + LON23 (50 pod mappings each) — use `scripts/eval_rl.py` (implemented); eval not yet run.
- [x] Plot training curves — `scripts/plot_results.py` Figure 2 reads `evaluations.npz` for all run-ids, overlays on same axes.
- [x] Plot per-MPD load time series comparison — `scripts/plot_results.py` Figure 3 + `scripts/diagnose.py` Plot 2.

---

## Phase 0.5 — Simulation Speed & Baseline Infrastructure

Goal: make `eval_baselines.py` fast enough to sweep all 10 traces × 4 topologies × 50
iterations in a single overnight run, and validate that greedy/PID numbers are stable
before any RL comparisons.

### Speed-up: `pooling_simulation` (`scripts/evaluate.py`)

The simulation is called thousands of times per sweep. Key bottlenecks to fix:

- [ ] **`mhd_list` recomputed per VM arrival** (inner loop): precompute
  `node_mhd_lists = {node_id: [...] for node_id in node_to_M}` once per simulation
  call and look up by index in the tick sweep.
- [ ] **`to_tick` datetime arithmetic in hot loops**: convert all `start_time`/`end_time`
  to integer ticks at the start of the simulation (one vectorised pass over `node_to_vms`),
  cache `(vm_tick_start, vm_tick_end, mem)` tuples — `to_tick` becomes integer floor
  division with no datetime objects in the hot loop.
- [ ] **`np.asarray(vm.rss)` called per VM twice** (HOTFIX pass + event build pass):
  cache `mem = float(vm.rss[MEM_IDX])` once per VM and reuse in both passes.
- [ ] **ctx dict allocated per VM**: `ctx` contains compile-time constants for greedy/PID.
  Replace with pre-built named args captured in a closure; remove per-call `dict()`.
- [ ] **`alloc_events_sim` as list-of-lists** (one slot per tick): most ticks are empty
  for sparse traces. Replace with a flat sorted list of `(tick, node_id, dealloc_tick,
  mem)` and advance a pointer — skips O(pod_dur) empty-slot iterations.

Target: ≥3× speedup per simulation call on AMS20 / AG16x6 measured before/after.

### Speed-up: `greedy_alloc` (`octopus/baselines.py`)

- [ ] Replace the Python `while` loop with a vectorised numpy version:
  1. `loads = cur_cxl_mem_vec[mhd_list]`
  2. `order = np.argsort(loads)` (O(degree log degree), degree ≤ 8)
  3. Sweep sorted loads once, fill level-by-level with cumsum arithmetic — no Python loop.
- [ ] Add a regression test (`tests/test_greedy_alloc.py`) that asserts
  `np.allclose(greedy_alloc_fast(...), greedy_alloc_ref(...))` on 20 random inputs
  before replacing the implementation.

### Parallelism: `eval_baselines.py`

- [ ] Add `--n-workers INT` (default: `os.cpu_count()`). Use `multiprocessing.Pool`
  to run independent `(policy, topology, trace)` combos in parallel.
- [ ] Each worker loads its own trace, runs its combo, returns a result dict.
  Main process collects results and appends to CSV atomically (write temp file + rename).
- [ ] `--n-workers 1` forces serial mode for debugging.

### Timing & logging

- [ ] Add `wall_sec` column to `output/baselines/results.csv`.
- [ ] Print per-combo timing on completion:
  ```
  [timing] greedy / AMS20 / AG16x6 : 12.3s  (50 iters, 0.25s/iter)
  [timing] TOTAL: 847s across 100 combos, 8 workers, wall=112s
  ```
- [ ] Print speedup factor (fast vs. ref greedy) once after first combo completes.

### Verification

```bash
# 1. Regression test: fast greedy must match reference
python -m pytest tests/test_greedy_alloc.py -v

# 2. Smoke: one combo, should finish in <30s
python scripts/eval_baselines.py --policies greedy --n-iter 5 \
  --traces AMS20PrdApp19-tround \
  --topologies data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv

# 3. Full sweep: all traces × AG16x6, greedy + pid, 50 iters
python scripts/eval_baselines.py --policies greedy pid --n-iter 50 --n-workers 16
# → output/baselines/results.csv should have 20 rows (2 × 10); re-run adds 0 rows
```

---

## Phase 1 — JAX Environment Rewrite

### Trace Loading
- [ ] Write trace loader: parse SQLite/pickle files → dense numpy arrays `(T, n_srv)`
- [ ] Encode VM arrivals as padded tensors: `arr_host[T, max_vms]`, `arr_size_gb[T, max_vms]`, `arr_lifetime[T, max_vms]`
- [ ] Profile VRAM usage of full trace as JAX array — confirm fits in 24 GB Titan
- [ ] `jax.device_put()` trace to GPU 0 at init time

### State Pytree (`OctopusState`)
- [ ] Implement `OctopusState` as NamedTuple / Flax struct
- [ ] Fields: `mpd_load`, `mpd_load_1h`, `mpd_load_2h`, `srv_pressure`, `srv_pressure_1h`, `srv_pressure_2h`, `slo_vio_24h`, `adj[n_srv, n_mpd]`, `tick` (int32), `key` (PRNG)

### Step Function
- [ ] Implement pure `step(state, action, trace) -> (next_state, obs, reward, done)` with no Python conditionals
- [ ] Read VM arrivals at tick `t` via index into pre-loaded trace tensor
- [ ] Apply allocation vector to `mpd_load`
- [ ] Roll delta buffers: `mpd_load_2h ← mpd_load_1h`, `mpd_load_1h ← mpd_load`
- [ ] Implement VM termination: release allocated GB from `mpd_load` at `tick + lifetime`
- [ ] Compute `slo_vio_24h` rolling window
- [ ] Implement `reset(trace, key) -> OctopusState`

### Reward
- [ ] Implement Option A (immediate): `R = α(1 − max_cap/cap_max) − β·slo_vio − λ·Var(mpd_load)`. See `docs/v1/v1.tex §Reward`.
- [ ] Implement Option B (immediate + terminal): scaled immediate signal + large terminal reward equal to actual pooling savings at episode end. Terminal dominates; immediate shapes credit assignment.
- [ ] Tune weights `α, β, λ` — enforce `α < β` (SLO priority over utilization)
- [ ] Validate reward is dense (non-zero gradient on every MPD, not just peak)
- [ ] Empirical comparison: run Option A and Option B on AMS20 (train) and LON23 (held-out). Decide by pooling savings, not reward magnitude.

### Topology
- [ ] Port `build_adjacency()` for random regular bipartite graph (degree=8)
- [ ] Add loader for actual Acadia topology CSV
- [ ] Add loader for AG16x6 expander quads topology (backward compatibility with v0 results)

### State extensions
- [ ] Add load deltas to observation: `[v_t, Δ_1h, Δ_2h]` per MPD — gives policy trend info to pre-emptively avoid filling MPDs
- [ ] Add cross-server memory pressure: aggregate load across all hosts in pod (policy currently has no peer-host visibility)
- [ ] Add day-of-week encoding: `sin(2π·dow/7)`, `cos(2π·dow/7)` alongside existing hour-of-day

---

## Phase 2 — GNN Policy Network

### Graph Construction
- [ ] Implement `build_graph(state) -> jraph.GraphsTuple`
- [ ] Server node features: `[srv_pressure_t, Δ1h, Δ2h, sin(hour), cos(hour)]`
- [ ] MPD node features: `[mpd_load_t, Δ1h, Δ2h, slo_vio_24h]`
- [ ] Edge features: `[1.0]` per CXL link (extend with bandwidth util later)
- [ ] Node layout: servers `[0..n_srv)`, MPDs `[n_srv..n_srv+n_mpd)` — offset receivers

### Message Passing
- [ ] Implement round 1: MPD → Server (aggregate load info at each server). Use mean aggregation for server nodes.
- [ ] Implement round 2: Server → MPD (aggregate pressure info at each MPD). Use sum aggregation for MPD nodes.
- [ ] Verify no Python branching in message functions (must be XLA-compatible)

### Action Head
- [ ] Implement edge MLP: per-edge scalar allocation weight
- [ ] Apply softmax over each server's outgoing edges (not global softmax)
- [ ] Confirm output satisfies `Σ wⱼ = 1` and `wⱼ = 0` for non-existent links
- [ ] Remove `conn_mask` post-hoc application — enforce topology structurally

### Validation
- [ ] Log per-edge allocation weights during eval
- [ ] Visualize allocation weights on bipartite graph (sanity check)
- [ ] Confirm same network weights work on 16-host and 96-host topologies without retraining
- [ ] Evaluate whether GNN significantly outperforms MLP (low priority until MLP converges)

---

## Phase 3 — Vectorization & Multi-GPU

### vmap
- [ ] Wrap `step()` with `jax.vmap` across N=1000 environment instances
- [ ] Wrap `reset()` with `jax.vmap` — each env gets own PRNG key via `jax.random.split`
- [ ] Assign GPU 0 to rollout via `jax.device_put(..., device=gpu_0)`

### pmap (GPU 1 + 2)
- [ ] Shard policy optimization across GPU 1 and GPU 2 with `jax.pmap`
- [ ] Implement `jax.lax.psum(grads)` for cross-device gradient sync
- [ ] Verify params stay in sync across devices after each update

### JIT compilation
- [ ] Wrap full rollout loop with `jax.lax.scan()` (eliminates Python loop overhead)
- [ ] JIT-compile the entire `train_step(params, state, trace) -> (params, metrics)` function
- [ ] Time first compilation (expect slow) vs. subsequent steps (expect fast)

---

## Phase 4 — Training Algorithm

### SAC (primary)
- [ ] Port SAC to PureJaxRL — actor, critic, entropy temperature
- [ ] Implement replay buffer as fixed-size JAX array (circular buffer)
- [ ] Uniform sampling from replay buffer via `jax.random.choice`
- [ ] Validate SAC convergence on single env before scaling to vmap

### Convergence infrastructure
- [ ] Held-out evaluation loop: run full pooling simulation on LON23 every N training episodes; log pooling savings, Var(**c**), policy entropy. This is the primary convergence signal.
- [ ] Weight statistics logging: during eval, log mean, variance, entropy of softmax weight vector **a**. Converging policy should show increasing differentiation.
- [ ] Early stopping on savings plateau: stop when held-out pooling savings haven't improved >0.001 for 50 consecutive eval checkpoints.

### LCPO (catastrophic forgetting)
- [ ] Implement context window buffer `D_old` (past experiences from different time-of-day regimes)
- [ ] Implement KL constraint: `KL(π_new ∥ π_old) ≤ δ` over `D_old` samples. See `docs/v1/v1.tex §LCPO`.
- [ ] Test LCPO retains morning-hour allocation patterns after training on evening data
- [ ] Compare LCPO vs. plain SAC on LON23 across full 14-day trace

### PPO (fallback)
- [ ] Implement PPO in PureJaxRL as fallback if SAC replay buffer struggles with non-stationarity
- [ ] Compare PPO vs. SAC on AMS20 — use whichever converges more stably

---

## Phase 5 — Evaluation & Scaling

### Per-datacenter policy evaluation
- [ ] Train on AMS20 and LVL01 separately (fast config, ~35 min each). AMS20 = low utilization; LVL01 = near capacity.
- [ ] Evaluate each trained policy on both traces — if AMS20 policy degrades on LVL01, motivates context-conditioning.
- [ ] Show greedy and PID have large variance in pooling savings across traces; RL adapts better. Key claim for per-DC section.

### 96-host Acadia scaling
- [ ] Train SAC with fast config on 96-host topology
- [ ] Confirm observation vector dimensions unchanged (`dmax=8` → same 20-dim v0, larger v1)
- [ ] Evaluate on Acadia-96 (N=4, m=120) — target: beat greedy by 1–2 pp savings
- [ ] Compare RL savings vs. greedy at 96 hosts against Figure 5 / Figure 6 curves

### Robustness
- [ ] Re-run link failure sweep (Table 6) with RL policy — does it degrade more or less gracefully than greedy?
- [ ] Test with domain randomization during training (random pod assignments) to improve generalization

---

## Phase 6 — State & Reward Extensions

- [ ] Add predicted VM lifetime to state (signal 6 in Table 2)
- [ ] Add peak-to-average ratio 24-hr summary (signal 7)
- [ ] Add predicted memory growth rate (signal 8)
- [ ] Add per-MPD bandwidth utilization to edge features (signal 2)
- [ ] Extend reward to account for per-VM bandwidth requirements
- [ ] Explore MORL for automatic weight tuning (removes manual `α, β, λ` tuning)
- [ ] Ablation: remove trend features, remove connectivity mask, vary topology type — quantify each signal's contribution to pooling savings

### GNN policy architecture (longer term)
- [ ] Replace MLP with GNN operating on host–MPD bipartite graph. Natural inductive bias; generalizes across pod sizes without retraining.
- [ ] Context conditioning: add datacenter embedding / cluster-level utilization stats as extra state signal. Enables single policy to generalize across regimes without retraining.
- [ ] LCPO: evaluate for catastrophic forgetting across diurnal regime shifts (if policy degrades between day/night windows).

---

## Hygiene

- [ ] Set up experiment tracking (W&B or equivalent) — log reward, savings %, MHD load variance per run
- [ ] Save best checkpoint by LON23 eval reward (generalization proxy, not training reward)
- [ ] Pin JAX + Jraph + Flax + optax versions in `requirements.txt`

---

## How to run things

### Training
```bash
# Fast (~34 min)
python3 scripts/train.py --fast --total-timesteps 1000000 \
  --topology data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
  --trace data/traces/AMS20PrdApp19-tround.sqlite.pkl \
  --eval-trace data/traces/LON23PrdApp01-troundgrt5m.sqlite.pkl \
  --save-dir output/checkpoints_fast --log-dir output/logs_fast --device cuda:1

# Slow (~6.5 hr)
python3 scripts/train.py --total-timesteps 1000000 \
  --topology data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
  --trace data/traces/AMS20PrdApp19-tround.sqlite.pkl \
  --eval-trace data/traces/LON23PrdApp01-troundgrt5m.sqlite.pkl \
  --save-dir output/checkpoints --log-dir output/logs --device cuda:0

# Monitor
tensorboard --logdir output/logs
```

| Parameter         | Slow        | Fast        |
|------------------|-------------|-------------|
| `train_freq`     | 1           | 16          |
| `net_arch`       | [256, 256]  | [128, 128]  |
| `buffer_size`    | 1,000,000   | 300,000     |
| `batch_size`     | 256         | 512         |
| `n_eval_eps`     | 3           | 1           |
| `checkpoint_freq`| 10,000      | 50,000      |
| `eval_freq`      | 20,000      | 100,000     |
| Wall time        | ~6.5 hr     | ~34 min     |
| Gradient steps   | ~999k       | ~62k        |

### Evaluation
```bash
python3 scripts/evaluate.py --policy greedy \
  --topology data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
  --trace data/traces/AMS20PrdApp19-tround.sqlite.pkl \
  --n-iter 50
```

### Plotting
```bash
python3 scripts/plot_demand.py                   # figs/demand_*.pdf
python3 scripts/plot_memory_pooling_sweep.py     # figs/fc_vs_octopus.pdf, 3panel_by_cxl.pdf
jupyter notebook notebooks/                      # memory-pooling-v0.ipynb
```

### Compile paper
```bash
cd docs/v1
pdflatex v1 && bibtex v1 && pdflatex v1 && pdflatex v1
```
