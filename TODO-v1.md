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
- [ ] **Weight vector distribution** — plot softmax output **a** over a held-out episode. Near-uniform = collapsed no-op; always routing to one MPD = degenerate greedy. Want diversity correlated with load.
- [ ] **Pooling savings curve** — evaluate on LON23 every 100 episodes during training. Training reward (Δpeak) can look reasonable while pooling savings are flat.
- [ ] **MPD load variance** Var(**c**_t) — plot over training steps. Should decrease as policy spreads load; growing = consolidating.
- [ ] **SAC entropy H(π)** — plot from TensorBoard (`output/logs/`). Healthy SAC maintains nonzero entropy; collapse to near-zero = loss of exploration before convergence.
- [ ] **Q-value vs actual return** — compare predicted Q to Monte Carlo returns on held-out episodes. Systematic overestimation = reward hacking.
- [ ] **Behavioral side-by-side** — run greedy, optimal, and RL on same trace segment; plot per-MPD load time series on one figure. Reveals whether RL tracks optimal, does something different, or ignores load.

### Reward
- [ ] Add variance shaping term: `R -= λ * jnp.var(mpd_load)`. Increase λ aggressively — current behavior may optimize average load without penalizing imbalance.
- [ ] Verify agent stops concentrating load — max/min MHD ratio should drop from ~20× toward greedy's ~2.3×.
- [ ] Verify reward is smooth and differentiable w.r.t. continuous action logits at every step. Non-smooth rewards (e.g. involving `argmax`) prevent SAC from learning a good policy gradient.

### Baselines
- [ ] Implement `pid_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, pid_state)` in `octopus/baselines.py` — proportional to inverse load with integral term tracking cumulative imbalance.
- [ ] Add `"pid"` to `--policy` choices in `scripts/evaluate.py` and add `make_pid_alloc_cb()` callback alongside `_greedy_alloc_cb`.
- [ ] Run greedy and optimal baselines on all 10 traces × all 4 topologies (50 pod mappings each) to establish per-cluster savings budget.
- [ ] Implement `make_optimal_alloc_cb(M)` in `scripts/evaluate.py` — builds tick-level demand accumulator and calls `find_optimal` at each step. Add `"optimal"` to `--policy` choices. Test on small trace first (optimal is O(flow) per tick).
- [ ] Fill `tab:greedy_vs_opt` in `docs/v1/v1.tex` with greedy avg/min/max/std for AG16x6 topology, then fill the gap column: `optimal_savings - greedy_savings`. This is the headline motivation number.

### Model comparison (slow vs. fast)
- [ ] Evaluate slow and fast checkpoints on AMS20 + LON23 (50 pod mappings each). See `docs/overview/PAPER_TODO.md §5` for parameter table.
- [ ] Plot training curves — both `output/logs/evaluations.npz` (slow) and `output/logs_fast/evaluations.npz` (fast) on same axes.
- [ ] Plot per-MPD load time series comparison (see `figs/mhd_timeseries_*.png` for format). Use `--save-timeseries` flag or notebook.

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
