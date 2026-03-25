# Phase 1–3 Implementation Plan

Agent instructions for implementing Phases 1, 2, and 3 of the Octopus CXL RL project.
All paths relative to `/home/pm3371/gitrepos/octopus-allocation/`. Run with venv active.

**Stack:** JAX + Flax + Optax + Jraph + PureJaxRL-style training loop.
**Hardware:** 3× TITAN RTX (24 GB each), kernel 5.15, CUDA 12.5.

**Target run:** one GNN-SAC training run (`v2_gnn_sac`) followed by LCPO deployment.
No MLP comparison runs are needed at this stage. The GNN-SAC policy is the sole output
of this pipeline; it is later served via the LCPO deployment path (see Phase 4).

**Key design constraint:** all v2 code lives under `octopus/v2/` and `scripts/v2/`
so the existing SB3 pipeline is untouched. The eval script (`scripts/v2/eval_rl_jax.py`)
writes `results.csv` in the same schema as `scripts/eval_rl.py` so `scripts/v2/plot_results_jax.py`
can overlay v1 SB3 bars with the v2 GNN bar if desired.

**Why GNN is the right inductive bias:**
The allocation problem has explicit bipartite structure: each server is connected to a
fixed subset of MPDs via CXL links. An MLP receives a padded flat vector and must learn
from scratch which entries correspond to which MPD — it has no way to represent that
"server 3 cannot use MPD 7" other than via a post-hoc mask. A GNN encodes the topology
as edges: MPD load information flows *along CXL links* to the server making the
allocation decision, and the action head produces one weight per existing edge — so
structural zeros in M are enforced architecturally, not by clamping.

---

## Prerequisites

- Phase 0 and 0.5 complete (baseline sweep done, SB3 MLP results in
  `output/rl_evals/*/results.csv`).
- Python packages: `jax[cuda12_pip]`, `flax`, `optax`, `jraph`, `chex`, `orbax-checkpoint`.
  Add to `requirements.txt` without removing existing deps.
- Verify GPU access: `python -c "import jax; print(jax.devices())"` should show 3 GPUs.

---

## Phase 1 — JAX Environment

### Task 1.1 — Trace preprocessing to JAX arrays

**New file:** `octopus/v2/jax_trace.py`

The existing pickle trace has VM objects with Python datetime fields. JAX needs
fixed-shape integer arrays on GPU.

Preprocessing steps (run once, cache to disk as `.npz`):

```
load_trace(name) → preprocess_trace(all_vms, node_to_vms, ...) → JaxTrace
```

`JaxTrace` is a dataclass:
```python
@dataclass
class JaxTrace:
    # Per-tick dense arrays, shape (T, max_vms_per_tick)
    arr_host:     np.ndarray   # int32, host index (pod-relative), -1 = empty slot
    arr_mem_gb:   np.ndarray   # float32, memory request in GB, 0.0 = empty slot
    arr_lifetime: np.ndarray   # int32, lifetime in ticks, 0 = empty slot
    # Scalars
    T: int                     # total ticks
    max_vms_per_tick: int      # padding width (95th percentile + 20%)
    pod_dram_gb: float         # total pod DRAM capacity (GB)
```

Implementation notes:
- Build a list of events per tick (same logic as `_build_events` in `env.py`).
- Apply HOTFIX (same filter as existing code — copy the logic, don't import from env.py).
- Sort events within each tick by (host, -size) for deterministic ordering.
- Pad each tick's event list to `max_vms_per_tick` with sentinel values.
- Cache preprocessed trace to `data/traces_v2/<name>.npz`; skip rebuild if file exists.
- `jax.device_put(trace_arrays, device=gpu_0)` at load time.

VRAM check: AMS20 is the largest trace. Profile shape and print GB before `device_put`.
Target: all 10 traces fit in 24 GB combined; each trace alone should be ≤ 3 GB.

**CLI:** `python octopus/v2/jax_trace.py --trace AMS20PrdApp19-tround` preprocesses and
prints shape + VRAM estimate. All 10 traces: `python octopus/v2/jax_trace.py --all`.

---

### Task 1.2 — `OctopusState` pytree

**New file:** `octopus/v2/jax_env.py`

```python
@flax.struct.dataclass
class OctopusState:
    mpd_load:       jnp.ndarray   # (num_mhd,) float32, current GB per MPD
    mpd_load_1h:    jnp.ndarray   # (num_mhd,) float32, load 12 ticks ago (1h)
    mpd_load_2h:    jnp.ndarray   # (num_mhd,) float32, load 24 ticks ago (2h)
    srv_pressure:   jnp.ndarray   # (pod_size,) float32, current VM demand per server
    srv_pressure_1h:jnp.ndarray   # (pod_size,) 1h lag
    srv_pressure_2h:jnp.ndarray   # (pod_size,) 2h lag
    slo_vio_24h:    jnp.ndarray   # (num_mhd,) float32, fraction of last 288 ticks over cap
    dealloc_buf:    jnp.ndarray   # (T, num_mhd) float32, scheduled deallocations
    tick:           jnp.ndarray   # () int32
    key:            jnp.ndarray   # (2,) uint32, JAX PRNG key
```

Keep `adj` (the topology matrix M) as a static Python/numpy object — it never changes
within a training run and should not be a traced JAX array.

---

### Task 1.3 — Pure `step` and `reset` functions

**File:** `octopus/v2/jax_env.py`

```python
def reset(trace: JaxTrace, adj: np.ndarray, key: jnp.ndarray) -> OctopusState:
    ...

def step(
    state: OctopusState,
    action: jnp.ndarray,      # (pod_size, max_degree) float32, pre-softmax logits
    trace: JaxTrace,
    adj: np.ndarray,           # (pod_size, num_mhd) int32, static
) -> Tuple[OctopusState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # returns (next_state, obs, reward, done)
    ...
```

Implementation:
1. Read VM arrivals at `state.tick` from `trace.arr_host`, `trace.arr_mem_gb`,
   `trace.arr_lifetime` — these are already padded arrays, iterate via `jnp.where`.
2. For each non-sentinel VM: apply softmax over the server's accessible MPD logits
   (use `adj[host]` as a mask, set logits of inaccessible MPDs to -1e9 before softmax),
   compute `alloc_gb = softmax_weights * vm_mem`.
3. Add to `mpd_load`; schedule deallocation by writing to `dealloc_buf[tick + lifetime]`.
4. Apply deallocations: `mpd_load -= dealloc_buf[tick]`.
5. Update rolling buffers: at ticks divisible by 12, roll `mpd_load_1h → mpd_load_2h`,
   `mpd_load → mpd_load_1h`.
6. Compute `slo_vio_24h`: fraction of `dealloc_buf` rows (last 288 ticks) where any
   MPD exceeded its fair-share cap. Use a sliding window sum on the existing `dealloc_buf`.
7. Reward (implement both, select via config):
   - **Option A** (immediate): `R = -peak_delta/fair_share - lambda*var(mpd_load/cap)`
     (matches existing SB3 reward + variance shaping, for a clean MLP→JAX comparison)
   - **Option B** (immediate + terminal): `R_t = 0.01 * R_A_t` + at done:
     `R_terminal = pooling_savings * 100` (raw savings as a large terminal signal).
   Start with Option A to establish the JAX MLP baseline matches SB3 MLP.
8. Build observation vector (same layout as `OctopusMemPoolEnv._get_obs` for MLP
   compatibility; GNN uses `state` directly — see Phase 2).
9. No Python conditionals in the hot path — use `jnp.where` throughout.

**No Python loop over ticks.** The rollout loop is in the trainer (Task 1.5), not here.

---

### Task 1.4 — Observation builder (MLP-compatible)

**File:** `octopus/v2/jax_env.py`

```python
def get_obs_mlp(state: OctopusState, event_host: int, event_mem: float,
                adj: np.ndarray, pod_dram: float, base_tick: int) -> jnp.ndarray:
```

Output layout matches `OctopusMemPoolEnv._get_obs` exactly (size `2*max_degree + 4`):
- accessible MPD loads (normalized, padded to max_degree)
- accessibility mask
- VM memory request (normalized)
- global peak (normalized)
- sin/cos hour-of-day

This lets the JAX MLP policy produce results directly comparable to the SB3 MLP.

---

### Task 1.5 — Rollout with `jax.lax.scan`

**File:** `octopus/v2/jax_env.py`

```python
def rollout(
    policy_fn,                 # (obs, params) -> action
    params,
    trace: JaxTrace,
    adj: np.ndarray,
    key: jnp.ndarray,
    policy_type: str = "mlp",  # "mlp" or "gnn"
) -> Tuple[OctopusState, Metrics]:
```

Use `jax.lax.scan` over ticks:
```python
def scan_body(carry, _):
    state, params = carry
    obs = get_obs_mlp(state, ...) if policy_type == "mlp" else state
    action = policy_fn(obs, params)
    next_state, obs, reward, done = step(state, action, trace, adj)
    return (next_state, params), (reward, done)

(final_state, _), (rewards, dones) = jax.lax.scan(scan_body, (init_state, params), None, length=trace.T)
```

**JIT:** `jax.jit(rollout, static_argnums=(0, 3, 5))` — policy_fn, adj, policy_type are
static. Trace once per (policy_type, topology) combination.

---

### Task 1.6 — Verify JAX MLP matches SB3 MLP

Before building the GNN, verify the JAX environment + MLP baseline produces similar
savings numbers to the SB3 MLP. Load the existing SB3 `best_model.zip` weights,
copy them into the JAX MLP parameter dict, run `scripts/v2/eval_rl_jax.py`-equivalent
on JAX env. The savings should agree within ±0.005.

This is the sanity check that the JAX step function is correct.

---

### Task 1.7 — State extensions (observation v1)

After the JAX MLP baseline is verified, extend the observation with the Phase 1 signals.
These are new signals not in the SB3 baseline — add them only to the JAX path.

New signals per MPD (added to MLP obs, used directly in GNN node features):
- `Δ_1h = mpd_load - mpd_load_1h` (trend: is this MPD filling up?)
- `Δ_2h = mpd_load - mpd_load_2h`

New signals per server:
- `srv_pressure_1h`, `srv_pressure_2h` (arrival rate trend)

New time encoding:
- `sin(2π·dow/7)`, `cos(2π·dow/7)` (day-of-week, alongside existing hour-of-day)

MLP obs size grows from `2*max_degree + 4` to `4*max_degree + 8` with the new signals.
This breaks SB3 MLP compatibility — that's fine, document the obs version in config.json.

---

## Phase 2 — GNN Policy

### Task 2.1 — Graph construction (`build_graph`)

**New file:** `octopus/v2/gnn_policy.py`

Build a `jraph.GraphsTuple` from `OctopusState` and the static `adj` matrix.

```
Nodes:
  - Servers [0 .. pod_size):
      features = [srv_pressure_t, Δ1h, Δ2h, sin(hour), cos(hour), sin(dow), cos(dow)]
                  shape: (pod_size, 7)
  - MPDs [pod_size .. pod_size+num_mhd):
      features = [mpd_load_t, Δ1h, Δ2h, slo_vio_24h]
                  shape: (num_mhd, 4)

Edges (one per CXL link):
  - senders:   MPD node indices (shifted by pod_size)
  - receivers: server node indices
  - features:  [1.0]  (binary link; extend with bandwidth util in Phase 6)
  - The edge list is built once from adj and cached as static — topology doesn't
    change within a training run.
```

Implementation:
```python
def build_edge_list(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (senders, receivers) for all non-zero entries in adj."""
    rows, cols = np.nonzero(adj)
    # servers = rows, mpds = cols (shifted by pod_size in graph node indexing)
    senders = cols + adj.shape[0]   # MPD → server direction
    receivers = rows
    return senders.astype(np.int32), receivers.astype(np.int32)

def build_graph(state: OctopusState, static_edges, pod_dram: float) -> jraph.GraphsTuple:
    senders, receivers = static_edges
    server_feats = jnp.stack([
        state.srv_pressure / pod_dram,
        (state.srv_pressure - state.srv_pressure_1h) / (pod_dram + 1e-9),
        (state.srv_pressure - state.srv_pressure_2h) / (pod_dram + 1e-9),
        jnp.full(state.srv_pressure.shape, jnp.sin(2*jnp.pi*hour/24)),
        jnp.full(state.srv_pressure.shape, jnp.cos(2*jnp.pi*hour/24)),
        jnp.full(state.srv_pressure.shape, jnp.sin(2*jnp.pi*dow/7)),
        jnp.full(state.srv_pressure.shape, jnp.cos(2*jnp.pi*dow/7)),
    ], axis=-1)
    cap = pod_dram / state.mpd_load.shape[0] + 1e-9
    mpd_feats = jnp.stack([
        state.mpd_load / cap,
        (state.mpd_load - state.mpd_load_1h) / cap,
        (state.mpd_load - state.mpd_load_2h) / cap,
        state.slo_vio_24h,
    ], axis=-1)
    node_feats = jnp.concatenate([server_feats, mpd_feats], axis=0)
    edge_feats = jnp.ones((len(senders), 1), dtype=jnp.float32)
    return jraph.GraphsTuple(
        nodes=node_feats, edges=edge_feats,
        senders=senders, receivers=receivers,
        n_node=jnp.array([node_feats.shape[0]]),
        n_edge=jnp.array([len(senders)]),
        globals=None,
    )
```

---

### Task 2.2 — Message passing (2 rounds)

**File:** `octopus/v2/gnn_policy.py`

Two rounds of message passing, implemented as Flax modules:

**Round 1: MPD → Server** (aggregate load context at each server)
```
message_fn_1: edge_feat ∥ mpd_feat[sender] → message_vec (hidden_dim,)
aggregate:    mean over all incoming MPD messages at each server node
update_fn_1:  server_feat ∥ agg_message → updated_server_feat (hidden_dim,)
```

**Round 2: Server → MPD** (aggregate demand pressure at each MPD)
- Reverse edge direction: senders = servers, receivers = MPDs.
- This tells each MPD which servers are putting pressure on it.
```
message_fn_2: edge_feat ∥ updated_server_feat[sender] → message_vec
aggregate:    sum over all incoming server messages at each MPD node
update_fn_2:  mpd_feat ∥ agg_message → updated_mpd_feat
```

Both `message_fn` and `update_fn` are 2-layer MLPs with LayerNorm + ReLU.
Hidden dim: 64 (tunable via config).

No Python conditionals in message/update functions — all ops are JAX-compatible.

Implement using `jraph.GraphNetwork` with custom `update_edge_fn`, `update_node_fn`.
For the two-round structure, apply the GraphNetwork twice with different weight sets.

---

### Task 2.3 — Action head (per-edge softmax)

**File:** `octopus/v2/gnn_policy.py`

After message passing, for each server, produce one allocation weight per connected MPD:

```python
def action_head(graph_after_mp: jraph.GraphsTuple, adj: np.ndarray,
                updated_node_feats: jnp.ndarray) -> jnp.ndarray:
    """
    Returns alloc_weights: (pod_size, num_mhd) float32.
    Zero for non-existent edges (enforced structurally, not by masking).
    Each row (server) sums to 1.0 over its accessible MPDs.
    """
```

Implementation:
- For each edge (server→MPD), compute a scalar logit:
  `logit = MLP(updated_server_feat[server] ∥ updated_mpd_feat[mpd])`
- Segment softmax over edges grouped by server (receiver):
  use `jraph.segment_softmax(logits, segment_ids=receivers_server_indices)`.
- Scatter weights back into a `(pod_size, num_mhd)` matrix using static edge indices.
- Structural zeros: non-existent edges are never in the graph → their weights are zero
  by construction, no post-hoc masking needed.

**This is the key architectural difference from MLP:** the MLP applies softmax over a
padded flat vector and clamps inaccessible entries to zero after the fact. The GNN
softmax only runs over edges that exist in M.

---

### Task 2.4 — GNN SAC actor/critic

**File:** `octopus/v2/gnn_policy.py`

For SAC we need a stochastic actor and two Q-critics.

**Actor** (GNN):
- Input: `OctopusState` → `build_graph` → 2-round GNN → action head → mean logits μ
- Add a learned log-std parameter (scalar per edge, clipped to [-5, 2]).
- Output: `Normal(μ, exp(log_std))` distribution; reparameterised sample + tanh squash.

**Critic** (GNN or MLP — use MLP for critic, GNN for actor only):
- Two separate Q-networks: `Q1(state, action)`, `Q2(state, action)`.
- Critic input: flatten `OctopusState` arrays + flatten action into a single vector.
  Using MLP for critic is standard (state-action value doesn't need graph structure —
  the action is already the allocation decision, not a raw logit).
- Architecture: [256, 256] MLP with ReLU, same as SB3 baseline.

Justification for MLP critic: the GNN inductive bias is most valuable at the *policy*
level (what allocation to make given link structure). The critic just needs to estimate
value given any state-action pair, which is a regression problem with no structural
constraint. Mixing GNN actor + MLP critic is standard practice (e.g. GNN-SAC in robot
manipulation literature).

---

### Task 2.5 — Training run

**Single run:** `v2_gnn_sac`

```
python scripts/v2/train_rl_jax.py \
    --run-id v2_gnn_sac \
    --policy-type gnn \
    --topology data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
    --trace AMS20PrdApp19-tround.sqlite \
    --total-timesteps 1000000 \
    --n-envs 512
```

All outputs go to `output/v2/checkpoints/v2_gnn_sac/` with `config.json`
(`"framework": "jax"`, `"policy_type": "gnn"`).

Evaluation on all 10 traces:
```
python scripts/v2/eval_rl_jax.py --run-id v2_gnn_sac --n-iter 50
```

Output: `output/v2/evals/v2_gnn_sac/results.csv` (same schema as v1 `results.csv`).

Plotting (optionally overlay v1 SB3 bars):
```
python scripts/v2/plot_results_jax.py \
    --v2-dirs output/v2/evals/v2_gnn_sac \
    --v1-dirs output/rl_evals/v1_sac_baseline   # optional
```

---

### Task 2.6 — LCPO deployment export

After `v2_gnn_sac` training is complete, export the GNN actor weights for LCPO
deployment:

```
python scripts/v2/export_lcpo.py --run-id v2_gnn_sac
```

**New file:** `scripts/v2/export_lcpo.py`

This script:
1. Loads the best orbax checkpoint from `output/v2/checkpoints/v2_gnn_sac/`.
2. Extracts the GNN actor params (not critic, not temperature).
3. Serialises to `output/v2/lcpo_export/v2_gnn_sac/actor_params.msgpack`
   plus `model_config.json` (topology, obs version, hidden_dim).
4. Prints a deployment command stub for the LCPO runtime.

The LCPO runtime calls `octopus/v2/gnn_policy.py:GNNActor.apply(params, graph)`
directly — no SB3 or Gymnasium dependency at inference time.

---

## Phase 3 — Vectorisation & Multi-GPU

### Task 3.1 — `vmap` over environment instances

**File:** `octopus/v2/jax_trainer.py`

```python
# Vectorise reset and step over N parallel envs
batch_reset = jax.vmap(reset, in_axes=(None, None, 0))  # 0 = batch key axis
batch_step  = jax.vmap(step,  in_axes=(0, 0, None, None))  # 0 = batch state/action axes
```

N = 512 (start here; increase to 1024 if VRAM allows on GPU 0).
Each env gets its own PRNG key via `jax.random.split(master_key, N)`.

GPU assignment: `jax.device_put(init_states, device=jax.devices()[0])` — all rollouts
on GPU 0.

Rollout with scan over batch:
```python
def batched_rollout(policy_fn, params, traces, adj, keys):
    init_states = batch_reset(traces, adj, keys)
    def scan_body(carry, _):
        states, params = carry
        obs_batch = jax.vmap(get_obs)(states, ...)
        actions = jax.vmap(policy_fn, in_axes=(0, None))(obs_batch, params)
        next_states, rewards, dones = batch_step(states, actions, traces, adj)
        return (next_states, params), (rewards, dones, obs_batch, actions)
    ...
```

**Note on trace batching:** all N envs use the *same* trace but different random pod
assignments (the random seed baked into `reset` selects which nodes form the pod). This
avoids loading N separate traces into VRAM.

---

### Task 3.2 — Replay buffer

**File:** `octopus/v2/jax_trainer.py`

SAC requires an experience replay buffer. Implement as a fixed-size circular buffer in
JAX (pure arrays, no Python object state):

```python
@flax.struct.dataclass
class ReplayBuffer:
    obs:     jnp.ndarray   # (capacity, obs_dim)
    actions: jnp.ndarray   # (capacity, action_dim)
    rewards: jnp.ndarray   # (capacity,)
    next_obs:jnp.ndarray   # (capacity, obs_dim)
    dones:   jnp.ndarray   # (capacity,) bool
    ptr:     jnp.ndarray   # () int32, write pointer
    size:    jnp.ndarray   # () int32, current fill
    capacity:int           # static
```

Add and sample ops must be JIT-compatible (`jnp.roll`-based circular write; random
index sampling via `jax.random.choice`).

Capacity: 500_000 transitions (≈ 3 GB for 128-dim obs float32 — fits in GPU VRAM).

---

### Task 3.3 — SAC in JAX

**File:** `octopus/v2/jax_trainer.py`

Port SAC update to pure JAX/Optax. All standard; key points:

- Actor loss: `L_π = E[α·log π(a|s) - min(Q1, Q2)(s, a)]`
- Critic loss: Bellman with target networks (soft update τ=0.005)
- Temperature: auto-tune α with target entropy `-dim(action_space)`
- One `jax.jit`-compiled `train_step(params, opt_states, buffer, key)` function
- `pmap` across all 3 GPUs for critic updates (rollout stays on GPU 0):

```python
# Gradient sync across all 3 GPUs
critic_update = jax.pmap(
    critic_grad_step,
    axis_name="devices",
    devices=jax.devices(),   # all 3 TITAN RTX
)
```

Use `jax.lax.pmean(grads, axis_name="devices")` for gradient averaging.

---

### Task 3.4 — `train_rl.py` (JAX version)

**New file:** `scripts/v2/train_rl_jax.py`

Arguments:
```
--run-id STR                required
--policy-type {gnn}         default: gnn   (mlp path kept but not the focus)
--trace STR                 default: AMS20PrdApp19-tround.sqlite
--eval-trace STR            default: LON23PrdApp01-troundgrt5m.sqlite
--topology STR              default: data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv
--total-timesteps INT       default: 1_000_000
--seed INT                  default: 42
--n-envs INT                default: 512
--buffer-size INT           default: 500_000
--reward-option {A, B}      default: A
--hidden-dim INT            default: 64   (GNN message passing width)
--checkpoint-freq INT       default: 50_000
--eval-freq INT             default: 50_000
```

All outputs go to `output/v2/checkpoints/<run-id>/` with `config.json`:
```json
{"framework": "jax", "policy_type": "gnn", "run_id": "v2_gnn_sac", ...}
```

Checkpoint format: Flax orbax checkpoint at `output/v2/checkpoints/<run-id>/ckpt_<step>/`.

---

### Task 3.5 — JIT compilation timing

After first training run, print:
```
[jit] First step compile time: 47.3s
[jit] Subsequent step time:    0.003s
[jit] Steps/sec (after warmup): 33,000
```

Target throughput: ≥ 10,000 steps/sec on GPU 0 with N=512 envs and JIT warmed up.
If below target, profile with `jax.profiler.trace` and identify the bottleneck
(usually either the replay buffer sample or the critic forward pass).

---

## Execution order

```
Phase 1:
  1.1 (trace preprocessing → data/traces_v2/)
      ↓
  1.2 + 1.3 (OctopusState + step/reset)  — implement together
      ↓
  1.4 (MLP obs builder) → 1.6 (verify JAX MLP ≈ SB3 MLP)
      ↓
  1.5 (rollout + scan)
      ↓
  1.7 (state extensions / obs v1)

Phase 2:
  2.1 (build_graph)           ← can start as soon as 1.2 done
      ↓
  2.2 (message passing)
      ↓
  2.3 (action head)
      ↓
  2.4 (GNN SAC actor/critic)
      ↓
  2.5 (launch v2_gnn_sac training run)
      ↓
  2.6 (export_lcpo.py — after training completes)

Phase 3:
  3.1 (vmap)  ← depends on 1.3
      ↓
  3.2 (replay buffer)
      ↓
  3.3 (SAC in JAX)
      ↓
  3.4 (train_rl_jax.py)  ← wires Phase 1–3 together; runs v2_gnn_sac
      ↓
  3.5 (JIT timing)

Phase 3 gates on Phase 1 (needs octopus/v2/jax_env.py).
Phase 2 runs concurrently with Phase 3 (needs build_graph from 2.1 only).
```

---

## Verification checklist

- **1.1**: `python octopus/v2/jax_trace.py --trace AMS20PrdApp19-tround` prints shape
  `(T, max_vms_per_tick)` and VRAM estimate in MB. Rerun prints "cache hit, skip".
- **1.3**: Single-step unit test — manually set state, action, check `next_state.mpd_load`
  matches manual calculation. Run in `tests/v2/test_jax_env.py`.
- **1.6**: JAX MLP savings within ±0.005 of SB3 MLP on AMS20/LON23 (5 seeds).
- **2.2**: `build_graph` + 2-round GNN forward pass runs without error on 16-host
  topology. Print node/edge shapes.
- **2.3**: `alloc_weights[server].sum() ≈ 1.0` for all servers; `alloc_weights[server, j] == 0`
  for all `j` where `adj[server, j] == 0`. Test with `tests/v2/test_gnn_action_head.py`.
- **2.5**: `output/v2/checkpoints/v2_gnn_sac/config.json` exists and contains
  `"framework": "jax"`, `"policy_type": "gnn"`. Training log shows steps/sec > 1,000.
- **2.6**: `output/v2/lcpo_export/v2_gnn_sac/actor_params.msgpack` exists;
  `model_config.json` contains topology path and hidden_dim.
- **3.1**: `batch_reset` + one `batch_step` run without OOM on GPU 0 with N=512.
- **3.5**: Steps/sec ≥ 10,000 after JIT warmup (printed to stdout).

---

## File layout: v1 (SB3/existing) vs v2 (JAX/new)

All new JAX code lives under `octopus/v2/` and `scripts/v2/` so the existing
SB3 pipeline (`octopus/env.py`, `scripts/train_rl.py`, `scripts/eval_rl.py`,
`scripts/plot_results.py`) is **untouched**.

```
# v1 — existing SB3 pipeline (DO NOT MODIFY)
octopus/env.py               # OctopusMemPoolEnv (Gymnasium + SB3)
scripts/train_rl.py          # SB3 SAC/PPO training
scripts/eval_rl.py           # SB3 checkpoint eval → results.csv
scripts/plot_results.py      # figures from CSV

# v2 — new JAX pipeline (all new files)
octopus/v2/__init__.py
octopus/v2/jax_trace.py      # trace preprocessing → JaxTrace NPZ cache
octopus/v2/jax_env.py        # OctopusState, reset, step, get_obs_mlp, rollout
octopus/v2/gnn_policy.py     # build_graph, GNN message passing, action head, SAC actor
octopus/v2/jax_trainer.py    # ReplayBuffer, SAC update, batched rollout, pmap setup

scripts/v2/train_rl_jax.py   # CLI entry point (mirrors scripts/train_rl.py)
scripts/v2/eval_rl_jax.py    # eval for JAX checkpoints → results.csv (same schema)
scripts/v2/plot_results_jax.py  # figures: handles both v1 and v2 run-ids side by side
scripts/v2/export_lcpo.py    # extract GNN actor weights → msgpack for LCPO runtime

tests/v2/test_jax_env.py        # unit tests for step correctness
tests/v2/test_gnn_action_head.py# structural zeros + softmax normalisation
```

**Cross-version plotting:** `scripts/v2/plot_results_jax.py` accepts both
`--v1-dirs` (SB3 `output/rl_evals/*/results.csv`) and `--v2-dirs` (JAX
`output/v2/evals/*/results.csv`) to overlay v1 SB3 bars with the v2 GNN bar.
The v1 `scripts/plot_results.py` is never called with v2 data.

**Data cache:** preprocessed NPZ files go to `data/traces_v2/` (separate from
`data/traces/` used by v1).

**Checkpoint output:** v2 runs write to `output/v2/checkpoints/<run-id>/`
(separate from `output/rl_evals/` used by v1).

**LCPO export:** `output/v2/lcpo_export/<run-id>/actor_params.msgpack` + `model_config.json`.

**requirements.txt** — add these without removing existing deps:
```
jax[cuda12_pip]
flax
optax
jraph
chex
orbax-checkpoint
```
