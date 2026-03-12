# RL-Based Memory Pooling: Research Roadmap v2
**Octopus CXL Systems — Iterative Design Document**

---

## Table of Contents
1. [Overview & Current Status](#1-overview--current-status)
2. [Step 1 — Fix State Space](#2-step-1--fix-state-space)
3. [Step 2 — Fix Reward Function](#3-step-2--fix-reward-function)
4. [Step 3 — More Training + Domain Randomization](#4-step-3--more-training--domain-randomization)
5. [Step 4 — Curriculum Learning](#5-step-4--curriculum-learning)
6. [Step 5 — GNN Architecture](#6-step-5--gnn-architecture)
7. [Step 6 — Ablation Study: SAC vs PPO vs LCPO](#7-step-6--ablation-study-sac-vs-ppo-vs-lcpo)
8. [Step 7 — Online Deployment with LCPO](#8-step-7--online-deployment-with-lcpo)
9. [Open Questions](#9-open-questions)

---

## 1. Overview & Current Status

### Problem
Allocate VM memory requests across MPDs in the sparse Octopus CXL bipartite graph to minimize peak per-MPD load (maximizing pooling savings), without migration.

### Why Current SAC Fails
The current SAC agent (1M timesteps, fast config) produces a 20× max/min MHD load imbalance — worse than greedy. Root causes identified:

| Cause | Evidence | Fix |
|---|---|---|
| Sparse reward signal | Only worst MPD gets gradient | Add Var(c) shaping term |
| Insufficient training | ~62k gradient updates | More training, better schedule |
| Missing temporal state | Agent cannot distinguish rising vs. falling loads | v1 state space |
| Missing VM pressure state | MPD load hides long-lived vs. short-lived VMs | New features below |
| Train/eval topology mismatch | Trained on 16 hosts, tested on varied mappings | Domain randomization |

---

## 2. Step 1 — Fix State Space

### 2.1 What v0 State Is Missing

The v0 state encodes current MPD loads as a single number `c_j`. This hides a critical distinction:

```
MPD A: 100 GB load
  └── VM1: 90 GB, 6 hrs old, lifetime = 7 hrs  → frees up in ~1 hr
  └── VM2: 10 GB, 1 hr old,  lifetime = 48 hrs → locked for days

MPD B: 100 GB load
  └── VM1: 50 GB, 1 hr old,  lifetime = 2 hrs  → frees up soon
  └── VM2: 50 GB, 1 hr old,  lifetime = 2 hrs  → frees up soon
```

Both MPDs look identical under v0. MPD B is about to free 100 GB. MPD A is essentially locked. A smart agent must prefer allocating new VMs to MPD A — but has zero signal for this under v0.

### 2.2 New Features: VM Pressure Profile per MPD

Since VM lifetimes are **known at arrival time**, we can compute forward-looking pressure features for each accessible MPD `j`:

| Feature | Formula | What it captures |
|---|---|---|
| Short-term release (1 hr) | `Σ mem_v · 1[remaining_lifetime_v < 1hr]` | Memory freeing up imminently |
| Medium-term release (4 hr) | `Σ mem_v · 1[remaining_lifetime_v < 4hr]` | Medium-horizon headroom |
| Weighted sticky load | `Σ mem_v · remaining_lifetime_v` | How "locked" is this MPD |
| VM count on MPD | `|VMs on j|` | Fragmentation signal |

All normalized by `D_pod` (total pod DRAM).

### 2.3 Full v2 State Space

Building on the v1 proposed state, the full observation vector is:

| # | Category | Signal | Temporal Treatment | Encoding |
|---|---|---|---|---|
| 1 | Resource Health | Per-MPD capacity util | 3 windows, 1-hr | `[v_t, Δ1h, Δ2h] × N_MPD` |
| 2 | Resource Health | Per-MPD bandwidth util | Current snapshot | `[v_t] × N_MPD` |
| 3 | Resource Health | Server memory pressure | 3 windows, 1-hr | `[v_t, Δ1h, Δ2h] × N_srv` |
| 4 | Resource Health | Recent SLO violations | 24-hr summary | Scalar (% allocs overflowing) |
| 5 | Workload | Incoming VM size | Current snapshot | Scalar (GB) |
| 6 | Workload | VM lifetime | Known at arrival | Scalar (hours) |
| 7 | Workload | Peak-to-average ratio | 24-hr summary | max / mean(usage_24h) |
| 8 | Workload | Predicted memory growth | Current snapshot | Scalar (GB/hr) |
| 9 | Topology | Server-to-MPD connectivity | Static | Binary mask ∈ {0,1}^N_MPD |
| **10** | **VM Pressure** | **Per-MPD short release (1hr)** | **Forward-looking** | **Scalar × N_MPD** |
| **11** | **VM Pressure** | **Per-MPD medium release (4hr)** | **Forward-looking** | **Scalar × N_MPD** |
| **12** | **VM Pressure** | **Per-MPD sticky load** | **Forward-looking** | **Scalar × N_MPD** |
| **13** | **VM Pressure** | **VM count per MPD** | **Current snapshot** | **Scalar × N_MPD** |

Features 10–13 (bold) are new additions. Features 1–9 are from v1.

### 2.4 Implementation Note

These features require iterating over VMs currently allocated to each MPD at decision time. This is O(N_VMs) per decision — acceptable at VM arrival frequency. Cache the per-MPD VM lists and update on arrivals/departures.

---

## 3. Step 2 — Fix Reward Function

### 3.1 The Core Problem: Credit Assignment

The true objective is:

```
Pooling Savings = 1 - (m · max_j PeakLoad_j) / Σ DRAM_i
```

This is computed **once, at the end of the full trace**. A bad allocation at hour 2 might not manifest as a peak until hour 14. The agent has made thousands of decisions in between — how does it know which one caused the peak?

This is the **credit assignment problem**.

### 3.2 Why the Current Reward Fails

Current reward:
```
R = α(1 - max_j c_j / cap_max) - β · slo_vio - γ · 1[migration]
```

**Problem 1 — Sparsity:** Only `max_j c_j` produces a non-zero gradient. If you have 192 MPDs and only one is penalized, the other 191 get no learning signal. The agent rationally ignores them — exactly what Figure 8 shows (20× load imbalance).

**Problem 2 — Myopia vs. true objective:** Minimizing the current-step peak `max_j c_j(t)` does not minimize the end-of-trace peak `max_j PeakLoad_j`. At midnight, all loads are low. An agent optimizing the per-step metric can pack midnight VMs onto one MPD (still looks fine at midnight) — but those are long-lived VMs still there at the 9am rush, spiking the true peak.

### 3.3 Term 1: Peak Penalty (immediate)

```
T1 = -α · max_j c_j(t) / cap_max
```

Penalizes the worst MPD at every step. This is the per-step proxy for the true objective. Gives the agent a dense signal pointing in the right direction.

**Why it's still needed even with Term 2:** Term 2 penalizes variance but doesn't penalize absolute load level. You need both — otherwise the agent could balance all MPDs at a high load level and still score well.

### 3.4 Term 2: Balance Penalty (immediate)

```
T2 = -β · Var(c(t))
```

Where `Var(c(t)) = (1/m) Σ_j (c_j(t) - mean_c)²` — variance of load across all MPDs.

This gives dense gradient signal for **every MPD**, not just the worst one. An agent that dumps load onto one MPD gets penalized even if that MPD isn't the global max yet.

**Reward hacking risk:** An agent could theoretically minimize variance by packing all load onto one MPD (variance = 0 if everything is equal). This is prevented by Term 1, which penalizes the resulting high peak. Together:
- Term 1 prevents high peaks → prevents packing onto one MPD
- Term 2 prevents imbalance → prevents uneven distribution
- Together they push toward uniform low load

**Validation requirement:** Before deploying this reward, verify empirically that `T1 + T2` correlates with true pooling savings. Run greedy, optimal, and random policies and confirm their reward ranking matches their savings ranking.

### 3.5 Term 3: SLO Violation Penalty (immediate)

```
T3 = -γ · slo_vio(t)
```

Where `slo_vio(t)` is the fraction of allocations in the current step that caused any MPD to exceed capacity.

**How Term 3 is calculated concretely:**

At each VM arrival event, the agent proposes an allocation vector `a`. Before committing, check:

```python
for j in accessible_mpds:
    new_load = current_load[j] + a[j]
    if new_load > mpd_capacity[j]:
        slo_vio += a[j] / total_vm_size  # fraction of this VM's memory causing overflow
```

This gives a continuous signal proportional to how much memory overflowed, rather than a binary 0/1. This is important — a binary signal would give zero gradient for allocations that are almost-but-not-quite violating, making it harder to learn the boundary.

**Weight ordering:** `γ >> α, β` — SLO violations are hard constraints. A single overflow is worse than any amount of imbalance.

### 3.6 Option B: Delayed Reward at VM Termination

The immediate reward alone still has a credit assignment gap for long-horizon consequences. Option B adds a **delayed component** triggered at VM termination:

```
R_t^total = R_t^immediate + (λ^T / T) · (-β · slo_vio_T - δ · 1[migrated])
```

Where:
- `T` = VM lifetime in hours (known at arrival)
- `λ` = discount factor
- `slo_vio_T` = total SLO violations caused by this VM over its lifetime
- `δ · 1[migrated]` = penalty if this VM was ever migrated

The `λ^T / T` term discounts and normalizes by lifetime — a longer-lived VM gets a proportionally smaller per-step delayed reward, but its total lifetime contribution is correctly weighted.

### 3.7 Full Integrated Reward

Combining all terms:

```
R_t = α · (1 - max_j c_j(t) / cap_max)    ← Term 1: immediate peak penalty
    - β · Var(c(t))                          ← Term 2: immediate balance penalty  
    - γ · slo_vio(t)                         ← Term 3: immediate SLO penalty
    + (λ^T / T) · (-γ · slo_vio_T           ← Option B: delayed SLO at termination
                   - δ · 1[migrated])        ← Option B: delayed migration penalty
```

**Weight priority ordering:** `γ >> β > α > δ`

Meaning: avoid SLO violations first, then balance load, then minimize peak, then avoid migration. This reflects the business objective ordering: correctness > efficiency > cost.

**Suggested starting weights:**
```
α = 0.1,  β = 0.3,  γ = 1.0,  δ = 0.05,  λ = 0.95
```

These are initial values — tune via ablation on held-out traces.

---

## 4. Step 3 — More Training + Domain Randomization

### 4.1 How Much Training

Current: ~62k gradient updates (train_freq=16, 1M steps) — clearly insufficient for a 14-day trace with complex diurnal structure.

Target: at minimum 10M environment steps, ideally 50M+. The LCPO paper uses 8-20M steps for simpler environments. A 14-day trace with VM-level granularity is substantially more complex.

Use the **fast configuration** (train_freq=16) as the default — the document already shows the slow config (train_freq=1) overfits. Consider train_freq=32 or 64 for even less overfitting at large timestep counts.

### 4.2 Domain Randomization Axes

Randomize across all of the following during training:

| Axis | Training range | Notes |
|---|---|---|
| Pod-to-node assignment | 50 random seeds per episode | Already implemented |
| Datacenter trace | Rotate AMS, LON, BLA, DSM, SYD | Different diurnal phases |
| CXL fraction | 10%, 30%, 50% | Vary within training |
| Pod size | 8 → 96 (via curriculum) | See Step 4 |
| Link failure rate | 0–5% random edge removal | Tests robustness |
| Time window | Random start within trace | Prevents memorizing specific days |

### 4.3 Train/Eval Split

- **Training traces:** AMS, BLA, DSM, SYD, YTO (5 datacenters)
- **Validation trace:** LON (held out — different geography and diurnal phase)
- **Test traces:** LVL, BN9, SG2, DUB (held out completely until final evaluation)

This prevents overfitting to a single datacenter's diurnal pattern.

---

## 5. Step 4 — Curriculum Learning

The idea: train on easy versions of the problem first, then progressively increase difficulty. The agent builds intuitions on small instances and transfers them to larger ones.

### 5.1 Difficulty Axes

**Axis 1 — Pod size** (primary axis)

| Stage | Pod size | MPDs | Why easier |
|---|---|---|---|
| 1 | 4 hosts | 6 MPDs | Nearly enumerable allocation space |
| 2 | 8 hosts | ~12 MPDs | Small but non-trivial |
| 3 | 16 hosts | 20 MPDs | Current experiment size |
| 4 | 32 hosts | ~40 MPDs | Getting complex |
| 5 | 96 hosts | 120 MPDs | Production target |

**Axis 2 — Trace complexity**

| Stage | Trace type | Why easier |
|---|---|---|
| 1 | Synthetic flat demand | No temporal dynamics to learn |
| 2 | Synthetic simple diurnal | Learn time-of-day patterns in clean setting |
| 3 | Real single-datacenter (AMS) | Real complexity, one geography |
| 4 | Multi-datacenter rotation | Different diurnal phases, harder generalization |

**Axis 3 — CXL fraction**

| Stage | CXL fraction | Why easier |
|---|---|---|
| 1 | 10% | Small footprint, allocation barely matters |
| 2 | 30% | Moderate pressure |
| 3 | 50% | Full production setting |

### 5.2 Practical Curriculum Schedule

For the initial experiments, use **staged training** — train to convergence at each stage before moving to the next:

```
Stage 1: pod_size=4, CXL=10%, synthetic flat trace
         → train until eval reward plateaus for 500K steps
         → save checkpoint C1

Stage 2: pod_size=8, CXL=30%, synthetic diurnal trace
         → initialize from C1, train until plateau
         → save checkpoint C2

Stage 3: pod_size=16, CXL=50%, AMS real trace
         → initialize from C2, train until plateau
         → save checkpoint C3

Stage 4: pod_size=96, CXL=50%, multi-datacenter rotation
         → initialize from C3, train to convergence
         → this is the final production candidate
```

Initializing from prior weights is the key mechanism — the agent transfers load balancing intuitions rather than starting from scratch.

### 5.3 Convergence Criterion

Use the validation trace (LON) reward as the convergence signal, not training reward. Advance to the next stage when:
- Validation reward has not improved by more than 1% over 500K steps, AND
- Agent beats greedy baseline on validation trace

The second condition prevents advancing when the agent has simply memorized the training trace without actually learning to allocate well.

### 5.4 Optional: Automatic Curriculum Learning

Instead of manual stages, use a progress threshold to advance automatically:

```python
if current_val_savings > greedy_savings + threshold:
    increase_pod_size()
    increase_cxl_fraction()
```

This is more flexible but harder to debug. Recommend starting with manual stages and moving to automatic if the manual schedule proves too rigid.

---

## 6. Step 5 — GNN Architecture

**Note: GNN should only be attempted after Step 3/4 confirms the MLP baseline is learning correctly.**

### 6.1 Why MLP is Insufficient

The Octopus bipartite graph has structure the MLP cannot natively exploit:
- Server nodes and MPD nodes are different types
- Connectivity is sparse and irregular
- Relevant information (neighboring server loads, MPD degrees) requires message passing across the graph

An MLP with a flattened feature vector loses the topological relationships.

### 6.2 Proposed Architecture: Bipartite Message Passing

The graph has two node types, requiring a **heterogeneous GNN**:

```
Server nodes:  features = [memory_pressure, CXL_demand, Δ1h, Δ2h, ...]
MPD nodes:     features = [capacity_util, bandwidth_util, short_release, sticky_load, ...]

Message passing:
  Round 1: MPD → Server  (each server aggregates state of its connected MPDs)
  Round 2: Server → MPD  (each MPD aggregates pressure from its connected servers)
  Round 3: MPD → Server  (final aggregation for allocation decision)

Output:
  For arriving VM on server i:
    → Read out MPD node embeddings for j ∈ N(i)
    → Pass through allocation head → weight vector w
```

This is 3 rounds of message passing through the bipartite graph.

### 6.3 Why This Matters for Octopus

A key failure mode in the sparse topology is that a hot subset of servers (simultaneously elevated demand) can't spread load if they share few MPDs. The GNN can learn to detect these "hot neighborhoods" and proactively avoid overloading their shared MPDs — something an MLP with a flattened observation vector cannot reason about structurally.

### 6.4 Implementation Recommendation

Use PyTorch Geometric's `HeteroConv` for heterogeneous message passing. This handles two-node-type graphs natively. The GNN encoder replaces the first MLP layer; the actor and critic heads remain unchanged.

**Ablation:** Compare GNN vs. MLP with the same state features to isolate the contribution of graph structure vs. richer features.

---

## 7. Step 6 — Ablation Study: SAC vs PPO vs LCPO

The goal is to understand which algorithm is best for **training** and which is best for **online deployment** separately, since these are different problems.

### 7.1 Training Comparison

Train each algorithm from scratch on the curriculum schedule (Step 4), same hyperparameters where possible, same random seeds.

| Algorithm | Expected strength | Expected weakness | Key hyperparameter |
|---|---|---|---|
| SAC | Sample efficient, good with continuous actions | Off-policy instability, replay buffer temporal correlation | Replay buffer size, train_freq |
| PPO | Stable on-policy updates, no replay issues | Less sample efficient, needs more steps | Clip ratio ε, rollout length |
| LCPO | CF-resistant, stable on-policy | More complex, needs OOD detector, TRPO overhead | c_anchor, σ (OOD threshold) |

**Metrics to compare:**
- Pooling savings vs. greedy at each curriculum stage
- Training stability (variance across seeds)
- Wallclock time per environment step
- Convergence speed (steps to beat greedy)

**Expected finding:** PPO and LCPO likely outperform SAC on training stability, especially with the diurnal non-stationarity in multi-datacenter traces. SAC may converge faster early but plateau or destabilize.

### 7.2 Deployment Comparison (Critical Distinction)

Training performance ≠ deployment performance. Deployment introduces:
- Continuous context shift (new days, new workload regimes)
- No resets — the agent must maintain performance indefinitely
- Catastrophic forgetting risk as the agent sees new data

Test each algorithm in a **rolling deployment simulation**:

```
Week 1: Train on AMS trace days 1-7
         → deploy, evaluate on days 8-14
Week 2: Continue online training on days 8-14
         → evaluate on days 15-21 (simulated)
...
```

| Algorithm | Deployment behavior |
|---|---|
| SAC | Off-policy: may overfit to recent days, forget early patterns |
| PPO | On-policy: stable but no explicit forgetting prevention |
| LCPO | On-policy + CF prevention: explicitly anchors on old context patterns |

**Metrics:**
- Pooling savings on weeks 2, 3, 4 (does performance hold?)
- Performance on morning hours after training on evening data (direct CF measurement)
- Recovery speed after context shift (new datacenter suddenly added to pod)

**Hypothesis:** SAC and PPO will show performance degradation on earlier time-of-day patterns as deployment continues. LCPO will not.

### 7.3 Ablation Matrix

| Condition | SAC | PPO | LCPO |
|---|---|---|---|
| Offline training only | ✓ | ✓ | ✓ |
| Online deployment, no retraining | ✓ | ✓ | ✓ |
| Online deployment, continuous training | ✓ | ✓ | ✓ |
| Link failure robustness | ✓ | ✓ | ✓ |
| New datacenter generalization | ✓ | ✓ | ✓ |

Run all cells with 10 random seeds, report mean ± 95% CI.

---

## 8. Step 7 — Online Deployment with LCPO

### 8.1 Context Definition for LCPO

The "context" `z_t` in LCPO maps naturally to the current workload regime:

```python
z_t = [
    sin(2π · hour / 24),      # time-of-day (circular encoding)
    cos(2π · hour / 24),      # time-of-day (circular encoding)
    cluster_load_level,        # normalized current cluster demand
    day_of_week / 7,           # weekly pattern
]
```

This is already partially in the state space (sin/cos hour encoding). The cluster load level can be computed as `mean(c_j) / cap_max`.

### 8.2 OOD Detection for This Problem

Two samples are OOD if their contexts are sufficiently different:

```python
def is_ood(z_old, z_current, sigma=1.5):
    # L2 distance on the context vector
    return ||z_old - z_current||_2 > sigma
```

**Concrete meaning:** A morning experience (8am, low load) is OOD relative to an evening batch (8pm, high load) — their time-of-day encodings are far apart in L2 space. When the agent trains on the evening batch, LCPO constrains it to not change its outputs on those morning anchor samples.

**Threshold tuning:** Start with `σ = 1.5` (roughly 6-hour difference in context space) and tune via the ablation in §7.3.

### 8.3 Buffer Pre-population (Cold Start)

LCPO's buffer `B_a` starts empty. In production, this means the first few days have no anchoring. Fix this by **pre-populating `B_a` from offline training data**:

```
Before deployment:
  → Collect experiences from offline training trace
  → Reservoir sample into B_a (targeting ~200K samples)
  → Ensure diverse time-of-day coverage (morning, afternoon, evening, overnight)
  → Deploy with pre-populated buffer
```

This ensures Day 1 of deployment already has anchor samples covering all diurnal phases.

### 8.4 Deployment Architecture

```
Offline Phase:
  Curriculum training (Steps 3-5)
  → Best checkpoint saved
  → B_a pre-populated from training trace
  → Policy exported

Online Phase (continuous):
  VM arrival → observe (s_t, z_t)
  → Forward pass → allocation action a_t
  → Execute allocation
  → Observe reward r_t
  → Add (s_t, z_t, a_t, r_t) to B_r (recent batch)
  → Every N arrivals: LCPO update
      → OOD sample from B_a → S_c
      → If S_c non-empty: constrained update
      → Else: unconstrained update
      → Reservoir sample B_r into B_a
  → Repeat indefinitely
```

Update frequency N: one LCPO update per ~200 VM arrivals (equivalent to one rollout epoch). At typical cloud VM arrival rates this is every few minutes — fast enough to adapt to load shifts.

### 8.5 Safety During Online Updates

The safety wrapper handles action-level limits (MPD capacity). But policy drift during online updates could lead to systematically bad allocations before the next safety check.

Add a **policy drift monitor**:
- After each LCPO update, evaluate the updated policy on a fixed reference batch of 1000 samples
- If policy outputs have shifted by more than `c_monitor` KL divergence from the last validated checkpoint → roll back to checkpoint
- Alert and freeze online updates pending investigation

This provides a safety net against online learning going badly without requiring manual intervention for small drifts.

---

## 9. Open Questions

These are unresolved issues to address in future iterations:

| Question | Priority | Notes |
|---|---|---|
| Does VM lifetime predictor need validation before using as state feature? | High | If inaccurate, could actively hurt policy |
| What is the right update frequency N for online LCPO? | High | Too frequent → unstable; too infrequent → slow adaptation |
| How to handle MPD addition/removal (topology changes)? | Medium | GNN naturally handles this; MLP requires retraining |
| Should B_a be stratified by time-of-day rather than uniform reservoir? | Medium | Ensures all diurnal phases represented even with bursty traces |
| Is migration-aware RL worth pursuing? | Medium | Bounded migration could close gap to optimal significantly |
| How to set reward weights α, β, γ, δ without extensive tuning? | Medium | Multi-objective RL (MORL) could automate this |
| Does LCPO benefit from separate context encoder? | Low | Could learn z_t representation rather than using raw time-of-day |

---

*Document version: v2. Last updated: March 2026. This is a living document — iterate as experiments complete.*
