# Full Conversation Summary
## LCPO Paper Study + CXL Memory Pooling Research Design
**Date: March 2026 | Iterative research session**

---

## Table of Contents
1. [LCPO Paper — Full Concept Map](#1-lcpo-paper--full-concept-map)
2. [RL Fundamentals Covered](#2-rl-fundamentals-covered)
3. [LCPO Algorithm Deep Dive](#3-lcpo-algorithm-deep-dive)
4. [CXL Memory Pooling Problem](#4-cxl-memory-pooling-problem)
5. [Original Proposed Roadmap and Critique](#5-original-proposed-roadmap-and-critique)
6. [State Space Design](#6-state-space-design)
7. [Reward Function Design](#7-reward-function-design)
8. [Curriculum Learning Design](#8-curriculum-learning-design)
9. [Algorithm Ablation Strategy](#9-algorithm-ablation-strategy)
10. [LCPO Online Deployment Design](#10-lcpo-online-deployment-design)
11. [Open Questions](#11-open-questions)

---

## 1. LCPO Paper — Full Concept Map

### 1.1 The Core Problem: Catastrophic Forgetting in Online RL

The paper (Hamadanian et al., ICLR 2025 Spotlight) addresses **online reinforcement learning in non-stationary, context-driven environments**.

**Non-stationarity** means the environment's dynamics change over time. The source of change is an **exogenous context process** `z_t` — a signal the agent can observe but cannot influence. Examples:
- Wind force on a robot
- Incoming workload composition in a datacenter
- Time-of-day demand patterns

**Online RL** means there is no separate training phase. The agent learns and deploys simultaneously. Each interaction is experienced exactly once — no replays of real-world data.

**Catastrophic Forgetting (CF):** When a neural network is updated sequentially on non-stationary data, weight updates for one context corrupt the learned behavior for prior contexts. An agent that mastered "high wind" behavior will forget it while learning "low wind" behavior. This is the central problem LCPO solves.

---

### 1.2 Formal Setup: Context-Driven MDP

The environment is a **context-driven MDP**: `M = (S, Z, A, {z_t}, T, d_0, r)`

| Symbol | Meaning |
|---|---|
| `S` | State space |
| `Z` | Context space |
| `A` | Action space |
| `{z_t}` | Context trace — arbitrary, no restrictions |
| `T(s' | s, z, a)` | Transition kernel — may depend on context |
| `r(s, z, a)` | Reward function |
| `π(s, z)` | Policy — takes both state and context as input |

**No restrictions on the context process.** It can be discrete, continuous, piecewise stationary, smooth, sudden, i.i.d., periodic, or arbitrary. This is broader than most prior work which assumes clean task boundaries.

---

### 1.3 Objective: Lifelong Return

The goal is not episodic performance — it is **lifelong return**: average performance across the entire deployment horizon.

For episodic environments (physical resets — robot falls over, job ends):
- Episode `i` runs timesteps `H·i` to `H·(i+1)-1`
- Return for episode `i`: `R_i = Σ_{t=1}^H r_t`
- Lifelong return: `J(π, z) = lim_{T→∞} (Σ_i R_i) / T`

Episodes exist because environments need physical resets — not as an artificial training construct.

**Prescient policy:** The optimal policy that maximizes lifelong return must know the full context trace `z` in advance — "cheating" by having future knowledge. Used in experiments as an upper bound, not a fair competitor. Among all online methods evaluated, LCPO is closest to the prescient bound.

---

### 1.4 Why Prior Approaches Fail

**1. Regularization (EWC, OGD):**
- Penalize changes to weights important for old tasks
- Require explicit task labels — clean boundaries marking where one task ends and another begins
- Brittle when contexts change smoothly (no boundaries to label)

**2. Separate parameters per task:**
- Maintain independent networks per task
- Require task labels to know when to spawn new networks
- Don't scale when number of contexts is large or unknown

**3. Off-policy replay (SAC, DQN, experience replay):**
- Naturally revisit old experiences via replay buffer
- BUT: off-policy RL is unstable due to bootstrapping + function approximation errors
- Hyperparameter-sensitive, especially with temporal correlation in replay buffers
- Cannot be safely used for continuous online deployment

**4. Model-based replay (MBPO, MBCD):**
- Learn environment model, generate synthetic old experiences
- MBPO fails in paper despite having a highly accurate model — the issue is the learning algorithm, not model quality
- MBCD requires piecewise stationary context with abrupt boundaries — brittle on smooth/noisy contexts

**5. Latent context inference:**
- These works try to infer `ẑ_t` when context is unobserved
- LCPO assumes context IS observed — orthogonal problem
- Future work: combine latent inference → feed `ẑ_t` into LCPO

**LCPO's position:** Uses a buffer of old experiences (like rehearsal), but ONLY to constrain updates — never to train on old data directly. Forgetting prevention without off-policy instability.

---

### 1.5 OOD Detection vs. Change-Point Detection

Change-point detection (CPD) finds boundaries between task regimes. It requires:
- Piecewise stationary context
- Infrequent, abrupt changes
- Well-defined boundaries

OOD detection needs only:
- A distance metric on context space
- A threshold `σ` — "these contexts are sufficiently different"

Figure 2 in the paper shows a smooth context process where CPD with threshold 3.1 finds 34 spurious change-points. OOD with threshold 1 cleanly identifies the meaningful outlier context. OOD is not a limitation or problem — it IS the solution.

---

### 1.6 Illustrative Example: Grid World

A 3×3 grid, two context modes, context is observed by the agent.

- **No Trap mode:** Optimal path through center cell. Return = −3.
- **Trap Active mode:** Center cell costs −10. Must go around. Return = −5.

**Context sequence:** 0-4K (No Trap) → 4K-16K (Trap Active) → 16K-20K (No Trap again)

**A2C (neural network):**
- Learns No Trap optimally by 4K
- When context switches to Trap Active, neural network weights are updated — and this overwrites the No Trap policy
- At 16K when No Trap returns, the agent must relearn from scratch. TV distance between learned and optimal policy → 1.0 during inactive period

**Tabular A2C:**
- Separate value table row per (state, context) pair: 9 cells × 2 contexts = 18 rows
- Updating row (s, z=1) never touches row (s, z=0). Zero CF by construction.
- TV distance stays near 0 during inactive periods. Perfect recovery at 16K.
- Only works for small discrete spaces — proves the concept LCPO approximates.

**LCPO:**
- Not quite as perfect as tabular, but near-zero TV distance throughout
- Instantly recovers at 16K switch — no relearning needed
- Anchoring on OOD samples from the opposite context prevents forgetting while adapting

---

### 1.7 Experimental Results Summary

**Gymnasium environments (Pendulum-v1, HalfCheetah, Hopper):**
- LCPO outperforms all online algorithms on CDF of normalized lifelong return
- A2C is the strongest baseline despite being the simplest — stability beats sample efficiency
- SAC and off-policy methods underperform due to instability
- MBCD spawns 3-7 policies per seed (brittle to threshold)

**Straggler mitigation (real datacenter task):**
- LCPO achieves lowest tail latency across both workloads
- All LCPO variants (Aggressive/Medium/Conservative) perform similarly
- MBPO fails despite accurate environment model

**Sensitivity to OOD threshold σ:**
- LCPO affected by σ but remains above A2C baseline across all tested values
- Mahalanobis distance outperforms L2 distance

**Sensitivity to buffer size `n_b`:**
- Performance maintained down to `n_b = 500` samples
- Drops significantly below 500
- Remarkable robustness given 8-20M total environment steps

**Computational cost:** ~1.5× A2C per step. Acceptable overhead.

---

## 2. RL Fundamentals Covered

### 2.1 A2C (Advantage Actor Critic)

Two neural networks:

**Actor (policy network):**
- Input: (state `s`, context `z`)
- Output: probability distribution over actions
- Example: `π(right|s,z) = 0.7, π(up|s,z) = 0.1, ...`

**Critic (value network):**
- Input: (state `s`, context `z`)
- Output: single scalar — expected future reward from this state
- Example: `V(start, NoTrap) = −3.0`

**The Advantage:**
```
A(s, a) = r + γ · V(s_next) − V(s)
```
- Positive advantage → action was better than expected → increase its probability
- Negative advantage → worse than expected → decrease probability
- More stable than raw returns because it normalizes by expectation

**Actor loss (policy gradient):**
```
L_actor = −log π(a|s,z) · A(s,a)
```

**Critic loss:**
```
L_critic = (r + γ · V(s_next) − V(s))²
```

**Why A2C causes CF:** Weight updates for context z=1 modify shared network weights, corrupting outputs for context z=0. Neural networks generalize across all inputs — you can't update one context without affecting others.

---

### 2.2 A2C vs. SAC

| Aspect | A2C | SAC |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Experience replay | No — discard after use | Yes — replay buffer |
| Entropy regularization | No | Yes (max entropy objective) |
| Training stability | More stable | Less stable, hyperparameter sensitive |
| Sample efficiency | Lower | Higher |
| Online CF behavior | High — sequential updates | Lower but unstable off-policy |

**Key finding in paper:** Despite lower sample efficiency, A2C's stability advantage makes it the best baseline in online settings. SAC's hyperparameter sensitivity is dangerous in continuous deployment. This motivates moving to PPO and LCPO for the CXL problem.

---

### 2.3 Is LCPO Model-Free?

**Yes — fully model-free.** LCPO never learns or uses `T(s' | s, z, a)`. All experiences come from real environment interactions. No synthetic rollout generation.

Contrast with MBPO which learns transition model ensembles to generate synthetic old experiences. MBPO fails in the paper despite a highly accurate model, confirming the failure mode is the learning algorithm not the model quality.

---

## 3. LCPO Algorithm Deep Dive

### 3.1 Core Intuition

When training on current batch `B_r`, **constrain** the policy update to not change outputs on experiences from different past contexts (OOD samples from `B_a`).

This anchoring:
- Does NOT require task labels
- Only requires a distance metric on context space
- Prevents forgetting without off-policy instability
- Is a hard constraint (not a soft regularization loss)

---

### 3.2 The Optimization Problem

```
min_θ  L_tot(θ; B_r)  =  L_PG(θ; B_r) + L_e(θ; B_r)
s.t.   D_KL(π_{θ_old}(·|s), π_θ(·|s); W(B_a, B_r)) ≤ c_anchor
```

Where:
- `L_PG` = policy gradient loss (maximize returns on current batch)
- `L_e` = entropy regularization loss
- `W(B_a, B_r)` = OOD samples from buffer `B_a` relative to current batch `B_r`
- `c_anchor` = max allowed KL divergence on OOD anchor samples
- The KL is evaluated at old policy `π_{θ_old}` (not actions — the full distribution)

**Why not anchor to past actions?** The policy may not have converged when those actions were taken. Anchoring to suboptimal past actions would lock in bad behavior.

**Why a hard constraint vs. regularization loss?** A loss term like `L_anchor = E[CrossEntropy(π_old, π_new)]` has zero gradient at `θ = θ_old`. It can't enforce the constraint at initialization. LCPO-P (PPO-based proximal variant) uses this approach and underperforms. Hard constraint is necessary.

---

### 3.3 Algorithm Line by Line

**Setup:** `θ_0` = initial weights, `B_a` = empty old buffer, `n_b` = buffer capacity (200K)

For each iteration:
1. **Collect `B_r`** — 200 fresh interaction steps `(s_t, z_t, a_t, r_t, s_{t+1})`
2. **Find OOD anchors `S_c ← W(B_a, B_r)`** — sample 5b candidates from `B_a`, keep those with context far from current batch context
3. **Compute unconstrained gradient** `v ← ∇_θ L_tot(θ; B_r)|_{θ_0}` — standard A2C update direction
4. **If `S_c` non-empty (anchors exist):**
   - Compute `g(x) = ∇_θ(x^T ∇_θ D_KL|_{θ_0})|_{θ_0}` — Fisher-vector product (curvature of KL on OOD samples)
   - `v_c ← conjgrad(v, g(·))` — project gradient onto feasible region respecting constraint
   - Line search: halve `v_c` until both satisfied:
     - `D_KL(π; S_c) ≤ c_anchor` (don't forget old contexts)
     - `D_KL(π; B_r) ≤ c_recent` (TRPO stability on current batch)
   - `θ_0 ← θ_0 + v_c`
5. **If `S_c` empty:** `θ_0 ← θ_0 + v` (standard update — no forgetting risk if nothing is OOD)
6. **Reservoir sample** `B_r` into `B_a`

---

### 3.4 OOD Detection

**L2 distance** (gym environments):
```
μ_w = E_{w~B_r}[w]
is_ood(w) := ||w − μ_w||_2 > σ
```

**Mahalanobis distance** (straggler mitigation):
```
μ_w = E_{w~B_r}[w]
Σ_w = E_{w~B_r}[(w − μ_w)²]
is_ood(w) := (w − μ_w)^T Σ_w^{−1} (w − μ_w) > σ²
```

Mahalanobis accounts for scale and correlation of the context distribution. More robust than raw L2.

**Sampling to find `S_c`:** Draw 5b candidates from `B_a`, keep OOD ones. If fewer than b found → skip constraint for this update.

---

### 3.5 Reservoir Sampling

Maintains buffer `B_a` of bounded size `n_b` with uniform probability guarantee.

**Algorithm:**
- While `|B_a| < n_b`: add every sample
- Once full: for each new sample, pick `i ~ Unif(0, n_s−1)`
  - If `i < n_b`: replace `B_a[i]` with new sample
  - If `i ≥ n_b`: discard

**Guarantee:** Every past interaction has equal probability `n_b / n_s` of being in `B_a` at any time. No context is systematically favored or forgotten.

**Practical scale:** 8-20M total steps, `n_b = 200K` (1% of samples). Even at 1%, the buffer captures full diurnal diversity via uniform random retention.

---

## 4. CXL Memory Pooling Problem

### 4.1 Setting

**Microsoft Azure** datacenters use **Compute Express Link (CXL)** to pool DRAM across servers. Rather than each server owning fixed local memory, **MPDs (Multi-Ported CXL Devices)** expose pooled memory that multiple hosts can share.

**Octopus architecture:** Uses MPDs instead of expensive CXL switches. Servers connect to MPDs via a **sparse bipartite graph** — each server connects to exactly 8 MPDs, each MPD connects to a small number of servers. The sparsity makes allocation non-trivial.

**Why pooling helps:** Individual server peaks are unsynchronized. Pooling lets aggregate capacity cover the population peak rather than the sum of individual peaks. Production traces show peak-to-mean ratios of 1.13–1.26× cluster-wide — that gap is recoverable capacity.

---

### 4.2 The Allocation Problem

When VM arrives on server `S_i`, requesting `x` GB of CXL memory:
- Can only draw from MPDs in `N(i)` — its physical neighbors
- Must decide how to split `x` across those MPDs
- Once allocated, migration between MPDs has real runtime cost — minimized

**Formal objective:**
```
Pooling Savings = 1 − (m · max_j PeakLoad_j) / Σ DRAM_i
```
Where `PeakLoad_j = max_t Σ_{i: M_ij=1} d_i(t)` — worst-case total load on MPD j across all time.

Minimize peak per-MPD load → maximize pooling savings → reduce required CXL capacity → real CapEx savings at hyperscale.

**Breakeven:** 3% savings offsets the cost of MPDs over plain memory expansion. Acadia-96 topology achieves ~16% savings with greedy allocation.

---

### 4.3 Baselines

**Greedy:** Allocate to currently least-loaded accessible MPD. No migration. Myopic — places today's VM without considering future arrivals. Max/min MPD ratio ~2.3×.

**Optimal:** Access to future VM arrivals, unlimited migration. Solved via binary search + Dinic's max-flow algorithm. Impractical in production but sets the upper bound. Greedy-to-optimal gap is what RL is trying to close.

**PID Controller:** Reactive feedback — bias allocations toward under-loaded MPDs proportional to load error:
```
u_j(t) = K_P · e_j(t) + K_I · Σ e_j(τ) + K_D · (e_j(t) − e_j(t−1))
```
Allocate proportional to softmax of scores. Stronger than greedy (uses trend) but no future anticipation. Gap between PID and RL isolates the value of learned temporal patterns.

**RL target:** Decisions at VM arrival time, current state only, no migration. Close the greedy-to-optimal gap by learning to anticipate future demand.

---

### 4.4 Azure Production Data (10 Datacenters)

Two weeks of December 2024 VM traces across AMS, LON, BLA, LVL, BN9, SG2, DSM, SYD, YTO, DUB.

**Key patterns observed:**
- **Diurnal cycling:** Clear 24-hour rhythm at every site. Demand rises during local business hours, falls overnight. Amplitude and phase vary by geography (AMS/LON peak in European daytime; SYD/SG2 peak independently).
- **Peak-to-mean ratio:** 1.13–1.26× cluster-wide. This is the headroom that pooling targets.
- **Per-server heterogeneity:** Individual servers span a much wider band than the aggregate. Their peaks are uncorrelated — exactly why pooling works.
- **Low inter-day variance:** The diurnal pattern is highly consistent across days. This regularity is what an RL agent can exploit.

---

### 4.5 Why Current SAC Agent Fails

The current agent (SAC, 1M steps, fast config) produces a **20× max/min MPD load imbalance** — dramatically worse than greedy.

Root causes identified:

| Cause | Evidence | Fix |
|---|---|---|
| Sparse reward | Only worst MPD gets gradient. Other 191 get nothing. | Add Var(c) shaping term |
| Insufficient training | ~62k gradient updates | 10-50M steps minimum |
| Missing temporal state | Can't distinguish rising vs. falling loads | v2 state space |
| Missing VM pressure state | Can't distinguish sticky vs. freeing MPDs | VM lifetime features |
| Topology mismatch | Trained 16 hosts, evaluated varied mappings | Domain randomization |

**Critical point:** These are NOT catastrophic forgetting failures. CF is a real deployment risk but not the current failure. The agent hasn't learned balanced allocation at all — reward engineering and training volume must be fixed first. LCPO cannot fix a bad reward signal.

---

## 5. Original Proposed Roadmap and Critique

### 5.1 Original Steps (from v1 paper)
1. Fix states (add temporal data)
2. Fix reward function
3. Add GNN to replace MLP
4. More training + domain randomization
5. LCPO training + online deployment

### 5.2 Critiques Applied

**GNN ordering:** Adding GNN before confirming MLP can learn is a mistake. If the MLP fails after reward + training fixes, the GNN might learn — but you won't know if it was the architecture or the reward/training that fixed it. Move GNN after training validation (Step 5 in revised roadmap).

**Missing PPO:** Jumping from SAC directly to LCPO skips an important middle step. PPO is on-policy (like LCPO), simpler to implement, and establishes whether on-policy learning can solve the base task. If PPO fails, LCPO will too for the same underlying reason. If PPO succeeds, LCPO adds exactly one thing on top: forgetting prevention.

**Missing curriculum learning:** Jumping from 16-host experiments to 96-host production is likely to fail. The agent needs to build intuitions on small pods first, then transfer to larger ones.

**Missing ablation structure:** The roadmap proposed "use LCPO" without establishing what it's better than and by how much. Need explicit SAC vs PPO vs LCPO comparison — and crucially, need to test training performance and deployment performance separately, as they are different problems.

### 5.3 Revised Roadmap (Final)

| Step | What | Why |
|---|---|---|
| 1 | Fix state space (v2) | Foundation — can't learn without good signals |
| 2 | Fix reward function | Most urgent bottleneck |
| 3 | More training + domain randomization | Confirm learning works before changing architecture |
| 4 | Curriculum learning | Transfer from small to large pods |
| 5 | GNN architecture | Meaningful ablation over working MLP baseline |
| 6 | Ablation: SAC vs PPO vs LCPO | Training AND deployment performance separately |
| 7 | LCPO online deployment | Full production deployment |

---

## 6. State Space Design

### 6.1 The Missing Signal

v0 state encodes current MPD load `c_j` as a single number. This hides critical information:

```
MPD A: 100 GB load
  └── VM1: 90 GB, remaining lifetime = 1 hr  → frees up soon
  └── VM2: 10 GB, remaining lifetime = 48 hrs → locked for days

MPD B: 100 GB load
  └── VM1: 50 GB, remaining lifetime = 2 hrs
  └── VM2: 50 GB, remaining lifetime = 2 hrs  → both freeing soon
```

Both show identical `c_j`. MPD B is about to release 100 GB. MPD A is locked for days. The agent must prefer placing new VMs where headroom will open — but has zero signal under v0/v1.

VM lifetimes are **known at arrival time**, making this computable without prediction.

---

### 6.2 New Features: VM Pressure Profile per MPD

For each accessible MPD `j`, compute at decision time:

| Feature | Formula | What it captures |
|---|---|---|
| Short-term release (1 hr) | `Σ mem_v · 1[remaining_lifetime_v < 1hr]` | Memory freeing up imminently |
| Medium-term release (4 hr) | `Σ mem_v · 1[remaining_lifetime_v < 4hr]` | Medium-horizon headroom |
| Weighted sticky load | `Σ mem_v · remaining_lifetime_v` | How "locked" is this MPD |
| VM count | `|VMs on j|` | Fragmentation signal |

All normalized by `D_pod`. Requires iterating over VMs currently on each MPD — O(N_VMs) per decision, acceptable at VM arrival frequency.

---

### 6.3 Full v2 State Space

| # | Category | Signal | Temporal Treatment | Encoding |
|---|---|---|---|---|
| 1 | Resource Health | Per-MPD capacity util | 3 windows, 1-hr delta | `[v_t, Δ1h, Δ2h] × N_MPD` |
| 2 | Resource Health | Per-MPD bandwidth util | Current snapshot | `[v_t] × N_MPD` |
| 3 | Resource Health | Server memory pressure | 3 windows, 1-hr delta | `[v_t, Δ1h, Δ2h] × N_srv` |
| 4 | Resource Health | Recent SLO violations | 24-hr summary | Scalar (% allocs overflowing) |
| 5 | Workload | Incoming VM size | Current | Scalar (GB) |
| 6 | Workload | VM lifetime | Known at arrival | Scalar (hours) |
| 7 | Workload | Peak-to-average ratio | 24-hr summary | max / mean(usage_24h) |
| 8 | Workload | Predicted memory growth | Current | Scalar (GB/hr) |
| 9 | Topology | Server-to-MPD connectivity | Static | Binary mask ∈ {0,1}^N_MPD |
| **10** | **VM Pressure** | **Per-MPD short release (1hr)** | **Forward-looking** | **Scalar × N_MPD** |
| **11** | **VM Pressure** | **Per-MPD medium release (4hr)** | **Forward-looking** | **Scalar × N_MPD** |
| **12** | **VM Pressure** | **Per-MPD sticky load** | **Forward-looking** | **Scalar × N_MPD** |
| **13** | **VM Pressure** | **VM count per MPD** | **Current** | **Scalar × N_MPD** |

Features 10–13 (bold) are new. The delta encoding (`[v_t, Δ1h, Δ2h]`) is intentional — makes velocity explicit rather than forcing the network to learn it from raw windows, reducing learning burden.

---

## 7. Reward Function Design

### 7.1 The Credit Assignment Problem

The true objective is:
```
max_j PeakLoad_j  evaluated once, at end of full trace
```

A bad allocation at hour 2 might not manifest as a peak until hour 14. The agent has made thousands of decisions in between. How does it connect the eventual reward to the causing decision? This is **credit assignment**.

Current reward fails in two ways:
1. **Sparsity:** Only `max_j c_j` produces gradient — 191 of 192 MPDs get zero signal
2. **Myopia vs. true objective:** Minimizing current-step peak `≠` minimizing end-of-trace peak

---

### 7.2 Full Integrated Reward

```
R_t = α · (1 − max_j c_j(t) / cap_max)      ← Term 1: immediate peak penalty
    − β · Var(c(t))                            ← Term 2: immediate balance penalty
    − γ · slo_vio(t)                           ← Term 3: immediate SLO penalty
    + (λ^T / T) · (−γ · slo_vio_T             ← Option B: delayed SLO at VM termination
                   − δ · 1[migrated])          ← Option B: delayed migration penalty
```

---

### 7.3 Term 1 — Immediate Peak Penalty

```
T1 = −α · max_j c_j(t) / cap_max
```

Per-step proxy for the true objective. Penalizes the worst MPD every step, giving a dense signal pointing in the right direction. Still needed even with Term 2 — Term 2 penalizes imbalance but not absolute load level. An agent could balance all MPDs at a high absolute load and score well on Term 2 alone.

---

### 7.4 Term 2 — Immediate Balance Penalty

```
T2 = −β · Var(c(t))
   = −β · (1/m) Σ_j (c_j(t) − mean_c)²
```

Provides dense gradient signal for **every MPD** — not just the worst one. An agent that dumps load onto one MPD gets penalized even if that MPD isn't yet the global max.

**Reward hacking protection:** Terms 1 and 2 work against each other. Minimizing variance alone could be achieved by making all MPDs equally loaded at a high level — Term 1 penalizes the resulting high peak. Together they push toward uniformly low load.

---

### 7.5 Term 3 — SLO Violation Penalty

```
T3 = −γ · slo_vio(t)
```

**How to calculate `slo_vio(t)` concretely:**

At each VM arrival, the agent proposes allocation vector `a`. Before committing:

```python
slo_vio = 0.0
for j in accessible_mpds:
    new_load = current_load[j] + a[j]
    if new_load > mpd_capacity[j]:
        slo_vio += a[j] / total_vm_size   # fraction of VM causing overflow
```

This is **continuous** (proportional to overflow magnitude), not binary. A binary 0/1 signal would give zero gradient for allocations that are nearly-but-not-quite violating, making it harder to learn the capacity boundary. Continuous signal provides meaningful gradients across the entire approach to the limit.

**Weight priority:** `γ >> β > α` — SLO violations are hard constraints that override all efficiency concerns.

---

### 7.6 Option B — Delayed Reward at VM Termination

Immediate reward still has a credit assignment gap for long-horizon consequences. Option B adds a delayed signal when each VM terminates:

```
R_t^delayed = (λ^T / T) · (−γ · slo_vio_T − δ · 1[migrated])
```

Where:
- `T` = VM lifetime in hours (known at arrival)
- `λ` = discount factor (0.95)
- `slo_vio_T` = total SLO violations caused by this VM over its entire lifetime
- `1[migrated]` = 1 if this VM was ever migrated

The `λ^T / T` term discounts by lifetime and normalizes — longer-lived VMs get a smaller per-step delayed reward but their full lifetime contribution is correctly accounted for.

**Why this matters:** A large long-lived VM placed at midnight looks fine immediately (loads are low). The delayed signal at termination (days later) captures the full impact of that placement decision on peak load.

---

### 7.7 Weight Summary

```
Weight ordering: γ >> β > α > δ
Suggested starting values: α=0.1, β=0.3, γ=1.0, δ=0.05, λ=0.95
```

Business priority encoding:
- Avoid SLO violations (correctness)
- Balance load (efficiency)  
- Minimize peak (capacity cost)
- Avoid migration (operational complexity)

**Future work:** Multi-objective RL (MORL) to remove need for manual weight tuning — directly addressing a known limitation of FleetIO [8].

---

## 8. Curriculum Learning Design

### 8.1 Core Idea

Train on easy versions of the problem first, progressively increase difficulty. Agent builds load balancing intuitions on small pods, then transfers them to larger ones via weight initialization.

---

### 8.2 Difficulty Axes

**Pod size (primary axis):**

| Stage | Pod size | MPDs | Why easier |
|---|---|---|---|
| 1 | 4 hosts | 6 MPDs | Nearly enumerable allocation space |
| 2 | 8 hosts | ~12 MPDs | Small but non-trivial |
| 3 | 16 hosts | 20 MPDs | Current experiment size |
| 4 | 32 hosts | ~40 MPDs | Approaching complex |
| 5 | 96 hosts | 120 MPDs | Production target |

**Trace complexity:**

| Stage | Trace | Why easier |
|---|---|---|
| 1 | Synthetic flat | No temporal dynamics |
| 2 | Synthetic diurnal | Clean time-of-day pattern |
| 3 | Real single datacenter (AMS) | Real complexity, one geography |
| 4 | Multi-datacenter rotation | Different diurnal phases, hard generalization |

**CXL fraction:** 10% → 30% → 50%

---

### 8.3 Staged Training Schedule

```
Stage 1: pod_size=4, CXL=10%, synthetic flat trace
         → train until validation plateau (500K steps without >1% improvement)
         → save checkpoint C1

Stage 2: pod_size=8, CXL=30%, synthetic diurnal trace
         → initialize from C1 weights → train → save C2

Stage 3: pod_size=16, CXL=50%, AMS real trace
         → initialize from C2 weights → train → save C3

Stage 4: pod_size=96, CXL=50%, multi-datacenter rotation
         → initialize from C3 weights → train to convergence
         → production candidate
```

**Convergence criterion:** Two conditions must both hold before advancing:
1. Validation (LON trace) reward not improved by >1% over 500K steps
2. Agent beats greedy baseline on LON trace

The second condition prevents advancing when the agent has merely memorized the training trace without learning generalizable allocation strategy.

---

### 8.4 Automatic Curriculum Learning (Advanced)

Instead of manual stage transitions, use a performance threshold:
```python
if current_val_savings > greedy_savings + threshold:
    increase_pod_size()
    increase_cxl_fraction()
```

More flexible but harder to debug. Recommended approach: start with manual stages, move to automatic if manual schedule proves too rigid.

---

## 9. Algorithm Ablation Strategy

### 9.1 Why SAC, PPO, then LCPO

**SAC (current):**
- Off-policy, sample efficient, continuous actions
- Failing now due to reward/training issues (not algorithm)
- Useful baseline once those are fixed
- Known weakness: temporal correlation in replay buffer, hyperparameter sensitivity

**PPO (intermediate step):**
- On-policy like LCPO — directly comparable
- No replay buffer → no temporal correlation issues
- Much simpler to implement than LCPO
- Establishes whether on-policy learning can solve the base task at all
- If PPO fails, LCPO will fail for the same reason
- If PPO succeeds, LCPO adds exactly one thing: forgetting prevention across contexts

**LCPO (final):**
- On-policy stability + CF prevention
- Adds OOD detector, reservoir buffer, constrained optimization
- Only meaningful if PPO baseline already works

---

### 9.2 Training vs. Deployment — Critical Distinction

Training performance ≠ deployment performance. They are different problems:

| Problem | When it matters | What fails |
|---|---|---|
| Training | Offline, before deployment | Agent doesn't learn balanced allocation |
| Deployment | Live, continuous | Agent forgets morning patterns by evening |

Both must be evaluated separately.

---

### 9.3 Deployment Test: Rolling Simulation

```
Week 1: Train on days 1-7 → freeze → evaluate on days 8-14
Week 2: Continue online training on days 8-14 → evaluate days 15-21
Week 3: Continue online training on days 15-21 → evaluate days 22-28
...
```

**Key metrics:**
- Pooling savings per week (does performance hold over time?)
- Morning-hour performance after evening training (direct CF measurement)
- Recovery speed after context shift (new datacenter added)

**Hypothesis:** SAC and PPO show degrading performance on earlier time-of-day patterns as deployment continues. LCPO maintains performance across all diurnal phases due to explicit anchoring.

---

### 9.4 Full Ablation Matrix

Run all conditions with 10 random seeds. Report mean ± 95% CI.

| Condition | SAC | PPO | LCPO |
|---|---|---|---|
| Offline training only | ✓ | ✓ | ✓ |
| Online deployment, no retraining | ✓ | ✓ | ✓ |
| Online deployment, continuous training | ✓ | ✓ | ✓ |
| Link failure robustness (1–10%) | ✓ | ✓ | ✓ |
| New datacenter generalization | ✓ | ✓ | ✓ |
| Curriculum vs. direct training | ✓ | ✓ | ✓ |

---

## 10. LCPO Online Deployment Design

### 10.1 Context Definition

```python
z_t = [
    sin(2π · hour / 24),    # time-of-day circular
    cos(2π · hour / 24),    # time-of-day circular
    cluster_load_level,      # normalized mean(c_j) / cap_max
    day_of_week / 7,         # weekly pattern
]
```

Time-of-day is circular-encoded (sin/cos) so that 11pm and midnight are adjacent in context space. Already partially in the state space — this makes it an explicit context feature for OOD detection.

---

### 10.2 OOD Detection

```python
def is_ood(z_old, z_current, sigma=1.5):
    return ||z_old - z_current||_2 > sigma
```

**Concrete meaning:** An 8am experience (low load, morning phase) is OOD relative to an 8pm batch (high load, evening phase). When the agent trains on evening data, LCPO constrains it to not change its policy outputs on those 8am anchor samples.

This prevents the agent from overwriting its morning allocation strategy while adapting to evening patterns.

**Threshold tuning:** Start `σ = 1.5` (roughly 6-hour context separation). Tune via ablation study.

---

### 10.3 Cold Start — Pre-Population of `B_a`

`B_a` starts empty. The first few days of deployment would have no anchors without pre-population.

**Fix:** Before deployment, reservoir-sample offline training experiences into `B_a` (~200K samples), ensuring diverse time-of-day coverage across all diurnal phases (morning, afternoon, evening, overnight, weekday, weekend).

This ensures Day 1 of production already has anchor samples for every context regime.

---

### 10.4 Online Update Loop

```
VM arrival
  → Observe (s_t, z_t)
  → Forward pass → allocation action a_t
  → Execute allocation
  → Observe reward r_t
  → Add (s_t, z_t, a_t, r_t) to B_r (recent batch)

Every N=200 VM arrivals:
  → OOD sample: S_c ← W(B_a, B_r)
  → If S_c non-empty:
      → Conjugate gradient + line search → constrained update
  → Else:
      → Standard unconstrained update
  → Reservoir sample B_r into B_a
  → Repeat indefinitely
```

Update frequency N: one LCPO update per ~200 VM arrivals. At typical cloud VM arrival rates, this is every few minutes — fast enough to adapt to load shifts without destabilizing from excessive updates.

---

### 10.5 Safety: Policy Drift Monitor

Online learning carries the risk of policy drift — systematic degradation before anyone notices.

**Monitor design:**
- After each LCPO update, evaluate policy on fixed reference batch of 1000 samples
- Compute KL divergence from last validated checkpoint
- If `D_KL > c_monitor` → roll back to checkpoint and freeze online updates
- Alert operations team pending investigation

This provides a safety net against online learning pathologies without requiring manual intervention for small, normal drifts.

---

### 10.6 Retraining Cadence

Online LCPO updates continuously adapt within deployment. Periodic **full retraining** should also occur:
- Recommended cadence: monthly
- Retrain from scratch on most recent month of production traces
- Use curriculum (start small pod size, scale up)
- Pre-populate new `B_a` from retraining data before deploying new checkpoint

This prevents slow drift over months that the per-update drift monitor might not catch individually.

---

## 11. Open Questions

These are unresolved issues requiring future investigation:

| Question | Priority | Notes |
|---|---|---|
| Should `B_a` be stratified by time-of-day rather than uniform reservoir? | High | Uniform may undersample rare overnight context; stratification ensures all diurnal phases covered |
| What is the right online update frequency N? | High | Too frequent → instability; too infrequent → slow adaptation to real-world drift |
| Does VM lifetime predictor need independent validation before use as state feature? | High | If inaccurate, could actively mislead policy on MPD stickiness |
| How to handle MPD addition/removal without full retraining? | Medium | GNN naturally handles topology changes; MLP requires retraining |
| Is migration-aware RL worth pursuing? | Medium | Bounded migration (e.g. 5% of VMs/hr) could significantly close greedy-to-optimal gap |
| Can reward weights be automated via MORL? | Medium | Removes need for manual α, β, γ, δ tuning — directly addresses FleetIO limitation |
| Should LCPO use a separate learned context encoder? | Low | Rather than raw time-of-day features, learn `z_t` representation from data |
| How to handle link failures gracefully during online LCPO? | Low | Pre-failure experiences should remain anchors; topology mask must update dynamically |

---

*Document version: v2 (full session). Covers: LCPO paper deep-dive, RL fundamentals, CXL problem formulation, full roadmap design including state space, reward function, curriculum learning, ablation strategy, and deployment design. March 2026.*