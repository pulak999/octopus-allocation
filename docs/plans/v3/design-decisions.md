# Data Augmentation: Design Decisions

Pre-implementation decisions document. Work through each section, mark decisions,
then update `plan-v1.md` accordingly.

---

## 1. Augmentation Methods (from `data_aug.tex` Section 12)

Six knobs were identified. Analysis and recommendation for each:

### Knob 1 — Trace ID selection (uniform or weighted)

Multi-trace sampling: which of the 10 traces feeds each episode.

- Currently training uses 1 trace; enabling multi-trace samples from all 10
- Simple, no downside — just a data sampling strategy
- Uniform sampling unless trace characterization suggests weighting

**Recommendation: Keep (P0).** Free diversity, no risk.

---

### Knob 2 — Pod seed (node subset sampling)

Random pod generation: which hosts are selected for the episode.

- **Already implemented** in `reset()` — pod seed is randomized every episode
- Gives different subsets of hosts, different load profiles
- No new work needed

**Recommendation: Already done (P0).** No changes.

---

### Knob 3 — Memory scale `s ∈ [0.7, 1.2]`

Uniform factor applied to all VM `rss[1]` in an episode.

- Triangular distribution centered at 1.0, support [0.7, 1.2]
- HOTFIX survival analysis proves the safe range (Fig 5):
  - s=0.7 → 83.6% survival
  - s=1.0 → 81.0% survival
  - s=1.2 → 76.7% survival
  - s=1.5 → 64.9% survival (too aggressive)
- Directly changes the load regime without breaking SKU structure
- Most closely maps to real-world variation (different clusters have different memory pressure)

**Recommendation: Keep (P0) — highest-value knob.** Always on during augmented training.

---

### Knob 4 — Arrival jitter (±1–2 ticks)

Small bounded tick shifts to decorrelate exact co-arrivals.

- Each tick ≈ 5 minutes
- Shifting by ±1 tick preserves chronological order
- Low risk since it's bounded
- **Benefit might be marginal**: episodes already have high event counts (500–1600 events),
  so individual arrival times matter less than aggregate load
- The load trajectory shape stays roughly the same; only exact co-arrival patterns change

**Recommendation: Keep but start disabled (P1).** Test as an ablation, not a default.
Unclear if it adds meaningful signal beyond what memory scaling already provides.

---

### Knob 5 — Lifetime perturbation (±5%)

Fractional noise on VM lifetimes (positive-lifetime VMs only).

- Shorter lifetimes → lower peak → easier episodes
- Longer lifetimes → higher peak → more HOTFIX dropout
- Interacts with memory scale in hard-to-predict ways
- Real VM lifetimes are noisy, so this is plausible
- But: at low noise (≤5%), the effect on episode difficulty may be negligible
- At high noise, it distorts the departure pattern and changes episode difficulty
  in ways that are hard to control

**Recommendation: Borderline (P2).** Include in the code, but default to 0.0.
Only enable if ablation shows it helps. If it just adds noise without changing
the learning signal, drop it.

---

### Knob 6 — Topology perturbation (1–5% link failures)

Random CXL link removal to simulate degraded connectivity.

- **Unique and valuable**: the only knob that changes action space semantics
  (which MPDs are reachable from which hosts)
- Forces the policy to learn robust allocation under degraded connectivity
- This is the entire point of CXL pooling — graceful degradation
- Safety: never disconnect a host from all its MPDs
- At 5% link failures, risk creating topologies where the agent is forced into
  bad allocations regardless of policy quality
- Start conservative (1–2%), increase in ablations

**Recommendation: Keep (P1).** Default 0.02 (2%). Valuable for robustness,
but needs careful testing — it changes the problem structure, not just the data.

---

### Knob 7 — Per-VM memory noise (NEW, proposed)

Add per-VM noise on top of the episode-level scale factor:
```
rss[1]' = s * rss[1] * (1 + ε),    ε ~ U(-σ, σ)
```
where `s` is the episode-level scale (Knob 3) and `σ` is the noise magnitude
(e.g. 0.05 = ±5%).

**What it does:** Episode-level scaling preserves exact ratios between VMs in an
episode (a 16 GB VM stays exactly 2× an 8 GB VM). Per-VM noise breaks this,
creating slight size variation within an episode. This simulates measurement noise,
real-world memory usage fluctuations, or slight differences between nominally
identical SKUs.

**Pros:**
- More realistic: real VMs don't use exactly their SKU allocation. Actual RSS
  fluctuates around the nominal size.
- Increases diversity in a different dimension than episode-level scaling. Scale
  changes the overall load level; noise changes the relative sizes within an episode.
- Cheap to compute (one multiply per event).

**Cons:**
- **Breaks SKU discreteness.** `rss[1]` values are powers of two (1024, 2048, 4096, ...).
  Adding continuous noise creates sizes like 4173 MB that don't exist in production.
  `data_aug.tex` Section 7 explicitly warns against this. However: the env only uses
  `rss[1]` as a continuous float for memory accounting, not as a discrete category.
  The agent never sees the raw `rss[1]` value — it sees normalized load fractions.
  So SKU-breaking may not matter in practice.
- **Overlap risk with different scale factors.** This is the key concern:

  Consider two augmented episodes:
  - Episode A: scale=0.9, noise=±10% → VM memory in range [0.81, 0.99] × base
  - Episode B: scale=1.0, no noise → VM memory = 1.0 × base

  These distributions **overlap**. A noisy episode at s=0.9 can look nearly identical
  to an unscaled episode. If the test set uses s=1.0 (no augmentation), training data
  with s=0.9+noise is "almost leaking" into the test distribution.

  **Mitigation:** Keep noise small (σ ≤ 0.03, i.e. ±3%). At this level:
  - s=0.9 with ±3% noise → range [0.873, 0.927] — still clearly below s=1.0
  - s=1.1 with ±3% noise → range [1.067, 1.133] — still clearly above s=1.0
  - Overlap with adjacent scale values is minimal

  **Rule of thumb:** noise σ should be much smaller than the scale range width.
  Scale range is [0.7, 1.2] (width 0.5). Noise σ = 0.03 is 6% of that width — safe.
  Noise σ = 0.10 would be 20% of the width — too much overlap.

- **Interacts with HOTFIX.** If noise pushes some VMs over per-node DRAM capacity,
  they get filtered out. But with σ ≤ 0.03, this effect is negligible (the episode-
  level scale factor dominates).

**Implementation:**
```python
def add_memory_noise(events, sigma, rng):
    """Add per-VM multiplicative noise to memory values.

    events: list of (tick, host, mem, dealloc)
    sigma: noise magnitude (0.03 = ±3%)
    """
    if sigma <= 0.0:
        return events
    noisy = []
    for tick, host, mem, dealloc in events:
        noise = 1.0 + rng.uniform(-sigma, sigma)
        noisy.append((tick, host, mem * max(noise, 0.01), dealloc))
    return noisy
```

Applied after `scale_memory()` and before HOTFIX in the augmentation pipeline.

**Recommendation: P1 (ablation).** Default σ=0.0 (disabled). Test at σ=0.02 and
σ=0.03. If it improves generalization beyond what episode-level scaling alone
provides, keep it. If not, drop it — the added complexity isn't worth marginal gains.

**The critical rule:** σ must stay well below the scale range width. If scale range
is [0.7, 1.2], keep σ ≤ 0.03.

---

### Summary: Recommended Phasing

| Priority | Knob | Default | Rationale |
|----------|------|---------|-----------|
| P0 (always on) | Memory scale [0.7, 1.2] | triangular | Highest value, well-understood |
| P0 (always on) | Multi-trace sampling | uniform | Free diversity |
| P0 (always on) | Pod seed randomization | (already done) | Already works |
| P1 (ablation) | Link failures | 0.02 | Tests robustness, unique value |
| P1 (ablation) | Per-VM memory noise | σ=0.0 | Complementary to scaling, keep σ ≤ 0.03 |
| P1 (ablation) | Arrival jitter | 1 tick | Low risk, unclear benefit |
| P2 (optional) | Lifetime perturbation | 0.0 | Marginal, interacts with scale |

**Decision needed:** Agree with this prioritization? Change ordering?

---

## 2. Train/Eval Data Split

### The Problem

10 production Azure VM traces. Need to decide what goes into training (augmented),
validation (not augmented, used for hyperparameter tuning), and test (not augmented,
used for final reporting only).

### Best Practices from Literature

#### Principle 1: Never augment eval/test data

Eval must measure performance on the **real Azure workload distribution**, not a
synthetic one. If you augment eval traces, you're measuring "how well does this
policy work on data that doesn't exist in production."

**Citations:**
- Laskin et al. (2020), "Reinforcement Learning with Augmented Data" (RAD), *NeurIPS* —
  Introduced systematic data augmentation for RL. Augmentations applied **only during
  training** (replay buffer / observation pipeline). Evaluation uses canonical,
  unaugmented environment. Now the standard protocol.
- Kostrikov et al. (2020), "Image Augmentation Is All You Need" (DrQ), *NeurIPS 2021* —
  Augmentation only on training transitions. Evaluation on clean observations. They
  explicitly note augmenting eval would "conflate the effect of augmentation with
  policy quality."
- Tobin et al. (2017), "Domain Randomization for Sim-to-Real Transfer," *IROS* —
  Foundational domain randomization paper. Evaluation is always on the real
  (unaugmented) domain. The entire point is that training diversity produces a
  policy robust enough to work on the unaugmented target.

#### Principle 2: Never split within a trace

Each trace is a temporal sequence. Splitting a trace between train and eval allows
temporal autocorrelation to leak across the boundary — the agent memorizes transition
patterns from training portions that predict eval portions.

**Citations:**
- Cerqueira et al. (2020), "Evaluating Time Series Forecasting Models: An Empirical
  Study on Performance Estimation Methods," *Machine Learning* — For time-series data,
  split at the **sequence level**, not within sequences. Random shuffling of timesteps
  destroys temporal structure and causes massive leakage.
- Mao et al. (2019), "Learning Scheduling Algorithms for Data Processing Clusters"
  (Decima), *SIGCOMM* — Workload-level split: entire workload traces held out,
  not individual jobs within traces. Directly analogous to your scenario.

#### Principle 3: Separate validation from test

If you tune augmentation parameters (scale range, jitter, etc.) to maximize val
performance, you're implicitly fitting to the val set. Need a separate test set
that is never used for any decision.

**Citations:**
- Cawley & Talbot (2010), "On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation," *JMLR* — Tuning on the validation set
  and reporting validation performance is a form of leakage. A separate test set
  must never be used for any decision.

#### Principle 4: Domain randomization works with limited sources

With as few as 5–10 source environments, domain randomization can produce policies
that generalize, **provided the randomization covers the axes of variation that
matter** (Peng et al., 2018).

**Citations:**
- Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization," *ICRA* — 5–10 source environments suffice with good randomization.
- Mehta et al. (2020), "Active Domain Randomization," *CoRL* — Not all randomizations
  are equally useful. Actively search for challenging parameters (future work).
- Cortez et al. (2017), "Resource Central: Understanding and Predicting Workloads
  for Improved Resource Management in Large Cloud Platforms," *SOSP* — Azure trace
  analysis showing VM lifetimes span 5 orders of magnitude, resource demands vary
  by 100x.

### Val vs Test: What's the Difference?

Both val and test use **unaugmented, real traces**. The difference is **when you
look at them and what decisions they inform**.

**Validation set (1 trace):**
- You look at val performance **during development** to make decisions
- "Should I use scale range [0.7, 1.2] or [0.8, 1.1]?" → compare val reward
- "Is SAC lr=3e-4 better than lr=1e-3?" → compare val reward
- "Should I add arrival jitter?" → compare val reward with and without
- "Is this model converging? Should I stop early?" → check val reward curve
- Val is your **steering wheel** — it tells you which direction to go
- **Problem:** every decision you make based on val performance slightly overfits
  to that trace. After 50 hyperparameter experiments, your best config is partly
  tailored to the val trace, not just to "good CXL allocation in general"

**Test set (2 traces):**
- You look at test performance **exactly once, at the very end**
- After ALL decisions are finalized (augmentation knobs, hyperparameters, model
  architecture, reward function), you run the final model on test traces
- The test number goes in the paper/report as "generalization performance"
- If test performance is much worse than val, you overfit to the val trace
- **Never go back and change decisions based on test results.** If you do, the
  test set becomes a second val set and you need a third set

**Analogy:** Val is the practice exam you can retake. Test is the final exam you
take once. If you memorize the practice exam answers (overfit to val), you'll do
well on practice but poorly on the final.

**Why 2 test traces instead of 1?** With 1 test trace, your "final exam" is a
single question — high variance. If that trace happens to be easy or hard, your
reported performance is misleading. 2 traces gives a more stable estimate.

### Recommended Split: 7 / 1 / 2

| Set | # Traces | Augmented? | Purpose |
|-----|----------|-----------|---------|
| **Train** | 7 | Yes | Policy learning; augmented every episode |
| **Val** | 1 | **No** | Hyperparameter tuning, early stopping, ablation comparisons |
| **Test** | 2 | **No** | Final reporting only; never tuned on, never augmented |

**Why 2 for test?** With only 1 test trace, results are high-variance. 2 traces
gives a more stable estimate and can cover different workload archetypes.

**Stratification:** Characterize each trace by workload type (bursty vs steady,
memory-heavy vs light, short-lived VMs vs long-lived). Ensure the 2 test traces
cover different archetypes than the majority of training traces. This gives a
harder, more honest test of generalization.

**LOOCV alternative:** For more robust val estimates, do leave-one-out on the
8 non-test traces (train on 7, val on 1, rotate 8 times). But this means 8
training runs per hyperparameter setting — expensive. Consider for final results
only.

### Data Leakage Prevention Checklist

1. **Trace-level splitting only** — never split within a trace
2. **No augmentation of eval/test traces** — not even with different parameters
3. **Separate test from val** — test traces used only for final reporting
4. **Fix augmentation parameters before test evaluation** — no retroactive tuning
5. **Log random seeds** — augmentation randomness must be reproducible
6. **Augmentation of train traces only** — even if the same trace appears in both
   train and val across LOOCV folds, only augment when it's in the train role

**Decision needed:** Agree with 7/1/2? Which 2 traces should be held out for test?
(Requires trace characterization — we should profile all 10 traces by workload type.)

---

## 3. Plan Scope: v1 = Testing, v2 = Ablations

### Current problem with plan-v1.md

The current plan-v1 includes Task 9 (training runs + evaluation). This mixes
implementation/testing with experimentation. The user wants:

- **plan-v1**: Build the augmentation system, validate correctness and performance
- **plan-v2**: Ablation studies, hyperparameter tuning, model comparison, training

### Revised plan-v1 scope (testing only)

| Task | Description | Status |
|------|-------------|--------|
| 1 | `AugmentationConfig` dataclass | To implement |
| 2 | Transform functions (scale, jitter, lifetime, link failures) | To implement |
| 3 | Integrate into `env.py` (init, reset, active_M) | To implement |
| 4 | Verify precompute compatibility (no changes needed) | To verify |
| 5 | Multi-trace support | To implement |
| 6 | CLI args in `train_rl.py` | To implement |
| 7 | Augmentation logging callback | To implement |
| 8 | Unit tests + integration tests | To implement |
| **NEW 9** | Performance benchmarks: `reset()` time with/without aug, verify <10ms | To implement |
| **NEW 10** | Correctness validation: 1000 augmented resets, verify guards, distribution shape | To implement |

**Removed:** Old Task 9 (training runs) → moves to plan-v2.

### Plan-v2 scope (ablations + training)

| Topic | Description |
|-------|-------------|
| Augmentation ablation grid | Each knob on/off, measure generalization |
| Hyperparameter tuning | SAC lr, buffer_size, batch_size, tau, target_entropy |
| Model comparison | MLP vs GNN policy |
| State/observation space selection | Which obs features matter |
| Reward function variants | Lambda sweep, alternative reward formulations |
| W&B logging + dashboards | Full monitoring setup |
| Training runs | Baseline, scale-only, full-aug, multi-trace |
| Evaluation pipeline | Val + test reporting |

**Decision needed:** Agree with this scope split? Anything to add/remove?

---

## 4. RL Training Monitoring (W&B)

This section documents best practices for monitoring SAC training. Will be
implemented in plan-v2 alongside training runs.

### W&B Dashboard Layout

#### Group 1 — Performance (primary)

| Metric | Source | What to watch |
|--------|--------|---------------|
| `eval/mean_reward` | `EvalCallback` (deterministic policy) | **Primary metric.** Must use unaugmented val trace |
| `rollout/ep_rew_mean` | SB3 built-in | Training reward (noisier due to exploration) |
| `rollout/ep_len_mean` | SB3 built-in | Changes reflect behavioral learning |

#### Group 2 — SAC Internals

| Metric | What to watch |
|--------|---------------|
| `train/critic_loss` | Should decrease and stabilize. Persistently increasing = failing |
| `train/actor_loss` | Should not diverge |
| `train/ent_coef` (alpha) | **Critical.** ≈0 = policy collapse. Growing unboundedly = maximally random |
| `train/ent_coef_loss` | Should stabilize near 0 |

**Alpha (entropy coefficient) is the most important SAC diagnostic.** It controls
the exploration-exploitation tradeoff. Healthy training: starts high, gradually
decreases, stabilizes at a non-zero value.

#### Group 3 — Q-Value Health

| Metric | What to watch |
|--------|---------------|
| `q1_mean`, `q2_mean` | Should grow proportionally to actual returns, not diverge |
| `q1_std` | Growing std = instability |
| Q1–Q2 spread | Should be small (twin critics should agree) |

Q-value divergence (overestimation) is a well-documented failure mode in off-policy RL.
SAC's twin critics mitigate it but don't eliminate it.

**Citation:** Fujimoto et al. (2018), "Addressing Function Approximation Error in
Actor-Critic Methods," *ICML*. Kumar et al. (2022), "DR3: Value-Based Deep RL
Requires Explicit Regularization," *ICLR*.

#### Group 4 — Augmentation Diversity

| Metric | What to watch |
|--------|---------------|
| `aug/scale_mean`, `aug/scale_std` | Verify augmentation is actually diverse |
| `aug/trace_id` histogram | Verify multi-trace coverage (all traces seen) |
| `aug/link_failures_applied` | Count of episodes with topology perturbation |

#### Group 5 — Environment-Specific (CXL)

| Metric | What to watch |
|--------|---------------|
| Peak memory utilization per episode | Direct outcome metric |
| Allocation success rate | Are VMs being placed? |
| Variance of MPD loads | Directly in your reward function |
| HOTFIX survival rate per episode | Augmentation quality metric |

### Failure Modes and Red Flags

| Signal | Meaning | Fix |
|--------|---------|-----|
| `ent_coef < 0.001` | Policy collapse — agent found one action and refuses to explore | Increase `target_entropy`, restart from checkpoint |
| Q-values > 10× max possible return | Q-value divergence / overestimation | Reduce lr, increase `tau` (target network update rate) |
| Critic loss increasing over time | Bellman backup diverging | Reduce lr, add regularization |
| Eval reward << training reward | Exploration inflating training reward | Check `deterministic=True` in eval |
| NaN in any loss | Numerical instability | Gradient clipping (`max_grad_norm=0.5`), check reward scale |
| Reward increases but env metrics degrade | Reward hacking | Check reward function, add env-specific monitoring |
| Performance drops after initial improvement | Catastrophic forgetting | Larger replay buffer, prioritized replay, checkpointing |

**Citations:**
- Haarnoya et al. (2018/2019), "Soft Actor-Critic" — entropy tuning mechanics
- Henderson et al. (2018), "Deep RL that Matters," *AAAI* — evaluation methodology
- Skalse et al. (2022), "Defining and Characterizing Reward Hacking," *NeurIPS*
- Lyle et al. (2023), "Understanding Plasticity in Neural Networks," *ICML* — catastrophic forgetting

### W&B + SB3 Integration Pattern

```python
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# 1. Init wandb with full config
run = wandb.init(
    project="cxl-memory-pooling",
    config={
        "algorithm": "SAC",
        "n_envs": 32,
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "target_entropy": "auto",
        "augmentation": True,
        "aug_scale_range": [0.7, 1.2],
        # ... all hyperparameters
    },
    sync_tensorboard=True,  # pulls SB3's built-in TB logs into wandb
)

# 2. Eval callback on UNAUGMENTED val trace (deterministic policy)
eval_callback = EvalCallback(
    eval_env,                          # separate env, no augmentation
    eval_freq=10_000 // n_envs,        # every 10k total steps
    n_eval_episodes=20,                # average over 20 episodes
    deterministic=True,                # mean action, no exploration
    best_model_save_path="./best/",
)

# 3. W&B callback for gradients + checkpoints
wandb_callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_path=f"models/{run.id}",
    model_save_freq=50_000,
)

# 4. Custom SAC diagnostics callback (Q-values, entropy details)
# 5. Augmentation logging callback (scale distribution, trace coverage)

model.learn(
    total_timesteps=2_000_000,
    callback=CallbackList([
        wandb_callback,
        eval_callback,
        sac_diagnostics_callback,
        augmentation_log_callback,
    ]),
)
```

### Pre-Training Sanity Checks

Before any real training run:

1. **Random policy baseline**: Run 10k steps with random actions. Record mean reward.
   The trained agent should beat this within the first 5–10% of training.
2. **Single-env overfit test**: Train with `n_envs=1` for a short run. If SAC cannot
   learn with 1 env, it won't learn with 32.
3. **Reward scale check**: SAC works best with rewards in [-10, 10]. Very large rewards
   (>1000) destabilize critic learning. Consider normalization if needed.
4. **Verify action space is continuous**: SAC requires continuous actions. Your env uses
   softmax over MPDs — confirm this maps to Box space correctly.

### SAC-Specific Hyperparameters to Tune (plan-v2)

| Parameter | Default | Range to sweep | Why |
|-----------|---------|---------------|-----|
| `learning_rate` | 3e-4 | [1e-4, 1e-3] | Most impactful single parameter |
| `buffer_size` | 1M | [500k, 2M] | With 32 envs, buffer fills fast |
| `batch_size` | 256 | [128, 512] | Larger = more stable gradients |
| `tau` | 0.005 | [0.001, 0.02] | Target network update speed |
| `target_entropy` | "auto" | ["auto", -dim/2, -dim] | Controls exploration |
| `gamma` | 0.99 | [0.95, 0.999] | Discount factor — how far ahead to plan |
| `learning_starts` | 100 | [10k, 50k] | Fill buffer before learning |
| `train_freq` | 1 | [1, 4] | Gradient steps per env step |
| `gradient_steps` | 1 | [1, 32] | Gradient steps per train call |

**Key references:**
- SB3 docs: `stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html`
- CleanRL SAC implementation: Huang et al. (2022), "CleanRL," *JMLR*
- W&B SB3 integration: `docs.wandb.ai/guides/integrations/stable-baselines-3`

---

## Decisions Summary

Mark each with your decision:

- [ ] **Augmentation knobs**: Agree with P0/P1/P2 prioritization?
- [ ] **Data split**: 7/1/2 (train/val/test)? Which 2 traces for test?
- [ ] **Plan scope**: Rewrite plan-v1 as testing-only, create plan-v2 skeleton for ablations?
- [ ] **W&B logging**: Goes into plan-v2 with training runs?
- [ ] **Trace characterization**: Run profiling to decide which traces go into test set?
