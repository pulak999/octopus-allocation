# Training, Ablations & Evaluation Plan

Agent instructions for training the augmented Octopus CXL RL agent, running ablation
studies, and evaluating generalization. All paths relative to
`/home/pm3371/gitrepos/octopus-allocation/`. Run with venv active.

**Prerequisite:** plan-v1 must be complete (augmentation system built and tested).

**Stack:** SB3 (SAC) + SubprocVecEnv (N=32) + W&B + existing `OctopusMemPoolEnv`.
**Hardware:** 3x TITAN RTX (24 GB each), kernel 5.15, CUDA 12.5.
**Reference:** `docs/plans/v3/design-decisions.md` (design rationale + literature).

---

## Data Split

| Set | # Traces | Augmented? | Purpose |
|-----|----------|-----------|---------|
| **Train** | 7 | Yes | Policy learning |
| **Val** | 1 | **No** | Hyperparameter tuning, early stopping, ablation comparison |
| **Test** | 2 | **No** | Final reporting only — touch once at the very end |

Trace assignments TBD after trace characterization (Task 1).

---

## Task 1 — Trace Characterization & Split Assignment

**Goal:** Profile all 10 traces to make an informed train/val/test split.

### 1a. Disable HOTFIX for training/eval

The HOTFIX filter removes VMs that would overflow per-node physical DRAM
(`machine_sz`). This constraint does not apply in the CXL pooling model —
MPD capacity is provisioned to handle expected load, and the whole point
of pooling is to exceed local DRAM limits. Removing HOTFIX:
- Includes all VMs → denser, harder episodes
- Eliminates distribution shift between augmented training (where HOTFIX
  drops more VMs at higher scale factors) and unaugmented eval
- Better reflects the production deployment scenario

**Action:** Add a `--skip-hotfix` flag to `train_rl.py` and `eval_rl.py`.
Keep the HOTFIX code in `_build_events()` behind the flag (do not delete).
All subsequent tasks in this plan run with `--skip-hotfix`.

### 1b. Trace characterization

For each trace, compute:
- Total VM count, unique VM type count
- Mean/std of `rss[1]` (memory demand profile)
- Mean/std of VM lifetime
- Arrival rate (VMs per tick): bursty vs steady
- Peak aggregate memory demand
- Episode event counts **with and without HOTFIX** (to quantify the
  difference and confirm HOTFIX removal doesn't produce degenerate episodes)

### 1c. Split assignment

Produce a summary table. Then assign:
- 2 test traces: pick different workload archetypes (e.g. one bursty + one steady,
  or one memory-heavy + one memory-light) to get a harder generalization test
- 1 val trace: representative of average workload
- 7 train traces: everything else

**Sealing the test set:** Write the 2 test trace names to
`data/splits/test_traces.sealed.json`. This file must NOT be read by the
executing agent until Task 10. Train and val trace names go in
`data/splits/train_val_traces.json` (readable immediately). The sealed file
keeps the agent honest — no peeking at test traces during Tasks 2–9.

**Output:** trace characterization table + split files.

---

## Task 2 — W&B Integration

**Goal:** Set up W&B logging for all training runs.

### 2a. Basic integration

```python
import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="cxl-memory-pooling",
    config={ ... },   # all hyperparams + aug config
    sync_tensorboard=True,
)
```

### 2b. Eval callback (unaugmented val trace)

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,                      # separate env, NO augmentation, val trace only
    eval_freq=10_000 // n_envs,
    n_eval_episodes=20,
    deterministic=True,
    best_model_save_path="./best/",
)
```

### 2c. SAC diagnostics callback

Log every N steps:
- `sac/ent_coef` — entropy coefficient (alpha)
- `sac/q1_mean`, `sac/q2_mean` — Q-value magnitudes
- `sac/q1_std` — Q-value spread
- `sac/q_spread` — Q1-Q2 disagreement
- Gradient norms (actor + critic)

### 2d. Augmentation logging callback

Log augmentation parameter diversity:
- `aug/scale_mean`, `aug/scale_std`, `aug/scale_min`, `aug/scale_max`
- `aug/trace_id` histogram (if multi-trace)
- `aug/link_failures_applied` count

### 2e. Environment-specific metrics

Log from `info` dict:
- Peak memory utilization per episode
- Variance of MPD loads (directly in reward)
- Number of events in episode
- HOTFIX survival count

### 2f. W&B dashboard template

Create saved dashboard with 5 panel groups:
1. **Performance:** eval/mean_reward, rollout/ep_rew_mean, rollout/ep_len_mean
2. **SAC Internals:** critic_loss, actor_loss, ent_coef, ent_coef_loss
3. **Q-Value Health:** q1_mean, q2_mean, q1_std, q_spread
4. **Augmentation:** scale distribution, trace coverage
5. **Environment:** peak utilization, MPD load variance

### 2g. Alerts

Set W&B alerts for:
- `ent_coef < 0.001` (policy collapse)
- Q-values > 10× max plausible return (divergence)
- NaN in any loss
- FPS drop > 50% (worker crash)

---

## Task 3 — Pre-Training Sanity Checks

Run these before any real training. All must pass.

### 3a. Random policy baseline

```bash
python scripts/train_rl.py \
    --run-id sanity_random \
    --total-timesteps 10000 \
    --n-envs 1 \
    --no-train
```

Record mean reward over 10k steps with random actions. This is the floor
the trained agent must beat within the first 5–10% of training.

### 3b. Single-env overfit test

```bash
python scripts/train_rl.py \
    --run-id sanity_overfit \
    --total-timesteps 50000 \
    --n-envs 1 \
    --trace <val_trace>
```

If SAC cannot learn anything with 1 env on a single trace, something is
fundamentally wrong. Reward should clearly exceed random baseline.

### 3c. Reward scale check

Check that rewards are in [-10, 10] range. SAC is sensitive to reward
magnitude. If rewards are outside this range, add reward normalization
or rescale the reward function.

### 3d. Augmented env smoke test

```bash
python scripts/train_rl.py \
    --run-id sanity_aug \
    --total-timesteps 10000 \
    --n-envs 4 \
    --augmentation \
    --aug-scale-lo 0.5 --aug-scale-hi 1.5
```

Verify augmented episodes produce similar reward distribution to
non-augmented. Large discrepancy = augmentation is too aggressive.

---

## Task 4 — Baseline Training Run (No Augmentation)

```bash
python scripts/train_rl.py \
    --run-id v3_baseline \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --trace <train_traces> \
    --variance-lambda 0.5 \
    --wandb
```

This is the control. All augmented runs are compared against this.

Evaluate on val trace:
```bash
python scripts/eval_rl.py --run-id v3_baseline --trace <val_trace> --n-iter 50
```

### Stop gate

**Before proceeding to Task 5:** verify that the baseline agent's val reward
clearly exceeds the random baseline from Task 3a. If it does not, the problem
is upstream (reward shaping, observation space, or SAC hyperparameters) and
must be diagnosed before running ablations. Specifically:
- Val reward must be > random baseline mean + 1 std within the first 500k steps.
- If not met, revisit reward function (Task 9 variants) and observation space
  (Task 8) before continuing.

---

## Task 5 — Augmentation Ablation Grid

Test each augmentation knob independently, then in combination.

### 5a. Memory scaling only (P0)

```bash
python scripts/train_rl.py \
    --run-id v3_aug_scale \
    --augmentation \
    --aug-scale-lo 0.5 --aug-scale-hi 1.5 \
    --aug-scale-dist uniform \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5b. Multi-trace only (P0)

```bash
python scripts/train_rl.py \
    --run-id v3_aug_multitrace \
    --multi-trace \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5c. Link failures only (P1)

```bash
python scripts/train_rl.py \
    --run-id v3_aug_links \
    --augmentation \
    --aug-link-failures 0.02 \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5d. Per-VM memory noise only (P1)

```bash
python scripts/train_rl.py \
    --run-id v3_aug_noise \
    --augmentation \
    --aug-memory-noise 0.03 \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5e. Arrival jitter only (P1)

```bash
python scripts/train_rl.py \
    --run-id v3_aug_jitter \
    --augmentation \
    --aug-jitter 1 \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5f. All P0 + P1 combined

```bash
python scripts/train_rl.py \
    --run-id v3_aug_full \
    --augmentation \
    --aug-scale-lo 0.5 --aug-scale-hi 1.5 \
    --aug-scale-dist uniform \
    --aug-link-failures 0.02 \
    --aug-memory-noise 0.03 \
    --aug-jitter 1 \
    --multi-trace \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --variance-lambda 0.5 \
    --wandb
```

### 5g. Ablation evaluation

Evaluate all runs on val trace:
```bash
for run in v3_baseline v3_aug_scale v3_aug_multitrace v3_aug_links \
           v3_aug_noise v3_aug_jitter v3_aug_full; do
    python scripts/eval_rl.py --run-id $run --trace <val_trace> --n-iter 50
done
```

Produce comparison table: **pooling_ratio** (primary), savings vs greedy baseline,
peak utilization, MPD load variance. Use pooling_ratio as the selection metric —
it is reward-formulation-independent, so runs across different reward variants
(Task 9) remain comparable. Pick the best augmentation configuration by lowest
pooling_ratio on val.

---

## Task 6 — Hyperparameter Tuning

Use W&B Sweeps on the best augmentation config from Task 5.

### Parameters to sweep

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [1e-4, 1e-3] | log |
| `buffer_size` | [500k, 2M] | log |
| `batch_size` | [128, 512] | categorical: 128, 256, 512 |
| `tau` | [0.001, 0.02] | log |
| `target_entropy` | ["auto", -dim/2, -dim] | categorical |
| `gamma` | [0.95, 0.999] | uniform |
| `learning_starts` | [10k, 50k] | log |
| `gradient_steps` | [1, 32] | categorical: 1, 4, 16, 32 |
| `n_envs` | [1, 8, 16, 32] | categorical — trades memory/reset cost vs sample diversity; profile RSS + FPS at each level before sweeping |

### Sweep config

```yaml
method: bayes
metric:
  name: eval/pooling_ratio
  goal: minimize  # lower = less peak memory needed = better savings
parameters:
  learning_rate:
    min: 0.0001
    max: 0.001
    distribution: log_uniform_values
  buffer_size:
    values: [500000, 1000000, 2000000]
  batch_size:
    values: [128, 256, 512]
  tau:
    min: 0.001
    max: 0.02
    distribution: log_uniform_values
  gamma:
    min: 0.95
    max: 0.999
```

Run 20–30 sweep trials. Each trial: 500k timesteps (shorter for speed).
Top 3 configs get full 2M-step runs.

---

## Task 7 — Model Comparison: MLP vs GNN

### 7a. MLP baseline (current)

Already running from Tasks 4–5.

### 7b. GNN policy

Implement GNN policy network that takes topology as input.
- Node features: host loads, MPD capacities
- Edge features: link bandwidth/latency
- Output: allocation scores per MPD

This is a larger implementation effort. Design details TBD based on
plan-v2 results (if MLP + augmentation already generalizes well,
GNN may not be needed).

**Decision point:** Only proceed with GNN if MLP + augmentation shows
poor generalization on test set.

---

## Task 8 — State/Observation Space Selection

Ablate observation features to find the minimal effective set.

Current obs space: `2*max_degree + 4` features:
- Normalized MPD loads
- Accessibility mask
- VM memory request
- Global peak
- sin/cos hour-of-day

### Ablation experiments

| Variant | Features removed | Hypothesis |
|---------|-----------------|-----------|
| No time | Remove sin/cos hour | Time may not help if augmentation provides diversity |
| No global peak | Remove global peak | Agent may learn this from MPD loads |
| No mask in obs | Remove accessibility mask | Mask is applied in action anyway |
| Minimal | Only MPD loads + VM request | Test if other features are noise |

---

## Task 9 — Reward Function Variants

### 9a. Lambda sweep

```
reward = -(new_peak - old_peak) / fair_share - λ * Var(mpd_loads)
```

Sweep λ ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0}.

### 9b. Alternative reward formulations

| Variant | Formula | Motivation |
|---------|---------|-----------|
| Peak-only | `-(new_peak - old_peak) / fair_share` | Is variance term helping? |
| Post-alloc peak | `-max(loads) / fair_share` | Directly minimizes what greedy minimizes (post-allocation max load). Needs normalization by cumulative allocated memory since raw peak grows over the episode. |
| Scale-invariant balance | `-(new_peak - old_peak) / fair_share - λ * CV(loads)` | Replace `Var(loads)` with coefficient of variation `std/mean` to fix non-stationarity: raw variance grows over the episode as more VMs arrive, making late-episode penalties dominate regardless of agent quality. CV is scale-invariant. |
| Sparse | `reward only at episode end` | Cleaner signal, harder credit assignment |

**Known issue with current reward:** The peak-delta term `-(new_peak - old_peak)` is
zero for most steps (only nonzero when a new global peak is set), so the agent is
guided almost entirely by the variance penalty — which itself grows non-stationary
over the episode. The post-alloc peak and CV variants above address both problems.

---

## Task 10 — Final Evaluation on Test Set

**This task runs EXACTLY ONCE, after all decisions are finalized.**

**Now open `data/splits/test_traces.sealed.json`.**

### 10a. Models to evaluate

Evaluate every model variant on the 2 held-out test traces:

| Model | Source |
|-------|--------|
| Greedy baseline | `octopus/baselines.py::greedy_alloc` |
| RL baseline (no augmentation) | Task 4 best checkpoint |
| RL + best single augmentation | Task 5 winner |
| RL + full augmentation | Task 5f best checkpoint |
| RL + best HP config | Task 6 best checkpoint |
| RL + best obs space variant | Task 8 winner (if different from default) |
| RL + each reward variant | Task 9 — one model per reward formulation |

### 10b. Metrics

For each model × each test trace, record:
- `pooling_ratio` (primary metric)
- Peak memory utilization
- MPD load variance (CV)
- Savings vs greedy (% reduction in pooling_ratio)
- Mean reward ± std (for reference, not comparison across reward variants)

### 10c. Persistence

Save all results to `output/final_eval/` so nothing needs to be rerun:

```
output/final_eval/
├── results_summary.csv          # one row per (model, trace), all metrics
├── episode_details/
│   ├── <model>_<trace>.csv      # per-episode breakdown (n_iter rows)
│   └── ...
├── models.json                  # maps model name → checkpoint path + config
└── eval_config.json             # n_iter, traces, topology, date, git SHA
```

`models.json` records the exact checkpoint path and full hyperparameter config
for each model so results are reproducible without re-training.

### 10d. Reporting

Produce a single comparison table sorted by `pooling_ratio` (ascending = better).

**If test performance is significantly worse than val, you overfit to the
val trace. Do NOT go back and retune — report honestly.**

---

## Execution Order

```
1  (Trace characterization + split)
   ↓
2  (W&B integration)
   ↓
3  (Pre-training sanity checks)
   ↓
4  (Baseline training — no augmentation)
   ↓
5  (Augmentation ablation grid)
   ↓
6  (Hyperparameter tuning on best aug config)
   ↓
7  (GNN — only if MLP generalizes poorly)
   ↓
8  (Observation space ablation)
   ↓
9  (Reward function ablation)
   ↓
10 (Final test evaluation — run once)
```

Tasks 7, 8, 9 can run in parallel if GPU resources allow.
Task 10 is always last.

---

## GPU Scheduling

**Hardware:** 3× TITAN RTX (`cuda:0`, `cuda:1`, `cuda:2`).
SB3 SAC is single-GPU, so we run independent jobs across GPUs to maximize
utilization. Each job pins its GPU via `CUDA_VISIBLE_DEVICES`.

### Scheduling principle

Never leave a GPU idle. When a job finishes, immediately backfill with the
next available run. The schedule below packs runs tightly across all 3 GPUs.

### Phase 1: Sequential (all 3 GPUs idle → low utilization OK)

| Step | GPU 0 | GPU 1 | GPU 2 | ~Wall hours |
|------|-------|-------|-------|-------------|
| Tasks 1–2 | W&B setup + trace characterization (CPU-only) | — | — | <0.5 |
| Task 3 | sanity_random | sanity_overfit | sanity_aug + reward check | <0.5 |

### Phase 2: Baseline + ablation grid (Tasks 4–5)

| Slot | GPU 0 | GPU 1 | GPU 2 | ~Wall hours |
|------|-------|-------|-------|-------------|
| 1 | **v3_baseline** (2M) | v3_aug_scale (2M) | v3_aug_multitrace (2M) | ~2 |
| 2 | v3_aug_links (2M) | v3_aug_noise (2M) | v3_aug_jitter (2M) | ~2 |
| 3 | v3_aug_full (2M) | *(idle or start Task 8/9 early)* | *(idle or start Task 8/9 early)* | ~2 |

Stop gate check after baseline completes (slot 1, GPU 0). If baseline fails
the gate, abort remaining ablation runs.

### Phase 3: HP sweep (Task 6)

Run 3 W&B sweep agents simultaneously, one per GPU, all pulling from the
same Bayesian sweep:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id>
# Terminal 3
CUDA_VISIBLE_DEVICES=2 wandb agent <sweep_id>
```

30 trials × 500k steps each. With 3 agents running concurrently: ~10 trials
per GPU → ~5 hours wall time. Top 3 configs then get full 2M-step runs
(one per GPU, ~2 hours).

### Phase 4: Parallel ablations (Tasks 7, 8, 9)

These are independent and run simultaneously:

| GPU 0 | GPU 1 | GPU 2 | ~Wall hours |
|-------|-------|-------|-------------|
| Task 8: obs ablation (4 runs × 1M, serial) | Task 9: reward ablation (6 runs × 1M, serial) | Task 7: GNN (if needed, 3 × 2M) | ~4–6 |

If GNN is not needed (Task 7 decision gate), GPU 2 backfills with
remaining Task 8 or 9 runs.

### Phase 5: Final eval (Task 10)

Single GPU, single run. ~0.5 hours.

---

## Estimated GPU Budget

| Task | Runs | Steps each | Total steps | ~GPU hours (1 TITAN RTX) |
|------|------|-----------|-------------|------------------------|
| 3 (sanity) | 4 | 10k–50k | 80k | <0.5 |
| 4 (baseline) | 1 | 2M | 2M | ~2 |
| 5 (ablation) | 7 | 2M | 14M | ~14 |
| 6 (HP sweep) | 30 | 500k | 15M | ~15 |
| 6 (top 3 full) | 3 | 2M | 6M | ~6 |
| 7 (GNN) | 3 | 2M | 6M | ~6 (if needed) |
| 8 (obs space) | 4 | 1M | 4M | ~4 |
| 9 (reward) | 6 | 1M | 6M | ~6 |
| **Total** | | | **~53M** | **~48** |

With 3 TITAN RTX packed per the schedule above: **~16 hours wall time**.
