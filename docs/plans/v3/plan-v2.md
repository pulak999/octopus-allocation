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

## Task 4 — Implement New State Space & Rewards

**Goal:** Implement the redesigned observation space (50-dim) and two new reward functions
from `docs/new-reward-statespace/new.tex`.

**Reference:** `docs/new-reward-statespace/new.tex` (full formulation).

### 4a. Per-MPD VM tracking

Add `self.mpd_vm_allocs: list[list[tuple[int, float]]]` to `octopus/env.py` —
per-MPD list of `(dealloc_tick, mem_gb)` for active VMs. Updated in `step()` on
allocation, purged in `_process_departures_through()`.

Also add per-host CXL load tracking:
- `self.cur_host_cxl_load = np.zeros(pod_size)`
- `self.host_dealloc_events = np.zeros((pod_dur, pod_size))`

These enable D_j (departure relief), S_j (sticky load), and P_j (neighborhood pressure).

### 4b. Static topology precomputation

In `__init__()` after `host_to_mhds`:
- `mhd_to_hosts`: invert `host_to_mhds` → `dict[int, list[int]]`
- `Q_j` array: for each MPD j, `Q_j = mean(1/deg(h) for h in mhd_to_hosts[j])`
- Extract into `_recompute_topology_derived()`, called from `__init__` and
  `_apply_augmentation`

### 4c. New reward functions

Add constructor params: `reward_variant: str = "current"`,
`lookahead_window: int = 200`, `reward_lambda: float = 0.2`.

In `step()`, compute D_j(t, W) once, then branch on `reward_variant`:
- **"current":** unchanged existing formula `-(Δpeak)/fair_share - λ·Var(loads)`
- **"A":** `R = -max(ĉ_j+(t) for j ∈ N(i))` where `ĉ_j = (c_j - D_j) / D_pod`
- **"B":** `R_A - λ · max(ĉ_j(t) for j ∉ N(i))`

Cache D_j values for reuse in `_get_obs()`.

### 4d. New observation space (50-dim for AG16x6)

When `reward_variant != "current"`, build `6·d_max + 2` obs:

Per-MPD slot k (zero-padded to `d_max`):
1. `c_j(t) / D_pod` — current load
2. `D_j(t,W) / D_pod` — time-weighted departure relief
3. `S_j(t) / D_pod` — sticky load (departs after W steps)
4. Connectivity mask (1/0)
5. `P_j(t) / D_pod` — neighborhood pressure
6. `Q_j` — neighbor scarcity (precomputed)

Global: `[global_peak / D_pod, vm_mem / D_pod]`

Update `observation_space` shape: `6 * max_degree + 2` when new obs active.

### 4e. CLI flags

Add to `scripts/train_rl.py` argparse:
- `--reward-variant {current,A,B}` default `"current"`
- `--lookahead-window` type=int default=200
- `--reward-lambda` type=float default=0.2

Pass to `_make_env()` → `OctopusMemPoolEnv()`.

### 4f. Eval script update

Update `scripts/evaluate.py` `make_rl_alloc_cb()` to accept `obs_variant` param.
Add `track_vm_allocs` option to `pooling_simulation` for per-MPD VM tracking in
ctx dict. Update `scripts/eval_rl.py` to pass reward variant through.

### 4g. Tests

- D_j and S_j computation with known VM sets
- Obs shape = 50 for new variants, 20 for current
- Reward A ∈ (-1, 0], Reward B ∈ (-(1+λ), 0]
- `--reward-variant current` produces identical trajectories to old code
- Augmentation correctly recomputes Q_j and mhd_to_hosts

---

## Task 5 — Pre-Ablation Timing

**Goal:** Measure wall-clock per reward variant to determine overnight timestep budget.

Run three 10k-step tests (sequential, any GPU):
```bash
python scripts/train_rl.py --run-id timing_current --reward-variant current \
    --total-timesteps 10000 --n-envs 32 --fast --skip-hotfix \
    --multi-trace --augmentation

python scripts/train_rl.py --run-id timing_A --reward-variant A \
    --lookahead-window 200 --total-timesteps 10000 --n-envs 32 --fast \
    --skip-hotfix --multi-trace --augmentation

python scripts/train_rl.py --run-id timing_B --reward-variant B \
    --lookahead-window 200 --reward-lambda 0.2 --total-timesteps 10000 \
    --n-envs 32 --fast --skip-hotfix --multi-trace --augmentation
```

Extrapolate: `max_timesteps = 10000 × (6 × 3600 / wall_seconds)`.
Use the **slowest** variant's rate for all three runs (so all finish within
6 hours).

---

## Task 6 — Overnight Experiments (3 GPUs, ~6 hours)

| GPU | Run ID | Reward | Obs Dim | Key Flags |
|-----|--------|--------|---------|-----------|
| 0 | `exp_baseline_v4` | current (Δpeak + λ·Var) | 20 | `--variance-lambda 0.5` |
| 1 | `exp_rewardA_v4` | A (local projected peak) | 50 | `--lookahead-window 200` |
| 2 | `exp_rewardB_v4` | B (A + global term) | 50 | `--lookahead-window 200 --reward-lambda 0.2` |

Common flags:
```bash
--n-envs 32 --skip-hotfix --multi-trace --augmentation --wandb --seed 42 \
--traces BLAPrdApp19-troundgrt5m BN9PrdApp18-troundgrt5m \
        DSM08PrdApp05-troundgrt5m DUB24PrdApp09-troundgrt5m \
        LON23PrdApp01-troundgrt5m SG2PrdApp35-troundgrt5m \
        SYD21PrdApp07-troundgrt5m \
--eval-trace YTO21PrdApp05-troundgrt5m \
--total-timesteps <from_timing>
```

Launch:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py --run-id exp_baseline_v4 ... &
CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py --run-id exp_rewardA_v4 ... &
CUDA_VISIBLE_DEVICES=2 python scripts/train_rl.py --run-id exp_rewardB_v4 ... &
```

### Morning check

Compare on W&B dashboard:
- `eval/pooling_ratio` curves (primary metric — lower = better)
- `rollout/ep_rew_mean` learning curves
- SAC diagnostics (entropy coefficient, Q-value magnitudes) for stability
- Check for divergence or policy collapse

---

## Task 7 — Post-Overnight Evaluation

Evaluate all three models on val trace:
```bash
for run in exp_baseline_v4 exp_rewardA_v4 exp_rewardB_v4; do
    python scripts/eval_rl.py --run-id $run \
        --traces YTO21PrdApp05-troundgrt5m --n-iter 50
done
```

Produce comparison table: **pooling_ratio** (primary), savings vs greedy,
peak utilization.

**Decision point:** Based on results, decide whether to:
- Continue with the full ablation grid (augmentation ablations)
- Extend winning reward variant to 96-host topology
- Tune W and λ further

---

## Execution Order

```
1  (Trace characterization + split)      — already done
   ↓
2  (W&B integration)                     — already done
   ↓
3  (Pre-training sanity checks)          — already done
   ↓
4  (Implement new state space + rewards)
   ↓
5  (Pre-ablation timing tests)
   ↓
6  (Overnight experiments × 3 GPUs)
   ↓
7  (Morning eval + decision point)
```

---

## GPU Scheduling

**Hardware:** 3× TITAN RTX (`cuda:0`, `cuda:1`, `cuda:2`).
SB3 SAC is single-GPU; pin via `CUDA_VISIBLE_DEVICES`.

**Phase 1 (Tasks 4–5):** Implementation + timing — single GPU, <1 hour.

**Phase 2 (Task 6):** All 3 GPUs occupied simultaneously, ~6 hours overnight.

**Phase 3 (Task 7):** Single GPU, ~30 min eval.

---

## Key Implementation Files

| File | What changes |
|------|-------------|
| `octopus/env.py` | Per-MPD VM tracking, mhd_to_hosts, Q_j, host CXL loads, new obs, new rewards |
| `scripts/train_rl.py` | `--reward-variant`, `--lookahead-window`, `--reward-lambda` CLI flags |
| `scripts/evaluate.py` | `make_rl_alloc_cb()` new obs support, `pooling_simulation` VM tracking |
| `scripts/eval_rl.py` | Pass reward variant config through |

## Risks

- **D_j/S_j per-step cost:** Iterating active VMs per MPD in `_get_obs` could
  slow training. Timing test (Task 5) will reveal. Mitigation: cache in step().
- **Eval script divergence:** `evaluate.py`'s `pooling_simulation` must match
  env.py's new obs logic exactly.
- **Reward B's global term is action-independent** at each step (documented in
  `new.tex` §5). This is the purpose of the ablation — comparing A vs B empirically.
