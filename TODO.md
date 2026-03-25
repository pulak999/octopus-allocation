# Training, Ablations & Evaluation — Tasks (plan-v2)

Source: `docs/plans/v3/plan-v2.md`

## Task 1 — Trace Characterization & Split Assignment

- [ ] 1a. Add `--skip-hotfix` flag to `train_rl.py` and `eval_rl.py`; guard HOTFIX code behind it
- [ ] 1b. Write trace characterization script (VM count, rss stats, lifetime stats, arrival rate, peak demand, event counts with/without HOTFIX)
- [ ] 1c. Run characterization on all 10 traces, produce summary table
- [ ] 1c. Assign train/val/test split (7/1/2); write `data/splits/test_traces.sealed.json` and `data/splits/train_val_traces.json`

## Task 2 — W&B Integration

- [ ] 2a. Add `wandb` to `requirements.txt`; add `--wandb` flag + `wandb.init()` to `train_rl.py`
- [ ] 2b. Configure `EvalCallback` for unaugmented val trace (n_eval_episodes=20, deterministic)
- [ ] 2c. Implement `SACDiagnosticsCallback` (ent_coef, Q-values, gradient norms)
- [ ] 2d. Extend `AugmentationLogCallback` (trace_id histogram, link_failures_applied)
- [ ] 2e. Log env-specific metrics from `info` dict (peak util, MPD load variance, event count)
- [ ] 2f. Create W&B dashboard template (5 panel groups)
- [ ] 2g. Set W&B alerts (policy collapse, divergence, NaN, FPS drop)

## Task 3 — Pre-Training Sanity Checks

- [ ] 3a. Add `--no-train` flag; run random policy baseline (10k steps, record mean reward)
- [ ] 3b. Single-env overfit test (50k steps, single trace, verify reward > random)
- [ ] 3c. Reward scale check (confirm rewards in [-10, 10])
- [ ] 3d. Augmented env smoke test (4 envs, 10k steps, compare reward distribution)
- [ ] Add `--n-envs` flag + SubprocVecEnv support to `train_rl.py`

## Task 4 — Baseline Training (No Augmentation)

- [ ] Train `v3_baseline` (2M steps, 32 envs, train traces, λ=0.5, W&B)
- [ ] Evaluate on val trace (50 iterations)
- [ ] Stop gate: verify val reward > random baseline mean + 1 std within 500k steps

## Task 5 — Augmentation Ablation Grid

- [ ] 5a. Memory scaling only (`v3_aug_scale`)
- [ ] 5b. Multi-trace only (`v3_aug_multitrace`)
- [ ] 5c. Link failures only (`v3_aug_links`)
- [ ] 5d. Per-VM memory noise only (`v3_aug_noise`)
- [ ] 5e. Arrival jitter only (`v3_aug_jitter`)
- [ ] 5f. All P0 + P1 combined (`v3_aug_full`)
- [ ] 5g. Evaluate all runs on val trace; produce comparison table (pooling_ratio)

## Task 6 — Hyperparameter Tuning

- [ ] Create W&B Sweep config (Bayesian, 9 params)
- [ ] Run 20–30 sweep trials (500k steps each, 3 GPUs)
- [ ] Full 2M-step runs for top 3 configs

## Task 7 — Model Comparison: MLP vs GNN

- [ ] Decision gate: only if MLP + augmentation shows poor generalization
- [ ] (Conditional) GNN policy implementation

## Task 8 — Observation Space Ablation

- [ ] No time (remove sin/cos hour)
- [ ] No global peak (remove peak feature)
- [ ] No mask in obs (remove accessibility mask)
- [ ] Minimal (only MPD loads + VM request)

## Task 9 — Reward Function Variants

- [ ] Parameterize reward selection in `env.py` via `--reward-variant` CLI arg
- [ ] 9a. Lambda sweep: λ ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0}
- [ ] 9b. Peak-only variant
- [ ] 9b. Post-alloc peak variant
- [ ] 9b. Scale-invariant balance (CV instead of Var)
- [ ] 9b. Sparse reward (episode end only)

## Task 10 — Final Evaluation on Test Set

- [ ] Open `test_traces.sealed.json` (run ONCE after all decisions finalized)
- [ ] Evaluate all model variants on 2 test traces
- [ ] Save results to `output/final_eval/` (summary CSV, episode details, models.json)
- [ ] Produce final comparison table sorted by pooling_ratio

## Completed (plan-v1)

- [x] AugmentationConfig dataclass
- [x] 6 transform functions in `augmentation.py`
- [x] `apply_augmentation()` pipeline
- [x] Env integration (`_apply_augmentation`, `_switch_trace`)
- [x] Augmentation CLI args in `train_rl.py`
- [x] `AugmentationLogCallback` (basic)
- [x] Multi-trace pool loading
- [x] 49 augmentation tests
