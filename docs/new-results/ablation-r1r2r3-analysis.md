# Ablation Batch 1 — R1 / R2 / R3 Analysis

Run date: 2026-05-06  
Config: 200k steps, n_envs=32, skip_hotfix=True, augmentation=True, multi_trace=True  
Scripts: `scripts/ablation_batch1.sh` → run IDs `ablation_R1_v1`, `ablation_R2_v1`, `ablation_R3_v1`  
Hardware: 3× TITAN RTX, one variant per GPU (CUDA_VISIBLE_DEVICES=0/1/2)

---

## Primary metric: pooling savings

| Variant | First +5% | Steady-state median | Final (200k) |
|---------|-----------|---------------------|--------------|
| R1      | ~38k steps | +5.0% | +5.17% |
| R2      | ~23k steps | +5.2% | +4.94% |
| R3      | ~22k steps | +5.2% | +5.22% |

All three converge to ~+5% savings and plateau early. R2 and R3 reach it slightly faster
than R1 (departure-aware obs helps early learning), but by 50k steps they are
statistically indistinguishable.

Compare to run7 (`aug_v4_rewardA_v2`, R2, single trace, skip_hotfix=False): plateau at
~+1.3%. The jump to +5% is likely due to `skip_hotfix=True` (different VM population,
easier problem) rather than the reward formulation — this needs to be controlled for.

---

## SAC health

| Metric | R1 | R2 | R3 |
|--------|----|----|-----|
| ent_coef start → end | 0.991 → 0.156 | 0.991 → 0.156 | 0.991 → 0.156 |
| critic_loss start → end | 10.77 → 0.018 | 14.85 → 0.020 | 14.84 → 0.010 |
| q1_mean start → end | 10.3 → 67.6 | 9.7 → 68.4 | 9.7 → 68.3 |
| fps | 52 | 52 | 49 |

- Critic loss converges cleanly in all three — no instability.
- Q-values grow proportionally — no divergence.
- FPS is stable at 52 (R1/R2) and 49 (R3) steps/sec end-to-end.

---

## Concerns

### 1. Entropy coef still decreasing at cutoff

All three end at `ent_coef = 0.156`, which is the minimum reached (still decreasing at
step 200k). The policy hasn't finished transitioning from exploration to exploitation.
200k steps is cutting off mid-convergence. The savings are stable (~5%) but the policy
may not have fully committed. Recommended: re-run at 500k steps minimum.

### 2. R1 / R2 / R3 indistinguishable at 200k

Steady-state savings differ by <0.2 percentage points across all three variants. This
is within single-run noise — a meaningful comparison requires either:
- Multiple seeds (≥3) per variant, or
- Longer runs (500k+) where variance averages out over more eval points.
A single 200k run per variant cannot support claims about which reward is better.

### 3. `eval/mean_reward` flat throughout — not a useful signal

The SB3 `eval/mean_reward` metric stays flat at ~-17 (R1/R3) and ~-13 (R2) for the
entire run with no trend. Only the async `eval/pooling_savings_mean` captures learning.
Do not use `eval/mean_reward` to judge training progress for these variants — the reward
scales differ across variants and the env reward doesn't map cleanly to pooling quality.

### 4. skip_hotfix confounds the +5% result

Run7 (R2, skip_hotfix=False, single trace) plateaued at +1.3%. These runs
(skip_hotfix=True, multi_trace) reach +5%. The confound: skip_hotfix=True removes the
VM HOTFIX filter, which changes the VM population and may make pooling easier. The
improvement could be skip_hotfix or multi_trace, not the reward formulation.
To isolate: run a current-reward baseline with the same flags (done in batch2 as
`ablation_current_v1`).

### 5. Checkpoint frequency too aggressive

`--checkpoint-freq 500` with 200k steps = 400 checkpoint saves per run, ~1.4 GB per
run. This slows down cleanup after training and wastes disk. Use 10k-20k for future
ablation runs.

---

## Next steps

- Run batch2 (R4, R5, current) with identical flags for apples-to-apples comparison.
- Compare all 6 variants on pooling_savings_mean — the `current` run will be the key
  reference point (same flags as R1-R5 but old reward formulation).
- If R4/R5 show different behavior, extend those runs to 500k steps.
- Consider multi-seed runs (3 seeds) before drawing conclusions on which variant wins.
