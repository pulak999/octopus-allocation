# RL Octopus — Next Iteration To-Dos

## 1. Diagnostics (do first)
- [ ] Plot actual weight vector **a** output by policy over time — check for mode collapse vs near-uniform
- [ ] Plot pooling savings on held-out trace every 100 episodes (not just reward)
- [ ] Plot MPD load variance Var(**c**_t) over training
- [ ] Plot SAC policy entropy H(π) over training steps
- [ ] Plot Q-value estimates vs actual returns — check for critic overestimation
- [ ] Run greedy, optimal, and RL side-by-side on same trace and plot per-MPD usage over time to get behavioral signature of what optimal actually looks like

## 2. State Redesign
- [ ] Add per-MPD load deltas at 1hr and 2hr windows — [v_t, Δ_1h, Δ_2h] encoding
- [ ] Add cross-server memory pressure trends — all servers in pod, not just requesting server
- [ ] Keep hour-of-day sin/cos encoding
- [ ] Add day-of-week encoding
- [ ] Evaluate whether observation should be restructured as a graph given bipartite topology

## 3. Reward Redesign
- [ ] Implement Option A (fully immediate, smooth)
- [ ] Implement Option B (immediate + delayed at VM termination)
- [ ] Tune variance shaping weight λ aggressively upward
- [ ] Ensure immediate reward is smooth and differentiable with respect to **a**
- [ ] Run both options on AMS20 and LON23 — let experiments decide

## 4. Per-Datacenter Policy Evaluation
- [ ] Train separate policies on AMS (easy — low utilization) and LVL (hard — near capacity)
- [ ] Compare performance — if same policy works on AMS but fails on LVL, motivates context conditioning
- [ ] Use as paper evidence that fixed policies (greedy/PID) are provably suboptimal across regimes

## 5. Convergence Infrastructure
- [ ] Build evaluation loop on held-out trace every N episodes logging pooling savings, load variance, and entropy
- [ ] Stop relying on training reward curve alone as convergence signal
- [ ] Log policy weight vector statistics (mean, variance, entropy of **a**) during evaluation

## 6. Longer Term / Paper Extensions
- [ ] Add datacenter context vector as state signal for regime-specific behavior
- [ ] Evaluate LCPO for catastrophic forgetting across diurnal regime shifts
- [ ] Run ablation studies: remove delta features, remove cross-server signals, remove conn_mask
- [ ] Consider GNN policy architecture to naturally encode bipartite topology

---
**Order of attack:** 1 → 2+3 in parallel → 4 → 5 alongside 2+3 → 6 once core results are solid