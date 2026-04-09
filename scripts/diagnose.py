#!/usr/bin/env python3
"""
Diagnostic tool: load a trained checkpoint + trace, run one full episode,
produce diagnostic plots.

Usage
-----
  python scripts/diagnose.py --model output/checkpoints/best_model \
      --trace LON23PrdApp01-troundgrt5m
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from octopus.data import load_trace, load_topology
from octopus.env import OctopusMemPoolEnv
from octopus.baselines import greedy_alloc
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
from scripts.evaluate import pooling_simulation, _greedy_alloc_cb


def _run_rl_episode(env, model):
    """Run one deterministic episode; return per-step data."""
    obs, _ = env.reset()
    weights_per_step = []
    loads_per_step = []
    entropies = []
    obs_list = []
    action_list = []
    reward_list = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        tick, node_in_pod_id, vm_mem, _ = env.events[env.event_idx]
        mhd_list = env.host_to_mhds[node_in_pod_id]
        n_acc = len(mhd_list)

        raw = action[:n_acc].astype(np.float64)
        raw = raw - raw.max()
        exp_raw = np.exp(raw)
        w = exp_raw / (exp_raw.sum() + 1e-12)

        # Record softmax weights (padded to max_degree, accessible MPDs only)
        w_full = np.zeros(env.max_degree)
        w_full[:n_acc] = w
        weights_per_step.append((mhd_list, w.copy()))

        # Record entropy proxy via actor distribution
        try:
            obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32).to(model.device)
            dist = model.actor.get_distribution(obs_t)
            ent = dist.distribution.entropy().mean().item()
            entropies.append(ent)
        except AttributeError:
            entropies.append(float("nan"))

        # Record obs, action, reward for Plot 6
        obs_list.append(obs.copy())
        action_list.append(action.copy())

        obs, reward, done, _, _ = env.step(action)
        loads_per_step.append(env.cur_cxl_mem_vec.copy())
        reward_list.append(float(reward))

    return weights_per_step, np.array(loads_per_step), entropies, obs_list, action_list, reward_list


def _run_greedy_episode(env):
    """Replay same pod with greedy policy; return per-step loads."""
    # Re-use the already-set-up env by resetting with same seed
    seed = env._seed
    obs, _ = env.reset(seed=seed)
    loads_per_step = []
    done = False

    while not done:
        tick, node_in_pod_id, vm_mem, _ = env.events[env.event_idx]
        mhd_list = env.host_to_mhds[node_in_pod_id]
        alloc = greedy_alloc(vm_mem, mhd_list, env.cur_cxl_mem_vec.copy())

        # Build a fake action that reproduces greedy allocation via the env step
        # Instead, directly apply allocation to env internal state (mirror step logic)
        old_peak = float(np.max(env.cur_cxl_mem_vec))
        for idx, mhd in enumerate(mhd_list):
            env.cur_cxl_mem_vec[mhd] += alloc[mhd]
        dealloc_tick = env.events[env.event_idx][3]
        if dealloc_tick < env.pod_dur:
            for mhd in range(env.num_mhd):
                if alloc[mhd] > 0:
                    env.dealloc_events[dealloc_tick, mhd] += alloc[mhd]

        env.event_idx += 1
        done = env.event_idx >= len(env.events)
        if not done:
            next_tick = env.events[env.event_idx][0]
            if next_tick > tick:
                env._process_departures_through(next_tick)

        loads_per_step.append(env.cur_cxl_mem_vec.copy())

    return np.array(loads_per_step)


def main():
    ap = argparse.ArgumentParser(description="Diagnose a trained RL checkpoint")
    ap.add_argument("--model", required=True, help="Path to SB3 checkpoint")
    ap.add_argument("--trace", required=True, help="Cluster name stem, e.g. LON23PrdApp01-troundgrt5m")
    ap.add_argument(
        "--topology",
        default="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output/diagnostics/")
    ap.add_argument(
        "--check-smoothness",
        action="store_true",
        default=False,
        help="Compute finite-difference reward smoothness check and print gradient table",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    trace_name = args.trace
    if not trace_name.endswith(".sqlite"):
        trace_name = trace_name + ".sqlite"

    print(f"Loading trace: {trace_name} …")
    trace_data = load_trace(trace_name)
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    print(f"Topology: {num_hosts} hosts, {num_pools} MPDs, max_degree={max_deg}")

    print(f"Loading model: {args.model} …")
    model = SAC.load(args.model)

    env = OctopusMemPoolEnv(
        all_vms=all_vms,
        node_to_vms=node_to_vms,
        node_to_machine=node_to_machine,
        machine_sz=machine_sz,
        M=M,
        seed=args.seed,
    )

    print("Running RL episode …")
    env.reset(seed=args.seed)
    rl_weights, rl_loads, entropies, obs_list, action_list, reward_list = _run_rl_episode(env, model)

    print("Running greedy episode …")
    env.reset(seed=args.seed)
    greedy_loads = _run_greedy_episode(env)

    num_mhd = env.num_mhd
    mhd_cap = env.pod_dram / num_mhd if num_mhd > 0 else 1.0
    steps_rl = len(rl_loads)
    steps_gr = len(greedy_loads)
    min_steps = min(steps_rl, steps_gr)

    # ── Plot 1: Softmax weight distribution ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    # Aggregate weights per accessible MPD over steps
    mhd_weight_series = {j: [] for j in range(num_mhd)}
    for step_idx, (mhd_list, w) in enumerate(rl_weights):
        for j in range(num_mhd):
            if j in mhd_list:
                local_idx = mhd_list.index(j)
                mhd_weight_series[j].append((step_idx, w[local_idx]))

    for j in range(num_mhd):
        if mhd_weight_series[j]:
            xs, ys = zip(*mhd_weight_series[j])
            ax.plot(xs, ys, alpha=0.7, label=f"MPD {j}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Softmax weight")
    ax.set_title("Softmax Allocation Weight Distribution per MPD")
    ax.legend(fontsize=7, ncol=4)
    out1 = os.path.join(args.out_dir, "weight_distribution.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Plot 2: Per-MPD load time series (RL vs Greedy) ──────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    rl_norm = rl_loads / (mhd_cap + 1e-12)
    gr_norm = greedy_loads / (mhd_cap + 1e-12)

    for j in range(num_mhd):
        axes[0].plot(rl_norm[:, j], alpha=0.7, label=f"MPD {j}")
    axes[0].set_ylabel("Normalized load")
    axes[0].set_title("Per-MPD load: RL")
    axes[0].legend(fontsize=7, ncol=4)

    for j in range(num_mhd):
        axes[1].plot(gr_norm[:, j], alpha=0.7, label=f"MPD {j}")
    axes[1].set_ylabel("Normalized load")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Per-MPD load: Greedy")
    axes[1].legend(fontsize=7, ncol=4)

    out2 = os.path.join(args.out_dir, "mpd_loads.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # ── Plot 3: Load variance over episode ───────────────────────────
    rl_var = np.var(rl_norm, axis=1)
    gr_var = np.var(gr_norm[:min_steps], axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rl_var, label="RL", alpha=0.8)
    ax.plot(gr_var, label="Greedy", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Var(MPD loads)")
    ax.set_title("Load Variance over Episode")
    ax.legend()
    out3 = os.path.join(args.out_dir, "load_variance.png")
    fig.tight_layout()
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    # ── Plot 4: Max/min MPD load ratio ───────────────────────────────
    rl_ratio = np.max(rl_norm, axis=1) / (np.min(rl_norm, axis=1) + 1e-9)
    gr_ratio = np.max(gr_norm[:min_steps], axis=1) / (np.min(gr_norm[:min_steps], axis=1) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rl_ratio, label="RL", alpha=0.8)
    ax.plot(gr_ratio, label="Greedy", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("max/min load ratio")
    ax.set_title("Max/Min MPD Load Ratio")
    ax.legend()
    out4 = os.path.join(args.out_dir, "load_ratio.png")
    fig.tight_layout()
    fig.savefig(out4, dpi=150)
    plt.close(fig)
    print(f"Saved: {out4}")

    # ── Plot 5: SAC action entropy proxy ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    steps5 = list(range(len(entropies)))
    ax.plot(steps5, entropies, alpha=0.8, label="Mean action entropy")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="entropy = 0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("SAC Action Entropy (proxy for policy stochasticity)")
    ax.legend()
    out5 = os.path.join(args.out_dir, "entropy_proxy.png")
    fig.tight_layout()
    fig.savefig(out5, dpi=150)
    plt.close(fig)
    print(f"Saved: {out5}")

    # ── Plot 6: Q-value vs Monte Carlo return ─────────────────────────
    rewards = np.array(reward_list)
    T = len(rewards)
    G = np.zeros(T)
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + 0.999 * G[t + 1]

    obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(model.device)
    act_batch = torch.tensor(np.array(action_list), dtype=torch.float32).to(model.device)
    with torch.no_grad():
        q1, q2 = model.critic(obs_batch, act_batch)
    q_vals = torch.min(q1, q2).cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(q_vals, G, alpha=0.4, s=10, label="Steps")
    q_min = min(q_vals.min(), G.min())
    q_max = max(q_vals.max(), G.max())
    ax.plot([q_min, q_max], [q_min, q_max], "r--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel("Predicted Q-value (min of two heads)")
    ax.set_ylabel("Monte Carlo Return (γ=0.999)")
    ax.set_title("Q-value vs Monte Carlo Return (above line = overestimation)")
    ax.legend()
    out6 = os.path.join(args.out_dir, "qvalue_vs_return.png")
    fig.tight_layout()
    fig.savefig(out6, dpi=150)
    plt.close(fig)
    print(f"Saved: {out6}")

    # ── Task B: Reward smoothness check ──────────────────────────────
    if args.check_smoothness:
        print("\n── Reward smoothness check ──")
        rng = np.random.default_rng(0)
        n_samples = min(100, len(obs_list))
        sample_idx = rng.choice(len(obs_list), size=n_samples, replace=False)
        sample_obs = [obs_list[i] for i in sample_idx]

        max_degree = env.max_degree
        eps = 1e-4

        # grad_matrix[sample, dim] = |∂R/∂a_dim|
        grad_matrix = np.zeros((n_samples, max_degree))

        for s_idx, obs_s in enumerate(sample_obs):
            # Use first max_degree elements as cur_cxl_mem_vec proxy
            cur_cxl_mem_vec_proxy = obs_s[:max_degree].copy()
            # vm_norm is at index 2*max_degree
            cxl_mem = float(obs_s[2 * max_degree]) if len(obs_s) > 2 * max_degree else 0.0

            old_peak = float(np.max(cur_cxl_mem_vec_proxy))

            def peak_change_reward(logits):
                """Compute peak-change reward component from raw action logits."""
                shifted = logits - logits.max()
                exp_l = np.exp(shifted)
                w = exp_l / (exp_l.sum() + 1e-12)
                # Allocate proportionally across all max_degree slots
                alloc = w * cxl_mem
                new_vec = cur_cxl_mem_vec_proxy + alloc
                new_peak = float(np.max(new_vec))
                return new_peak - old_peak

            # Use the recorded action for this obs as the base logits
            base_action = action_list[sample_idx[s_idx]].copy().astype(np.float64)
            # Pad or truncate to max_degree
            logits = np.zeros(max_degree)
            logits[:len(base_action)] = base_action[:max_degree]

            for dim in range(max_degree):
                logits_p = logits.copy()
                logits_p[dim] += eps
                logits_m = logits.copy()
                logits_m[dim] -= eps
                grad_matrix[s_idx, dim] = abs(
                    (peak_change_reward(logits_p) - peak_change_reward(logits_m)) / (2 * eps)
                )

        mean_grad = grad_matrix.mean(axis=0)
        dead_dims = [d for d in range(max_degree) if np.all(grad_matrix[:, d] == 0.0)]

        print(f"\n{'Dim':>4}  {'mean |∂R/∂a_i|':>18}  {'Dead?':>6}")
        print("-" * 35)
        for d in range(max_degree):
            dead_flag = "DEAD" if d in dead_dims else ""
            print(f"{d:>4}  {mean_grad[d]:>18.6f}  {dead_flag:>6}")

        if dead_dims:
            print(f"\nWARNING: {len(dead_dims)} dead reward dimension(s) found: {dead_dims}")
        else:
            print("\nAll action dimensions have nonzero gradient signal across sampled obs.")


if __name__ == "__main__":
    main()
