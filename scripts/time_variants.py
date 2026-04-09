#!/usr/bin/env python3
"""
Task 5: Measure env rollout throughput per reward variant.

The bottleneck is CPU env throughput (pure Python), not GPU.
We time:
  1. steps/sec for each variant (DummyVecEnv, 1 env, no SubprocVecEnv overhead)
  2. Extrapolate to n_envs=32 SubprocVecEnv (linear in n_envs)
  3. Deduct ~15% for IPC overhead in SubprocVecEnv
  4. Compute 6-hour max_timesteps budget
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from octopus.data import load_trace, load_topology
from octopus.env import OctopusMemPoolEnv

import os as _os

TRACE = "AMS20PrdApp19-tround.sqlite"
TOPOLOGY = str(_REPO_ROOT / "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv")
_os.chdir(_REPO_ROOT)  # ensure relative data paths resolve
TIMING_STEPS = 5_000   # steps per variant timing run
N_ENVS_PLAN = 32       # target for overnight runs
IPC_OVERHEAD = 0.15    # conservative SubprocVecEnv IPC cost
HOURS = 6.0


def _make_env_fn(trace_data, M, seed, reward_variant, lookahead_window=200):
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data
    def _init():
        return OctopusMemPoolEnv(
            all_vms=all_vms,
            node_to_vms=node_to_vms,
            node_to_machine=node_to_machine,
            machine_sz=machine_sz,
            M=M,
            seed=42,
            skip_hotfix=True,
            reward_variant=reward_variant,
            lookahead_window=lookahead_window,
        )
    return _init


def time_variant(variant, trace_data, M):
    print(f"\n--- Timing reward_variant='{variant}' (1 env, {TIMING_STEPS:,} steps) ---")
    vec_env = DummyVecEnv([_make_env_fn(trace_data, M, seed=42, reward_variant=variant)])

    # Warm up: one reset to build event cache
    print("  warm-up reset … ", end="", flush=True)
    t_reset = time.time()
    vec_env.reset()
    print(f"done ({time.time()-t_reset:.1f}s)")

    # Time env-only rollout (random actions, no model)
    print("  timing rollout … ", end="", flush=True)
    t0 = time.time()
    obs = vec_env.reset()
    for _ in range(TIMING_STEPS):
        action = [vec_env.action_space.sample()]
        obs, _, done, _ = vec_env.step(action)
    elapsed = time.time() - t0
    vec_env.close()

    # 1-env throughput
    sps_1env = TIMING_STEPS / elapsed
    # Scale to n_envs=32, apply IPC overhead
    sps_32 = sps_1env * N_ENVS_PLAN * (1.0 - IPC_OVERHEAD)
    budget = int(sps_32 * HOURS * 3600)

    print(f"done ({elapsed:.1f}s)")
    print(f"  1-env:         {sps_1env:>8,.0f} steps/sec")
    print(f"  n_envs=32 est: {sps_32:>8,.0f} steps/sec  (×{N_ENVS_PLAN}, -{IPC_OVERHEAD*100:.0f}% IPC)")
    print(f"  6-hour budget: {budget:>10,} timesteps")
    return sps_32, budget


def time_full_sac(variant, trace_data, M):
    """Time actual SAC training (env + model updates) for 2k steps."""
    print(f"\n--- SAC training timing for '{variant}' ---")
    vec_env = DummyVecEnv([_make_env_fn(trace_data, M, seed=42, reward_variant=variant)])
    model = SAC(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, buffer_size=100_000, learning_starts=500,
        batch_size=256, gamma=0.999, tau=0.005, ent_coef="auto",
        train_freq=1, gradient_steps=1,
        verbose=0, device="cuda:0", seed=42,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    t0 = time.time()
    model.learn(total_timesteps=2_000, progress_bar=False)
    elapsed = time.time() - t0
    vec_env.close()
    sps = 2_000 / elapsed
    print(f"  SAC 2k steps: {elapsed:.1f}s → {sps:.0f} steps/sec (1 env, incl. model updates)")
    return sps


def main():
    print(f"Loading trace: {TRACE}")
    trace_data = load_trace(TRACE)
    M, num_hosts, num_pools = load_topology(TOPOLOGY)
    max_deg = max(sum(row) for row in M)
    print(f"Topology: {num_hosts} hosts, {num_pools} MPDs, max_degree={max_deg}")
    print(f"Plan: n_envs={N_ENVS_PLAN}, timing_steps={TIMING_STEPS:,}")
    print(f"Bottleneck: CPU env rollout (pure Python). GPU only for SAC policy updates.")

    results = {}
    for variant in ("current", "A", "B"):
        sps, budget = time_variant(variant, trace_data, M)
        sac_sps = time_full_sac(variant, trace_data, M)
        results[variant] = (sps, budget, sac_sps)

    print("\n" + "=" * 65)
    print("TIMING SUMMARY")
    print("=" * 65)
    print(f"{'Variant':<10} {'env sps (32)':>14} {'SAC sps (1)':>14} {'6h budget':>14}")
    print("-" * 55)
    for v, (sps, bud, sac_sps) in results.items():
        print(f"{v:<10} {sps:>14,.0f} {sac_sps:>14,.0f} {bud:>14,}")

    min_budget = min(b for _, b, _ in results.values())
    rounded = int(min_budget // 100_000) * 100_000
    print(f"\nConservative 6h budget (min across variants): {min_budget:,}")
    print(f"Suggested --total-timesteps for overnight runs: {rounded:,}")

    # Write results for reference
    import json, datetime
    out = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "n_envs_plan": N_ENVS_PLAN,
        "timing_steps": TIMING_STEPS,
        "results": {v: {"env_sps_32": sps, "sac_sps_1env": s, "budget_6h": b}
                    for v, (sps, b, s) in results.items()},
        "recommended_total_timesteps": rounded,
    }
    import os; os.makedirs("output", exist_ok=True)
    with open("output/task5_timing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → output/task5_timing.json")


if __name__ == "__main__":
    main()
