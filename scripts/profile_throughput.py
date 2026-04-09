#!/usr/bin/env python3
"""
Throughput profiling script — diagnoses the 75 fps regression.

Runs four ablations (5k steps each) and reports fps + per-reset timing:
  1. Baseline: single trace, no aug, no multi-trace  (expected ~1221 fps)
  2. + Multi-trace only
  3. + Augmentation only
  4. Both (current training config)

Usage:
  python scripts/profile_throughput.py
"""

import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from octopus.data import load_trace, load_topology, to_arrays
from octopus.env import OctopusMemPoolEnv

TRACE_SINGLE = "AMS20PrdApp19-tround.sqlite"
TRACE_POOL_NAMES = [
    "AMS20PrdApp19-tround.sqlite",
    "BLAPrdApp19-troundgrt5m.sqlite",
    "BN9PrdApp18-troundgrt5m.sqlite",
    "DSM08PrdApp05-troundgrt5m.sqlite",
    "DUB24PrdApp09-troundgrt5m.sqlite",
    "LON23PrdApp01-troundgrt5m.sqlite",
    "LVL01PrdApp05-troundgrt5m.sqlite",
    "SG2PrdApp35-troundgrt5m.sqlite",
    "SYD21PrdApp07-troundgrt5m.sqlite",
    "YTO21PrdApp05-troundgrt5m.sqlite",
]
N_ENVS = 32
N_STEPS = 5_000
TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"


def make_env_fn(trace_arrays, M, seed, aug_config=None, trace_pool=None):
    def _init():
        env = OctopusMemPoolEnv(
            trace_arrays=trace_arrays,
            M=M,
            seed=seed,
            aug_config=aug_config,
            trace_pool=trace_pool,
        )
        env._reset_timing = True   # enable per-reset timing on first reset
        return env
    return _init


def run_ablation(label, env_fns, n_steps):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    vec_env = SubprocVecEnv(env_fns) if N_ENVS > 1 else DummyVecEnv(env_fns)
    obs = vec_env.reset()

    t_start = time.perf_counter()
    for step in range(n_steps):
        actions = np.array([vec_env.action_space.sample() for _ in range(N_ENVS)])
        obs, rewards, dones, infos = vec_env.step(actions)
    elapsed = time.perf_counter() - t_start

    total_transitions = n_steps * N_ENVS
    fps = total_transitions / elapsed
    print(f"\n  fps = {fps:.0f}  ({total_transitions:,} transitions in {elapsed:.1f}s)")
    vec_env.close()
    return fps


def main():
    print("Loading topology…")
    M, _, _ = load_topology(TOPOLOGY)

    print(f"Loading single trace ({TRACE_SINGLE})…")
    single_arrays = to_arrays(load_trace(TRACE_SINGLE))

    print("Loading 10-trace pool…")
    pool = []
    for name in TRACE_POOL_NAMES:
        print(f"  {name}")
        pool.append(to_arrays(load_trace(name)))
    print(f"  Pool ready ({len(pool)} traces)")

    # ── Augmentation config ─────────────────────────────────────────────
    from octopus.augmentation import AugmentationConfig
    aug_config = AugmentationConfig(
        scale_range=(0.7, 1.2),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=True,
    )

    results = {}

    # 1. Baseline
    env_fns = [make_env_fn(single_arrays, M, seed=i) for i in range(N_ENVS)]
    results["1_baseline"] = run_ablation(
        "1. Baseline: single trace, no aug, no multi-trace", env_fns, N_STEPS
    )

    # 2. Multi-trace only
    aug_multi_only = AugmentationConfig(
        scale_range=(1.0, 1.0),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=False,   # aug disabled but multi_trace flag still used for pool switching
    )
    # Note: multi_trace switching happens in reset() only when aug_config.multi_trace=True
    # AND aug_config is not None. We need enabled=True for the pool-switch path to fire.
    aug_multi_only_enabled = AugmentationConfig(
        scale_range=(1.0, 1.0),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=True,
    )
    env_fns = [make_env_fn(single_arrays, M, seed=i,
                           aug_config=aug_multi_only_enabled,
                           trace_pool=pool)
               for i in range(N_ENVS)]
    results["2_multi_trace"] = run_ablation(
        "2. Multi-trace only (no scale/noise/jitter aug)", env_fns, N_STEPS
    )

    # 3. Augmentation only (single trace)
    aug_single = AugmentationConfig(
        scale_range=(0.7, 1.2),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=False,
        enabled=True,
    )
    env_fns = [make_env_fn(single_arrays, M, seed=i, aug_config=aug_single)
               for i in range(N_ENVS)]
    results["3_aug_only"] = run_ablation(
        "3. Augmentation only (single trace, scale aug)", env_fns, N_STEPS
    )

    # 4. Both (current training config)
    env_fns = [make_env_fn(single_arrays, M, seed=i,
                           aug_config=aug_config,
                           trace_pool=pool)
               for i in range(N_ENVS)]
    results["4_both"] = run_ablation(
        "4. Both: multi-trace + augmentation (current config)", env_fns, N_STEPS
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v:.0f} fps")
    print(f"\n  Expected baseline: ~1221 fps")
    print(f"  Current training:  ~75 fps")


if __name__ == "__main__":
    main()
