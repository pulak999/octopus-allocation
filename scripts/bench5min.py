#!/usr/bin/env python3
"""
5-minute env throughput benchmark using run7 config.
Timing excludes trace loading. Run from repo root.
"""
import sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from octopus.data import load_trace, load_topology, to_arrays, precompute_pod_events_arrays
from octopus.env import OctopusMemPoolEnv
from octopus.augmentation import AugmentationConfig

TRACE_NAMES = [
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
TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"
N_ENVS = 32
BENCH_SECONDS = 300  # 5 minutes
REPORT_EVERY = 500   # steps between progress prints


def main():
    print("=== Loading traces (excluded from timing) ===", flush=True)
    t0 = time.perf_counter()
    M, _, _ = load_topology(TOPOLOGY)
    trace_pool = []
    for name in TRACE_NAMES:
        print(f"  {name}", flush=True)
        trace_pool.append(to_arrays(load_trace(name)))
    load_elapsed = time.perf_counter() - t0
    print(f"Traces loaded in {load_elapsed:.1f}s", flush=True)

    N_SEEDS = 128
    CACHE_DIR = "data/cache/pod_events"
    print(f"Precomputing event caches ({N_SEEDS} seeds × {len(trace_pool)} traces, excluded from timing)...", flush=True)
    t_cache_start = time.perf_counter()
    trace_pool_events = []
    for i, ta in enumerate(trace_pool):
        print(f"  trace {i+1}/{len(trace_pool)}...", flush=True)
        trace_pool_events.append(
            precompute_pod_events_arrays(ta, M, range(N_SEEDS), cache_dir=CACHE_DIR)
        )
    cache_elapsed = time.perf_counter() - t_cache_start
    print(f"Caches ready in {cache_elapsed:.1f}s {'(from disk)' if cache_elapsed < 5 else '(built fresh)'}\n", flush=True)

    aug_config = AugmentationConfig(
        scale_range=(0.7, 1.2),
        scale_distribution="triangular",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=True,
    )

    # trace_pool is in global scope of main process; workers get it via serialization at spawn
    def _make_env(idx):
        def _init():
            return OctopusMemPoolEnv(
                trace_arrays=trace_pool[idx % len(trace_pool)],
                M=M,
                seed=42 + idx,
                reward_variant="A",
                lookahead_window=200,
                reward_lambda=0.2,
                aug_config=aug_config,
                trace_pool=trace_pool,
                trace_pool_events=trace_pool_events,
            )
        return _init

    print(f"Building SubprocVecEnv ({N_ENVS} workers)...", flush=True)
    env = SubprocVecEnv([_make_env(i) for i in range(N_ENVS)])
    obs = env.reset()
    print(f"Env ready. Starting {BENCH_SECONDS//60}-minute timing window...\n", flush=True)

    step = 0
    fps_samples = []
    t_start = time.perf_counter()
    t_last = t_start

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= BENCH_SECONDS:
            break
        actions = np.array([env.action_space.sample() for _ in range(N_ENVS)])
        env.step(actions)
        step += 1

        if step % REPORT_EVERY == 0:
            now = time.perf_counter()
            window_fps = REPORT_EVERY * N_ENVS / (now - t_last)
            fps_samples.append(window_fps)
            print(f"  step={step*N_ENVS:>10,}  elapsed={elapsed:5.0f}s  fps={window_fps:.0f}", flush=True)
            t_last = now

    env.close()

    total_elapsed = time.perf_counter() - t_start
    total_transitions = step * N_ENVS
    overall_fps = total_transitions / total_elapsed
    steady_fps = np.mean(fps_samples[2:]) if len(fps_samples) > 2 else overall_fps

    print(f"\n{'='*55}")
    print(f"  5-MINUTE BENCHMARK — run7 config")
    print(f"{'='*55}")
    print(f"  Config:              reward_variant=A, n_envs={N_ENVS}")
    print(f"                       multi_trace=True, augmentation=True")
    print(f"  Cache prep time:     {cache_elapsed:.1f}s  ({N_SEEDS} seeds × {len(trace_pool)} traces)")
    print(f"  Total transitions:   {total_transitions:,}")
    print(f"  Wall time (bench):   {total_elapsed:.1f}s")
    print(f"  Overall fps:         {overall_fps:.0f}")
    print(f"  Steady-state fps:    {steady_fps:.0f}  (post-warmup mean)")
    print(f"  Min / Max fps:       {min(fps_samples):.0f} / {max(fps_samples):.0f}")
    print(f"\n  Baseline (run7):     ~53 fps")
    print(f"  Target (plan):       185 fps (3.5×)")
    print(f"  Speedup vs baseline: {overall_fps/53:.1f}×")
    print(f"{'='*55}", flush=True)


if __name__ == "__main__":
    main()
