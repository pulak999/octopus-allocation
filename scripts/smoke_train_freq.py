#!/usr/bin/env python3
"""
Smoke test: measure env-step latency vs gradient-update latency at different
train_freq values, and report fps for each.

Usage:
  python scripts/smoke_train_freq.py
"""

import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from octopus.data import load_trace, load_topology, to_arrays
from octopus.env import OctopusMemPoolEnv

TRACE      = "AMS20PrdApp19-tround.sqlite"
TOPOLOGY   = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"
N_ENVS     = 32
N_STEPS    = 3_000          # total timesteps per run (small for quick test)
TRAIN_FREQS = [1, 4, 8, 16, 32]


# ── Timing state (shared via closure) ───────────────────────────────────────
_grad_times   = []
_env_t_start  = None
_env_times    = []


def _patch_sac_train(model):
    """Monkey-patch SAC.train() to record gradient update durations."""
    original_train = model.train.__func__

    def timed_train(self, gradient_steps, batch_size):
        t0 = time.perf_counter()
        result = original_train(self, gradient_steps, batch_size)
        _grad_times.append((time.perf_counter() - t0) * 1e3)
        return result

    import types
    model.train = types.MethodType(timed_train, model)


def _patch_collect_rollouts(model):
    """Monkey-patch to record per-env-step duration."""
    original_collect = model.collect_rollouts.__func__

    def timed_collect(self, *args, **kwargs):
        global _env_t_start
        _env_t_start = time.perf_counter()
        result = original_collect(self, *args, **kwargs)
        _env_times.append((time.perf_counter() - _env_t_start) * 1e3)
        return result

    import types
    model.collect_rollouts = types.MethodType(timed_collect, model)


def run(train_freq):
    global _grad_times, _env_times
    _grad_times = []
    _env_times  = []

    ta = to_arrays(load_trace(TRACE))
    M, _, _ = load_topology(TOPOLOGY)

    env_fns = [
        (lambda s: (lambda: OctopusMemPoolEnv(trace_arrays=ta, M=M, seed=s)))(i)
        for i in range(N_ENVS)
    ]
    vec_env = SubprocVecEnv(env_fns)

    model = SAC(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,        # small buffer — fills quickly, stable sampling
        learning_starts=500,
        batch_size=256,
        gamma=0.999,
        train_freq=train_freq,
        gradient_steps=1,
        verbose=0,
        device="auto",
        seed=42,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    _patch_sac_train(model)
    _patch_collect_rollouts(model)

    t_start = time.perf_counter()
    model.learn(total_timesteps=N_STEPS)
    elapsed = time.perf_counter() - t_start

    vec_env.close()

    fps = N_STEPS / elapsed

    g = np.array(_grad_times)
    e = np.array(_env_times)

    print(f"\n  train_freq={train_freq:2d}  |  fps={fps:6.0f}  |  "
          f"grad: n={len(g)} mean={g.mean():.0f}ms p95={np.percentile(g,95):.0f}ms  |  "
          f"collect: n={len(e)} mean={e.mean():.0f}ms")

    return fps, g, e


def main():
    print(f"Loading trace ({TRACE})…")
    # pre-load once so timing is stable across runs
    # (actual load happens inside run() — kept simple)

    print(f"\n{'='*75}")
    print(f"  train_freq sweep  |  N_ENVS={N_ENVS}  N_STEPS={N_STEPS}")
    print(f"{'='*75}")

    results = {}
    for tf in TRAIN_FREQS:
        results[tf] = run(tf)

    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"  {'train_freq':>10}  {'fps':>8}  {'speedup vs tf=1':>16}")
    base_fps = results[1][0]
    for tf, (fps, _, __) in results.items():
        print(f"  {tf:>10}  {fps:>8.0f}  {fps/base_fps:>15.1f}x")

    print(f"\n  Expected baseline (env only, no grad): ~2800 fps")
    print(f"  Previous training run (train_freq=1):  ~75 fps")


if __name__ == "__main__":
    main()
