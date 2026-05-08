#!/usr/bin/env python3
"""JAX env-only throughput benchmark — Chunk C gate.

Measures steps/sec for the JAX lax.scan + vmap rollout and compares against
the SB3 SubprocVecEnv baseline (2,757 fps at n_envs=32).

Usage:
    python scripts/benchmark_jax_env.py
    python scripts/benchmark_jax_env.py --n-envs 32 --horizon 256 --n-iters 50
    python scripts/benchmark_jax_env.py --trace data/traces/AMS20PrdApp19-tround.sqlite.pkl
"""

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax
import jax.numpy as jnp
import numpy as np

from octopus.data import load_trace, load_topology, to_arrays
from octopus.jax.env import make_episode, make_rollout_fn, reset_fn

DEFAULT_TRACE    = "AMS20PrdApp19-tround.sqlite"
DEFAULT_TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"
SB3_BASELINE_FPS = 2757   # steady-state fps from SPEEDUP_PLAN (n_envs=32, random actions)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs",  type=int, default=32)
    p.add_argument("--horizon", type=int, default=256,
                   help="Steps per rollout chunk (H)")
    p.add_argument("--n-iters", type=int, default=50,
                   help="Timed iterations (after 1 warm-up)")
    p.add_argument("--trace",    default=DEFAULT_TRACE)
    p.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    p.add_argument("--seed",     type=int, default=0,
                   help="Episode seed (same episode replicated across all n_envs)")
    return p.parse_args()


def load_data(trace_name: str, topology_path: str):
    raw = load_trace(trace_name)
    ta  = to_arrays(raw)
    # load_topology() returns (matrix, num_hosts, num_pools). The env/JAX code
    # expects the adjacency matrix itself.
    M_mat, _, _ = load_topology(topology_path)
    return ta, np.array(M_mat, dtype=np.int32)


def build_batched_state(sc, n_envs: int):
    """Stack n_envs copies of reset_fn(sc) into a batched OctopusState."""
    state0, _ = reset_fn(sc)
    # Replicate the same initial state across all envs
    return jax.tree.map(lambda x: jnp.broadcast_to(x[None], (n_envs,) + x.shape), state0)


def run_benchmark(args):
    print(f"JAX env-only benchmark")
    print(f"  n_envs  = {args.n_envs}")
    print(f"  horizon = {args.horizon} steps/chunk")
    print(f"  n_iters = {args.n_iters} timed iterations")
    print(f"  trace   = {args.trace}")
    print(f"  device  = {jax.default_backend()}")
    print()

    # ----- load data -----
    print("Loading trace and topology...", end="", flush=True)
    ta, M = load_data(args.trace, args.topology)
    print(f" done  ({len(ta.node_ids)} nodes, {M.shape} topology)")

    # ----- build episode -----
    print("Building StaticConfig...", end="", flush=True)
    sc = make_episode(
        seed=args.seed, trace_arrays=ta, M=M,
        reward_variant="current", lookahead_window=200, reward_lambda=0.2,
        skip_hotfix=False,
    )
    print(f" done  (num_events={sc.num_events}, pod_size={sc.pod_size}, "
          f"num_mhd={sc.num_mhd}, max_degree={sc.max_degree})")

    # ----- build batched initial state -----
    batched_state = build_batched_state(sc, args.n_envs)

    # ----- build action sequence -----
    rng = np.random.default_rng(42)
    actions_np = rng.standard_normal(
        (args.n_envs, args.horizon, sc.max_degree)
    ).astype(np.float32)
    batched_actions = jnp.asarray(actions_np)

    # ----- compile rollout -----
    print("Compiling rollout fn (JIT warm-up)...", end="", flush=True)
    rollout = make_rollout_fn(sc)
    t0 = time.perf_counter()
    final_state, transitions = rollout(batched_state, batched_actions)
    # Block until computation completes
    jax.block_until_ready(transitions)
    compile_sec = time.perf_counter() - t0
    print(f" {compile_sec:.2f}s  (includes JIT compilation)")

    # ----- timed iterations -----
    # Re-use the same batched_state and batched_actions for all iterations.
    # This measures pure throughput of the compiled kernel.
    times = []
    for i in range(args.n_iters):
        t0 = time.perf_counter()
        _, transitions = rollout(batched_state, batched_actions)
        jax.block_until_ready(transitions)
        times.append(time.perf_counter() - t0)

    steps_per_iter = args.n_envs * args.horizon
    fps_per_iter   = [steps_per_iter / t for t in times]

    fps_mean   = float(np.mean(fps_per_iter))
    fps_median = float(np.median(fps_per_iter))
    fps_p10    = float(np.percentile(fps_per_iter, 10))

    print()
    print("=" * 50)
    print(f"Throughput (steps/sec):")
    print(f"  mean   : {fps_mean:>8,.0f} fps")
    print(f"  median : {fps_median:>8,.0f} fps")
    print(f"  p10    : {fps_p10:>8,.0f} fps")
    print()
    print(f"SB3 baseline (n_envs={args.n_envs}, random):  {SB3_BASELINE_FPS:>8,.0f} fps")
    ratio = fps_median / SB3_BASELINE_FPS
    sign  = "+" if ratio >= 1.0 else "-"
    print(f"JAX / SB3 ratio: {ratio:.2f}x  ({sign}{abs(ratio-1)*100:.0f}%)")
    print("=" * 50)

    if fps_median > SB3_BASELINE_FPS:
        print("RESULT: JAX beats SB3 baseline ✓")
    else:
        print("RESULT: JAX below SB3 baseline (CPU-only; GPU would close the gap)")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
