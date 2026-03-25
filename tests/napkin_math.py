from __future__ import annotations

# Allow running this file directly without installing the package.
# (When run as `python3 napkin_math.py`, Python doesn't automatically
# add `octopus-allocation/` to sys.path.)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # octopus-allocation/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gc
import os
import pickle
import time
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from octopus.data import load_trace, load_topology, precompute_pod_events
from octopus.env import OctopusMemPoolEnv


# ---- ENV FACTORY ----
def make_env(trace_data, M, seed=0, variance_lambda=0.5, precomputed_events=None):
    all_vms, node_to_vms, node_to_machine, _vm_type_sz, machine_sz = trace_data

    def _init():
        return OctopusMemPoolEnv(
            all_vms=all_vms,
            node_to_vms=node_to_vms,
            node_to_machine=node_to_machine,
            machine_sz=machine_sz,
            M=M,
            seed=seed,
            variance_lambda=variance_lambda,
            precomputed_events=precomputed_events,
        )
    return _init


def _gpu_util():
    """Return GPU utilization % if CUDA is available, else None."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.utilization(0)
        except Exception:
            return None
    return None


def _bench_throughput(env, model_kwargs, n_steps=5_000):
    """Train for n_steps and return steps/sec."""
    model = SAC("MlpPolicy", env, verbose=0, **model_kwargs)
    # Warm up
    model.learn(total_timesteps=500)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    model.learn(total_timesteps=n_steps)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


def main():
    print("Loading data...")
    trace_data = load_trace("AMS20PrdApp19-tround.sqlite")
    M, _, _ = load_topology("data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv")

    # ------------------------------------------------------------------ #
    # Step 0a: Pickle round-trip check                                    #
    # ------------------------------------------------------------------ #
    print("\n--- Pickle check ---")
    buf = pickle.dumps(trace_data)
    td2 = pickle.loads(buf)
    assert len(td2[0]) == len(trace_data[0]), "all_vms mismatch after pickle round-trip"
    assert len(td2[1]) == len(trace_data[1]), "node_to_vms mismatch after pickle round-trip"
    print(f"✓ trace_data is picklable ({len(buf)/1e6:.1f} MB)")

    # ------------------------------------------------------------------ #
    # Q1: Env step time (single env)                                      #
    # ------------------------------------------------------------------ #
    print("\n--- Q1: Env step time ---")
    env = DummyVecEnv([make_env(trace_data, M)])
    obs = env.reset()
    action = [env.action_space.sample()]

    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        obs, _, _, _ = env.step(action)
    elapsed = time.perf_counter() - start

    step_time = elapsed / N
    print(f"Avg env.step(): {step_time*1000:.3f} ms")
    print(f"Steps/sec (env only): {1/step_time:.1f}")

    # ------------------------------------------------------------------ #
    # Q1b: reset() time — no precomputation                              #
    # ------------------------------------------------------------------ #
    print("\n--- Q1b: reset() time (no precomputation) ---")
    R = 20
    start = time.perf_counter()
    for _ in range(R):
        env.reset()
    elapsed = time.perf_counter() - start
    reset_ms_cold = elapsed / R * 1000
    print(f"Avg env.reset() cold: {reset_ms_cold:.1f} ms  ({R} trials)")

    env.close()

    # ------------------------------------------------------------------ #
    # Q1c: reset() time — with precomputed events                        #
    # ------------------------------------------------------------------ #
    print("\n--- Q1c: reset() time (with precomputed events) ---")
    seeds_to_precompute = list(range(R))
    print(f"  Precomputing {len(seeds_to_precompute)} seeds …")
    t0 = time.perf_counter()
    precomputed = precompute_pod_events(trace_data, M, seeds_to_precompute)
    precompute_time = time.perf_counter() - t0
    print(f"  Precompute took: {precompute_time*1000:.0f} ms total  "
          f"({precompute_time/len(seeds_to_precompute)*1000:.1f} ms/seed)")

    env_pre = DummyVecEnv([make_env(trace_data, M, seed=0, precomputed_events=precomputed)])
    # Warm the first reset
    env_pre.reset()

    # Measure R resets cycling through precomputed seeds
    # Each reset() increments episode_count → uses seed 0,1,2,...
    start = time.perf_counter()
    for _ in range(R):
        env_pre.reset()
    elapsed = time.perf_counter() - start
    reset_ms_warm = elapsed / R * 1000
    env_pre.close()

    print(f"Avg env.reset() precomputed: {reset_ms_warm:.1f} ms  ({R} trials)")
    print(f"Speedup: {reset_ms_cold / reset_ms_warm:.1f}x")

    # ------------------------------------------------------------------ #
    # Q2: GPU info                                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Q2: GPU info ---")
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU util at rest: {_gpu_util()}%")

    # ------------------------------------------------------------------ #
    # Q3: Training throughput benchmark suite                             #
    # ------------------------------------------------------------------ #
    print("\n--- Q3: Training throughput benchmark suite ---")
    print("Each config runs 5,000 steps (+ 500 warmup).\n")

    BENCH_STEPS = 5_000
    results = []

    def _run(label, n_envs, train_freq, gradient_steps, batch_size):
        env_fns = [make_env(trace_data, M, seed=i) for i in range(n_envs)]
        if n_envs > 1:
            venv = SubprocVecEnv(env_fns, start_method="fork")
        else:
            venv = DummyVecEnv(env_fns)
        kwargs = dict(
            device=device,
            policy_kwargs=dict(net_arch=[128, 128]),
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            buffer_size=100_000,
            learning_starts=200,
        )
        sps = _bench_throughput(venv, kwargs, n_steps=BENCH_STEPS)
        util = _gpu_util()
        venv.close()
        results.append((label, n_envs, train_freq, batch_size, sps, util))
        tag = f"{util}%" if util is not None else "N/A"
        print(f"  {label:<35} {sps:>8.0f} steps/sec   GPU util: {tag}")

    _run("Baseline  (n=1, tf=1,  bs=256)",  1,  1,  1,  256)
    _run("n=1, tf=16, gs=16, bs=512",        1, 16, 16,  512)
    _run("n=4,  SubprocVecEnv, tf=4",         4,  4,  4,  512)
    _run("n=8,  SubprocVecEnv, tf=8",         8,  8,  8,  512)
    _run("n=16, SubprocVecEnv, tf=16",        16, 16, 16, 512)
    _run("n=32, SubprocVecEnv, tf=32",        32, 32, 32, 512)

    print("\n=== Summary ===")
    print(f"{'Config':<35} {'Steps/sec':>10} {'GPU%':>6}")
    print("-" * 55)
    baseline_sps = results[0][4]
    for label, n_envs, tf, bs, sps, util in results:
        speedup = sps / baseline_sps
        tag = f"{util}%" if util is not None else "N/A"
        print(f"{label:<35} {sps:>10.0f} {tag:>6}   ({speedup:.1f}x baseline)")


def scale_stress_test():
    """
    Q4: 60x scale stress test.

    Measures the four things that could become new bottlenecks when training
    on ~60x more data (all available traces combined):

      Q4a  Trace load time + pickle size per trace
      Q4b  Precompute time per seed as trace size grows
      Q4c  dealloc_events matrix memory per env (pod_dur x num_mhd)
      Q4d  SubprocVecEnv throughput with N=32 on all traces combined
           vs. single-trace baseline — does IPC cost grow?
    """
    TRACE_DIR = "data/traces"
    TOPO = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"
    SEEDS = list(range(10))   # 10 precompute samples per trace
    N_ENVS = 32

    M, _, _ = load_topology(TOPO)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    pkl_files = sorted(
        f for f in os.listdir(TRACE_DIR) if f.endswith(".pkl")
    )
    print(f"\n{'='*60}")
    print(f"Q4: 60x scale stress test  ({len(pkl_files)} traces found)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Q4a: load time + pickle size per trace                              #
    # ------------------------------------------------------------------ #
    print("--- Q4a: Trace load time + size ---")
    print(f"{'Trace':<45} {'Load(s)':>8} {'pkl MB':>8} {'n_vms':>8} {'n_nodes':>8}")
    print("-" * 80)

    trace_records = []   # (name, trace_data, load_s, pkl_mb, n_vms, n_nodes)
    total_vms = 0
    for fname in pkl_files:
        stem = fname.replace(".pkl", "")
        t0 = time.perf_counter()
        td = load_trace(stem, trace_dir=TRACE_DIR)
        load_s = time.perf_counter() - t0

        buf = pickle.dumps(td)
        pkl_mb = len(buf) / 1e6
        n_vms = len(td[0])
        n_nodes = len(td[1])
        total_vms += n_vms
        del buf
        gc.collect()

        print(f"  {stem:<43} {load_s:>8.2f} {pkl_mb:>8.1f} {n_vms:>8,} {n_nodes:>8,}")
        trace_records.append((stem, td, load_s, pkl_mb, n_vms, n_nodes))

    print(f"\n  Total VMs across all traces: {total_vms:,}")
    single_vms = trace_records[0][4]
    print(f"  Scale vs. single trace ('{trace_records[0][0]}'): "
          f"{total_vms / single_vms:.1f}x\n")

    # ------------------------------------------------------------------ #
    # Q4b: precompute time per seed, per trace                            #
    # ------------------------------------------------------------------ #
    print("--- Q4b: Precompute time per seed ---")
    print(f"{'Trace':<45} {'ms/seed':>10} {'total(s)':>10}")
    print("-" * 70)

    for stem, td, *_ in trace_records:
        t0 = time.perf_counter()
        precompute_pod_events(td, M, SEEDS)
        elapsed = time.perf_counter() - t0
        ms_per_seed = elapsed / len(SEEDS) * 1000
        print(f"  {stem:<43} {ms_per_seed:>10.1f} {elapsed:>10.2f}")

    print()

    # ------------------------------------------------------------------ #
    # Q4c: dealloc_events memory per env                                  #
    # (pod_dur x num_mhd x 8 bytes, worst-case pod_dur across seeds)     #
    # ------------------------------------------------------------------ #
    print("--- Q4c: dealloc_events matrix memory per env ---")
    print(f"{'Trace':<45} {'med pod_dur':>12} {'p95 pod_dur':>12} {'med MB':>8} {'p95 MB':>8}")
    print("-" * 85)

    num_mhd = len(M[0])
    for stem, td, *_ in trace_records:
        cache = precompute_pod_events(td, M, list(range(30)))
        pod_durs = [v[1] for v in cache.values() if v[1] > 0]
        if not pod_durs:
            print(f"  {stem:<43}  (no valid pods)")
            continue
        med_dur = int(np.median(pod_durs))
        p95_dur = int(np.percentile(pod_durs, 95))
        med_mb  = med_dur * num_mhd * 8 / 1e6
        p95_mb  = p95_dur * num_mhd * 8 / 1e6
        print(f"  {stem:<43} {med_dur:>12,} {p95_dur:>12,} {med_mb:>8.3f} {p95_mb:>8.3f}")

    # With N=32 envs each holding one dealloc_events matrix:
    sample_cache = precompute_pod_events(trace_records[0][1], M, list(range(30)))
    sample_durs  = [v[1] for v in sample_cache.values() if v[1] > 0]
    p95_dur_ref  = int(np.percentile(sample_durs, 95)) if sample_durs else 4000
    total_mb_32  = N_ENVS * p95_dur_ref * num_mhd * 8 / 1e6
    print(f"\n  With N={N_ENVS} envs (p95 pod_dur from '{trace_records[0][0]}'):")
    print(f"  Total dealloc_events RAM: {total_mb_32:.1f} MB\n")

    # ------------------------------------------------------------------ #
    # Q4d: SubprocVecEnv throughput — single trace vs. all traces         #
    # Each worker randomly picks a trace per episode (simulated by        #
    # assigning worker i to trace i % n_traces).                          #
    # ------------------------------------------------------------------ #
    print("--- Q4d: SubprocVecEnv throughput (N=32) ---")

    def _sps(env_fns, label):
        venv = SubprocVecEnv(env_fns, start_method="fork")
        kw = dict(
            device=device,
            policy_kwargs=dict(net_arch=[128, 128]),
            train_freq=N_ENVS,
            gradient_steps=N_ENVS,
            batch_size=512,
            buffer_size=100_000,
            learning_starts=200,
        )
        sps = _bench_throughput(venv, kw, n_steps=3_000)
        venv.close()
        print(f"  {label:<50} {sps:>8.0f} steps/sec")
        return sps

    # Single trace (baseline, matches Q3)
    single_fns = [make_env(trace_records[0][1], M, seed=i) for i in range(N_ENVS)]
    sps_single = _sps(single_fns, f"N=32, single trace ({trace_records[0][0][:20]})")

    # All traces — worker i uses trace i % n_traces
    n_traces = len(trace_records)
    multi_fns = [
        make_env(trace_records[i % n_traces][1], M, seed=i)
        for i in range(N_ENVS)
    ]
    sps_multi = _sps(multi_fns, f"N=32, all {n_traces} traces round-robin")

    print(f"\n  IPC overhead from larger trace data: "
          f"{(1 - sps_multi/sps_single)*100:+.1f}% vs single-trace")

    print("\nQ4 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-test", action="store_true",
                    help="Run Q4: 60x scale stress test (loads all traces, slow)")
    args = ap.parse_args()

    main()
    if args.scale_test:
        scale_stress_test()
