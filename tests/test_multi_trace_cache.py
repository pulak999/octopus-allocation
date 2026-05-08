"""Tests for Chunk 2: multi-trace precompute cache correctness."""

import numpy as np
import pytest

from octopus.data import TraceArrays, precompute_pod_events_arrays
from octopus.env import OctopusMemPoolEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace_arrays(n_nodes=4, n_vms_per_node=6, seed=0):
    """Synthetic TraceArrays: n_nodes nodes, n_vms_per_node VMs each."""
    rng = np.random.default_rng(seed)
    n_vms = n_nodes * n_vms_per_node

    # VMs span ~14 days in 5-min ticks from epoch
    BASE = 0
    vm_start = rng.integers(BASE, BASE + 4000 * 300, size=n_vms, dtype=np.int64)
    vm_end   = vm_start + rng.integers(300, 10_000 * 300, size=n_vms, dtype=np.int64)
    vm_mem   = rng.uniform(1.0, 50.0, size=n_vms).astype(np.float64)

    node_ids    = np.arange(n_nodes, dtype=np.int64)
    node_dram   = np.full(n_nodes, 200.0, dtype=np.float64)
    node_offsets = np.arange(0, n_nodes * n_vms_per_node + 1,
                             n_vms_per_node, dtype=np.int64)
    vm_ptrs     = np.arange(n_vms, dtype=np.int64)

    return TraceArrays(
        vm_start=vm_start,
        vm_end=vm_end,
        vm_mem=vm_mem,
        node_ids=node_ids,
        node_dram=node_dram,
        node_offsets=node_offsets,
        vm_ptrs=vm_ptrs,
    )


def _make_env(ta, M, seed=0):
    return OctopusMemPoolEnv(trace_arrays=ta, M=M, seed=seed, reward_variant="A")


# ---------------------------------------------------------------------------
# 1. Cache correctness: events match live _build_events for all seeds
# ---------------------------------------------------------------------------

def test_cached_events_match_live_build():
    """precompute_pod_events_arrays must produce events identical to _build_events."""
    M = [[1, 1], [1, 1], [1, 0], [0, 1]]   # 4 hosts, 2 MPDs
    ta = _make_trace_arrays(n_nodes=4, seed=1)
    seeds = list(range(20))

    cache = precompute_pod_events_arrays(ta, M, seeds)

    env = _make_env(ta, M, seed=0)
    for seed in seeds:
        env._generate_pod(seed)
        env._build_events()
        live_events = list(env.events)
        live_pod_dur = env.pod_dur
        live_pod_dram = env.pod_dram

        assert seed in cache, f"seed {seed} missing from cache"
        c_events, c_pod_dur, c_pod_dram, _, _ = cache[seed]

        assert c_events == live_events, \
            f"seed {seed}: cached events differ from live build"
        assert c_pod_dur == live_pod_dur, \
            f"seed {seed}: cached pod_dur {c_pod_dur} != live {live_pod_dur}"
        assert c_pod_dram == pytest.approx(live_pod_dram), \
            f"seed {seed}: cached pod_dram {c_pod_dram} != live {live_pod_dram}"


# ---------------------------------------------------------------------------
# 2. _switch_trace atomically swaps both trace_arrays and _precomputed_events
# ---------------------------------------------------------------------------

def test_switch_trace_swaps_both_fields():
    M = [[1, 1], [1, 1]]
    ta1 = _make_trace_arrays(n_nodes=2, seed=10)
    ta2 = _make_trace_arrays(n_nodes=2, seed=20)
    cache2 = {0: ([], 0, 0.0, None, 0)}

    env = _make_env(ta1, M, seed=0)
    assert env.trace_arrays is ta1
    assert env._precomputed_events is None

    env._switch_trace(ta2, cache2)
    assert env.trace_arrays is ta2
    assert env._precomputed_events is cache2


def test_switch_trace_clears_cache_when_none():
    M = [[1, 1], [1, 1]]
    ta1 = _make_trace_arrays(n_nodes=2, seed=10)
    ta2 = _make_trace_arrays(n_nodes=2, seed=20)
    cache1 = {0: ([], 0, 0.0, None, 0)}

    env = _make_env(ta1, M, seed=0)
    env._switch_trace(ta1, cache1)
    assert env._precomputed_events is cache1

    env._switch_trace(ta2, None)
    assert env._precomputed_events is None


# ---------------------------------------------------------------------------
# 3. Reset uses cached events when cache is present
# ---------------------------------------------------------------------------

def test_reset_uses_cache_when_available():
    """With cache present, reset() must use the cached events (not rebuild)."""
    M = [[1, 1], [1, 1]]
    ta = _make_trace_arrays(n_nodes=2, seed=5)
    seeds = list(range(10))
    cache = precompute_pod_events_arrays(ta, M, seeds)

    env = OctopusMemPoolEnv(
        trace_arrays=ta, M=M, seed=0,
        precomputed_events=cache,
        reward_variant="A",
    )
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        c_events, _, _, _, _ = cache[seed]
        assert env.events == c_events, \
            f"seed {seed}: env.events diverged from cache after reset"


def test_reset_falls_through_on_cache_miss():
    """Seeds not in the cache must still produce valid events via _build_events."""
    M = [[1, 1], [1, 1]]
    ta = _make_trace_arrays(n_nodes=2, seed=7)
    # Only precompute seeds 0-4; env will use seed 99 which is not cached
    cache = precompute_pod_events_arrays(ta, M, range(5))

    env = OctopusMemPoolEnv(
        trace_arrays=ta, M=M, seed=99,
        precomputed_events=cache,
        reward_variant="A",
    )
    obs, info = env.reset()
    assert isinstance(env.events, list)
    assert obs.shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# 4. Multi-trace + cache: events come from the correct per-trace cache
# ---------------------------------------------------------------------------

def test_multi_trace_uses_per_trace_cache():
    """With trace_pool_events, _switch_trace must swap the matching cache."""
    from octopus.augmentation import AugmentationConfig

    M = [[1, 1], [1, 1]]
    ta0 = _make_trace_arrays(n_nodes=2, seed=0)
    ta1 = _make_trace_arrays(n_nodes=2, seed=1)
    seeds = list(range(20))
    cache0 = precompute_pod_events_arrays(ta0, M, seeds)
    cache1 = precompute_pod_events_arrays(ta1, M, seeds)

    aug_config = AugmentationConfig(
        multi_trace=True, enabled=False,  # no scale/noise, just pool switching
    )
    # enabled=False means scale=1 etc. but multi_trace switch still fires in reset()
    # We need enabled=True for the reset() branch to execute:
    aug_config = AugmentationConfig(
        scale_range=(1.0, 1.0),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=True,
    )

    env = OctopusMemPoolEnv(
        trace_arrays=ta0,
        M=M,
        seed=0,
        reward_variant="A",
        aug_config=aug_config,
        trace_pool=[ta0, ta1],
        trace_pool_events=[cache0, cache1],
    )

    # After each reset, env._precomputed_events should be whichever cache
    # matches the chosen trace (ta0→cache0 or ta1→cache1).
    for _ in range(20):
        env.reset()
        if env.trace_arrays is ta0:
            assert env._precomputed_events is cache0
        else:
            assert env._precomputed_events is cache1


# ---------------------------------------------------------------------------
# 5. Smoke: 100 resets with multi-trace + cache, no errors, obs finite
# ---------------------------------------------------------------------------

def test_smoke_multi_trace_cache():
    from octopus.augmentation import AugmentationConfig

    M = [[1, 1], [1, 1]]
    ta0 = _make_trace_arrays(n_nodes=2, seed=42)
    ta1 = _make_trace_arrays(n_nodes=2, seed=43)
    seeds = list(range(150))
    cache0 = precompute_pod_events_arrays(ta0, M, seeds)
    cache1 = precompute_pod_events_arrays(ta1, M, seeds)

    aug_config = AugmentationConfig(
        scale_range=(1.0, 1.0),
        scale_distribution="uniform",
        memory_noise_sigma=0.0,
        arrival_jitter_ticks=0,
        lifetime_noise_frac=0.0,
        link_failure_ratio=0.0,
        multi_trace=True,
        enabled=True,
    )

    env = OctopusMemPoolEnv(
        trace_arrays=ta0,
        M=M,
        seed=0,
        reward_variant="A",
        aug_config=aug_config,
        trace_pool=[ta0, ta1],
        trace_pool_events=[cache0, cache1],
    )

    for _ in range(100):
        obs, info = env.reset()
        assert np.all(np.isfinite(obs))
        # Take a few steps
        for _ in range(5):
            if env.event_idx >= len(env.events):
                break
            _, reward, done, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if done:
                break
