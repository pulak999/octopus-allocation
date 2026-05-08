"""Tests for Chunk 1: flat MPD arrays and vectorized env hot paths."""

import numpy as np
import pytest

from octopus.data import to_arrays
from octopus.env import OctopusMemPoolEnv, _INITIAL_MPD_SLOTS


# ---------------------------------------------------------------------------
# Minimal env fixture
# ---------------------------------------------------------------------------

def _make_env(reward_variant="A", seed=0):
    """Create a minimal 2-host / 2-MPD env using synthetic trace data."""
    # 2 hosts, 2 MPDs, fully connected
    M = np.array([[1, 1], [1, 1]], dtype=np.int32)

    # Synthetic trace: 2 nodes, 4 VMs each
    n_vms = 8
    vm_start = np.array([0, 300, 600, 900,  0,  300,  600,  900], dtype=np.int64)
    vm_end   = np.array([3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], dtype=np.int64)
    vm_mem   = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    node_ids    = np.array([0, 1], dtype=np.int64)
    node_dram   = np.array([200.0, 200.0], dtype=np.float64)
    node_offsets = np.array([0, 4, 8], dtype=np.int64)
    vm_ptrs     = np.arange(n_vms, dtype=np.int64)

    from octopus.data import TraceArrays
    ta = TraceArrays(
        vm_start=vm_start,
        vm_end=vm_end,
        vm_mem=vm_mem,
        node_ids=node_ids,
        node_dram=node_dram,
        node_offsets=node_offsets,
        vm_ptrs=vm_ptrs,
    )
    env = OctopusMemPoolEnv(
        trace_arrays=ta,
        M=M,
        seed=seed,
        reward_variant=reward_variant,
        lookahead_window=200,
    )
    env.reset()
    return env


# ---------------------------------------------------------------------------
# 1. Flat array initialisation
# ---------------------------------------------------------------------------

def test_flat_arrays_initialized_at_reset():
    env = _make_env()
    assert hasattr(env, "_mpd_dt")
    assert hasattr(env, "_mpd_mem")
    assert hasattr(env, "_mpd_n")
    assert env._mpd_dt.shape == (env.num_mhd, _INITIAL_MPD_SLOTS)
    assert env._mpd_mem.shape == (env.num_mhd, _INITIAL_MPD_SLOTS)
    assert env._mpd_n.shape == (env.num_mhd,)
    # All counts start at 0
    assert (env._mpd_n == 0).all()
    # dt sentinel is inf, mem sentinel is 0
    assert np.all(np.isinf(env._mpd_dt))
    assert np.all(env._mpd_mem == 0.0)


def test_no_mpd_vm_allocs_attribute():
    env = _make_env()
    assert not hasattr(env, "mpd_vm_allocs"), \
        "mpd_vm_allocs list-of-lists must be removed"


def test_no_cached_D_j_attribute():
    env = _make_env()
    assert not hasattr(env, "_cached_D_j"), \
        "_cached_D_j dead field must be removed"


# ---------------------------------------------------------------------------
# 2. Slot write after allocation
# ---------------------------------------------------------------------------

def test_slot_written_after_step():
    env = _make_env(reward_variant="A")
    n_before = env._mpd_n.copy()
    env.step(env.action_space.sample())
    # At least one MPD should have gained a VM
    assert (env._mpd_n >= n_before).all()
    added = env._mpd_n - n_before
    for j in range(env.num_mhd):
        if added[j] > 0:
            slot = added[j] - 1
            assert np.isfinite(env._mpd_dt[j, slot])
            assert env._mpd_mem[j, slot] > 0.0


# ---------------------------------------------------------------------------
# 3. _compute_D_j matches reference Python loop
# ---------------------------------------------------------------------------

def _reference_D_j(env, tick):
    """Reference: old Python nested loop over mpd_vm_allocs equivalent."""
    D = np.zeros(env.num_mhd, dtype=np.float64)
    W = env.lookahead_window
    t_plus_W = tick + W
    for j in range(env.num_mhd):
        n = env._mpd_n[j]
        for s in range(n):
            dt = env._mpd_dt[j, s]
            mem = env._mpd_mem[j, s]
            if dt <= t_plus_W:
                D[j] += mem * (1.0 - (dt - tick) / W)
    return D


@pytest.mark.parametrize("n_steps", [1, 5, 20])
def test_compute_D_j_matches_reference(n_steps):
    env = _make_env(reward_variant="A")
    for _ in range(n_steps):
        if env.event_idx >= len(env.events):
            break
        tick = env.events[env.event_idx][0]
        D_vec = env._compute_D_j(tick)
        D_ref = _reference_D_j(env, tick)
        np.testing.assert_allclose(D_vec, D_ref, rtol=1e-12, atol=1e-12)
        env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# 4. _get_obs D_j / S_j match reference
# ---------------------------------------------------------------------------

def _reference_D_S_j(env, tick):
    W = env.lookahead_window
    t_plus_W = tick + W
    D = np.zeros(env.num_mhd, dtype=np.float64)
    S = np.zeros(env.num_mhd, dtype=np.float64)
    for j in range(env.num_mhd):
        n = env._mpd_n[j]
        for s in range(n):
            dt = env._mpd_dt[j, s]
            mem = env._mpd_mem[j, s]
            if dt <= t_plus_W:
                D[j] += mem * (1.0 - (dt - tick) / W)
            else:
                S[j] += mem
    return D, S


def test_get_obs_D_S_j_match_reference():
    """After several steps, obs D_j/S_j components must match reference."""
    env = _make_env(reward_variant="A")
    for _ in range(10):
        if env.event_idx >= len(env.events):
            break
        tick = env.events[env.event_idx][0]
        norm = env.pod_dram if env.pod_dram > 0 else 1.0
        mhd_list = env.host_to_mhds[env.events[env.event_idx][1]]

        D_ref, S_ref = _reference_D_S_j(env, tick)
        obs = env._get_obs()

        for k, mhd in enumerate(mhd_list):
            base = k * 6
            assert obs[base + 1] == pytest.approx(D_ref[mhd] / norm, abs=1e-6)
            assert obs[base + 2] == pytest.approx(S_ref[mhd] / norm, abs=1e-6)

        env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# 5. _process_departures_through compacts correctly
# ---------------------------------------------------------------------------

def test_departures_compact_expired_vms():
    env = _make_env(reward_variant="A")

    # Manually inject two VMs into MPD 0: one expiring at tick 5, one at tick 1000
    env._mpd_dt[0, 0] = 5.0
    env._mpd_mem[0, 0] = 20.0
    env._mpd_dt[0, 1] = 1000.0
    env._mpd_mem[0, 1] = 30.0
    env._mpd_n[0] = 2

    env._process_departures_through(10)

    assert env._mpd_n[0] == 1
    assert env._mpd_dt[0, 0] == pytest.approx(1000.0)
    assert env._mpd_mem[0, 0] == pytest.approx(30.0)
    assert np.isinf(env._mpd_dt[0, 1])
    assert env._mpd_mem[0, 1] == pytest.approx(0.0)


def test_departures_keeps_all_when_none_expired():
    env = _make_env()
    env._mpd_dt[0, 0] = 500.0
    env._mpd_mem[0, 0] = 10.0
    env._mpd_n[0] = 1
    env._process_departures_through(5)
    assert env._mpd_n[0] == 1
    assert env._mpd_dt[0, 0] == pytest.approx(500.0)


def test_departures_clears_all_when_all_expired():
    env = _make_env()
    env._mpd_dt[0, 0] = 3.0
    env._mpd_dt[0, 1] = 4.0
    env._mpd_mem[0, 0] = 10.0
    env._mpd_mem[0, 1] = 20.0
    env._mpd_n[0] = 2
    env._process_departures_through(10)
    assert env._mpd_n[0] == 0
    assert np.isinf(env._mpd_dt[0, 0])
    assert env._mpd_mem[0, 0] == 0.0


# ---------------------------------------------------------------------------
# 6. _ensure_mpd_capacity doubles arrays without data loss
# ---------------------------------------------------------------------------

def test_capacity_growth():
    env = _make_env()
    original_cap = env._mpd_capacity

    # Fill MPD 0 to capacity
    env._mpd_n[0] = original_cap
    env._mpd_dt[0, :original_cap] = np.arange(original_cap, dtype=np.float64) + 100.0
    env._mpd_mem[0, :original_cap] = np.arange(original_cap, dtype=np.float64) + 1.0

    env._ensure_mpd_capacity(0)

    assert env._mpd_capacity == original_cap * 2
    assert env._mpd_dt.shape[1] == original_cap * 2
    assert env._mpd_mem.shape[1] == original_cap * 2

    # Existing data preserved
    np.testing.assert_array_equal(
        env._mpd_dt[0, :original_cap],
        np.arange(original_cap, dtype=np.float64) + 100.0
    )
    np.testing.assert_array_equal(
        env._mpd_mem[0, :original_cap],
        np.arange(original_cap, dtype=np.float64) + 1.0
    )
    # New slots initialised correctly
    assert np.all(np.isinf(env._mpd_dt[0, original_cap:]))
    assert np.all(env._mpd_mem[0, original_cap:] == 0.0)


def test_no_growth_when_not_full():
    env = _make_env()
    cap_before = env._mpd_capacity
    env._mpd_n[0] = cap_before - 1
    env._ensure_mpd_capacity(0)
    assert env._mpd_capacity == cap_before


# ---------------------------------------------------------------------------
# 7. Smoke: 1000 steps with reward_variant=A, no errors, rewards finite
# ---------------------------------------------------------------------------

def test_smoke_1000_steps_reward_A():
    env = _make_env(reward_variant="A", seed=42)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape

    total_steps = 0
    episodes = 0
    while total_steps < 1000:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        assert np.isfinite(reward), f"non-finite reward at step {total_steps}"
        assert obs.shape == env.observation_space.shape
        total_steps += 1
        if done:
            obs, _ = env.reset()
            episodes += 1

    assert total_steps == 1000


def test_smoke_1000_steps_reward_B():
    env = _make_env(reward_variant="B", seed=7)
    env.reset()
    for i in range(1000):
        _, reward, done, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward), f"non-finite reward at step {i}"
        if done:
            env.reset()
