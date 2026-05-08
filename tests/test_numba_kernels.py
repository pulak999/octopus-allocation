"""Tests for Chunk 3: Numba JIT kernels in octopus/kernels.py."""

import numpy as np
import pytest

from octopus.kernels import compute_D_j, compute_D_S_j


# ---------------------------------------------------------------------------
# Reference (pure Python, equivalent to old _compute_D_j logic)
# ---------------------------------------------------------------------------

def _ref_D_j(mpd_dt, mpd_mem, mpd_n, tick, W):
    num_mhd = mpd_n.shape[0]
    D = np.zeros(num_mhd, dtype=np.float64)
    t_plus_W = tick + W
    for j in range(num_mhd):
        for s in range(mpd_n[j]):
            dt = mpd_dt[j, s]
            mem = mpd_mem[j, s]
            if dt <= t_plus_W:
                D[j] += mem * (1.0 - (dt - tick) / W)
    return D


def _ref_D_S_j(mpd_dt, mpd_mem, mpd_n, tick, W):
    num_mhd = mpd_n.shape[0]
    D = np.zeros(num_mhd, dtype=np.float64)
    S = np.zeros(num_mhd, dtype=np.float64)
    t_plus_W = tick + W
    for j in range(num_mhd):
        for s in range(mpd_n[j]):
            dt = mpd_dt[j, s]
            mem = mpd_mem[j, s]
            if dt <= t_plus_W:
                D[j] += mem * (1.0 - (dt - tick) / W)
            else:
                S[j] += mem
    return D, S


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_arrays(num_mhd=6, capacity=64, seed=0):
    rng = np.random.default_rng(seed)
    mpd_dt  = np.full((num_mhd, capacity), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((num_mhd, capacity), dtype=np.float64)
    mpd_n   = np.zeros(num_mhd, dtype=np.int32)
    # Populate some slots
    for j in range(num_mhd):
        n = int(rng.integers(0, 30))
        mpd_n[j] = n
        mpd_dt[j, :n]  = rng.integers(50, 500, size=n).astype(np.float64)
        mpd_mem[j, :n] = rng.uniform(1.0, 50.0, size=n)
    return mpd_dt, mpd_mem, mpd_n


# ---------------------------------------------------------------------------
# 1. compute_D_j correctness
# ---------------------------------------------------------------------------

def test_compute_D_j_empty_arrays():
    mpd_dt  = np.full((4, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((4, 64), dtype=np.float64)
    mpd_n   = np.zeros(4, dtype=np.int32)
    D = compute_D_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    np.testing.assert_array_equal(D, np.zeros(4))


def test_compute_D_j_single_vm_at_window_midpoint():
    """VM at dt = tick + W/2 contributes mem * 0.5."""
    mpd_dt  = np.full((2, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((2, 64), dtype=np.float64)
    mpd_n   = np.zeros(2, dtype=np.int32)

    mpd_dt[0, 0] = 100.0   # dt = tick(0) + W/2 with W=200
    mpd_mem[0, 0] = 8.0
    mpd_n[0] = 1

    D = compute_D_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    assert D[0] == pytest.approx(8.0 * 0.5)
    assert D[1] == pytest.approx(0.0)


def test_compute_D_j_beyond_window_excluded():
    mpd_dt  = np.full((2, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((2, 64), dtype=np.float64)
    mpd_n   = np.zeros(2, dtype=np.int32)

    mpd_dt[0, 0] = 201.0   # dt = tick(0) + W(200) + 1 → beyond window
    mpd_mem[0, 0] = 10.0
    mpd_n[0] = 1

    D = compute_D_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    assert D[0] == pytest.approx(0.0)


def test_compute_D_j_at_window_boundary_included():
    mpd_dt  = np.full((2, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((2, 64), dtype=np.float64)
    mpd_n   = np.zeros(2, dtype=np.int32)

    mpd_dt[0, 0] = 200.0   # dt == tick(0) + W(200) → exactly on boundary
    mpd_mem[0, 0] = 10.0
    mpd_n[0] = 1

    D = compute_D_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    assert D[0] == pytest.approx(10.0 * 0.0)  # weight = 1 - W/W = 0


@pytest.mark.parametrize("seed", range(10))
def test_compute_D_j_matches_reference(seed):
    mpd_dt, mpd_mem, mpd_n = _make_arrays(num_mhd=6, capacity=64, seed=seed)
    tick, W = 42, 200
    D_nb  = compute_D_j(mpd_dt, mpd_mem, mpd_n, tick, W)
    D_ref = _ref_D_j(mpd_dt, mpd_mem, mpd_n, tick, W)
    np.testing.assert_allclose(D_nb, D_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. compute_D_S_j correctness
# ---------------------------------------------------------------------------

def test_compute_D_S_j_empty_arrays():
    mpd_dt  = np.full((4, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((4, 64), dtype=np.float64)
    mpd_n   = np.zeros(4, dtype=np.int32)
    D, S = compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    np.testing.assert_array_equal(D, np.zeros(4))
    np.testing.assert_array_equal(S, np.zeros(4))


def test_compute_D_S_j_near_goes_to_D_far_goes_to_S():
    mpd_dt  = np.full((2, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((2, 64), dtype=np.float64)
    mpd_n   = np.zeros(2, dtype=np.int32)

    mpd_dt[0, 0] = 50.0    # near (dt <= 200)
    mpd_mem[0, 0] = 4.0
    mpd_dt[0, 1] = 300.0   # far (dt > 200)
    mpd_mem[0, 1] = 7.0
    mpd_n[0] = 2

    D, S = compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    assert D[0] == pytest.approx(4.0 * (1.0 - 50.0 / 200.0))
    assert S[0] == pytest.approx(7.0)
    assert D[1] == pytest.approx(0.0)
    assert S[1] == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(10))
def test_compute_D_S_j_matches_reference(seed):
    mpd_dt, mpd_mem, mpd_n = _make_arrays(num_mhd=6, capacity=64, seed=seed)
    tick, W = 37, 200
    D_nb, S_nb   = compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick, W)
    D_ref, S_ref = _ref_D_S_j(mpd_dt, mpd_mem, mpd_n, tick, W)
    np.testing.assert_allclose(D_nb,  D_ref,  rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(S_nb,  S_ref,  rtol=1e-12, atol=1e-12)


def test_D_S_partition_total_memory():
    """D_j[j] + S_j[j] = sum of all mem on MPD j (up to weighting; only exact when W→∞)."""
    mpd_dt  = np.full((3, 64), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((3, 64), dtype=np.float64)
    mpd_n   = np.zeros(3, dtype=np.int32)

    # All VMs far from window → D=0, S=total mem
    mpd_dt[1, 0] = 9999.0
    mpd_dt[1, 1] = 9998.0
    mpd_mem[1, 0] = 10.0
    mpd_mem[1, 1] = 20.0
    mpd_n[1] = 2

    D, S = compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick=0, W=200)
    assert D[1] == pytest.approx(0.0)
    assert S[1] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 3. End-to-end: env uses kernels and produces correct output
# ---------------------------------------------------------------------------

def test_env_uses_kernels_correctly():
    """After Chunk 3, env._compute_D_j and _get_obs must still match reference."""
    import numpy as np
    from octopus.data import TraceArrays
    from octopus.env import OctopusMemPoolEnv

    M = np.array([[1, 1], [1, 1]], dtype=np.int32)
    n_vms = 8
    vm_start = np.zeros(n_vms, dtype=np.int64)
    vm_end   = np.full(n_vms, 3600 * 24 * 14, dtype=np.int64)
    vm_mem   = np.full(n_vms, 10.0, dtype=np.float64)
    ta = TraceArrays(
        vm_start=vm_start,
        vm_end=vm_end,
        vm_mem=vm_mem,
        node_ids=np.array([0, 1], dtype=np.int64),
        node_dram=np.array([200.0, 200.0], dtype=np.float64),
        node_offsets=np.array([0, 4, 8], dtype=np.int64),
        vm_ptrs=np.arange(n_vms, dtype=np.int64),
    )
    env = OctopusMemPoolEnv(
        trace_arrays=ta, M=M, seed=0, reward_variant="A", lookahead_window=200
    )
    env.reset()

    for _ in range(50):
        if env.event_idx >= len(env.events):
            break
        tick = env.events[env.event_idx][0]
        D_env = env._compute_D_j(tick)
        D_ref = _ref_D_j(env._mpd_dt, env._mpd_mem, env._mpd_n, tick, env.lookahead_window)
        np.testing.assert_allclose(D_env, D_ref, rtol=1e-12, atol=1e-12)
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if done:
            break
