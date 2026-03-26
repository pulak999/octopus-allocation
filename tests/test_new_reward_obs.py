"""Tests for plan-v2 Task 4: new state space, topology precomputation, and reward variants."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: minimal topology + env construction without trace files
# ---------------------------------------------------------------------------

def _make_minimal_env(reward_variant="current", **kwargs):
    """Create a minimal OctopusMemPoolEnv with synthetic trace data (no files)."""
    from octopus.env import OctopusMemPoolEnv
    import datetime as dt

    # 2 hosts, 3 MPDs
    # Host 0 → MPDs 0, 1
    # Host 1 → MPDs 1, 2
    M = [[1, 1, 0],
         [0, 1, 1]]

    # Minimal VM data: 2 VMs, one per host
    base = dt.datetime(2024, 1, 1, 0, 0)

    class FakeVM:
        def __init__(self, start_offset_min, end_offset_min, mem_gb):
            self.start_time = base + dt.timedelta(minutes=start_offset_min)
            self.end_time = base + dt.timedelta(minutes=end_offset_min)
            self.rss = [0, mem_gb, 0, 0]

    all_vms = {
        "vm0": FakeVM(0, 60, 4.0),   # host 0, 4 GB, lasts 60 min = 12 ticks
        "vm1": FakeVM(0, 30, 2.0),   # host 1, 2 GB, lasts 30 min = 6 ticks
    }
    node_to_vms = {0: ["vm0"], 1: ["vm1"]}
    node_to_machine = {0: "mtype", 1: "mtype"}
    machine_sz = {"mtype": [0, 100.0, 0, 0]}  # 100 GB DRAM (mem_idx=1)

    env = OctopusMemPoolEnv(
        all_vms=all_vms,
        node_to_vms=node_to_vms,
        node_to_machine=node_to_machine,
        machine_sz=machine_sz,
        M=M,
        seed=0,
        reward_variant=reward_variant,
        skip_hotfix=True,
        **kwargs,
    )
    return env, M


# ---------------------------------------------------------------------------
# 4b — mhd_to_hosts and Q_j
# ---------------------------------------------------------------------------

def test_mhd_to_hosts_correct():
    env, M = _make_minimal_env()
    # Host 0 → MPDs 0, 1  →  MPD 0 has [host 0], MPD 1 has [host 0, host 1]
    assert env.mhd_to_hosts[0] == [0]
    assert set(env.mhd_to_hosts[1]) == {0, 1}
    assert env.mhd_to_hosts[2] == [1]


def test_Q_j_values():
    env, M = _make_minimal_env()
    # Host 0 has degree 2, host 1 has degree 2
    # Q_j[0] = mean(1/deg(h) for h in [0]) = 1/2 = 0.5
    # Q_j[1] = mean(1/2, 1/2) = 0.5
    # Q_j[2] = mean(1/2) = 0.5
    assert env.Q_j[0] == pytest.approx(0.5)
    assert env.Q_j[1] == pytest.approx(0.5)
    assert env.Q_j[2] == pytest.approx(0.5)


def test_Q_j_asymmetric_topology():
    """Host with degree 1 should increase Q_j for its MPD."""
    from octopus.env import OctopusMemPoolEnv
    import datetime as dt

    # 2 hosts, 2 MPDs
    # Host 0 → MPD 0 only (degree 1 — no alternatives)
    # Host 1 → MPDs 0, 1 (degree 2)
    M = [[1, 0], [1, 1]]

    base = dt.datetime(2024, 1, 1)
    class FakeVM:
        def __init__(self):
            self.start_time = base
            self.end_time = base + dt.timedelta(minutes=60)
            self.rss = [0, 1.0, 0, 0]

    env = OctopusMemPoolEnv(
        all_vms={"v0": FakeVM()},
        node_to_vms={0: ["v0"], 1: []},
        node_to_machine={0: "t", 1: "t"},
        machine_sz={"t": [0, 100.0, 0, 0]},
        M=M,
        seed=0,
        skip_hotfix=True,
    )
    # Q_j[0]: hosts connected = [0, 1]; deg(0)=1, deg(1)=2
    #         mean(1/1, 1/2) = mean(1.0, 0.5) = 0.75
    assert env.Q_j[0] == pytest.approx(0.75)
    # Q_j[1]: hosts connected = [1]; deg(1)=2
    #         mean(1/2) = 0.5
    assert env.Q_j[1] == pytest.approx(0.5)


def test_recompute_topology_derived_after_link_failure():
    """After a link failure, Q_j should reflect the new topology."""
    env, M = _make_minimal_env()
    # Simulate removing link host0 → MPD1 by directly modifying host_to_mhds
    env.host_to_mhds[0] = [0]  # host 0 now only connects to MPD 0
    env._recompute_topology_derived()
    # MPD 1 now only has host 1
    assert env.mhd_to_hosts[1] == [1]
    # Q_j[1] = mean(1/deg(1)) = mean(1/2) = 0.5 (host 1 still has degree 2 in host_to_mhds)
    # but host 1's host_to_mhds is still [1, 2]
    assert env.Q_j[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4a — mpd_vm_allocs and cur_host_cxl_load tracking
# ---------------------------------------------------------------------------

def test_mpd_vm_allocs_populated_on_step():
    """After stepping through a VM arrival, mpd_vm_allocs should have entries."""
    env, _ = _make_minimal_env(reward_variant="A")
    obs, info = env.reset(seed=0)
    assert sum(len(env.mpd_vm_allocs[j]) for j in range(env.num_mhd)) == 0
    action = env.action_space.sample()
    env.step(action)
    total_entries = sum(len(env.mpd_vm_allocs[j]) for j in range(env.num_mhd))
    assert total_entries > 0


def test_cur_host_cxl_load_updated():
    """After stepping, cur_host_cxl_load should be non-zero for the host that just placed."""
    env, _ = _make_minimal_env(reward_variant="A")
    env.reset(seed=0)
    assert float(env.cur_host_cxl_load.sum()) == pytest.approx(0.0)
    env.step(env.action_space.sample())
    assert float(env.cur_host_cxl_load.sum()) > 0.0


def test_mpd_vm_allocs_purged_on_departure():
    """VM entries in mpd_vm_allocs should be removed once their dealloc_tick is processed."""
    env, _ = _make_minimal_env(reward_variant="A", lookahead_window=200)
    env.reset(seed=0)
    # Step through all events
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
    # Any remaining entries must have dealloc_tick > _last_depart_tick
    # (VMs still "active" past the last processed departure tick)
    processed_up_to = env._last_depart_tick
    for j in range(env.num_mhd):
        for dt, m in env.mpd_vm_allocs[j]:
            assert dt > processed_up_to, (
                f"MPD {j} has VM with dealloc_tick={dt} <= processed_up_to={processed_up_to}"
            )


# ---------------------------------------------------------------------------
# 4d — observation space shape
# ---------------------------------------------------------------------------

def test_obs_shape_current():
    env, _ = _make_minimal_env(reward_variant="current")
    obs, _ = env.reset(seed=0)
    # 2*max_degree + 4 = 2*2 + 4 = 8 (M is 2x3, max_degree=2)
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == env.max_degree * 2 + 4


def test_obs_space_shape_new_variants():
    """observation_space declares the right shape for new variants (obs not yet built)."""
    for variant in ("A", "B"):
        env, _ = _make_minimal_env(reward_variant=variant)
        # 6*max_degree + 2 (M is 2x3, max_degree=2 → 14)
        assert env.observation_space.shape[0] == env.max_degree * 6 + 2


def test_reward_variant_invalid():
    with pytest.raises(AssertionError):
        _make_minimal_env(reward_variant="invalid")


# ---------------------------------------------------------------------------
# 4c — D_j / S_j computation
# ---------------------------------------------------------------------------

def test_compute_D_j_empty():
    """No VMs allocated → D_j = 0 for all MPDs."""
    env, _ = _make_minimal_env(reward_variant="A", lookahead_window=100)
    env.reset(seed=0)
    D_j = env._compute_D_j(tick=0)
    assert np.all(D_j == pytest.approx(0.0))


def test_compute_D_j_exact():
    """D_j formula: VM departing at t+W/2 contributes mem * 0.5."""
    env, _ = _make_minimal_env(reward_variant="A", lookahead_window=10)
    env.reset(seed=0)
    # Manually inject a VM into MPD 0 departing at tick 5 (W=10, tick=0 → weight = 1 - 5/10 = 0.5)
    env.mpd_vm_allocs[0].append((5, 4.0))
    D_j = env._compute_D_j(tick=0)
    assert D_j[0] == pytest.approx(4.0 * 0.5)
    assert D_j[1] == pytest.approx(0.0)


def test_compute_D_j_beyond_window():
    """VM departing beyond W is sticky (not in D_j)."""
    env, _ = _make_minimal_env(reward_variant="A", lookahead_window=10)
    env.reset(seed=0)
    env.mpd_vm_allocs[0].append((15, 4.0))  # dt=15 > tick+W=10
    D_j = env._compute_D_j(tick=0)
    assert D_j[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4c — reward ranges
# ---------------------------------------------------------------------------

def test_reward_A_in_range():
    """Reward A must be in (-1, 0] on every step."""
    env, _ = _make_minimal_env(reward_variant="A", lookahead_window=200)
    env.reset(seed=0)
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        assert reward <= 0.0 + 1e-9, f"Reward A > 0: {reward}"
        assert reward >= -1.0 - 1e-9, f"Reward A < -1: {reward}"


def test_reward_B_in_range():
    """Reward B must be in (-(1+λ), 0] on every step."""
    lam = 0.3
    env, _ = _make_minimal_env(reward_variant="B", lookahead_window=200, reward_lambda=lam)
    env.reset(seed=0)
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        assert reward <= 0.0 + 1e-9, f"Reward B > 0: {reward}"
        assert reward >= -(1.0 + lam) - 1e-9, f"Reward B < -(1+λ): {reward}"


# ---------------------------------------------------------------------------
# 4c — current variant produces identical trajectories (regression)
# ---------------------------------------------------------------------------

def test_current_variant_identical_to_old():
    """reward_variant='current' must produce the same obs/reward as the pre-plan-v2 code."""
    env1, _ = _make_minimal_env(reward_variant="current")
    env2, _ = _make_minimal_env(reward_variant="current")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.allclose(obs1, obs2)
    rng = np.random.default_rng(0)
    done1, done2 = False, False
    while not (done1 or done2):
        action = rng.uniform(-1, 1, size=env1.action_space.shape)
        obs1, r1, done1, _, _ = env1.step(action)
        obs2, r2, done2, _, _ = env2.step(action)
        assert np.allclose(obs1, obs2)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# 4d — new obs contents
# ---------------------------------------------------------------------------

def test_new_obs_shape_from_reset():
    """_get_obs() returns 6*max_degree+2 for new variants."""
    for variant in ("A", "B"):
        env, _ = _make_minimal_env(reward_variant=variant)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.max_degree * 6 + 2,)
        assert obs.shape == env.observation_space.shape


def test_new_obs_mask_slot():
    """Mask slots (index 3 within each MPD block) are 1 for accessible MPDs, 0 for padding."""
    env, _ = _make_minimal_env(reward_variant="A")
    obs, _ = env.reset(seed=0)
    n_acc = len(env.host_to_mhds[env.events[0][1]])
    for k in range(env.max_degree):
        mask_val = obs[k * 6 + 3]
        if k < n_acc:
            assert mask_val == pytest.approx(1.0)
        else:
            assert mask_val == pytest.approx(0.0)


def test_new_obs_global_features():
    """Last two elements of new obs are global_peak/D_pod and vm_mem/D_pod."""
    env, _ = _make_minimal_env(reward_variant="A")
    obs, _ = env.reset(seed=0)
    norm = env.pod_dram
    expected_peak = float(np.max(env.cur_cxl_mem_vec)) / norm
    expected_vm = float(env.events[env.event_idx][2]) / norm
    assert obs[-2] == pytest.approx(expected_peak, abs=1e-6)
    assert obs[-1] == pytest.approx(expected_vm, abs=1e-6)


# ---------------------------------------------------------------------------
# 4g — obs-sync: make_rl_alloc_cb produces identical obs to _get_obs()
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal stub with model.predict() returning a fixed zero action."""
    def __init__(self, n_actions):
        self._n = n_actions

    def predict(self, obs, deterministic=True):
        return np.zeros(self._n, dtype=np.float32), None


def test_obs_sync_initial_step():
    """make_rl_alloc_cb must build the same obs as env._get_obs() at the first VM arrival."""
    from scripts.evaluate import make_rl_alloc_cb

    env, M = _make_minimal_env(reward_variant="A", lookahead_window=100)
    env_obs, _ = env.reset(seed=0)

    # Reconstruct the ctx that pooling_simulation would pass at the first event tick.
    # At reset time the env has: no VMs allocated, mpd_vm_allocs empty, host_cxl_load zeros.
    tick, host_id_in_pod, vm_mem, _ = env.events[env.event_idx]
    mhd_list = env.host_to_mhds[host_id_in_pod]
    num_mhd = env.num_mhd
    norm = env.pod_dram

    ctx = {
        "tick": tick,
        "pod_start_ts": 0,
        "base_time": env.events[0][0],  # not used in new obs path
        "num_mhd": num_mhd,
        "pod_rss_mem": float(norm),
        "mpd_vm_allocs": [list(a) for a in env.mpd_vm_allocs],  # copy (empty at reset)
        "host_cxl_load": env.cur_host_cxl_load.copy(),
    }

    captured_obs = [None]

    class _CapturingModel:
        def predict(self, obs, deterministic=True):
            captured_obs[0] = obs.copy()
            return np.zeros(env.max_degree, dtype=np.float32), None

    cb = make_rl_alloc_cb(
        _CapturingModel(),
        max_degree=env.max_degree,
        obs_variant="A",
        mhd_to_hosts=env.mhd_to_hosts,
        Q_j=env.Q_j,
        lookahead_window=env.lookahead_window,
    )
    cb(vm_mem, mhd_list, env.cur_cxl_mem_vec.copy(), ctx)

    assert captured_obs[0] is not None, "model.predict was never called"
    assert captured_obs[0].shape == env_obs.shape, (
        f"obs shape mismatch: evaluate={captured_obs[0].shape} env={env_obs.shape}"
    )
    assert np.allclose(captured_obs[0], env_obs, atol=1e-5), (
        f"obs mismatch at first VM arrival:\n  evaluate={captured_obs[0]}\n  env={env_obs}"
    )
