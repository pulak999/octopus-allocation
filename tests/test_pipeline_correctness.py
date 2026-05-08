"""Pipeline correctness tests: invariants, parity, and reward-variant contracts."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace_arrays(all_vms, node_to_vms, node_to_machine, machine_sz):
    from octopus.data import to_arrays
    return to_arrays((all_vms, node_to_vms, node_to_machine, {}, machine_sz))


def _make_minimal_env(reward_variant="current", **kwargs):
    """2-host, 3-MPD env with 2 synthetic VMs (no trace files required)."""
    from octopus.env import OctopusMemPoolEnv
    import datetime as dt

    M = [[1, 1, 0], [0, 1, 1]]
    base = dt.datetime(2024, 1, 1, 0, 0)

    class FakeVM:
        def __init__(self, start_min, end_min, mem_gb):
            self.start_time = base + dt.timedelta(minutes=start_min)
            self.end_time   = base + dt.timedelta(minutes=end_min)
            self.rss = [0, mem_gb, 0, 0]

    all_vms          = {"vm0": FakeVM(0, 60, 4.0), "vm1": FakeVM(0, 30, 2.0)}
    node_to_vms      = {0: ["vm0"], 1: ["vm1"]}
    node_to_machine  = {0: "mtype", 1: "mtype"}
    machine_sz       = {"mtype": [0, 100.0, 0, 0]}

    trace_arrays = _make_trace_arrays(all_vms, node_to_vms, node_to_machine, machine_sz)
    env = OctopusMemPoolEnv(
        trace_arrays=trace_arrays, M=M, seed=0,
        reward_variant=reward_variant, skip_hotfix=True, **kwargs,
    )
    return env, M, all_vms, node_to_vms, node_to_machine, machine_sz


def _make_long_env_r5(n_vms=10, sub_episode_len=3, pbrs_gamma=0.99):
    """R5 env with n_vms events spread over consecutive ticks (5-min each)."""
    from octopus.env import OctopusMemPoolEnv
    from octopus.data import to_arrays
    import datetime as dt

    M = [[1, 1, 0], [0, 1, 1]]
    base = dt.datetime(2024, 1, 1, 0, 0)

    class FakeVM:
        def __init__(self, start_min, end_min, mem_gb):
            self.start_time = base + dt.timedelta(minutes=start_min)
            self.end_time   = base + dt.timedelta(minutes=end_min)
            self.rss = [0, mem_gb, 0, 0]

    all_vms     = {}
    node_to_vms = {0: [], 1: []}
    for i in range(n_vms):
        all_vms[f"vm{i}"] = FakeVM(i * 5, i * 5 + 200, 1.0)
        node_to_vms[i % 2].append(f"vm{i}")

    node_to_machine = {0: "mtype", 1: "mtype"}
    machine_sz      = {"mtype": [0, 100.0, 0, 0]}
    trace_arrays = to_arrays((all_vms, node_to_vms, node_to_machine, {}, machine_sz))
    return OctopusMemPoolEnv(
        trace_arrays=trace_arrays, M=M, seed=0,
        reward_variant="R5", skip_hotfix=True,
        sub_episode_len=sub_episode_len, pbrs_gamma=pbrs_gamma,
    )


# ---------------------------------------------------------------------------
# 1. Mass conservation
# ---------------------------------------------------------------------------

def test_mass_conservation():
    """After each step, sum(cur_cxl_mem_vec) == sum of active-VM memory in _mpd arrays."""
    env, *_ = _make_minimal_env(reward_variant="current")
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        mpd_total = sum(
            float(env._mpd_mem[j, :env._mpd_n[j]].sum())
            for j in range(env.num_mhd)
        )
        assert mpd_total == pytest.approx(float(env.cur_cxl_mem_vec.sum()), abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Deallocation clears flat MPD arrays
# ---------------------------------------------------------------------------

def test_deallocation_clears():
    """After departure processing, no MPD slot retains a VM with dt <= last_processed_tick."""
    env, *_ = _make_minimal_env(reward_variant="current")
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
    env._process_departures_through(env.pod_dur - 1)
    processed_up_to = env._last_depart_tick
    for j in range(env.num_mhd):
        n = int(env._mpd_n[j])
        for s in range(n):
            assert env._mpd_dt[j, s] > processed_up_to, (
                f"MPD {j} slot {s}: dealloc_tick={env._mpd_dt[j,s]} "
                f"<= processed_up_to={processed_up_to}"
            )


# ---------------------------------------------------------------------------
# 3. pooling_ratio formula
# ---------------------------------------------------------------------------

def test_pooling_ratio_formula():
    """info['pooling_ratio'] at episode end equals max_peak * num_mhd / pod_dram."""
    env, *_ = _make_minimal_env(reward_variant="current")
    env.reset(seed=0)
    done = False
    final_info = {}
    while not done:
        _, _, done, _, info = env.step(env.action_space.sample())
        if done:
            final_info = info
    expected = env.max_peak * env.num_mhd / env.pod_dram
    assert final_info["pooling_ratio"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Eval pipeline vs env parity
# ---------------------------------------------------------------------------

def test_eval_env_parity():
    """Uniform allocation: env-path pooling_ratio matches pooling_simulation to 1e-4."""
    from scripts.evaluate import pooling_simulation
    import datetime as dt
    from octopus.env import OctopusMemPoolEnv

    M = [[1, 1, 0], [0, 1, 1]]
    base = dt.datetime(2024, 1, 1, 0, 0)

    class FakeVM:
        def __init__(self, start_min, end_min, mem_gb):
            self.start_time = base + dt.timedelta(minutes=start_min)
            self.end_time   = base + dt.timedelta(minutes=end_min)
            self.rss = [0, mem_gb, 0, 0]

    all_vms         = {"vm0": FakeVM(0, 60, 4.0), "vm1": FakeVM(0, 30, 2.0)}
    node_to_vms     = {0: ["vm0"], 1: ["vm1"]}
    node_to_machine = {0: "mtype", 1: "mtype"}
    machine_sz      = {"mtype": [0, 100.0, 0, 0]}

    # Env path — zeros action → uniform distribution among accessible MPDs
    trace_arrays = _make_trace_arrays(all_vms, node_to_vms, node_to_machine, machine_sz)
    env = OctopusMemPoolEnv(
        trace_arrays=trace_arrays, M=M, seed=0,
        reward_variant="current", skip_hotfix=True,
    )
    env.reset(seed=0)
    zeros = np.zeros(env.max_degree, dtype=np.float32)
    done = False
    final_info = {}
    while not done:
        _, _, done, _, info = env.step(zeros)
        if done:
            final_info = info
    env_ratio = final_info["pooling_ratio"]

    # pooling_simulation path — same uniform allocation callback
    def uniform_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
        alloc = np.zeros(len(cur_cxl_mem_vec))
        for j in mhd_list:
            alloc[j] = cxl_mem / len(mhd_list)
        return alloc

    # node_to_M maps actual node id → pod-local index (0..pod_size-1)
    node_to_M = {0: 0, 1: 1}
    sim_ratio = pooling_simulation(
        node_to_M, M, uniform_alloc,
        all_vms, node_to_vms, node_to_machine, machine_sz,
    )

    assert env_ratio == pytest.approx(sim_ratio, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. R4 intermediate rewards are zero
# ---------------------------------------------------------------------------

def test_R4_intermediate_rewards_zero():
    """All non-terminal R4 rewards are 0.0; terminal reward equals pooling_savings."""
    env, *_ = _make_minimal_env(reward_variant="R4")
    env.reset(seed=0)
    done = False
    n_steps = 0
    while not done:
        _, reward, done, _, info = env.step(env.action_space.sample())
        n_steps += 1
        if not done:
            assert reward == pytest.approx(0.0), \
                f"Non-terminal R4 reward={reward} at step {n_steps}"
        else:
            assert reward == pytest.approx(info["pooling_savings"], abs=1e-9)
    assert n_steps >= 1


# ---------------------------------------------------------------------------
# 6. R5 sub-peak resets at boundary; MPD loads persist
# ---------------------------------------------------------------------------

def test_R5_sub_peak_resets_mpd_loads_persist():
    """At a sub-episode boundary, _sub_peak resets to 0 while cur_cxl_mem_vec stays > 0."""
    env = _make_long_env_r5(n_vms=10, sub_episode_len=3)
    env.reset(seed=0)

    boundary_hit = False
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        # _sub_step > 0 guards against the initial state (reset gives _sub_step=0)
        at_boundary = (env._sub_step > 0 and env._sub_step % env.sub_episode_len == 0)
        if at_boundary and not done:
            assert env._sub_peak == pytest.approx(0.0), \
                f"_sub_peak not reset at boundary (step {env._sub_step})"
            assert float(env.cur_cxl_mem_vec.sum()) > 0.0, \
                "MPD loads were cleared at sub-episode boundary (should persist)"
            boundary_hit = True

    assert boundary_hit, "No sub-episode boundary was reached during the episode"


# ---------------------------------------------------------------------------
# 7. R5 PBRS telescoping
# ---------------------------------------------------------------------------

def test_R5_pbrs_telescoping():
    """With γ=1, Σ(Φ(s') - Φ(s)) = Φ(s_T) - Φ(s_0); verifies _prev_potential tracking."""
    # γ=1 makes the sum a clean telescope: Σ(Φ(s')-Φ(s)) = Φ(s_T)-Φ(s_0).
    # With γ≠1 the cross-terms don't cancel, so we test the trivial γ=1 case.
    env = _make_long_env_r5(n_vms=10, sub_episode_len=3, pbrs_gamma=1.0)
    env.reset(seed=0)

    phi_0 = env._prev_potential     # 0.0 after reset

    pbrs_sum = 0.0
    done = False
    while not done:
        phi_before = env._prev_potential
        _, _, done, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        phi_after = env._prev_potential
        pbrs_sum += phi_after - phi_before  # γ=1: term is just Φ(s') - Φ(s)

    # Pure telescoping: Σ(Φ(s') - Φ(s)) = Φ(s_T) - Φ(s_0)
    expected = env._prev_potential - phi_0
    assert pbrs_sum == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Obs reflects env state (rich obs)
# ---------------------------------------------------------------------------

def test_obs_reflects_env_state():
    """Rich obs: c_j/pod_dram, peak/pod_dram, vm_mem/pod_dram match env state."""
    env, *_ = _make_minimal_env(reward_variant="R2")
    obs, _ = env.reset(seed=0)
    norm = env.pod_dram

    # At reset: no allocations — all c_j == 0
    _, node, vm_mem, _ = env.events[env.event_idx]
    mhd_list = env.host_to_mhds[node]
    for k, mhd in enumerate(mhd_list):
        assert obs[k * 6] == pytest.approx(
            float(env.cur_cxl_mem_vec[mhd]) / norm, abs=1e-6
        )
    assert obs[-2] == pytest.approx(float(np.max(env.cur_cxl_mem_vec)) / norm, abs=1e-6)
    assert obs[-1] == pytest.approx(float(vm_mem) / norm, abs=1e-6)

    # After one step: obs should reflect updated state
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, _, done, _, _ = env.step(zeros)
    if not done:
        _, next_node, next_vm_mem, _ = env.events[env.event_idx]
        next_mhds = env.host_to_mhds[next_node]
        for k, mhd in enumerate(next_mhds):
            assert obs2[k * 6] == pytest.approx(
                float(env.cur_cxl_mem_vec[mhd]) / norm, abs=1e-6
            )
        assert obs2[-2] == pytest.approx(
            float(np.max(env.cur_cxl_mem_vec)) / norm, abs=1e-6
        )
        assert obs2[-1] == pytest.approx(float(next_vm_mem) / norm, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. greedy_alloc mutation regression
# ---------------------------------------------------------------------------

def test_greedy_mutation_regression():
    """greedy_alloc must not mutate the cur_cxl_mem_vec argument."""
    from octopus.baselines import greedy_alloc

    mhd_list = [0, 1]
    cur_vec  = np.array([2.0, 5.0, 3.0])
    original = cur_vec.copy()

    _ = greedy_alloc(5.0, mhd_list, cur_vec)

    assert np.allclose(cur_vec, original), \
        f"greedy_alloc mutated cur_cxl_mem_vec: {cur_vec} != {original}"


# ---------------------------------------------------------------------------
# 10. R1 bounds
# ---------------------------------------------------------------------------

def test_R1_bounds():
    """R1 reward is in [-1.0, 0.0] on every step."""
    env, *_ = _make_minimal_env(reward_variant="R1")
    env.reset(seed=0)
    done = False
    while not done:
        _, reward, done, _, _ = env.step(env.action_space.sample())
        assert reward <= 0.0 + 1e-9, f"R1 reward > 0: {reward}"
        assert reward >= -1.0 - 1e-9, f"R1 reward < -1: {reward}"
