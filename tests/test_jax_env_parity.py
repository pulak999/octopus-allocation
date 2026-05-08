"""JAX env parity tests — Chunks A through D.

All tests run under jax_enable_x64=True (float64 mode) to allow direct
comparison with SB3's float64 state.  Production training uses float32.

Chunk A: _compute_D_S_j correctness (4 tests)
Chunk B: reset_fn / step_fn smoke tests (5 tests)
Chunk D: step-level parity vs SB3 OctopusMemPoolEnv (5 tests) — added later
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import datetime as dt

from octopus.jax.env import (
    MAX_TICKS, MAX_EVENTS,
    _compute_D_S_j, _compute_D_j,
    make_episode, reset_fn, step_fn, rollout_fn, make_rollout_fn,
)
from octopus.kernels import compute_D_S_j as _nb_compute_D_S_j
from octopus.env import OctopusMemPoolEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mpd_state_to_dealloc_buf(mpd_dt, mpd_mem, mpd_n, num_mhd):
    """Convert SB3 flat MPD arrays into dealloc_buf equivalent.

    dealloc_buf[t, j] = sum of mem for VMs on MPD j with dealloc_tick == t.
    """
    buf = np.zeros((MAX_TICKS, num_mhd), dtype=np.float64)
    for j in range(num_mhd):
        for s in range(int(mpd_n[j])):
            t = int(mpd_dt[j, s])
            if 0 <= t < MAX_TICKS:
                buf[t, j] += mpd_mem[j, s]
    return buf


def _make_synthetic_trace(n_nodes=4, n_vms_per_node=3, seed=42):
    """Minimal synthetic TraceArrays + topology for smoke tests (no file I/O)."""
    from octopus.data import to_arrays

    rng = np.random.default_rng(seed)
    base = dt.datetime(2024, 1, 1, 0, 0)

    class FakeVM:
        def __init__(self, start_offset_min, end_offset_min, mem_gb):
            self.start_time = base + dt.timedelta(minutes=start_offset_min)
            self.end_time   = base + dt.timedelta(minutes=end_offset_min)
            self.rss = [0, mem_gb, 0, 0]

    all_vms: dict = {}
    node_to_vms: dict = {}
    for node in range(n_nodes):
        node_to_vms[node] = []
        for i in range(n_vms_per_node):
            vmkey = f"vm_{node}_{i}"
            start = float(rng.integers(0, 500))        # 0..500 min
            dur   = float(rng.integers(10, 200))       # 10..200 min
            mem   = float(rng.uniform(1.0, 8.0))
            all_vms[vmkey] = FakeVM(start, start + dur, mem)
            node_to_vms[node].append(vmkey)

    node_to_machine = {n: "mtype" for n in range(n_nodes)}
    machine_sz = {"mtype": [0, 512.0, 0, 0]}  # 512 GB DRAM per node

    ta = to_arrays((all_vms, node_to_vms, node_to_machine, {}, machine_sz))

    # 4-node pod, each host connected to 2 MPDs (2 MPDs total shared)
    # M: 4 hosts × 2 MPDs — every host can use either MPD
    M = np.ones((n_nodes, 2), dtype=np.int32)
    return ta, M


# ---------------------------------------------------------------------------
# Chunk D — reset determinism + float64 parity vs Gymnasium env
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_trace_for_parity():
    """Bigger synthetic episode so we can run ~200 steps for parity checks."""
    ta, M = _make_synthetic_trace(n_nodes=4, n_vms_per_node=60, seed=7)
    return ta, M


def _make_env_and_jax(static_reward_variant: str, ta, M, *, seed: int = 0,
                       lookahead_window: int = 100, reward_lambda: float = 0.2):
    """Construct matching Gym and JAX episode objects."""
    sc = make_episode(
        seed=seed,
        trace_arrays=ta,
        M=M,
        reward_variant=static_reward_variant,
        lookahead_window=lookahead_window,
        reward_lambda=reward_lambda,
        skip_hotfix=True,
    )

    env = OctopusMemPoolEnv(
        trace_arrays=ta,
        M=M,
        seed=seed,
        reward_variant=static_reward_variant,
        lookahead_window=lookahead_window,
        reward_lambda=reward_lambda,
        skip_hotfix=True,
    )
    return env, sc


def _run_parity_episode_compare(env: OctopusMemPoolEnv, sc, reward_variant: str, *,
                                n_steps: int = 200, action_seed: int = 12345,
                                obs_atol: float = 1e-5, reward_atol: float = 1e-6):
    """Compare per-step JAX outputs vs Gymnasium env outputs."""
    # Reset both
    state, obs_jax = reset_fn(sc)
    obs_gym, _info = env.reset(seed=int(env._seed))

    np.testing.assert_allclose(
        np.array(obs_jax), np.array(obs_gym),
        atol=obs_atol, rtol=0.0,
        err_msg=f"reset obs mismatch for reward_variant={reward_variant!r}",
    )

    # Fixed action sequence
    max_degree = int(sc.max_degree)
    rng = np.random.default_rng(action_seed)
    actions_np = rng.standard_normal((n_steps, max_degree)).astype(np.float32)
    actions = jnp.asarray(actions_np)

    # Run up to the desired step budget or the episode end.
    # JAX: done when event_idx increments past num_events
    n_env_events = getattr(env, "events", [])
    steps_to_run = min(n_steps, int(sc.num_events), len(n_env_events))
    done_jax = False
    done_gym = False

    for k in range(steps_to_run):
        a_jax = actions[k]
        a_gym = actions_np[k]

        state, obs_jax, reward_jax, done_jax_arr, _ = step_fn(state, a_jax, sc)
        obs_gym, reward_gym, done_gym, _trunc, _info = env.step(a_gym)

        # Outputs
        np.testing.assert_allclose(
            np.array(obs_jax), np.array(obs_gym),
            atol=obs_atol, rtol=0.0,
            err_msg=f"obs mismatch at step {k} for reward_variant={reward_variant!r}",
        )
        np.testing.assert_allclose(
            float(reward_jax), float(reward_gym),
            atol=reward_atol, rtol=0.0,
            err_msg=f"reward mismatch at step {k} for reward_variant={reward_variant!r}",
        )
        assert bool(done_jax_arr) == bool(done_gym), \
            f"done mismatch at step {k} for reward_variant={reward_variant!r}"

    return True


def test_reset_deterministic(synthetic_trace_for_parity):
    """Same seed → same first obs (host-side pod/event generation is deterministic)."""
    ta, M = synthetic_trace_for_parity
    reward_variant = "current"
    env1, sc1 = _make_env_and_jax(reward_variant, ta, M, seed=0)
    env2, sc2 = _make_env_and_jax(reward_variant, ta, M, seed=0)

    _, obs_jax_1 = reset_fn(sc1)
    _, obs_jax_2 = reset_fn(sc2)
    np.testing.assert_allclose(np.array(obs_jax_1), np.array(obs_jax_2), atol=1e-7, rtol=0.0)

    obs_gym_1, _ = env1.reset(seed=0)
    obs_gym_2, _ = env2.reset(seed=0)
    np.testing.assert_allclose(np.array(obs_gym_1), np.array(obs_gym_2), atol=1e-7, rtol=0.0)


def test_jax_step_matches_gym_current(synthetic_trace_for_parity):
    ta, M = synthetic_trace_for_parity
    env, sc = _make_env_and_jax("current", ta, M, seed=0)
    _run_parity_episode_compare(env, sc, "current", n_steps=200)


def test_jax_step_matches_gym_A(synthetic_trace_for_parity):
    ta, M = synthetic_trace_for_parity
    env, sc = _make_env_and_jax("A", ta, M, seed=0)
    _run_parity_episode_compare(env, sc, "A", n_steps=200)


def test_jax_step_matches_gym_B(synthetic_trace_for_parity):
    ta, M = synthetic_trace_for_parity
    env, sc = _make_env_and_jax("B", ta, M, seed=0)
    _run_parity_episode_compare(env, sc, "B", n_steps=200)


# ---------------------------------------------------------------------------
# Chunk A — _compute_D_S_j unit tests
# ---------------------------------------------------------------------------

def test_dsj_empty():
    """All-zero dealloc_buf → D_j = S_j = 0 for all MPDs."""
    buf = jnp.zeros((MAX_TICKS, 4), dtype=jnp.float64)
    D_j, S_j = _compute_D_S_j(buf, last_depart_tick=0, tick=0, W=100)
    np.testing.assert_allclose(np.array(D_j), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(S_j), 0.0, atol=1e-14)


def test_dsj_matches_numba():
    """D_j/S_j from dealloc_buf matches Numba kernel on equivalent _mpd_dt/_mpd_mem state."""
    num_mhd = 4
    cap = 32
    mpd_dt  = np.full((num_mhd, cap), np.inf, dtype=np.float64)
    mpd_mem = np.zeros((num_mhd, cap), dtype=np.float64)
    mpd_n   = np.zeros(num_mhd, dtype=np.int32)

    # Insert VMs spanning near/far/edge cases
    vms = [
        (0, 10,  4.0),   # MPD 0, dt=10  → D_j: weight=1-10/100=0.9
        (1, 50,  2.5),   # MPD 1, dt=50  → D_j: weight=0.5
        (1, 200, 3.0),   # MPD 1, dt=200 → S_j (beyond W=100)
        (2, 100, 1.0),   # MPD 2, dt=100 = tick+W → D_j: weight=0 (contributes nothing)
        (3, 101, 7.0),   # MPD 3, dt=101 > tick+W → S_j
    ]
    for j, dt_val, mem in vms:
        s = int(mpd_n[j])
        mpd_dt[j, s]  = float(dt_val)
        mpd_mem[j, s] = mem
        mpd_n[j] += 1

    tick, W = 0, 100
    D_ref, S_ref = _nb_compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick, W)

    buf = jnp.array(_mpd_state_to_dealloc_buf(mpd_dt, mpd_mem, mpd_n, num_mhd))
    D_jax, S_jax = _compute_D_S_j(buf, last_depart_tick=tick, tick=tick, W=W)

    np.testing.assert_allclose(np.array(D_jax), D_ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.array(S_jax), S_ref, atol=1e-12, rtol=1e-12)


def test_dsj_weight_at_edge():
    """VM departing at exactly tick+W: weight=0 → not in D_j, not in S_j."""
    buf_np = np.zeros((MAX_TICKS, 2), dtype=np.float64)
    tick, W = 0, 100
    buf_np[tick + W, 0] = 5.0   # dealloc at exactly tick+W
    buf_jax = jnp.array(buf_np)
    D_j, S_j = _compute_D_S_j(buf_jax, last_depart_tick=tick, tick=tick, W=W)
    # weight = 1 - W/W = 0 → D_j = 0; t = tick+W is not > tick+W → S_j = 0
    assert float(D_j[0]) == pytest.approx(0.0, abs=1e-14)
    assert float(S_j[0]) == pytest.approx(0.0, abs=1e-14)


def test_dsj_excludes_past_departures():
    """Entries at t ≤ last_depart_tick are excluded (already applied to mpd_load)."""
    buf_np = np.zeros((MAX_TICKS, 2), dtype=np.float64)
    tick, W = 50, 100
    buf_np[30, 0] = 10.0   # t=30 ≤ last_depart_tick=50 → excluded
    buf_np[60, 0] =  5.0   # t=60 > 50, ≤ 150 → D_j: weight=1-(60-50)/100=0.9
    buf_np[160, 0] = 3.0   # t=160 > tick+W=150 → S_j
    buf_jax = jnp.array(buf_np)
    D_j, S_j = _compute_D_S_j(buf_jax, last_depart_tick=tick, tick=tick, W=W)
    assert float(D_j[0]) == pytest.approx(5.0 * 0.9, abs=1e-12)
    assert float(S_j[0]) == pytest.approx(3.0, abs=1e-12)
    assert float(D_j[1]) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Chunk B — reset_fn / step_fn smoke tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_episode():
    """Build a StaticConfig from the synthetic trace (shared across Chunk B tests)."""
    ta, M = _make_synthetic_trace(n_nodes=4, n_vms_per_node=5, seed=7)
    sc = make_episode(seed=0, trace_arrays=ta, M=M, reward_variant="current",
                      lookahead_window=100, reward_lambda=0.2, skip_hotfix=True)
    return sc


def test_make_episode_builds_valid_config(synthetic_episode):
    """make_episode produces a StaticConfig with correct shapes and num_events > 0."""
    sc = synthetic_episode
    assert sc.num_events > 0, "expected at least one event in synthetic episode"
    assert sc.events.shape == (MAX_EVENTS, 4)
    assert sc.host_to_mhds.shape == (sc.pod_size, sc.max_degree)
    assert sc.mhd_to_hosts.shape == (sc.num_mhd, sc.max_conn)
    assert sc.Q_j.shape == (sc.num_mhd,)
    # Sentinel rows: events beyond num_events have tick == -1
    assert float(sc.events[sc.num_events - 1, 0]) >= 0, "last real event has non-negative tick"
    assert float(sc.events[sc.num_events, 0]) == pytest.approx(-1.0)


def test_reset_returns_correct_shapes(synthetic_episode):
    """reset_fn returns (OctopusState, obs) with shapes matching the topology."""
    sc = synthetic_episode
    state, obs = reset_fn(sc)

    assert state.mpd_load.shape == (sc.num_mhd,)
    assert state.dealloc_buf.shape == (MAX_TICKS, sc.num_mhd)
    assert state.host_load.shape == (sc.pod_size,)
    assert state.host_dealloc_buf.shape == (MAX_TICKS, sc.pod_size)
    assert state.max_peak.shape == ()
    assert state.event_idx.shape == ()
    assert int(state.event_idx) == 0

    expected_obs_dim = sc.max_degree * 2 + 4  # "current" variant
    assert obs.shape == (expected_obs_dim,)
    assert obs.dtype == jnp.float32


def test_step_returns_correct_shapes_and_terminates(synthetic_episode):
    """step_fn returns correct shapes; episode terminates after num_events steps."""
    sc = synthetic_episode
    state, obs = reset_fn(sc)

    action = np.zeros(sc.max_degree, dtype=np.float32)
    n_steps = 0
    done = False
    while not done and n_steps < sc.num_events + 5:
        state, obs, reward, done, info = step_fn(state, action, sc)
        assert obs.shape == (sc.max_degree * 2 + 4,)
        assert reward.shape == ()
        assert done.shape == ()
        n_steps += 1

    assert bool(done), "episode should terminate"
    assert n_steps == sc.num_events, f"expected {sc.num_events} steps, got {n_steps}"


def test_step_reward_variant_A(synthetic_episode):
    """step_fn with reward variant A runs without error and returns finite rewards."""
    ta, M = _make_synthetic_trace(n_nodes=4, n_vms_per_node=5, seed=7)
    sc = make_episode(seed=0, trace_arrays=ta, M=M, reward_variant="A",
                      lookahead_window=100, reward_lambda=0.2, skip_hotfix=True)
    state, _ = reset_fn(sc)
    action = np.zeros(sc.max_degree, dtype=np.float32)
    for _ in range(min(10, sc.num_events)):
        state, obs, reward, done, _ = step_fn(state, action, sc)
        assert np.isfinite(float(reward)), f"reward is not finite: {float(reward)}"
        assert obs.shape == (sc.max_degree * 6 + 2,)


def test_step_reward_variant_B(synthetic_episode):
    """step_fn with reward variant B runs without error and returns finite rewards."""
    ta, M = _make_synthetic_trace(n_nodes=4, n_vms_per_node=5, seed=7)
    sc = make_episode(seed=0, trace_arrays=ta, M=M, reward_variant="B",
                      lookahead_window=100, reward_lambda=0.2, skip_hotfix=True)
    state, _ = reset_fn(sc)
    action = np.zeros(sc.max_degree, dtype=np.float32)
    for _ in range(min(10, sc.num_events)):
        state, obs, reward, done, _ = step_fn(state, action, sc)
        assert np.isfinite(float(reward)), f"reward is not finite: {float(reward)}"
        assert obs.shape == (sc.max_degree * 6 + 2,)


def test_jit_step_compiles_and_runs(synthetic_episode):
    """jax.jit(step_fn, static_argnums=2) compiles without error and gives consistent output."""
    sc = synthetic_episode
    state, _ = reset_fn(sc)
    jit_step = jax.jit(step_fn, static_argnums=2)

    action = jnp.zeros(sc.max_degree, dtype=jnp.float32)
    # First call triggers compilation
    state2, obs2, r2, done2, _ = jit_step(state, action, sc)
    # Second call uses the compiled kernel
    state3, obs3, r3, done3, _ = jit_step(state2, action, sc)

    # Results should be identical to non-jit version
    state_ref, obs_ref, r_ref, done_ref, _ = step_fn(state, action, sc)
    np.testing.assert_allclose(np.array(obs2), np.array(obs_ref), atol=1e-6)
    np.testing.assert_allclose(float(r2), float(r_ref), atol=1e-6)
    assert bool(done2) == bool(done_ref)


# ---------------------------------------------------------------------------
# Chunk C — lax.scan rollout + make_rollout_fn smoke tests
# ---------------------------------------------------------------------------

def test_scan_matches_manual_loop(synthetic_episode):
    """rollout_fn(state, actions, sc) produces the same trajectory as H manual step_fn calls."""
    sc = synthetic_episode
    state, _ = reset_fn(sc)

    H = min(10, sc.num_events)
    rng = np.random.default_rng(99)
    actions_np = rng.standard_normal((H, sc.max_degree)).astype(np.float32)
    actions = jnp.asarray(actions_np)

    # -- lax.scan rollout --
    _, (obs_scan, rew_scan, done_scan) = rollout_fn(state, actions, sc)

    # -- manual loop --
    obs_loop   = []
    rew_loop   = []
    done_loop  = []
    s = state
    for k in range(H):
        s, obs, rew, done, _ = step_fn(s, actions[k], sc)
        obs_loop.append(np.array(obs))
        rew_loop.append(float(rew))
        done_loop.append(bool(done))

    np.testing.assert_allclose(
        np.array(obs_scan), np.array(obs_loop), atol=1e-5,
        err_msg="scan obs differs from manual loop"
    )
    np.testing.assert_allclose(
        np.array(rew_scan), np.array(rew_loop), atol=1e-5,
        err_msg="scan rewards differ from manual loop"
    )
    assert list(np.array(done_scan)) == done_loop, "scan done flags differ from manual loop"


def test_make_rollout_fn_batched(synthetic_episode):
    """make_rollout_fn produces a JIT+vmap'd fn whose outputs match single-env rollout."""
    sc = synthetic_episode
    state0, _ = reset_fn(sc)

    H      = min(8, sc.num_events)
    n_envs = 3
    rng    = np.random.default_rng(7)
    actions_np = rng.standard_normal((n_envs, H, sc.max_degree)).astype(np.float32)
    batched_actions = jnp.asarray(actions_np)

    # Replicate state0 across n_envs
    batched_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (n_envs,) + x.shape), state0
    )

    rollout = make_rollout_fn(sc)
    _, (obs_batch, rew_batch, done_batch) = rollout(batched_state, batched_actions)

    assert obs_batch.shape  == (n_envs, H, sc.max_degree * 2 + 4)
    assert rew_batch.shape  == (n_envs, H)
    assert done_batch.shape == (n_envs, H)

    # Env 0's outputs should match a single-env rollout with the same actions
    _, (obs_single, rew_single, _) = rollout_fn(state0, batched_actions[0], sc)
    np.testing.assert_allclose(
        np.array(obs_batch[0]), np.array(obs_single), atol=1e-5,
        err_msg="batched env 0 differs from single-env rollout"
    )
    np.testing.assert_allclose(
        np.array(rew_batch[0]), np.array(rew_single), atol=1e-5,
        err_msg="batched env 0 rewards differ from single-env rollout"
    )


def test_jaxpr_has_no_python_callbacks(synthetic_episode):
    """make_jaxpr of rollout_fn succeeds and contains no python_callback primitives."""
    sc = synthetic_episode
    state, _ = reset_fn(sc)

    H = 4
    actions = jnp.zeros((H, sc.max_degree), dtype=jnp.float32)

    jaxpr = jax.make_jaxpr(lambda s, a: rollout_fn(s, a, sc))(state, actions)
    jaxpr_str = str(jaxpr)

    assert "python_callback" not in jaxpr_str, \
        "rollout_fn JAXPR contains python_callback — Python is in the hot path!"
    assert "io_callback" not in jaxpr_str, \
        "rollout_fn JAXPR contains io_callback — Python I/O is in the hot path!"
