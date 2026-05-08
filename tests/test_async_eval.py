"""Tests for async eval worker (async-plan).

All tests run on CPU (device_str="cpu") so no GPU is required.
The worker function is exercised end-to-end via multiprocessing.spawn.
"""

import copy
import datetime as dt
import multiprocessing as mp
import queue
import time

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level fake policy (must be at module level for spawn pickling)
# ---------------------------------------------------------------------------

class _FakePolicy(nn.Module):
    """Tiny MLP policy usable as a drop-in for spawn-based worker tests."""
    def __init__(self, obs_dim: int = 8, act_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Linear(obs_dim, act_dim)

    def set_training_mode(self, mode: bool):
        self.train(mode)

    def _predict(self, obs, deterministic=True):
        with torch.no_grad():
            return self.net(obs)


def _make_minimal_policy(obs_dim: int = 8, act_dim: int = 2):
    return _FakePolicy(obs_dim=obs_dim, act_dim=act_dim)


def _make_minimal_trace():
    """Tiny synthetic trace: 2 hosts, 2 VMs, no file I/O.

    Uses octopus.data.VM (module-level, picklable) for spawn compatibility.
    """
    from octopus.data import VM
    base = dt.datetime(2024, 1, 1, 0, 0)

    def _vm(start_min, end_min, mem_gb):
        v = VM()
        v.start_time = base + dt.timedelta(minutes=start_min)
        v.end_time = base + dt.timedelta(minutes=end_min)
        v.rss = [0, mem_gb, 0, 0]
        return v

    all_vms = {"vm0": _vm(0, 300, 4.0), "vm1": _vm(0, 120, 2.0)}
    node_to_vms = {0: ["vm0"], 1: ["vm1"]}
    node_to_machine = {0: "mt", 1: "mt"}
    vm_type_sz = {}
    machine_sz = {"mt": [0, 512.0, 0, 0]}
    return all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz


# 2-host, 3-MPD topology (max_degree = 2)
_M = [[1, 1, 0],
      [0, 1, 1]]
_MAX_DEGREE = 2


# ---------------------------------------------------------------------------
# Helper: spawn worker, send one eval, collect result, stop
# ---------------------------------------------------------------------------

def _run_worker_roundtrip(n_iter=2, obs_variant="current", timeout=60):
    """Spawn _eval_worker_main, dispatch one eval, return (mean, std, snap_step)."""
    from scripts.train_rl import _eval_worker_main

    ctx = mp.get_context("spawn")
    cmd_q = ctx.Queue()
    res_q = ctx.Queue()

    obs_dim = _MAX_DEGREE * 2 + 4 if obs_variant == "current" else _MAX_DEGREE * 6 + 2
    policy_cpu = _make_minimal_policy(obs_dim=obs_dim, act_dim=_MAX_DEGREE)

    worker = ctx.Process(
        target=_eval_worker_main,
        args=(cmd_q, res_q, policy_cpu, _make_minimal_trace(),
              _M, _MAX_DEGREE, n_iter, obs_variant, 200, "cpu"),
        daemon=True,
    )
    worker.start()

    # Give the worker a moment to initialise before sending the first command
    time.sleep(0.5)

    sd_cpu = {k: v.cpu() for k, v in policy_cpu.state_dict().items()}
    cmd_q.put(("eval", sd_cpu, 100_000))

    result = res_q.get(timeout=timeout)

    cmd_q.put(("stop",))
    worker.join(timeout=10)
    return result, worker.exitcode


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorkerRoundtrip:
    def test_returns_mean_std_snap_step(self):
        result, exitcode = _run_worker_roundtrip(n_iter=2)
        mean, std, snap_step, worker_wall_s = result
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert snap_step == 100_000
        assert isinstance(worker_wall_s, float)
        assert worker_wall_s >= 0.0
        assert exitcode == 0

    def test_mean_in_valid_range(self):
        mean, std, _, _wall = _run_worker_roundtrip(n_iter=3)[0]
        # savings = 1 - pooling_ratio; pooling_ratio can be > 1, savings can be < 0
        # but must be finite
        assert np.isfinite(mean)
        assert std >= 0.0

    def test_stop_exits_cleanly(self):
        _, exitcode = _run_worker_roundtrip(n_iter=1)
        assert exitcode == 0

    def test_multiple_evals_in_sequence(self):
        """Worker handles multiple eval commands correctly."""
        from scripts.train_rl import _eval_worker_main

        ctx = mp.get_context("spawn")
        cmd_q = ctx.Queue()
        res_q = ctx.Queue()

        obs_dim = _MAX_DEGREE * 2 + 4
        policy_cpu = _make_minimal_policy(obs_dim=obs_dim, act_dim=_MAX_DEGREE)

        worker = ctx.Process(
            target=_eval_worker_main,
            args=(cmd_q, res_q, policy_cpu, _make_minimal_trace(),
                  _M, _MAX_DEGREE, 1, "current", 200, "cpu"),
            daemon=True,
        )
        worker.start()
        time.sleep(0.5)

        sd = {k: v.cpu() for k, v in policy_cpu.state_dict().items()}
        for step in [100_000, 200_000]:
            cmd_q.put(("eval", sd, step))

        results = []
        for _ in range(2):
            results.append(res_q.get(timeout=60))

        cmd_q.put(("stop",))
        worker.join(timeout=10)

        assert len(results) == 2
        assert results[0][2] == 100_000
        assert results[1][2] == 200_000
        assert worker.exitcode == 0


class TestCallbackLifecycle:
    """Test PoolingSavingsCallback async lifecycle without a real SB3 model."""

    def _make_callback(self, eval_device=None):
        from scripts.train_rl import PoolingSavingsCallback
        return PoolingSavingsCallback(
            eval_trace_data=_make_minimal_trace(),
            M=_M,
            max_degree=_MAX_DEGREE,
            eval_freq=100_000,
            n_iter=1,
            obs_variant="current",
            eval_device=eval_device,
            verbose=0,
        )

    def _fake_model(self):
        obs_dim = _MAX_DEGREE * 2 + 4
        policy = _make_minimal_policy(obs_dim=obs_dim, act_dim=_MAX_DEGREE)

        class _FakeLogger:
            def record(self, key, val):
                pass

        class _FakeModel:
            pass

        m = _FakeModel()
        m.policy = policy
        m.logger = _FakeLogger()
        return m

    def test_worker_starts_and_stops(self):
        cb = self._make_callback(eval_device="cpu")
        cb.model = self._fake_model()
        cb.on_training_start({}, {})
        assert cb._worker is not None
        assert cb._worker.is_alive()
        cb.on_training_end()
        assert not cb._worker.is_alive()

    def test_eval_device_none_mirrors_model_device(self):
        cb = self._make_callback(eval_device=None)
        cb.model = self._fake_model()
        # Should not raise; device_str will be "cpu" since policy is on CPU
        cb.on_training_start({}, {})
        cb.on_training_end()

    def test_on_step_dispatches_at_eval_freq(self):
        cb = self._make_callback(eval_device="cpu")
        cb.model = self._fake_model()
        cb.on_training_start({}, {})

        # Simulate SB3 calling _on_step at num_timesteps = eval_freq
        cb.num_timesteps = 100_000
        cb._on_step()

        assert cb._pending_step == 100_000

        # Wait for worker to finish and put result
        result = cb._res_q.get(timeout=60)
        assert result[2] == 100_000

        cb.on_training_end()

    def test_on_training_end_drains_pending_result(self):
        cb = self._make_callback(eval_device="cpu")
        cb.model = self._fake_model()
        cb.on_training_start({}, {})

        # Dispatch eval and wait for worker to produce a result
        cb.num_timesteps = 100_000
        cb._on_step()
        result = cb._res_q.get(timeout=60)  # drain it so we can re-inject

        # Re-inject so on_training_end has something to drain
        cb._res_q.put(result)

        # on_training_end should drain the result and stop the worker cleanly
        cb.on_training_end()
        assert not cb._worker.is_alive()
