# Async Eval Plan — Overlapping Callbacks with Training

## Problem

`PoolingSavingsCallback._run_eval()` blocks the SB3 training loop completely.
After fixing both O(num_mhd) bugs (run5), each callback takes ~133s:
  - inference: ~122s (320k policy calls × 0.167ms/call on GPU)
  - numpy/purge overhead: ~10s

With 20 callbacks over 2M timesteps, that is 20 × 133s = **44min of dead training time**
out of a projected ~69min total. Training itself (rollout + SAC update) takes only ~14min.
The callbacks are the bottleneck.

## Goal

Run eval callbacks in a background process so they overlap with training.
Training resumes the moment a callback is triggered; the result is collected at the
next trigger (100k steps later). Net wall time drops from ~69min to roughly ~20min
(startup ~11min + training ~14min, callbacks fully hidden behind training time).

One-step metric lag (policy at step N, result logged at step N+100k) is acceptable.

---

## Architecture: Persistent Eval Worker

Do not re-spawn a process per callback. Spawn **one persistent worker process**
at `on_training_start()` that waits on a command queue. This avoids re-pickling
large trace data and re-initialising CUDA on every eval.

```
Main process (training)                    Worker process (eval)
──────────────────────────────             ──────────────────────────────
on_training_start()
  spawn worker, send INIT(policy_cpu,
    trace_data, topology, config)   ──→    receive INIT
                                           move policy to GPU (own context)
                                           enter wait loop

every 100k steps:
  collect result_queue (non-blocking) ←──  put (mean, std, snap_step) on result_queue
  log collected metrics (if any)
  deepcopy policy state_dict → CPU
  cmd_queue.put(('eval', state_dict))  ──→ receive 'eval'
  return True (training resumes)           load state_dict onto GPU policy
                                           run 10× pooling_simulation
                                           put result on result_queue

on_training_end():
  cmd_queue.put(('stop',))             ──→ exit
  worker.join()
  drain result_queue, log final result
```

---

## Time Budget Comparison

| Phase           | Run5 (sync)  | Async        |
|-----------------|-------------|--------------|
| Startup         | ~11min       | ~11min       |
| Training        | ~14min       | ~14min       |
| Callbacks       | ~44min       | ~0min (hidden) |
| **Total**       | **~69min**   | **~25min**   |

The worker init (CUDA context, one deepcopy) costs ~5–10s, counted once at startup.
State_dict transfer per callback: ~4MB (256×256 network) — negligible.

---

## Implementation

### New top-level function: `_eval_worker_main`

Location: `scripts/train_rl.py`, before `PoolingSavingsCallback`.

Signature:
```python
def _eval_worker_main(cmd_q, res_q, policy_cpu, eval_trace_data, M,
                      max_degree, n_iter, obs_variant, lookahead_window,
                      device_str):
```

Steps inside the worker:
1. Import torch, evaluate module (deferred to worker process — avoids pickling issues).
2. Move `policy_cpu` to `device_str` (worker gets its own CUDA context via spawn).
3. Set `policy.set_training_mode(False)`.
4. Pre-allocate `_obs_buf` once (same as the current `make_rl_alloc_cb` optimisation).
5. Enter `while True: cmd = cmd_q.get()` loop:
   - `('eval', state_dict_cpu, snap_step)`:
     - Load state_dict onto policy: `policy.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})`
     - Build `make_rl_alloc_cb`-equivalent closure using the local policy.
     - Run `n_iter` × `pooling_simulation` iterations.
     - `res_q.put((float(mean), float(std), snap_step))`
   - `('stop',)`: break

The worker does not call `make_rl_alloc_cb(model, ...)` — it builds the inference
closure directly from its local policy reference to avoid importing the model object.

### Changes to `PoolingSavingsCallback`

**`__init__` — add fields:**
```python
self._worker: mp.Process | None = None
self._cmd_q:  mp.Queue   | None = None
self._res_q:  mp.Queue   | None = None
self._pending_step: int = -1
self._eval_device: str | None = None   # set from --eval-device arg; None = mirror training device
```

**New `on_training_start()`:**
```python
def on_training_start(self, locals_, globals_) -> None:
    import copy, multiprocessing as mp
    ctx = mp.get_context("spawn")         # spawn avoids CUDA-fork issues
    self._cmd_q = ctx.Queue()
    self._res_q = ctx.Queue()
    policy_cpu = copy.deepcopy(self.model.policy).cpu()
    # Use dedicated eval GPU if specified, otherwise mirror training device
    device_str = self._eval_device or str(next(self.model.policy.parameters()).device)
    self._worker = ctx.Process(
        target=_eval_worker_main,
        args=(self._cmd_q, self._res_q, policy_cpu,
              self._eval_trace_data, self._M, self._max_degree,
              self._n_iter, self._obs_variant, self._lookahead_window,
              device_str),
        daemon=True,
    )
    self._worker.start()
    _log(f"[PoolingSavings] async worker started on {device_str}")
```

**Replace `_on_step()`:**
```python
def _on_step(self) -> bool:
    # 1. Collect any completed result (non-blocking)
    try:
        mean, std, snap_step = self._res_q.get_nowait()
        self.logger.record("eval/pooling_savings_mean", mean)
        self.logger.record("eval/pooling_savings_std",  std)
        if self.verbose:
            _log(f"  [PoolingSavings @ {snap_step}] "
                 f"mean={mean:.4f}  std={std:.4f}  (collected at {self.num_timesteps})")
        self._pending_step = -1
    except Exception:
        pass   # queue.Empty or worker not done yet

    # 2. Trigger new eval if due
    if self.num_timesteps - self._last_eval_step >= self._eval_freq:
        self._last_eval_step = self.num_timesteps
        import copy
        sd_cpu = {k: v.cpu() for k, v in self.model.policy.state_dict().items()}
        self._cmd_q.put(('eval', sd_cpu, self.num_timesteps))
        self._pending_step = self.num_timesteps
        if self.verbose:
            _log(f"  [PoolingSavings @ {self.num_timesteps}] eval dispatched (async)")
    return True
```

**New `on_training_end()`:**
```python
def on_training_end(self) -> None:
    if self._worker is None:
        return
    self._cmd_q.put(('stop',))
    self._worker.join(timeout=300)
    # Drain any pending result
    try:
        mean, std, snap_step = self._res_q.get_nowait()
        self.logger.record("eval/pooling_savings_mean", mean)
        self.logger.record("eval/pooling_savings_std",  std)
        _log(f"  [PoolingSavings @ {snap_step}] final result collected on shutdown")
    except Exception:
        pass
```

**Remove `_run_eval()`** — no longer called.

### New CLI argument: `--eval-device`

Add to `argparse` in `main()`:
```python
ap.add_argument(
    "--eval-device",
    default="cuda:1",
    help="Torch device for the async eval worker (default: cuda:1). "
         "Set to '' or omit to mirror --device.",
)
```

Pass `args.eval_device or None` into `PoolingSavingsCallback` so it lands in
`self._eval_device`. With 3× TITAN RTX available, the recommended split is:

| GPU      | Role                          |
|----------|-------------------------------|
| `cuda:0` | SAC training (rollout + update) |
| `cuda:1` | Async eval worker (inference)  |
| `cuda:2` | Free — parallel run / ablation |

The worker initialises its own CUDA context via `spawn`, so there is no
shared-memory or stream conflict between the two processes.

### Multiprocessing start method

Add near the top of `main()` (before any CUDA is initialised):
```python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

`spawn` is required because PyTorch's CUDA context must not be forked (causes
silent corruption or hangs). `SubprocVecEnv` also uses spawn on this codebase.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `mp.set_start_method("spawn")` conflicts with SubprocVecEnv | SB3's SubprocVecEnv on Linux uses `forkserver` or `fork`. Check whether SB3 sets its own context. If conflict, pass `context=ctx` to SubprocVecEnv explicitly. |
| Worker falls behind (eval takes longer than 100k step interval) | Non-blocking `get_nowait()` skips silently. The main process never blocks; a slow worker just means some eval windows miss a report. Add a counter to log missed windows. |
| CUDA OOM from two contexts on same GPU | Moot with GPU split: training on `cuda:0`, eval worker on `cuda:1`. Each gets a full 24GB TITAN RTX with no contention. |
| Deepcopy of GPU policy is slow | `.cpu()` after deepcopy: measured at <1s for this network size. Happens once at startup only. State_dict copy per trigger is faster (~50ms). |
| Worker crashes silently | `daemon=True` means it dies with the main process. Add `_worker.exitcode` check in `_on_step` to detect crashes and fall back to synchronous eval. |
| `pooling_simulation` imports in worker | All `from scripts.evaluate import ...` calls happen inside the worker (after spawn), so the main process module state is not inherited. Worker must add `_REPO_ROOT` to `sys.path` before importing. Pass `repo_root` as an arg or hardcode in `_eval_worker_main`. |

---

## Testing

Run two full episodes and record per-episode training time and eval callback time.
Average both across the two episodes and compare.

**Procedure:**
1. Instrument `_on_step` (async) and `_run_eval` (sync baseline) to record wall-clock
   duration for each eval trigger.
2. Record rollout+update wall time per episode from SB3's built-in `time/fps` or a
   manual `time.perf_counter` wrap around the learn loop.
3. Run both sync (run5 config) and async (run6 config) for exactly two episodes.
4. Average training time and eval time over the two episodes for each variant.
5. Verify async eval time ≈ 0s (worker is non-blocking) and training time is
   unchanged vs sync.

**Pass criteria:**
- Worker spawns and logs "async worker started on cuda:1".
- At least one "collected at" line appears per episode.
- Final result drained on shutdown.
- Average async eval wall time < 1s per trigger (non-blocking).
- `pooling_savings_mean` values within noise between sync and async runs.
- `nvidia-smi` during training shows activity on both `cuda:0` (training) and
  `cuda:1` (eval worker) with no activity on `cuda:2`.

**GPU split verification:**
```bash
# In a second terminal while training runs:
watch -n 2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```
Expect `cuda:0` busy during rollout/update phases, `cuda:1` busy in short bursts
during eval, `cuda:2` idle.

---

## Alternatives Considered

**Threading instead of multiprocessing**: The simulation loop is Python CPU-bound and
competes for the GIL with training's Python overhead (rollout collection, env step
coordination). Rejected in favour of spawn-process isolation.

**CUDA streams** (same process, concurrent kernel execution): Requires coordinating
two separate CUDA stream contexts in one process. PyTorch GIL means kernel dispatch
is still serialized. Complex for marginal gain.

**Async RL (IMPALA-style decoupled actors/learner)**: Changes the fundamental RL
algorithm (introduces policy lag in rollouts). Offers at most 1.3× training speedup.
Rejected — the bottleneck is callback time, not rollout/update throughput.

**Thread pool for the 10 pooling_simulation iterations** (parallelise within one
callback): Would require deep changes to the simulation (each iter currently runs
sequentially and is pure Python). Not helpful if the callback itself is overlapped.
