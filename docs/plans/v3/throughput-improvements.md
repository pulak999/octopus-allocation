# Throughput Improvements: Numba JIT + Async Rollouts

Current baseline: ~244 fps (timesteps/sec) with 32 SubprocVecEnv workers.
Both changes below target the env-side bottleneck, not the policy network.

---

## 1. Numba JIT on hot env loops

### What to change

Two methods in `octopus/env.py` are pure numeric loops that Numba can compile to C speed:

**`_process_departures_through`** — called between every VM arrival event, iterates
over ticks and subtracts from `dealloc_events`. Currently a Python range loop.

**`_compute_D_j`** — called on every `step()` in reward variants A/B, iterates over
all active VMs per MPD. Nested Python loops.

### Obstacle: `mpd_vm_allocs`

`mpd_vm_allocs` is a `list[list[tuple]]` — Numba can't compile over Python objects.
It needs to become a padded NumPy array:

```
mpd_vm_allocs_dt:  int32[num_mhd, max_active]   # dealloc ticks
mpd_vm_allocs_mem: float32[num_mhd, max_active]  # memory GB
mpd_vm_allocs_n:   int32[num_mhd]                # count of active VMs per MPD
```

`max_active` can be sized conservatively (e.g. 512) and checked at episode start.
Insertions and deletions become index-tracked writes rather than list appends.

### What the JIT'd functions look like

```python
from numba import njit

@njit
def process_departures_jit(dealloc_events, host_dealloc, cur_cxl, cur_host,
                            start, end):
    for t in range(start, end + 1):
        for j in range(cur_cxl.shape[0]):
            cur_cxl[j] -= dealloc_events[t, j]
            if cur_cxl[j] < 0.0:
                cur_cxl[j] = 0.0
        for h in range(cur_host.shape[0]):
            cur_host[h] -= host_dealloc[t, h]
            if cur_host[h] < 0.0:
                cur_host[h] = 0.0

@njit
def compute_D_j_jit(alloc_dt, alloc_mem, alloc_n, tick, W, num_mhd):
    D = np.zeros(num_mhd)
    t_plus_W = tick + W
    for j in range(num_mhd):
        for k in range(alloc_n[j]):
            dt = alloc_dt[j, k]
            if dt <= t_plus_W:
                D[j] += alloc_mem[j, k] * (1.0 - (dt - tick) / W)
    return D
```

Numba compiles on first call (~1s, cached). All subsequent calls run at native speed.

### Expected gain

`_process_departures_through` and `_compute_D_j` are the dominant CPU cost per step.
Expect 5–20× speedup on those paths → overall env step ~3–10× faster depending on
how much time is spent elsewhere (event sorting, augmentation, reset).

---

## 2. Async double-buffered rollouts

### The problem with SubprocVecEnv

GPU and CPU alternate: GPU idles while 32 envs step, envs idle while GPU runs inference.
With an expensive env step, roughly half of wallclock time is wasted on each side.

### The fix: double-buffered workers

Split envs into two buffers A and B (16 envs each, same total).

```
GPU:  [ inference(A) ][ inference(B) ][ inference(A) ] ...
CPU:  [ B.step()     ][ A.step()     ][ B.step()     ] ...
```

While the GPU processes buffer A's observations, buffer B's envs are already executing
their steps. GPU and CPU overlap completely — neither idles.

### Implementation path

PufferLib's `PufferEnv` wrapper implements this in C with OMP threads. Two options:

**Option A — use PufferLib directly**  
Wrap `OctopusMemPoolEnv` as a PufferLib environment. Requires conforming to PufferLib's
env interface (close to Gymnasium, minor adapter needed). Gets async workers, pinned
observation memory, and C-native threading for free.

**Option B — implement in Python**  
Two `ThreadPoolExecutor` workers, each owning half the envs. Main loop alternates:
submit step actions to worker B, run GPU inference on A's obs, collect B results,
submit to worker A, run inference on B, repeat. Simpler than it sounds; no C required.
Pinned memory (`torch.zeros(...).pin_memory()`) for obs buffers avoids an extra
host→device copy per step.

Option A is less work and gets pinned memory automatically. Option B avoids a new
dependency and is easier to debug.

### Stale observation note

Actions applied to buffer B are computed from obs that are one rollout step old.
This is standard in async RL and does not meaningfully hurt SAC or PPO.

### Expected gain

PufferLib reported +2M sps (out of ~15M total) from this change alone — roughly 15%.
On our setup where env steps are more expensive relative to the policy, the overlap
benefit is proportionally larger. Rough estimate: 1.5–2.5× throughput improvement
on top of Numba gains.

---

## Combined effect

| Change | Estimated multiplier |
|---|---|
| Numba JIT (env loops) | 3–10× |
| Async rollouts (overlap) | 1.5–2.5× |
| Combined | ~5–25× |

At the low end: 244 × 5 ≈ 1,200 fps → 2M steps in ~28 min (vs 2h17m now).  
At the high end: 244 × 25 ≈ 6,000 fps → 2M steps in ~6 min.

Profile after Numba to confirm whether env or GPU is the new bottleneck before
investing in async rollouts.
