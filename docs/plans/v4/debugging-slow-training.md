# RL Training Slowness — Debug Session

## 1. What I Know

### Codebase entrypoints (read in this order)

| File | What it does |
|------|-------------|
| `scripts/train_rl.py` | **Main entrypoint.** Parses args, builds env, instantiates SAC/PPO, calls `model.learn()`. Start here. |
| `octopus/env.py` | The Gymnasium env. `reset()` samples a pod and builds the event list. `step()` places one VM and computes reward. |
| `octopus/data.py` | `load_trace()`, `load_topology()`, `precompute_pod_events()`. The precompute cache is the key speed fix. |
| `octopus/augmentation.py` | Episode-level augmentation transforms (scale, jitter, link failures, etc.). Applied inside `reset()`. |
| `tests/napkin_math.py` | **Benchmark script.** Measures env step time, reset time (cold vs. cached), and training throughput across N=1..32. Run this first. |

### How training works (simplified)

```
train_rl.py
  └── SubprocVecEnv([env_fn] * N)   # N parallel workers
        └── OctopusMemPoolEnv.reset()  # sample pod, build events, apply augmentation
        └── OctopusMemPoolEnv.step()   # place VM, compute reward → (obs, reward, done)
  └── SAC.learn(total_timesteps)
        └── collect N transitions → replay buffer
        └── every train_freq steps → gradient update from buffer (batch_size=512)
```

### The known bottleneck

`env.reset()` is expensive because it runs an `O(|VMs|)` filter-and-sort pipeline to build the event list for a randomly sampled pod. With a large trace (271k VMs) this dominates wall-clock time between episodes.

---

## 2. Minimal Reproducible Setup

### Activate environment

```bash
cd /home/pm3371/gitrepos/octopus-allocation
source venv/bin/activate
```

### Run the benchmark (shows exact slowness numbers)

```bash
python tests/napkin_math.py
```

This prints:
- `Q1` — env step time (should be ~0.1 ms, i.e. fast)
- `Q1b` — reset time **without** precomputed cache (the slow case)
- `Q1c` — reset time **with** precomputed cache (should be ~7 ms)
- `Q2` — GPU info and utilization at rest
- `Q3` — training throughput for N=1,4,8,16,32 with steps/sec and GPU%

### Run a short training session with timing instrumentation

Add these print statements to `scripts/train_rl.py` around `model.learn()`:

```python
import time, torch

t0 = time.perf_counter()
model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
elapsed = time.perf_counter() - t0

sps = args.total_timesteps / elapsed
print(f"\n[TIMING] total_timesteps={args.total_timesteps}")
print(f"[TIMING] elapsed={elapsed:.1f}s  →  {sps:.0f} steps/sec")
if torch.cuda.is_available():
    print(f"[TIMING] GPU util (end): {torch.cuda.utilization(0)}%")
```

Then run:

```bash
# Baseline (slow path — N=1, no precompute)
python scripts/train_rl.py --run-id debug_baseline --fast --total-timesteps 2000

# With parallelism
python scripts/train_rl.py --run-id debug_n32 --fast --total-timesteps 2000 --n-envs 32
```

### Add per-reset timing inside `env.py`

In `OctopusMemPoolEnv.reset()`, wrap the event-building path:

```python
import time

def reset(self, seed=None, options=None):
    t0 = time.perf_counter()

    # ... existing reset logic ...

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > 50:   # only print slow resets
        print(f"[SLOW RESET] {elapsed_ms:.1f} ms  seed={self._episode_seed}")

    return obs, {}
```

---

## 3. Exact Setup and What Has Been Tried

### Hardware (hulk)

- 3× NVIDIA TITAN RTX (24 GB each)
- CUDA 12.5, kernel 5.15.0
- Pin GPU with `CUDA_VISIBLE_DEVICES=0`

### Software

- Python 3.10, virtualenv at `venv/`
- Stable-Baselines3 (SAC + PPO)
- Algorithm: SAC (off-policy, replay buffer)

### What was tried

| Attempt | Result |
|---------|--------|
| Increase `train_freq` and `batch_size` alone (N=1) | ~1× — no improvement. CPU/Python overhead dominates, GPU is idle. |
| `SubprocVecEnv` with N=4,8,16,32 | Near-linear speedup. N=32 is the sweet spot on this machine. |
| `precompute_pod_events()` cache for reset | Reset time 120 ms → 7 ms (16.6×). Biggest single win. |
| Combining both (N=32 + precompute) | 26.5× total speedup over baseline. |
| Augmentation (scale, jitter, link failures) | Added inside `reset()` after cache load. Cost ~1 ms — negligible. |

### What still doesn't work / is unclear

- Training is faster but **policy quality is not improving** (20× MPD imbalance, greedy beats SAC). This is a reward/state problem, not a throughput problem.
- GPU utilization remains low even at N=32 — the bottleneck has shifted from reset to IPC overhead between the 32 subprocess workers.
- `train_freq` and `gradient_steps` scaling with N is untested at large N.

---

## 4. Benchmark Numbers

All measured on a single TITAN RTX with SAC, `net_arch=[128,128]`, 5,000 steps.

| Configuration | Steps / sec | Speedup | GPU util |
|---------------|-------------|---------|----------|
| Baseline `DummyVecEnv` (N=1, tf=1, bs=256) | 46 | 1× | <10% |
| N=1, tf=16, gs=16, bs=512 | 50 | 1.1× | <10% |
| `SubprocVecEnv` N=4 | 191 | 4.1× | — |
| `SubprocVecEnv` N=8 | 369 | 8.0× | — |
| `SubprocVecEnv` N=16 | 657 | 14.2× | — |
| `SubprocVecEnv` N=32 | **1221** | **26.5×** | — |

**Reset time:**

| Mode | Time |
|------|------|
| Cold (no cache, AMS20 trace ~23k VMs) | ~120 ms |
| Precomputed cache | ~7 ms |
| Speedup | 16.6× |

**Env step time (isolated, no training loop):** ~0.1 ms → env itself is not the problem.

**Key finding:** `train_freq` and `batch_size` alone give no gain. The bottleneck is Python/CPU serial throughput in the training loop, not GPU kernel size. Parallelism via `SubprocVecEnv` is the only lever that helps.

---

## 5. Follow-up Q&A

### Q: How does the callback-based eval work? What is `PoolingSavingsCallback`?

There are two parallel eval mechanisms running during training:

**`EvalCallback` (SB3 built-in)**
- Runs every `eval_freq` steps on a separate `DummyVecEnv` eval env
- Measures mean episode reward (the RL reward signal)
- Saves `best_model.zip` when mean reward improves

**`PoolingSavingsCallback` (custom)**
- Also runs every `eval_freq` steps
- Computes domain-specific metric: `pooling_savings_mean` and `pooling_savings_std` (i.e. how much better than baseline greedy the policy is, in terms of pooling ratio)
- Logged to W&B as `eval/pooling_savings_mean`
- Does NOT save a checkpoint — it's a diagnostic signal only

The two can diverge: reward can improve while pooling savings stagnate (reward shaping issue) or vice versa. Watch both.

---

### Q: How do I run augmentation across all training traces with W&B checkpointing?

The ten preprocessed pickles under `data/traces/` use **one AMS trace plus nine other datacenters** (BLA, BN9, …), not `AMS20PrdApp01`–`06`. Stems must match `*.sqlite.pkl` filenames.

Either **omit `--aug-traces`** (uses the same default list as `train_rl.py`) or pass the stems explicitly:

```bash
cd /home/pm3371/gitrepos/octopus-allocation
source venv/bin/activate

python scripts/train_rl.py \
  --run-id aug_all_traces \
  --augmentation \
  --multi-trace \
  --aug-traces \
    AMS20PrdApp19-tround \
    BLAPrdApp19-troundgrt5m \
    BN9PrdApp18-troundgrt5m \
    DSM08PrdApp05-troundgrt5m \
    DUB24PrdApp09-troundgrt5m \
    LON23PrdApp01-troundgrt5m \
    LVL01PrdApp05-troundgrt5m \
    SG2PrdApp35-troundgrt5m \
    SYD21PrdApp07-troundgrt5m \
    YTO21PrdApp05-troundgrt5m \
  --wandb \
  --n-envs 32 \
  --total-timesteps 500000
```

**To verify W&B checkpointing is happening:**
- Look for `WandbCallback` in the callback list printed at startup
- In W&B dashboard: check the `Artifacts` tab for your run — model checkpoints should appear as `model-<run-id>` artifacts
- Locally: `runs/<run-id>/` should accumulate `.zip` files

**Flags:**
- `--augmentation` — enables episode-level augmentation (scale, jitter, etc.)
- `--multi-trace` — samples from multiple traces per episode instead of a fixed one
- `--aug-traces` — list of trace stems matching `data/traces/<stem>.sqlite.pkl` (omit this flag to use the built-in default list)
- `--wandb` — activates `WandbCallback`, which logs metrics and saves model artifacts

---

### Q: Is inference on CPU or GPU? Why is GPU utilization low?

**Inference runs on GPU** when `--device auto` is passed (default) and CUDA is available. The policy forward pass (obs → action) happens in the main process on GPU.

**Why GPU utilization stays low at N=32:**

The bottleneck is IPC round-trips, not compute:

```
Main process (GPU)
  ├── sends obs request to 32 subprocesses    ← 32 IPC round-trips
  ├── waits for all 32 env steps to return    ← serialized by slowest worker
  └── runs one GPU forward pass (tiny batch)  ← ~0.1 ms, GPU is done instantly
```

Each IPC round-trip (pickle → pipe → subprocess → step → pipe → unpickle) takes ~0.3–1 ms. With N=32 workers the GPU waits ~10–30 ms between forward passes, giving <10% utilization despite running on CUDA.

**To actually saturate the GPU:** would need much larger `batch_size` for gradient updates (currently 512) or vectorized env that avoids subprocess IPC (e.g. Isaac Gym-style GPU envs). Neither is practical here given the Python env logic.
