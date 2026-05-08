# Throughput log — SB3 vs JAX comparison

Env-only benchmarks (random actions, no policy inference, no SAC updates).
Script: `scripts/bench5min.py` — 300s timing window, excludes trace loading and cache prep.
Hardware: 3× NVIDIA TITAN RTX, kernel 5.15, Python 3.10, n_envs=32.
Config: `reward_variant=A`, `lookahead_window=200`, `reward_lambda=0.2`,
        `multi_trace=True`, `augmentation=True` (scale triangular 0.7–1.2).
Topology: `AG16x6_expander_quads_r5_sym_fixed.csv` (16 hosts, 6 MPDs, max_degree=8, obs_dim=50).
Traces: 10 cleaned Azure VM traces from `data/traces/cleaned_traces/`.

---

## Run 1 — SB3 pre-SPEEDUP_PLAN (2026-05-05)

**Stack:** `mpd_vm_allocs` list-of-lists, no precompute cache, no Numba JIT.
No `trace_pool_events` passed (every reset calls `_build_events()` ~120ms).

| Window | Elapsed (s) | Window fps |
|--------|------------|-----------|
| 16,000 | 7 | 2,444 |
| 32,000 | 17 | 1,545 |
| 48,000 | 30 | 1,242 |
| 64,000 | 45 | 1,078 |
| 80,000 | 60 | 1,017 |
| 96,000 | 76 | 1,012 |
| 112,000 | 94 | 901 |
| 128,000 | 111 | 927 |
| 144,000 | 127 | 983 |
| 160,000 | 143 | 1,041 |
| 176,000 | 159 | 969 |
| 192,000 | 174 | 1,055 |
| 208,000 | 192 | 929 |
| 224,000 | 208 | 1,012 |
| 240,000 | 223 | 1,037 |
| 256,000 | 238 | 1,030 |
| 272,000 | 255 | 992 |
| 288,000 | 271 | 960 |
| 304,000 | 288 | 965 |

**Total transitions:** 316,256  
**Wall time:** 300.3s  
**Overall fps:** 1,053  
**Steady-state fps:** 1,009 (post-warmup mean, windows 3+)  
**Min / Max:** 901 / 2,444  
**Cache prep:** N/A (no cache)  

---

## Run 2 — SB3 post-SPEEDUP_PLAN Chunks 1+2+3 (2026-05-05)

**Stack:** flat `_mpd_dt/_mpd_mem/_mpd_n` arrays (Chunk 1), precompute cache with
disk persistence (Chunk 2, 128 seeds × 10 traces), Numba JIT kernels `compute_D_j` /
`compute_D_S_j` (Chunk 3).

**Cache prep:** 379.5s first run (built fresh); subsequent runs ~3s from disk.

| Window | Elapsed (s) | Window fps |
|--------|------------|-----------|
| 16,000 | 2 | 7,775 |
| 32,000 | 5 | 5,035 |
| 48,000 | 11 | 2,936 |
| 64,000 | 17 | 2,686 |
| 80,000 | 23 | 2,683 |
| 96,000 | 29 | 2,354 |
| 112,000 | 37 | 2,208 |
| 128,000 | 42 | 2,802 |
| 144,000 | 48 | 2,777 |
| 160,000 | 54 | 2,853 |
| 176,000 | 59 | 2,792 |
| 192,000 | 65 | 2,687 |
| 208,000 | 71 | 2,805 |
| 224,000 | 77 | 2,894 |
| 240,000 | 82 | 2,734 |
| 256,000 | 88 | 2,846 |
| 272,000 | 94 | 2,840 |
| 288,000 | 99 | 2,934 |
| 304,000 | 105 | 2,872 |
| 320,000 | 110 | 2,849 |
| 336,000 | 116 | 2,923 |
| 352,000 | 122 | 2,771 |
| 368,000 | 128 | 2,603 |
| 384,000 | 133 | 2,886 |
| 400,000 | 139 | 2,826 |
| 416,000 | 145 | 2,675 |
| 432,000 | 151 | 2,795 |
| 448,000 | 156 | 2,781 |
| 464,000 | 162 | 2,786 |
| 480,000 | 168 | 2,599 |
| 496,000 | 174 | 2,911 |
| 512,000 | 180 | 2,639 |
| 528,000 | 186 | 2,848 |
| 544,000 | 191 | 2,765 |
| 560,000 | 197 | 2,766 |
| 576,000 | 203 | 2,533 |
| 592,000 | 210 | 2,627 |
| 608,000 | 215 | 2,782 |
| 624,000 | 221 | 2,853 |
| 640,000 | 227 | 2,797 |
| 656,000 | 232 | 2,736 |
| 672,000 | 238 | 2,827 |
| 688,000 | 244 | 2,762 |
| 704,000 | 250 | 2,781 |
| 720,000 | 256 | 2,675 |
| 736,000 | 262 | 2,664 |
| 752,000 | 267 | 2,959 |
| 768,000 | 273 | 2,912 |
| 784,000 | 278 | 2,801 |
| 800,000 | 284 | 2,739 |
| 816,000 | 290 | 2,683 |
| 832,000 | 296 | 2,580 |

**Overall fps:** 2,799  
**Steady-state fps:** 2,757 (post-warmup mean, windows 3+)  
**Min / Max (steady):** 2,533 / 2,959  
**Cache prep (first run):** 379.5s → ~3s on subsequent runs  
**Speedup vs Run 1:** 2.73× env-only  
**Speedup vs run7 training baseline (~53 fps):** 52.8× env-only  

---

## Summary

| Run | Stack | Steady fps | vs Run 1 |
|-----|-------|-----------|---------|
| 1 | SB3 pre-SPEEDUP | 1,009 | 1.0× |
| 2 | SB3 post-SPEEDUP (Chunks 1+2+3) | **2,757** | **2.73×** |
| 3 | JAX (Phase 2+) | TBD | TBD |

**Notes:**
- All numbers are env-only (no policy inference). Full SAC training fps is lower (GPU bottleneck).
- Run7 full training was ~53 fps; SPEEDUP_PLAN target was 185 fps training. These env-only numbers
  confirm the env is no longer the training bottleneck — GPU policy inference is.
- JAX target: exceed 2,757 fps env-only at n_envs=32, then measure end-to-end training fps
  with JIT-compiled rollout + updates to see if it moves the training fps needle.

---

## Phase 4 training snapshot (2026-05-05)

End-to-end training runs at matched high-level settings:
`reward_variant=current`, `lookahead_window=200`, `reward_lambda=0.2`,
`trace=AMS20PrdApp19-tround.sqlite`, `n_envs=32`, `train_freq=32`, `total_timesteps≈50k`.

| Run | Stack | Steps | Wall time | End-to-end fps |
|-----|-------|------:|----------:|---------------:|
| 4 | JAX (`scripts/train_jax.py`) | 50,016 | 72.8s | **687** |
| 5 | SB3 (`scripts/train_rl.py`) | 50,176 | 58s | **865** |

Current snapshot: SB3 is ~1.26x faster at this 50k training configuration.
