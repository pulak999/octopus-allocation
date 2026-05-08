"""Numba JIT kernels for env hot paths.

Operates on the flat MPD arrays (_mpd_dt, _mpd_mem, _mpd_n) from Chunk 1.
Uses explicit loops inside @njit — faster than numpy vectorisation inside JIT
because it avoids temporary array allocations.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def compute_D_j(mpd_dt, mpd_mem, mpd_n, tick, W):
    """D_j(t, W): time-weighted departure relief per MPD.

    D_j[j] = sum_{s in active VMs on j, dt_s <= tick+W}
                mem_s * (1 - (dt_s - tick) / W)

    Parameters
    ----------
    mpd_dt  : float64 (num_mhd, capacity)  — dealloc ticks
    mpd_mem : float64 (num_mhd, capacity)  — alloc GB
    mpd_n   : int32   (num_mhd,)           — valid-slot count
    tick    : int
    W       : int                          — lookahead window
    """
    num_mhd = mpd_n.shape[0]
    D = np.zeros(num_mhd, dtype=np.float64)
    t_plus_W = np.float64(tick + W)
    f_tick = np.float64(tick)
    f_W = np.float64(W)
    for j in range(num_mhd):
        n = mpd_n[j]
        acc = 0.0
        for s in range(n):
            dt = mpd_dt[j, s]
            if dt <= t_plus_W:
                acc += mpd_mem[j, s] * (1.0 - (dt - f_tick) / f_W)
        D[j] = acc
    return D


@njit(cache=True)
def compute_D_S_j(mpd_dt, mpd_mem, mpd_n, tick, W):
    """D_j and S_j in a single pass — avoids iterating the arrays twice.

    S_j[j] = sum of mem for active VMs on j with dt > tick+W  (sticky pressure).
    """
    num_mhd = mpd_n.shape[0]
    D = np.zeros(num_mhd, dtype=np.float64)
    S = np.zeros(num_mhd, dtype=np.float64)
    t_plus_W = np.float64(tick + W)
    f_tick = np.float64(tick)
    f_W = np.float64(W)
    for j in range(num_mhd):
        n = mpd_n[j]
        d_acc = 0.0
        s_acc = 0.0
        for s in range(n):
            dt = mpd_dt[j, s]
            mem = mpd_mem[j, s]
            if dt <= t_plus_W:
                d_acc += mem * (1.0 - (dt - f_tick) / f_W)
            else:
                s_acc += mem
        D[j] = d_acc
        S[j] = s_acc
    return D, S
