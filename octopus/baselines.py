"""
Baseline allocation policies.

greedy_alloc_ref — fill least-loaded accessible MPDs (matches notebook exactly).
greedy_alloc     — vectorized numpy argsort-based implementation (matches ref).
pid_alloc        — PID controller: distribute proportional to inverse load with
                   integral tracking of cumulative imbalance.
"""

import numpy as np


def greedy_alloc_ref(cxl_mem, mhd_list, cur_cxl_mem_vec):
    """Greedy allocation: fill the least-loaded accessible MPDs first.

    Parameters
    ----------
    cxl_mem : float
        Memory request in GB.
    mhd_list : list[int]
        Indices of accessible MPDs for this host.
    cur_cxl_mem_vec : np.ndarray
        Current load on every MPD (GB).  **Modified in-place** for the
        caller's bookkeeping (matches notebook convention).

    Returns
    -------
    alloc_arr : np.ndarray
        Allocation vector (same length as *cur_cxl_mem_vec*).
    """
    alloc_arr = np.zeros(len(cur_cxl_mem_vec))
    cur_cxl_mem_vec = np.maximum(cur_cxl_mem_vec, 0.0)

    while cxl_mem > 0:
        min_mhd = mhd_list[0]
        for mhd in mhd_list:
            if cur_cxl_mem_vec[mhd] < cur_cxl_mem_vec[min_mhd]:
                min_mhd = mhd

        num_min_mhd = 0
        next_min_mhd = None
        for mhd in mhd_list:
            if cur_cxl_mem_vec[mhd] == cur_cxl_mem_vec[min_mhd]:
                num_min_mhd += 1
            if cur_cxl_mem_vec[mhd] > cur_cxl_mem_vec[min_mhd] and (
                next_min_mhd is None
                or cur_cxl_mem_vec[mhd] < cur_cxl_mem_vec[next_min_mhd]
            ):
                next_min_mhd = mhd

        assert num_min_mhd > 0
        assert (
            next_min_mhd is None
            or cur_cxl_mem_vec[min_mhd] < cur_cxl_mem_vec[next_min_mhd]
        )

        min_val = cur_cxl_mem_vec[min_mhd]
        if next_min_mhd is None or cxl_mem <= num_min_mhd * (
            cur_cxl_mem_vec[next_min_mhd] - cur_cxl_mem_vec[min_mhd]
        ):
            for mhd in mhd_list:
                if cur_cxl_mem_vec[mhd] == min_val:
                    alloc_arr[mhd] += cxl_mem / num_min_mhd
                    cur_cxl_mem_vec[mhd] += cxl_mem / num_min_mhd
            cxl_mem = 0
        else:
            to_alloc = cur_cxl_mem_vec[next_min_mhd] - cur_cxl_mem_vec[min_mhd]
            for mhd in mhd_list:
                if cur_cxl_mem_vec[mhd] == min_val:
                    alloc_arr[mhd] += to_alloc
                    cur_cxl_mem_vec[mhd] += to_alloc
            cxl_mem -= to_alloc * num_min_mhd

    return alloc_arr


def greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec):
    mhd_arr = np.asarray(mhd_list)
    loads = np.maximum(cur_cxl_mem_vec[mhd_arr], 0.0).copy()
    n = len(mhd_arr)
    alloc = np.zeros(len(cur_cxl_mem_vec))

    # Sort by load ascending; water-fill from lowest level up
    order = np.argsort(loads, kind="stable")
    sorted_loads = loads[order]

    remaining = float(cxl_mem)
    # Water fill: maintain water level, progressively include more MPDs
    # At step i: MPDs order[i..n-1] are all at or above sorted_loads[i].
    # We raise the water level from sorted_loads[i] toward sorted_loads[i+1].
    # active_count = number of MPDs whose original load <= current water level.
    # We process each "step" where a new MPD joins the active pool.
    water_level = sorted_loads[0]
    # alloc_per_mhd tracks how much each MPD (by sorted index) has received
    alloc_per_sorted = np.zeros(n)

    for i in range(n):
        if remaining <= 0.0:
            break
        # At this point, MPDs order[0..i] have been brought up to water_level
        # (and water_level == sorted_loads[i] at the start of each iteration)
        # Active pool: all MPDs from 0..i (they're all at water_level now)
        # Next barrier: sorted_loads[i+1] (or inf if i == n-1)
        next_barrier = sorted_loads[i + 1] if i + 1 < n else np.inf
        active_count = i + 1

        gap = (next_barrier - water_level) * active_count
        if remaining <= gap:
            each = remaining / active_count
            alloc_per_sorted[:active_count] += each
            water_level += each
            remaining = 0.0
        else:
            each = next_barrier - water_level
            alloc_per_sorted[:active_count] += each
            water_level = next_barrier
            remaining -= gap
            # Next iteration: i+1 joins the pool already at water_level

    alloc[mhd_arr[order]] = alloc_per_sorted

    # Update cur_cxl_mem_vec in-place to match ref behavior
    cur_cxl_mem_vec[mhd_arr] += alloc[mhd_arr]
    return alloc


def pid_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, pid_state,
              kp=1.0, ki=0.01, kd=0.1):
    """PID allocation: distribute cxl_mem across accessible MPDs in mhd_list,
    proportional to inverse load with integral tracking cumulative imbalance.

    pid_state: dict with keys 'integral' (np.array len num_mhd), 'prev_error' (np.array).
               Mutated in-place each call.
    Returns: allocation vector (length = num_mhd, zero for inaccessible MPDs).
    """
    num_mhd = len(cur_cxl_mem_vec)
    if "integral" not in pid_state:
        pid_state["integral"] = np.zeros(num_mhd, dtype=np.float64)
    if "prev_error" not in pid_state:
        pid_state["prev_error"] = np.zeros(num_mhd, dtype=np.float64)

    acc_loads = np.array([cur_cxl_mem_vec[j] for j in mhd_list], dtype=np.float64)
    target_load = float(np.mean(acc_loads))
    error = target_load - acc_loads  # positive = underloaded

    dt = 1
    for i, j in enumerate(mhd_list):
        pid_state["integral"][j] += error[i] * dt

    d_error = error - pid_state["prev_error"][mhd_list]

    w = kp * error + ki * pid_state["integral"][mhd_list] + kd * d_error
    w = np.maximum(w, 0.0)

    w_sum = w.sum()
    if w_sum <= 0:
        # Fallback: uniform allocation
        w = np.ones(len(mhd_list), dtype=np.float64)
        w_sum = float(len(mhd_list))

    proportions = w / w_sum

    alloc_arr = np.zeros(num_mhd, dtype=np.float64)
    for i, j in enumerate(mhd_list):
        alloc_arr[j] = proportions[i] * cxl_mem

    for i, j in enumerate(mhd_list):
        pid_state["prev_error"][j] = error[i]

    return alloc_arr
