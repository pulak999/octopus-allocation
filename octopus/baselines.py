"""
Baseline allocation policies.

greedy_alloc  — fill least-loaded accessible MPDs (matches notebook exactly).
"""

import numpy as np


def greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec):
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
