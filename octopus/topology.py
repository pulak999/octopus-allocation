"""
Topology utilities: pod generation, matrix expansion, link-failure injection.

All functions match the notebook (memory-pooling-v0.ipynb) exactly.
"""

import random
import copy
import numpy as np


def generate_pod_to_nodes(node_to_vms, pod_size, seed):
    """Randomly partition nodes into pods of *pod_size*.

    Parameters
    ----------
    node_to_vms : dict
        ``{node_id: [vm_keys …]}``.  Only the keys are used.
    pod_size : int
    seed : int

    Returns
    -------
    pod_to_nodes : dict[int -> list[int]]
        ``{pod_id: [node_id, …]}`` for each full pod.
    """
    random.seed(seed)
    node_list = [int(node_id) for node_id in node_to_vms.keys()]
    random.shuffle(node_list)

    pod_to_nodes = {}
    for i in range(len(node_list) // pod_size):
        pod_to_nodes[i] = list(node_list[i * pod_size : (i + 1) * pod_size])
    return pod_to_nodes


def expand_M_to_all_nodes(M, pod_to_nodes):
    """Expand a per-pod topology matrix into a block-diagonal global matrix.

    Returns
    -------
    node_to_M : dict[node_id -> row_index_in_expanded_M]
    expanded_M : list[list[int]]
    """
    pod_list = sorted(pod_to_nodes.keys())
    num_nodes_per_pod = len(pod_to_nodes[pod_list[0]])
    assert len(M) == num_nodes_per_pod
    num_mhd_per_pod = len(M[0])
    num_pods = len(pod_to_nodes)

    node_to_M = {}
    expanded_M = []
    for pod in pod_list:
        local_M = []
        for node in range(num_nodes_per_pod):
            M_row = []
            node_to_M[pod_to_nodes[pod][node]] = len(expanded_M) + len(local_M)
            for mhd_pod in range(num_pods):
                if mhd_pod == pod:
                    M_row.extend(M[node])
                else:
                    M_row.extend([0] * num_mhd_per_pod)
            local_M.append(M_row)
        expanded_M.extend(local_M)
    return node_to_M, expanded_M


def remove_ones(matrix, ratio, seed=None):
    """Remove edges from a topology (simulating CXL link failures).

    Exactly ``floor(total_ones * ratio)`` ones are removed, while ensuring
    every row that originally had at least one ``1`` keeps at least one.

    Raises
    ------
    ValueError
        If the requested removal count is infeasible.
    """
    if seed is not None:
        random.seed(seed)

    ones_positions = [
        (i, j)
        for i, row in enumerate(matrix)
        for j, val in enumerate(row)
        if val == 1
    ]
    total_ones = len(ones_positions)
    target_remove = int(total_ones * ratio)
    if total_ones == 0 or target_remove == 0:
        return copy.deepcopy(matrix)

    must_keep = set()
    rows_with_ones = 0
    for i, row in enumerate(matrix):
        row_ones = [(i, j) for j, val in enumerate(row) if val == 1]
        if row_ones:
            rows_with_ones += 1
            must_keep.add(random.choice(row_ones))

    max_removable = total_ones - rows_with_ones
    if target_remove > max_removable:
        raise ValueError(
            f"Infeasible: requested to remove {target_remove} ones "
            f"but at least one '1' must remain in each of the {rows_with_ones} "
            f"rows that originally had a '1' (max removable = {max_removable})."
        )

    removable = [pos for pos in ones_positions if pos not in must_keep]
    positions_to_remove = set(random.sample(removable, target_remove))

    new_matrix = copy.deepcopy(matrix)
    for i, j in positions_to_remove:
        new_matrix[i][j] = 0
    return new_matrix
