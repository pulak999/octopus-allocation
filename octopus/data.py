"""
Data loading utilities for Octopus CXL memory pooling.

Provides:
  - VM class (must be defined before unpickling trace files)
  - load_trace()  — load a pre-processed Azure VM trace pickle
  - load_topology() — load a topology CSV and return adjacency matrix
  - precompute_pod_events() — precompute per-seed event lists so env.reset() is fast
"""

import datetime
import pickle
import csv
import random as _random
import numpy as np


# ---------------------------------------------------------------------------
# VM class — must match the structure used when the pickles were created
# ---------------------------------------------------------------------------
class VM:
    """Virtual machine record from an Azure production trace."""

    def __init__(self):
        self.vm_id = -1
        self.node_id = -1
        self.cores = -1
        self.memory = -1
        self.nic = -1
        self.rss = list()
        self.ssd = -1
        self.max_mem_ts = list()
        self.avg_mem_ts = list()
        self.max_cpu_ts = list()
        self.avg_cpu_ts = list()
        self.start_time = datetime.datetime(2024, 1, 1, 1, 1)
        self.end_time = datetime.datetime(2020, 1, 1, 1, 1)
        self.instance_role = ""
        self.subscription_id = ""


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
class _VMUnpickler(pickle.Unpickler):
    """Custom unpickler that maps ``__main__.VM`` → ``octopus.data.VM``.

    The trace pickles were created with ``VM`` defined in ``__main__``.
    When we load them from a different module we need this redirect.
    """

    def find_class(self, module, name):
        if name == "VM":
            return VM
        return super().find_class(module, name)


def load_trace(cluster_name, trace_dir="data/traces"):
    """Load a pre-processed Azure VM trace pickle.

    Parameters
    ----------
    cluster_name : str
        Cluster file stem, e.g. ``"AMS20PrdApp19-tround.sqlite"``.
    trace_dir : str
        Directory containing ``<cluster_name>.pkl`` files.

    Returns
    -------
    all_vms : dict[vm_key -> VM]
    node_to_vms : dict[node_id -> list[vm_key]]
    node_to_machine : dict[node_id -> machine_type]
    vm_type_sz : dict
    machine_sz : dict[machine_type -> list of 4 resource capacities]
    """
    path = f"{trace_dir}/{cluster_name}.pkl"
    with open(path, "rb") as f:
        all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = (
            _VMUnpickler(f).load()
        )
    return all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz


# ---------------------------------------------------------------------------
# Topology loading
# ---------------------------------------------------------------------------
def precompute_pod_events(trace_data, M, seeds, mem_idx=1, skip_hotfix=False):
    """Precompute VM arrival event lists for a set of pod-assignment seeds.

    This runs the same logic as OctopusMemPoolEnv._generate_pod() +
    _build_events(), but offline so that env.reset() can look up the result
    instantly instead of recomputing it.

    Parameters
    ----------
    trace_data : tuple
        As returned by load_trace().
    M : list[list[int]]
        Adjacency matrix (num_hosts × num_pools).
    seeds : iterable[int]
        The set of seeds to precompute.
    mem_idx : int
        Index into rss / machine_sz for memory (default 1).

    Returns
    -------
    cache : dict[int, tuple]
        Mapping seed → (events, pod_dur, pod_dram, base_time, pod_start_ts)
        where events is the sorted list of (tick, pod_id, vm_mem, dealloc_tick).
    """
    import datetime as _dt
    from datetime import timedelta

    all_vms, node_to_vms, node_to_machine, _vm_type_sz, machine_sz = trace_data
    pod_size = len(M)

    cache = {}
    node_list_all = [int(nid) for nid in node_to_vms.keys()]

    for seed in seeds:
        # --- Reproduce _generate_pod ---
        _random.seed(seed)
        shuffled = node_list_all[:]
        _random.shuffle(shuffled)
        pod_nodes = shuffled[:pod_size]
        node_to_pod_id = {n: i for i, n in enumerate(pod_nodes)}

        pod_dram = 0.0
        for node in pod_nodes:
            cap = np.asarray(machine_sz[node_to_machine[node]], dtype=float)
            pod_dram += float(cap[mem_idx])

        # --- Reproduce _build_events ---
        n_start = _dt.datetime.max
        n_end = _dt.datetime.min
        any_vm = False
        for node in pod_nodes:
            for vmkey in node_to_vms.get(node, []):
                vm = all_vms[vmkey]
                any_vm = True
                if vm.start_time < n_start:
                    n_start = vm.start_time
                if vm.end_time > n_end:
                    n_end = vm.end_time

        if not any_vm:
            cache[seed] = ([], 0, pod_dram, _dt.datetime(2024, 1, 1), 0)
            continue

        base_time = _dt.datetime(
            n_start.year, n_start.month, n_start.day,
            n_start.hour, n_start.minute,
        )

        def to_tick(t):
            return int((t - base_time).total_seconds() // 300)

        pod_start_ts = to_tick(n_start)
        pod_end_ts = to_tick(n_end)
        pod_dur = pod_end_ts - pod_start_ts + 1

        if pod_dur <= 0:
            cache[seed] = ([], pod_dur, pod_dram, base_time, pod_start_ts)
            continue

        # HOTFIX: filter VMs that exceed per-node DRAM
        vmkey_to_skip: set = set()

        if not skip_hotfix:
            for node in pod_nodes:
                node_alloc = [[] for _ in range(pod_dur)]
                node_dealloc = np.zeros(pod_dur, dtype=np.float64)
                node_cap = float(
                    np.asarray(machine_sz[node_to_machine[node]], dtype=float)[mem_idx]
                )
                for vmkey in node_to_vms.get(node, []):
                    vm = all_vms[vmkey]
                    vm_s = to_tick(vm.start_time) - pod_start_ts
                    vm_e = to_tick(vm.end_time) - pod_start_ts
                    if vm_e < 0:
                        vmkey_to_skip.add(vmkey)
                        continue
                    mem = float(np.asarray(vm.rss, dtype=float)[mem_idx])
                    node_alloc[vm_s].append((vmkey, vm_e + 1, mem))
                    if vm_e + 1 < pod_dur:
                        node_dealloc[vm_e + 1] += mem
                cur = 0.0
                for ts in range(pod_dur):
                    cur -= node_dealloc[ts]
                    if cur < 0:
                        cur = 0.0
                    for vmkey, end_ts, mem in node_alloc[ts]:
                        cur += mem
                        if cur > node_cap:
                            cur -= mem
                            vmkey_to_skip.add(vmkey)
                            if end_ts < pod_dur:
                                node_dealloc[end_ts] -= mem

        events = []
        for node in pod_nodes:
            pod_id = node_to_pod_id[node]
            for vmkey in node_to_vms.get(node, []):
                if vmkey in vmkey_to_skip:
                    continue
                vm = all_vms[vmkey]
                vm_s = to_tick(vm.start_time) - pod_start_ts
                vm_e = to_tick(vm.end_time) - pod_start_ts
                if vm_e < 0:
                    continue  # VM ends before pod window — always skip
                mem = float(np.asarray(vm.rss, dtype=float)[mem_idx])
                if mem <= 0:
                    continue
                events.append((vm_s, pod_id, mem, vm_e + 1))
        events.sort(key=lambda e: (e[0], e[1], -e[2]))

        cache[seed] = (events, pod_dur, pod_dram, base_time, pod_start_ts)

    return cache


def load_topology(csv_path):
    """Load a topology CSV (``server,pool``) and return adjacency matrix.

    Returns
    -------
    matrix : list[list[int]]
        Adjacency matrix of shape ``[num_hosts, num_pools]``.
    num_hosts : int
    num_pools : int
    """
    with open(csv_path, "r") as fp:
        reader = csv.reader(fp)
        rows = [row for row in reader][1:]  # skip header
    server_dev_tup_list = [(int(row[0]), int(row[1])) for row in rows]

    num_pools = len(set(tup[1] for tup in server_dev_tup_list))
    num_hosts = len(set(tup[0] for tup in server_dev_tup_list))
    matrix = [[0 for _ in range(num_pools)] for _ in range(num_hosts)]
    for host, dev in server_dev_tup_list:
        matrix[host][dev] = 1
    return matrix, num_hosts, num_pools
