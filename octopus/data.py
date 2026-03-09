"""
Data loading utilities for Octopus CXL memory pooling.

Provides:
  - VM class (must be defined before unpickling trace files)
  - load_trace()  — load a pre-processed Azure VM trace pickle
  - load_topology() — load a topology CSV and return adjacency matrix
"""

import datetime
import pickle
import csv
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
