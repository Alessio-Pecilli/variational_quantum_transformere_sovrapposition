from multiprocessing import Pool, cpu_count
import numpy as np
import os
import time

def get_hpc_workers_max():
    omp = os.environ.get('OMP_NUM_THREADS')
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm and int(slurm) > 1:
        return int(slurm)
    elif omp and int(omp) > 1:
        return int(omp)
    else:
        return max(1, cpu_count() - 1)

def create_smart_batches(num_params, num_workers):
    target_batches = min(num_workers * 2, 20)
    batch_size = max(1, num_params // target_batches)
    return [list(range(i, min(i + batch_size, num_params)))
            for i in range(0, num_params, batch_size)]

def compute_gradient_batch(batch_data):
    from quantum_mpi_utils import _compute_single_gradient_component
    param_indices, params, shift, states_calc, U, Z, num_layers, dim, circuit_func = batch_data
    grads = []
    for idx in param_indices:
        grad = _compute_single_gradient_component(idx, params, shift, states_calc, U, Z, num_layers, dim, circuit_func)
        grads.append(grad)
    return param_indices, grads
