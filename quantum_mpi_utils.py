"""
MPI utilities for distributed quantum circuit training.
Implements data-parallel synchronous training following DDP pattern.
"""

import numpy as np
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from quantum_utils import get_params
from quantum_circuits import get_circuit_function


def build_variational_ansatz(n_qubits, num_layers):
    """
    Build the variational ansatz (parametric part) once with ParameterVector.
    This is the reusable template that will be bound with different parameters.
    
    Args:
        n_qubits (int): Number of qubits
        num_layers (int): Number of ansatz layers
        
    Returns:
        tuple: (qc_template, theta_vec)
            qc_template: QuantumCircuit with parameterized gates
            theta_vec: ParameterVector of variational parameters
    """
    # Calculate total number of parameters (V + K)
    param_shape = get_params(n_qubits, num_layers).shape
    n_params_per_type = int(np.prod(param_shape))
    num_params = 2 * n_params_per_type  # V and K parameters
    
    # Create parameter vector
    theta_vec = ParameterVector('θ', num_params)
    
    # Create quantum circuit template
    # Note: This is a placeholder - actual implementation depends on your ansatz structure
    qc_template = QuantumCircuit(n_qubits)
    
    # Simple example ansatz with RY + RZ rotations and entanglement
    # Adjust based on your actual ansatz in quantum_circuits.py
    param_idx = 0
    for layer in range(num_layers):
        # Single qubit rotations
        for q in range(n_qubits):
            qc_template.ry(theta_vec[param_idx], q)
            param_idx += 1
            qc_template.rz(theta_vec[param_idx], q)
            param_idx += 1
        
        # Entanglement (CNOT ladder)
        for q in range(n_qubits - 1):
            qc_template.cx(q, q + 1)
    
    return qc_template, theta_vec


def build_input_unitary(psi, U, Z):
    """
    Build the input encoding as a UnitaryGate from sentence states.
    This encoding is sentence-dependent and represents the quantum state of the input sentence.
    
    The encoding follows the structure used in quantum_circuits.py:
    - psi: initial state unitaries (Ψ vectors representing word combinations)
    - U: next word prediction unitaries (U† matrices)
    - Z: current word unitaries (Z† matrices)
    
    For compatibility with Qiskit's UnitaryGate, the matrix must be 2^n × 2^n dimensional.
    We use the FIRST psi matrix as the primary encoding, which represents the combined
    state of all words in the sentence (created by kronecker products in process_sentence_states).
    
    Args:
        psi (list): List of unitary matrices for initial states (from process_sentence_states)
        U (list): List of unitary matrices for next word prediction (U_dagger)
        Z (list): List of unitary matrices for current word (Z_dagger)
        
    Returns:
        UnitaryGate: Unitary encoding of the input sentence states
    """
    # Determine dimensions from input
    if not psi or len(psi) == 0:
        # Fallback to identity for empty input
        n_qubits = 4
        unitary_matrix = np.eye(2**n_qubits, dtype=complex)
        return UnitaryGate(unitary_matrix, label="InputEncoding")
    
    # Use the first psi matrix as the primary encoding
    # In quantum_circuits.py, each psi represents a superposition of word states
    # The first one encodes the most important transition
    unitary_matrix = psi[0].copy()
    
    # Verify dimension is a power of 2 (required for UnitaryGate)
    dim = unitary_matrix.shape[0]
    n_qubits = int(np.log2(dim))
    
    if 2**n_qubits != dim:
        raise ValueError(f"Matrix dimension {dim} is not a power of 2! Cannot create UnitaryGate.")
    
    # Optional: Add information from U and Z through phase modulation
    # This enriches the encoding without changing dimensionality
    if len(U) > 0 and len(Z) > 0:
        # Extract phase information from U[0] and Z[0]
        # Note: U and Z might have different dimensions than psi (e.g., 4x4 vs 16x16)
        # We only use the phase information that fits
        U_dim = min(U[0].shape[0], dim)
        Z_dim = min(Z[0].shape[0], dim)
        min_dim = min(U_dim, Z_dim)
        
        # Extract phase information from U[0] and Z[0]
        U_phases = np.angle(np.diag(U[0][:min_dim, :min_dim]))
        Z_phases = np.angle(np.diag(Z[0][:min_dim, :min_dim]))
        
        # Create phase modulation matrix (extend to full dimension)
        combined_phases = np.zeros(dim)
        combined_phases[:min_dim] = 0.5 * (U_phases + Z_phases)
        phase_matrix = np.diag(np.exp(1j * combined_phases))
        
        # Apply phase modulation: Ψ' = Ψ * Phase
        unitary_matrix = unitary_matrix @ phase_matrix
    
    # Ensure the matrix is unitary (should already be, but verify)
    # Check unitarity: U†U should be close to identity
    unitary_check = unitary_matrix.conj().T @ unitary_matrix
    identity = np.eye(dim)
    unitarity_error = np.linalg.norm(unitary_check - identity, 'fro')
    
    if unitarity_error > 1e-6:
        print(f"⚠️  Warning: Input encoding unitarity error: {unitarity_error:.2e}")
        print(f"   Applying QR correction...")
        
        # Apply Gram-Schmidt to enforce unitarity
        # Use QR decomposition which produces an orthonormal basis
        Q, R = np.linalg.qr(unitary_matrix)
        
        # Adjust phases to match original as closely as possible
        # Extract diagonal phases from R
        phases = np.angle(np.diag(R))
        phase_matrix = np.diag(np.exp(1j * phases))
        
        unitary_matrix = Q @ phase_matrix
        
        # Verify unitarity after correction
        unitary_check_corrected = unitary_matrix.conj().T @ unitary_matrix
        unitarity_error_corrected = np.linalg.norm(unitary_check_corrected - identity, 'fro')
        
        if unitarity_error_corrected > 1e-10:
            print(f"   ❌ Unitarity error after QR correction: {unitarity_error_corrected:.2e}")
        else:
            print(f"   ✅ Unitarity restored (error: {unitarity_error_corrected:.2e})")
    
    return UnitaryGate(unitary_matrix, label="InputEncoding")




def compose_full_circuit(qc_input, qc_template):
    """
    Compose input encoding + variational ansatz.
    
    Args:
        qc_input (QuantumCircuit): Input encoding circuit
        qc_template (QuantumCircuit): Variational ansatz template
        
    Returns:
        QuantumCircuit: Full circuit (encoding + ansatz)
    """
    qc_full = qc_input.copy()
    qc_full.compose(qc_template, inplace=True)
    return qc_full


def simulate_probabilities(qc_bound):
    """
    Simulate circuit and return probability distribution.
    
    Args:
        qc_bound (QuantumCircuit): Circuit with bound parameters
        
    Returns:
        np.ndarray: Probability distribution
    """
    # Use Statevector for fast simulation
    sv = Statevector.from_instruction(qc_bound)
    return sv.probabilities()


def _compute_single_gradient_component(param_index, params, shift, psi, U, Z, num_layers, dim, circuit_function):
    """
    Helper function to compute a single gradient component using parameter-shift rule.
    This function is designed to be called in parallel via multiprocessing.
    
    Args:
        param_index (int): Index of parameter to compute gradient for
        params (np.ndarray): Current parameter values
        shift (float): Parameter shift amount (π/2)
        psi (list): Initial state unitaries
        U (list): Next word unitaries
        Z (list): Current word unitaries
        num_layers (int): Number of layers
        dim (int): Dimension parameter
        circuit_function (callable): Circuit function for loss calculation
        
    Returns:
        float: Gradient component for this parameter
    """
    import os
    import time
    worker_pid = os.getpid()
    start_time = time.time()
    
    # Determine parameter shape
    n_params_single = len(params) // 2
    param_shape = get_params(2, num_layers).shape
    
    # Shift parameter up
    params_plus = params.copy()
    params_plus[param_index] += shift
    
    pV_plus = params_plus[:n_params_single].reshape(param_shape)
    pK_plus = params_plus[n_params_single:].reshape(param_shape)
    
    loss_plus = circuit_function(psi, U, Z, pV_plus, pK_plus, num_layers, dim)
    
    # Shift parameter down
    params_minus = params.copy()
    params_minus[param_index] -= shift
    
    pV_minus = params_minus[:n_params_single].reshape(param_shape)
    pK_minus = params_minus[n_params_single:].reshape(param_shape)
    
    loss_minus = circuit_function(psi, U, Z, pV_minus, pK_minus, num_layers, dim)
    
    # Parameter-shift formula: ∂L/∂θᵢ = 0.5 * (L(θ+π/2) - L(θ-π/2))
    gradient_component = 0.5 * (loss_plus - loss_minus)
    
    elapsed = time.time() - start_time
    print(f"    Worker PID:{worker_pid} param[{param_index}] = {gradient_component:.6f} ({elapsed:.3f}s)")
    
    return gradient_component


def loss_and_grad_for_sentence(sentence_data, params, qc_template, theta_vec, circuit_function, num_layers, dim, 
                                parallel=True, n_workers=None):
    """
    Calculate loss and gradient for a single sentence using parameter-shift rule.
    
    Supports parallel gradient computation using multiprocessing for intra-rank speedup.
    This provides a 2-level parallelization:
    - Inter-rank: MPI distributes sentences across ranks
    - Intra-rank: multiprocessing parallelizes gradient components within each rank
    
    Args:
        sentence_data (tuple): (psi, U, Z) for the sentence
        params (np.ndarray): Current parameter values
        qc_template (QuantumCircuit): Variational ansatz template (not used currently)
        theta_vec (ParameterVector): Parameter vector (not used currently)
        circuit_function (callable): Circuit function for loss calculation
        num_layers (int): Number of layers
        dim (int): Dimension parameter
        parallel (bool): Whether to use parallel gradient computation (default: True)
        n_workers (int): Number of worker processes (default: None = auto-detect)
        
    Returns:
        tuple: (loss, gradient)
            loss: float
            gradient: np.ndarray of same shape as params
    """
    psi, U, Z = sentence_data
    
    # Calculate loss at current parameters
    loss_current = circuit_function(psi, U, Z, 
                                   params[:len(params)//2].reshape(get_params(2, num_layers).shape),
                                   params[len(params)//2:].reshape(get_params(2, num_layers).shape),
                                   num_layers, dim)
    
    # Calculate gradient using parameter-shift rule
    gradient = np.zeros_like(params)
    shift = np.pi / 2
    
    if parallel and len(params) > 1:  # Parallelizza sempre (anche per pochi parametri)
        # Use parallel computation for gradient - ALWAYS when parallel=True
        # Determine number of workers DINAMICAMENTE (HPC-ready)
        if n_workers is None:
            # Priority order for HPC environments:
            # 1. OMP_NUM_THREADS (OpenMP standard)
            # 2. SLURM_CPUS_PER_TASK (SLURM workload manager)  
            # 3. PBS_NP (PBS/Torque)
            # 4. cpu_count() (fallback auto-detection)
            omp_threads = os.environ.get('OMP_NUM_THREADS')
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            pbs_cpus = os.environ.get('PBS_NP')
            
            if omp_threads:
                n_workers = int(omp_threads)
            elif slurm_cpus:
                n_workers = int(slurm_cpus)
            elif pbs_cpus:
                n_workers = int(pbs_cpus)
            else:
                n_workers = min(cpu_count(), len(params))  # Don't spawn more workers than params
        
        print(f"🔧 PARALLEL GRADIENT: {len(params)} parameters, {n_workers} workers")
        print(f"   CPU count: {cpu_count()}")
        print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print(f"   SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}")
        print(f"   PBS_NP: {os.environ.get('PBS_NP', 'not set')}")
        
        # Determine source of worker count for transparency
        if os.environ.get('OMP_NUM_THREADS'):
            worker_source = "OMP_NUM_THREADS"
        elif os.environ.get('SLURM_CPUS_PER_TASK'):
            worker_source = "SLURM_CPUS_PER_TASK"
        elif os.environ.get('PBS_NP'):
            worker_source = "PBS_NP"
        else:
            worker_source = "cpu_count() auto-detect"
        print(f"   Workers source: {worker_source}")
        
        # Create partial function with fixed arguments
        compute_grad = partial(_compute_single_gradient_component,
                              params=params,
                              shift=shift,
                              psi=psi,
                              U=U,
                              Z=Z,
                              num_layers=num_layers,
                              dim=dim,
                              circuit_function=circuit_function)
        
        # Parallel computation of all gradient components
        import time
        start_time = time.time()
        with Pool(processes=n_workers) as pool:
            gradient = np.array(pool.map(compute_grad, range(len(params))))
        parallel_time = time.time() - start_time
        print(f"   ⚡ Parallel gradient computed in {parallel_time:.3f}s with {n_workers} workers")
    
    else:
        # Sequential computation (for small parameter counts or when parallel=False)
        print(f"🔧 SEQUENTIAL GRADIENT: {len(params)} parameters (parallel={parallel})")
        import time
        start_time = time.time()
        for i in range(len(params)):
            gradient[i] = _compute_single_gradient_component(
                i, params, shift, psi, U, Z, num_layers, dim, circuit_function
            )
        sequential_time = time.time() - start_time
        print(f"   🐌 Sequential gradient computed in {sequential_time:.3f}s")
    
    return loss_current, gradient


def shard_dataset(data_list, rank, size):
    """
    Partition dataset across MPI ranks.
    
    Args:
        data_list (list): Full dataset
        rank (int): MPI rank
        size (int): Total number of MPI ranks
        
    Returns:
        list: Subset of data for this rank
    """
    # Use numpy array_split for balanced partitioning
    shards = np.array_split(data_list, size)
    return list(shards[rank]) if rank < len(shards) else []


def save_checkpoint(params, epoch, lr, seed, checkpoint_dir='checkpoints'):
    """
    Save training checkpoint.
    
    Args:
        params (np.ndarray): Current parameters
        epoch (int): Current epoch
        lr (float): Learning rate
        seed (int): Random seed
        checkpoint_dir (str): Directory for checkpoints
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'qtransformer_ckpt_epoch{epoch}.npz')
    np.savez(checkpoint_path,
             params=params,
             epoch=epoch,
             lr=lr,
             random_state=seed)
    
    return checkpoint_path


def load_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Load the latest checkpoint if available.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        
    Returns:
        dict or None: Checkpoint data if found, None otherwise
    """
    import os
    import glob
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'qtransformer_ckpt_epoch*.npz'))
    
    if not checkpoint_files:
        return None
    
    # Get latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    
    # Load checkpoint
    data = np.load(latest_checkpoint)
    return {
        'params': data['params'],
        'epoch': int(data['epoch']),
        'lr': float(data['lr']),
        'random_state': int(data['random_state'])
    }


def log_training_metrics(epoch, loss_mean, grad_norm, elapsed_time, log_file='logs/train_log.csv'):
    """
    Log training metrics to CSV file.
    
    Args:
        epoch (int): Current epoch
        loss_mean (float): Mean loss
        grad_norm (float): Gradient norm
        elapsed_time (float): Time elapsed for epoch
        log_file (str): Path to log file
    """
    import os
    
    # Create logs directory if needed
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Write header if file doesn't exist
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a') as f:
        if not file_exists:
            f.write('epoch,loss_mean,grad_norm,time_seconds\n')
        f.write(f'{epoch},{loss_mean:.6f},{grad_norm:.6f},{elapsed_time:.2f}\n')
