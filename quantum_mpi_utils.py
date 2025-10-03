"""
MPI utilities for distributed quantum circuit training.
Implements data-parallel synchronous training following DDP pattern.
"""

import numpy as np
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
    Build the input encoding as a UnitaryGate.
    This encoding is sentence-dependent and needs to be reconstructed for each sentence.
    
    Args:
        psi (list): Initial state unitaries
        U (list): Next word unitaries
        Z (list): Current word unitaries
        
    Returns:
        UnitaryGate: Unitary encoding of the input
    """
    # This is a placeholder - actual implementation depends on how you encode inputs
    # Typically this would be based on the state preparation from your data
    
    # For now, return identity as placeholder
    # You need to implement the actual encoding based on psi, U, Z
    n_qubits = 4  # Adjust based on your system
    unitary_matrix = np.eye(2**n_qubits, dtype=complex)
    
    return UnitaryGate(unitary_matrix)


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


def loss_and_grad_for_sentence(sentence_data, params, qc_template, theta_vec, circuit_function, num_layers, dim):
    """
    Calculate loss and gradient for a single sentence using parameter-shift rule.
    
    Args:
        sentence_data (tuple): (psi, U, Z) for the sentence
        params (np.ndarray): Current parameter values
        qc_template (QuantumCircuit): Variational ansatz template
        theta_vec (ParameterVector): Parameter vector
        circuit_function (callable): Circuit function for loss calculation
        num_layers (int): Number of layers
        dim (int): Dimension parameter
        
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
    
    for i in range(len(params)):
        # Shift parameter up
        params_plus = params.copy()
        params_plus[i] += shift
        
        loss_plus = circuit_function(psi, U, Z,
                                     params_plus[:len(params)//2].reshape(get_params(2, num_layers).shape),
                                     params_plus[len(params)//2:].reshape(get_params(2, num_layers).shape),
                                     num_layers, dim)
        
        # Shift parameter down
        params_minus = params.copy()
        params_minus[i] -= shift
        
        loss_minus = circuit_function(psi, U, Z,
                                      params_minus[:len(params)//2].reshape(get_params(2, num_layers).shape),
                                      params_minus[len(params)//2:].reshape(get_params(2, num_layers).shape),
                                      num_layers, dim)
        
        # Parameter-shift formula
        gradient[i] = 0.5 * (loss_plus - loss_minus)
    
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
