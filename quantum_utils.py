"""
Utility functions for quantum circuit operations and parameter management.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def get_unitary_from_tk(psi):
    """
    Generate a unitary matrix from a given state vector using Gram-Schmidt process.
    
    Args:
        psi (array): Input state vector
        
    Returns:
        numpy.ndarray: Unitary matrix with psi as first column
    """
    psi = psi / np.linalg.norm(psi)
    dim = len(psi)
    base = [psi]

    while len(base) < dim:
        vec = np.random.rand(dim) + 1j * np.random.rand(dim)

        for i, b in enumerate(base):
            coeff = np.vdot(b, vec)
            vec -= coeff * b

        norm = np.linalg.norm(vec)

        if norm < 1e-12:
            continue
        vec /= norm
        base.append(vec)

    U = np.column_stack(base)
    return U


def get_params(num_qubits, num_layers):
    """
    Generate parameter array for quantum circuit ansatz.
    
    Args:
        num_qubits (int): Number of qubits
        num_layers (int): Number of layers in the ansatz
        
    Returns:
        numpy.ndarray: Shaped parameter array
    """
    x = get_param_resolver(num_qubits, num_layers)
    params = get_params_shape(x, num_qubits, num_layers)
    return params


def get_param_resolver(num_qubits, num_layers):
    """
    Create parameter dictionary for optimization.
    
    Args:
        num_qubits (int): Number of qubits
        num_layers (int): Number of layers
        
    Returns:
        dict: Parameter dictionary mapping symbols to values
    """
    num_angles = 12 * num_qubits * num_layers
    angs = np.pi * (2 * np.random.rand(num_angles) - 1)
    params = ParameterVector('θ', num_angles)
    param_dict = dict(zip(params, angs))
    return param_dict


def get_params_shape(param_list, num_qubits, num_layers):
    """
    Reshape parameter values into the required structure.
    
    Args:
        param_list (dict): Parameter dictionary
        num_qubits (int): Number of qubits
        num_layers (int): Number of layers
        
    Returns:
        numpy.ndarray: Reshaped parameter array
    """
    param_values = np.array(list(param_list.values()))
    x = param_values.reshape(num_layers, 2, num_qubits // 2, 12)
    x_reshaped = x.reshape(num_layers, 2, num_qubits // 2, 4, 3)
    return x_reshaped


def get_circuit_ux_dagger_from_tk(t_k):
    """
    Create a quantum circuit with unitary dagger gate from state vector.
    
    Args:
        t_k (array): State vector
        
    Returns:
        QuantumCircuit: Circuit with unitary dagger gate
    """
    t_k = np.array(t_k, dtype=complex)
    t_k = t_k / np.linalg.norm(t_k)
    dim = len(t_k)
    n = int(np.log2(dim))

    U = get_unitary_from_tk(t_k)
    U_dagger = U.conj().T
    gate = UnitaryGate(U_dagger)
    qc = QuantumCircuit(n, name="U†_x")
    qc.append(gate, range(n))

    return qc


def build_controlled_unitary(U, controls, targets, label, activate_on):
    """
    Build a controlled unitary gate.
    
    Args:
        U (numpy.ndarray): Unitary matrix
        controls (list): Control qubit indices
        targets (list): Target qubit indices
        label (str): Gate label
        activate_on (str): Control state to activate on
        
    Returns:
        Gate: Controlled unitary gate
    """
    gate = UnitaryGate(U, label=label).control(len(controls), ctrl_state=activate_on)
    return gate


def safe_controlled_unitary(U, control_indices, target_indices, label):
    """
    Create a controlled unitary gate safely.
    
    Args:
        U (numpy.ndarray): Unitary matrix
        control_indices (list): Control qubit indices
        target_indices (list): Target qubit indices
        label (str): Gate label
        
    Returns:
        Gate: Controlled gate
    """
    n_target = len(target_indices)
    inner = QuantumCircuit(n_target)
    inner.unitary(U, range(n_target), label=label)
    gate = inner.to_gate(label=label)
    ctrl_gate = gate.control(len(control_indices))
    return ctrl_gate


def wrap_angles(theta):
    """
    Wrap angles to [-π, π] range.
    
    Args:
        theta (array): Input angles
        
    Returns:
        array: Wrapped angles
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def calculate_loss_from_statevector(qc):
    """
    Calculate loss from quantum circuit by computing |0...0> probability.
    
    Args:
        qc (QuantumCircuit): Quantum circuit
        
    Returns:
        float: Negative log probability loss
    """
    sv = Statevector.from_instruction(qc)
    prob = float(abs(sv.data[0]) ** 2)
    return -np.log(prob + 1e-12)