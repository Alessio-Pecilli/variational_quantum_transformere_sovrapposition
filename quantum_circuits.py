"""
Quantum circuit creation functions for different word configurations.
"""
import numpy as np
import datetime
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from layer import AnsatzBuilder
from quantum_utils import build_controlled_unitary, calculate_loss_from_statevector
import visualization


def create_circuit_2words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 2 words (1 transition).
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 1  # 1 control qubit for 2 words
    a = 2
    
    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for different states
    qc.compose(
        build_controlled_unitary(psi[0], control_indices, target_indices, "Ψ_0", activate_on='0'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(psi[1], control_indices, target_indices, "ψ_1", activate_on='1'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    qc.compose(
        build_controlled_unitary(U[0], control_indices, list(range(0, 2)), "next_0", activate_on='0'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[0], control_indices, list(range(2, 4)), "current_0", activate_on='0'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(U[1], control_indices, list(range(0, 2)), "next_1", activate_on='1'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[1], control_indices, list(range(2, 4)), "current_1", activate_on='1'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_4words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 4 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 2  # 2 control qubits for 4 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for 4 different combinations
    states = ['00', '10', '01', '11']
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_8words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 8 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 3  # 3 control qubits for 8 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5, 6]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for 8 different combinations
    states = ['000', '001', '010', '011', '100', '101', '110', '111']
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_16words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 16 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 4  # 4 control qubits for 16 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5, 6, 7]
    target_indices = list(range(0, 4))

    # Generate all 16 binary combinations
    activate_states = [f'{i:04b}' for i in range(16)]

    # Apply controlled unitaries for 16 different combinations
    for i, state in enumerate(activate_states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(activate_states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_experimental_circuit(psi, U, Z, params_v, params_k, params_f, num_layers, dim=4):
    """
    Create experimental quantum circuit with additional phase parameters.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        params_f (array): Additional phase parameters
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 1
    a = 2
    
    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)

    # Create diagonal phase matrix
    print("paramsF:", np.diag(params_f))
    diag_matrix = np.diag(np.exp(1j * np.array(params_f)))
    ansatz_f = UnitaryGate(diag_matrix, label="φ")
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries
    qc.compose(
        build_controlled_unitary(psi[0], control_indices, target_indices, "ψ_0", activate_on='0'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(psi[1], control_indices, target_indices, "ψ_1", activate_on='1'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    qc.compose(
        build_controlled_unitary(U[0], control_indices, list(range(0, 2)), "next_0", activate_on='0'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[0], control_indices, list(range(2, 4)), "current_0", activate_on='0'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(U[1], control_indices, list(range(0, 2)), "next_1", activate_on='1'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[1], control_indices, list(range(2, 4)), "current_1", activate_on='1'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    # Apply experimental phase gate
    qc.compose(ansatz_f, list(range(qtarget, n_qubits)), inplace=True)
    
    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    # Calculate probabilities for analysis
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    
    # Calculate ancilla probabilities
    prob_ancilla0 = sum(p for state, p in probs.items() if state[-1] == '0')
    prob_ancilla1 = sum(p for state, p in probs.items() if state[-1] == '1')
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f}")
    
    return loss


# Circuit function mapping
CIRCUIT_FUNCTIONS = {
    2: create_circuit_2words,
    4: create_circuit_4words,
    8: create_circuit_8words,
    16: create_circuit_16words
}


def get_circuit_function(num_words):
    """
    Get the appropriate circuit function based on number of words.
    
    Args:
        num_words (int): Number of words to handle
        
    Returns:
        function: Circuit creation function
        
    Raises:
        ValueError: If number of words is not supported
    """
    if num_words in CIRCUIT_FUNCTIONS:
        return CIRCUIT_FUNCTIONS[num_words]
    else:
        raise ValueError(f"Number of words ({num_words}) not supported. Maximum 16 words.")
