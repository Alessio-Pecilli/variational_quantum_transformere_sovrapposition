import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumRegister, QuantumCircuit

class AnsatzBuilder:
    """
    Quantum ansatz builder for variational quantum circuits.
    
    This class creates parameterized quantum circuits using rotation gates
    and CNOT gates arranged in layers.
    """
    
    def __init__(self, num_qubits, params, num_layers):
        """
        Initialize the ansatz builder.
        
        Args:
            num_qubits (int): Number of qubits in the circuit
            params (array): Parameters for the ansatz layers
            num_layers (int): Number of ansatz layers
        """
        self._num_qubits = int(num_qubits)
        self.qubits = QuantumRegister(self._num_qubits)
        self.unitary_circ = QuantumCircuit(self.qubits)
        
        for layer_idx in range(num_layers):
            self.add_layer(params[layer_idx][0], params[layer_idx][1])
        
        
    


    def get_ansatz(self):
        """
        Get the constructed quantum circuit.
        
        Returns:
            QuantumCircuit: The parameterized quantum circuit
        """
        return self.unitary_circ

    def add_layer(self, params, shifted_params):
        """
        Add a variational layer to the circuit.
        
        Args:
            params (array): Parameters for even qubit pairs
            shifted_params (array): Parameters for odd qubit pairs
        """
        n = self._num_qubits
        if params.size != self.num_angles_required_for_layer() / 2:
            raise ValueError(f"Params.size: {params.size}, required angles: {self.num_angles_required_for_layer() / 2}")

        # Apply gates to even qubit pairs (0,1), (2,3), etc.
        for ii in range(0, n - 1, 2):
            qubits = [self.qubits[ii], self.qubits[ii + 1]]
            gate_params = params[ii // 2]
            self._apply_two_qubit_gate(qubits, gate_params)

        # Apply gates to odd qubit pairs (1,2), (3,4), etc. with wraparound
        if n >= 2:
            for ii in range(1, n, 2):
                qubits = [self.qubits[ii], self.qubits[(ii + 1) % n]]
                self._apply_two_qubit_gate(qubits, shifted_params[ii // 2])

    def _apply_two_qubit_gate(self, qubits, params):
        """
        Apply a parameterized two-qubit gate.
        
        Args:
            qubits (list): Two qubits to apply the gate to
            params (array): Parameters for the gate
        """
        # First rotation on qubit 0
        self._apply_rotation(qubits[0], params[0])
        # First rotation on qubit 1
        self._apply_rotation(qubits[1], params[1])
        # First CNOT
        self.unitary_circ.cx(qubits[0], qubits[1])
        # Second rotation on qubit 0
        self._apply_rotation(qubits[0], params[2])
        # Second rotation on qubit 1
        self._apply_rotation(qubits[1], params[3])
        # Second CNOT
        self.unitary_circ.cx(qubits[0], qubits[1])

    def _apply_rotation(self, qubit, params):
        """
        Apply rotation gates (RX, RY, RZ) to a qubit.
        
        Args:
            qubit: Target qubit
            params (array): Rotation parameters [rx_angle, ry_angle, rz_angle]
        """
        self.unitary_circ.rx(params[0], qubit)
        self.unitary_circ.ry(params[1], qubit)
        self.unitary_circ.rz(params[2], qubit)

    def num_angles_required_for_layer(self):
        """
        Calculate the number of parameters required for one layer.
        
        Returns:
            int: Number of parameters per layer
        """
        return 12 * self._num_qubits

    def get_unitary(self, circuit_name):
        """
        Convert the circuit to a unitary instruction.
        
        Args:
            circuit_name (str): Name for the circuit instruction
            
        Returns:
            Instruction: Unitary instruction representing the circuit
        """
        combined_circuit = QuantumCircuit(self._num_qubits, name=circuit_name)
        combined_circuit.compose(self.unitary_circ, inplace=True)
        return combined_circuit.to_instruction()


def generate_vqe_ansatz(qc, num_qubits, params):
    """
    Generate a VQE ansatz circuit.
    
    Args:
        qc (QuantumCircuit): Circuit to add the ansatz to
        num_qubits (int): Number of qubits
        params (array): Parameters for the ansatz
    """
    param_idx = 0
    
    # Apply rotation gates to all qubits
    for qubit in range(num_qubits):
        qc.rx(params[param_idx], qubit)
        param_idx += 1
        qc.ry(params[param_idx], qubit)
        param_idx += 1
    
    # Apply entangling gates
    for qubit in range(num_qubits - 1):
        qc.cx(qubit, qubit + 1)
