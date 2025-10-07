

from tkinter import Image
from qiskit.visualization import circuit_drawer
import numpy as np
import datetime
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from typing import List, Dict, Tuple, Optional
import math

from config import DEFAULT_SENTENCES
from encoding import Encoding
from layer import AnsatzBuilder
from quantum_utils import build_controlled_unitary, calculate_loss_from_statevector, get_params, get_unitary_from_tk
import visualization


class GeneralizedQuantumCircuitBuilder:
    """
    Builder per circuiti quantici generalizzati che supporta embedding dimensions variabili.
    
    """
    
    def __init__(self, embedding_dim: int, sentence_length: int):
        """
        Inizializza builder per dimensione embedding e lunghezza frase specifiche.
        
        Args:
            embedding_dim: Dimensione embedding (deve essere potenza di 2)
            sentence_length: Lunghezza frase (numero parole)
        """
        
        self.embedding_dim = embedding_dim
        self.sentence_length = sentence_length
        
        # Calcola numero qubits target da embedding dimension
        self.n_target_qubits = int(math.ceil(math.log2(embedding_dim*embedding_dim)))
        
        # Calcola numero qubits control da lunghezza frase
        self.n_control_qubits = int(math.ceil(math.log2(sentence_length)))
        print("Control qubits: ", self.n_control_qubits)
        self.n_states = 2**self.n_control_qubits
        
        # Totale qubits
        self.n_total_qubits = self.n_target_qubits + self.n_control_qubits
        
        # Indici
        self.target_indices = list(range(self.n_target_qubits))
        self.control_indices = list(range(self.n_target_qubits, self.n_total_qubits))
        
        # Calcola dimensioni per ansatz (basato su target qubits)
        self.ansatz_dim = self.n_target_qubits // 2 if self.n_target_qubits >= 2 else 1
        
        # Genera stati di controllo
        self.control_states = [f'{i:0{self.n_control_qubits}b}' for i in range(min(sentence_length, self.n_states))]
        
        print(f"🔧 Generalized Circuit Builder:")
        print(f"   Embedding dim: {self.embedding_dim} -> {self.n_target_qubits} target qubits")
        print(f"   Sentence length: {self.sentence_length} -> {self.n_control_qubits} control qubits")  
        print(f"   Total qubits: {self.n_total_qubits}")
        print(f"   Ansatz dim: {self.ansatz_dim}")
        print(f"   Control states: {len(self.control_states)}")
        
        
        
    def create_generalized_circuit(self, psi: List[np.ndarray], U: List[np.ndarray], 
                                 Z: List[np.ndarray], params_v: np.ndarray, 
                                 params_k: np.ndarray, num_layers: int) -> float:
        """
        Crea circuito quantico generalizzato per qualsiasi embedding dimension.
        
        Args:
            psi: Liste di unitarie per stati iniziali (len = sentence_length)
            U: Liste di unitarie per predizione parola successiva  
            Z: Liste di unitarie per parola corrente
            params_v: Parametri per ansatz V
            params_k: Parametri per ansatz K
            num_layers: Numero layer ansatz
            use_experimental_phase: Se usare gate di fase sperimentale
            params_f: Parametri fase (se use_experimental_phase=True)
            
        Returns:
            Loss calcolata dal circuito
        """
        print(f"🚀 Creazione circuito quantico a {self.n_total_qubits} qubits")
        # Verifica dimensioni input
        self._validate_input_dimensions(psi, U, Z, params_v, params_k)
        
        # Costruisci ansatz con dimensioni corrette
        ansatz_v = AnsatzBuilder(self.n_target_qubits, params_v, num_layers)
        ansatz_k = AnsatzBuilder(self.n_target_qubits, params_k, num_layers)
        print("✅ Ansatz costruiti")
        # Crea circuito quantico
        qc = QuantumCircuit(self.n_total_qubits, self.n_total_qubits)
        print("🔧 Costruzione circuito...")
        # 1. Inizializza control qubits in superposition
        self._initialize_control_superposition(qc)
        print("✅ Control qubits in superposition")
        # 2. Applica controlled unitaries per stati iniziali
        self._apply_controlled_initial_states(qc, psi)
        print("✅ Stati iniziali applicati")
        # 3. Applica ansatz variazionali
        
        ansatz_v = AnsatzBuilder(self.n_target_qubits, params_v, num_layers)
        ansatz_k = AnsatzBuilder(self.n_target_qubits, params_k, num_layers)
        print("✅ Ansatz costruiti")
        qc.compose(ansatz_v.get_unitary("V"), self.target_indices, inplace=True)
        qc.compose(ansatz_k.get_unitary("W"), self.target_indices, inplace=True)
        print("✅ Ansatz applicati")
        # 4. Applica controlled prediction unitaries
        self._apply_controlled_predictions(qc, U, Z)
        print("✅ Unitarie di predizione applicate")
        # 5. Gate di fase sperimentale (opzionale)
        
        # 6. Hadamard finali sui control qubits
        self._apply_final_hadamards(qc)
        print("✅ Circuito costruito")
        # 7. Calcola e ritorna loss
        loss = calculate_loss_from_statevector(qc)
        
        print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')} "
              f"[{self.embedding_dim}D, {self.sentence_length}W]")
        
        img_path = f"quantum_attention_circuit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        circuit_drawer(qc, output="mpl", filename=img_path)
        Image.open(img_path).show()

        return loss
    
    def _validate_input_dimensions(self, psi: List[np.ndarray], U: List[np.ndarray], 
                                  Z: List[np.ndarray], params_v: np.ndarray, 
                                  params_k: np.ndarray):
        """Valida che le dimensioni degli input siano corrette."""
        
        expected_unitary_dim = self.embedding_dim
        
        # Verifica unitarie
        for name, unitaries in [("psi", psi), ("U", U), ("Z", Z)]:
            
            if len(unitaries) != self.sentence_length:
                raise ValueError(f"{name} deve avere almeno {self.sentence_length} unitarie ma ne ha {len(unitaries)}")
            
            
        
        # Verifica parametri ansatz (dipendono dall'implementazione di AnsatzBuilder)
        print(f"Parametri V: {len(params_v)}, K: {len(params_k)}")
    
    def _build_ansatz(self, params_v: np.ndarray, params_k: np.ndarray, 
                     num_layers: int) -> Tuple:
        """Costruisce ansatz con dimensioni corrette per l'embedding."""
        
        # Per semplicità, creiamo ansatz diretti invece di usare AnsatzBuilder complesso
        # Questo bypassa i problemi di interfaccia e ci dà controllo completo
        
        try:
            ansatz_v = self._create_simple_ansatz(self.ansatz_dim, params_v, "V")
            ansatz_k = self._create_simple_ansatz(self.ansatz_dim, params_k, "W")  # Nome "W" per compatibilità
            return ansatz_v, ansatz_k
            
        except Exception as e:
            print(f"⚠️ Errore costruzione ansatz semplice: {e}")
            
            # Fallback: ansatz identità
            return self._create_identity_ansatz("V"), self._create_identity_ansatz("W")
    
    def _create_simple_ansatz(self, num_qubits: int, params: np.ndarray, name: str):
        """Crea ansatz semplice con rotazioni parametriche."""
        
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        
        # Calcola numero parametri necessari (3 rotazioni per qubit)
        params_needed = num_qubits * 3
        
        # Assicura abbastanza parametri
        if len(params) < params_needed:
            # Pad con parametri random piccoli
            padding = np.random.random(params_needed - len(params)) * 0.01
            params = np.concatenate([params, padding])
        
        # Usa solo i primi parametri necessari
        params = params[:params_needed]
        
        # Crea circuito ansatz
        qc = QuantumCircuit(num_qubits, name=name)
        
        # Applica rotazioni parametriche
        for qubit in range(num_qubits):
            param_idx = qubit * 3
            qc.rx(params[param_idx], qubit)
            qc.ry(params[param_idx + 1], qubit)
            qc.rz(params[param_idx + 2], qubit)
        
        # Aggiungi entangling gates se abbiamo più qubits
        if num_qubits > 1:
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            # Connessione circolare per maggiore entanglement
            if num_qubits > 2:
                qc.cx(num_qubits - 1, 0)
        
        # Crea oggetto compatibile con interfaccia esistente
        class SimpleAnsatz:
            def __init__(self, circuit):
                self.circuit = circuit
            
            def get_unitary(self, circuit_name):
                """Ritorna instruction del circuito."""
                return self.circuit.to_instruction()
        
        return SimpleAnsatz(qc)
    
    def _create_identity_ansatz(self, name: str):
        """Crea ansatz identità come fallback."""
        
        from qiskit import QuantumCircuit
        
        # Crea circuito vuoto (identità)
        qc = QuantumCircuit(1, name=name)  # Singolo qubit per semplicità
        
        class IdentityAnsatz:
            def __init__(self, circuit):
                self.circuit = circuit
            
            def get_unitary(self, circuit_name):
                """Ritorna instruction identità."""
                return self.circuit.to_instruction()
        
        return IdentityAnsatz(qc)
    
    def _initialize_control_superposition(self, qc: QuantumCircuit):
        """Inizializza control qubits in superposition."""
        for qubit in self.control_indices:
            qc.h(qubit)
        
    def _apply_controlled_initial_states(self, qc: QuantumCircuit, psi: List[np.ndarray]):
        """Applica controlled unitaries per stati iniziali."""
        print("Applico un numero di stati pari a ", len(psi))
        for i, state in enumerate(self.control_states):
            
            if i < len(psi):
                unitary = psi[i]
                unitary_dim = unitary.shape[0]
                print(f"Unitary dimension for state {state}: {unitary_dim}")
                
                print(f"Applying initial state for control state {state} requiring {self.n_target_qubits} target qubits")
                # CORREZIONE: Usa abbastanza target qubits per la dimensione unitaria
                if self.n_target_qubits <= len(self.target_indices):
                    target_qubits_for_unitary = self.target_indices[:self.n_target_qubits]
                else:
                    # Estendi con qubits aggiuntivi se necessario
                    target_qubits_for_unitary = self.target_indices + list(range(self.n_total_qubits, self.n_total_qubits + required_qubits - len(self.target_indices)))
                
                try:
                    controlled_unitary = build_controlled_unitary(
                        unitary, 
                        self.control_indices, 
                        target_qubits_for_unitary,  # CORRETTO!
                        f"ψ_{state}", 
                        activate_on=state
                    )
                    
                    qubits_list = self.control_indices + target_qubits_for_unitary
                    qc.compose(controlled_unitary, qubits=qubits_list, inplace=True)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")


    def _apply_variational_ansatz(self, qc: QuantumCircuit, ansatz_v, ansatz_k):
        """Applica ansatz variazionali sui target qubits."""
        
        # Dividi target qubits per V e K ansatz
        if self.n_target_qubits >= 4:
            # Abbastanza qubits per dividere
            v_qubits = self.target_indices[:self.n_target_qubits//2]
            k_qubits = self.target_indices[self.n_target_qubits//2:]
            
        elif self.n_target_qubits == 2:
            # 2 qubits: assegna 1 per V e 1 per K
            v_qubits = [self.target_indices[0]]
            k_qubits = [self.target_indices[1]]
            
        else:
            # 1 qubit: applica solo V
            v_qubits = self.target_indices
            k_qubits = []
        
        # Applica ansatz
        if v_qubits:
            try:
                qc.compose(ansatz_v.get_unitary("V"), v_qubits, inplace=True)
            except Exception as e:
                print(f"⚠️ Errore applicazione ansatz V: {e}")
        
        if k_qubits:
            try:
                qc.compose(ansatz_k.get_unitary("W"), k_qubits, inplace=True)
            except Exception as e:
                print(f"⚠️ Errore applicazione ansatz K: {e}")
    
    def _apply_controlled_predictions(self, qc: QuantumCircuit, U: List[np.ndarray], 
                                    Z: List[np.ndarray]):
        """Applica controlled prediction unitaries."""
        
        # Per embedding grandi, dividiamo target qubits per U e Z
        if self.n_target_qubits >= 4:
            u_qubits = self.target_indices[:self.n_target_qubits//2]
            z_qubits = self.target_indices[self.n_target_qubits//2:]
            
        elif self.n_target_qubits == 2:
            u_qubits = [self.target_indices[0]]
            z_qubits = [self.target_indices[1]]
            
        else:
            u_qubits = self.target_indices
            z_qubits = []
        
        # Applica prediction unitaries per ogni stato
        for i, state in enumerate(self.control_states):
            if i < len(U):
                # Predizione next word
                if u_qubits:
                    qc.compose(
                        build_controlled_unitary(
                            self._adapt_unitary_dimension(U[i], len(u_qubits)),
                            self.control_indices,
                            u_qubits,
                            f"next_{state}",
                            activate_on=state
                        ),
                        qubits=self.control_indices + u_qubits,
                        inplace=True
                    )
                
                # Current word
                if z_qubits and i < len(Z):
                    qc.compose(
                        build_controlled_unitary(
                            self._adapt_unitary_dimension(Z[i], len(z_qubits)),
                            self.control_indices,
                            z_qubits,
                            f"current_{state}",
                            activate_on=state
                        ),
                        qubits=self.control_indices + z_qubits,
                        inplace=True
                    )
    
    def _adapt_unitary_dimension(self, unitary: np.ndarray, n_qubits: int) -> np.ndarray:
        """Adatta dimensione unitaria per numero qubits specifico."""
        
        expected_dim = 2**n_qubits
        current_dim = unitary.shape[0]
        
        if current_dim == expected_dim:
            return unitary
        
        elif current_dim > expected_dim:
            # Tronca se troppo grande
            print(f"⚠️ Truncating unitary from {current_dim}x{current_dim} to {expected_dim}x{expected_dim}")
            return unitary[:expected_dim, :expected_dim]
        
        else:
            # Estendi se troppo piccolo (padding con identità)
            print(f"🔧 Extending unitary from {current_dim}x{current_dim} to {expected_dim}x{expected_dim}")
            extended = np.eye(expected_dim, dtype=complex)
            extended[:current_dim, :current_dim] = unitary
            return extended
    
    def _apply_experimental_phase(self, qc: QuantumCircuit, params_f: np.ndarray):
        """Applica gate di fase sperimentale."""
        
        try:
            # Crea matrice diagonale di fase
            n_control_states = 2**self.n_control_qubits
            phase_params = params_f[:n_control_states] if len(params_f) >= n_control_states else np.pad(params_f, (0, n_control_states - len(params_f)))
            
            diag_matrix = np.diag(np.exp(1j * phase_params))
            phase_gate = UnitaryGate(diag_matrix, label="φ")
            
            qc.compose(phase_gate, self.control_indices, inplace=True)
            
        except Exception as e:
            print(f"⚠️ Errore applicazione fase sperimentale: {e}")
    
    def _apply_final_hadamards(self, qc: QuantumCircuit):
        """Applica Hadamard finali sui control qubits."""
        for qubit in self.control_indices:
            qc.h(qubit)
    
    def get_circuit_info(self) -> Dict:
        """Ritorna informazioni sul circuito configurato."""
        
        return {
            "embedding_dim": self.embedding_dim,
            "sentence_length": self.sentence_length,
            "n_target_qubits": self.n_target_qubits,
            "n_control_qubits": self.n_control_qubits,
            "n_total_qubits": self.n_total_qubits,
            "ansatz_dim": self.ansatz_dim,
            "n_states": len(self.control_states),
            "control_states": self.control_states,
            "target_indices": self.target_indices,
            "control_indices": self.control_indices
        }


class AdaptiveQuantumCircuitFactory:
    """
    Factory per creare circuiti quantici adattivi basati su embedding dimension e sentence length.
    """
    
    @staticmethod
    def create_circuit_builder(embedding_dim: int, sentence_length: int) -> GeneralizedQuantumCircuitBuilder:
        """
        Crea circuit builder ottimizzato per embedding dimension e sentence length.
        
        Args:
            embedding_dim: Dimensione embedding (4, 16, 64, 256, etc.)
            sentence_length: Lunghezza frase (3, 5, 9, 17, etc.)
            
        Returns:
            GeneralizedQuantumCircuitBuilder configurato
        """
        
        # Valida embedding dimension
        if embedding_dim <= 0 or (embedding_dim & (embedding_dim - 1)) != 0:
            # Trova potenza di 2 più vicina
            next_power = 2**int(math.ceil(math.log2(embedding_dim)))
            print(f"⚠️ Embedding dim {embedding_dim} non è potenza di 2, usando {next_power}")
            embedding_dim = next_power
        
        # Limiti pratici
        max_embedding = 256  # 8 qubits target
        if embedding_dim > max_embedding:
            print(f"⚠️ Embedding dim {embedding_dim} > {max_embedding}, limitando a {max_embedding}")
            embedding_dim = max_embedding
        
        
        return GeneralizedQuantumCircuitBuilder(embedding_dim, sentence_length)
    
    @staticmethod
    def get_optimal_embedding_dim(vocab_size: int, target_qubits: Optional[int] = None) -> int:
        """
        Suggerisce embedding dimension ottimale per vocabulary size.
        
        Args:
            vocab_size: Dimensione vocabolario
            target_qubits: Numero target qubits desiderato (opzionale)
            
        Returns:
            Embedding dimension ottimale
        """
        
        if target_qubits is not None:
            return 2**target_qubits
        
        # Euristica basata su vocabulary size
        if vocab_size <= 1000:
            return 4    # 2 qubits
        elif vocab_size <= 10000:
            return 16   # 4 qubits 
        elif vocab_size <= 50000:
            return 64   # 6 qubits
        else:
            return 256  # 8 qubits
    
    @staticmethod
    def estimate_circuit_complexity(embedding_dim: int, sentence_length: int) -> Dict:
        """
        Stima complessità computazionale del circuito.
        
        Returns:
            Dict con stime di complessità
        """
        
        n_target = int(math.log2(embedding_dim))
        n_control = int(math.ceil(math.log2(sentence_length)))
        n_total = n_target + n_control
        
        # Stime approssimate
        n_gates_estimate = sentence_length * (n_target * 5 + n_control * 2)  # Gates per unitaries + Hadamards
        memory_estimate_mb = (2**n_total * 16) / (1024**2)  # Statevector complex128
        
        return {
            "total_qubits": n_total,
            "target_qubits": n_target,
            "control_qubits": n_control,
            "estimated_gates": n_gates_estimate,
            "estimated_memory_mb": memory_estimate_mb,
            "complexity_score": n_total * sentence_length,  # Metrica semplice
            "is_feasible": n_total <= 15 and memory_estimate_mb <= 1000  # Limiti pratici
        }


def process_sentence_states(states):
    """
    Process sentence states to create unitaries for quantum circuit.
    
    Args:
        states (list): List of state vectors for each word in sentence
        
    Returns:
        tuple: (states_calculated, U, Z) - processed unitaries
    """
    states_calculated = []
    U = []
    Z = []
    print("Inizio a fare gli stati, la lunghezza è ", len(states))
    for i in range(1, len(states)):
        print("=" * 60)
        print(f"[ITERATION {i}]")

        psi = None

        # Construction of psi
        for j in range(i):
            t = states[j]
            norm = np.linalg.norm(t)
            print(f"  word {j}, norm={norm:.4f}")
            if norm == 0 or np.isnan(norm):
                print("   [WARN] Null or ill-defined vector, skipping")
                continue
            t = t / norm
            kron = np.kron(t, t)
            print(f"   kron shape={kron.shape}")
            psi = kron if psi is None else psi + kron

        # Normalize psi
        norm_psi = np.linalg.norm(psi)
        print(f"  norm(psi) before normalization = {norm_psi:.4f}")
        psi = psi / norm_psi
        print(f"  psi normalized, shape={psi.shape}, norm={np.linalg.norm(psi):.4f}")

        # Target and origin
        x = states[i]
        z = states[i - 1]
        print(f"  target x = word {i}, shape={x.shape}, norm={np.linalg.norm(x):.4f}")
        print(f"  origin z = word {i-1}, shape={z.shape}, norm={np.linalg.norm(z):.4f}")

        # Calculate unitaries
        try:
            U_dagger = get_unitary_from_tk(x).conj().T
            Z_dagger = get_unitary_from_tk(z).conj().T
            unitary_psi = get_unitary_from_tk(psi)


        except Exception as e:
            print(f"[ERROR] Unitary calculation failed: {e}")
            continue

        # Append to lists
        print("  Saving triplet (U, Z, psi)")
        
        U.append(U_dagger)
        Z.append(Z_dagger)
        states_calculated.append(unitary_psi)

        print("  [OK] Triplet saved in U, Z, states_calculated")
    return states_calculated, U, Z


def _generate_default_unitaries(n_unitaries: int, embedding_dim: int, 
                               unitary_type: str) -> List[np.ndarray]:
    """Genera unitarie default per testing."""
    
    unitaries = []
    
    for i in range(n_unitaries):
        # Genera matrice random unitaria
        if embedding_dim == 4:
            # 2x2 unitaries per 2 qubits
            angles = np.random.random(3) * 2 * np.pi
            unitary = np.array([
                [np.cos(angles[0]) * np.exp(1j * angles[2]), np.sin(angles[0]) * np.exp(1j * (angles[1] + angles[2]))],
                [-np.sin(angles[0]) * np.exp(1j * (angles[1] - angles[2])), np.cos(angles[0]) * np.exp(-1j * angles[2])]
            ])
            
            # Estendi a 4x4 con prodotto tensoriale
            I = np.eye(2)
            if i % 2 == 0:
                full_unitary = np.kron(unitary, I)
            else:
                full_unitary = np.kron(I, unitary)
                
        else:
            # Genera unitaria random per dimensione generale
            # Usa decomposizione QR per garantire unitarietà
            random_matrix = np.random.randn(embedding_dim, embedding_dim) + 1j * np.random.randn(embedding_dim, embedding_dim)
            q, r = np.linalg.qr(random_matrix)
            d = np.diag(r)
            ph = d / np.abs(d)
            full_unitary = q @ np.diag(ph)
        
        unitaries.append(full_unitary)
    
    print(f"🔧 Generated {len(unitaries)} default {unitary_type} unitaries ({embedding_dim}x{embedding_dim})")
    
    return unitaries

def test_quantum_circuit_loss(embedding_dim: int, sentence_idx: int = 0, num_layers: int = 2) -> float:
        """
        Test quantum circuit and return loss for given embedding dimension and sentence.
        
        Args:
            embedding_dim: Embedding dimension (4, 8, 16, 32, etc.)
            sentence_idx: Index of sentence to process (default: 0)
            num_layers: Number of ansatz layers (default: 2)
            
        Returns:
            float: Calculated loss from quantum circuit
        """
        print(f"🧪 Testing quantum circuit: embedding_dim={embedding_dim}, sentence_idx={sentence_idx}")
        
        try:
            # Create encoding
            enc = Encoding(DEFAULT_SENTENCES, embeddingDim=embedding_dim)
            
            # Process sentence states
            states = enc.stateVectors[sentence_idx]
            states_calculated, U, Z = process_sentence_states(states)
            sentence_length = len(states_calculated)
            
            # Create circuit builder
            builder = AdaptiveQuantumCircuitFactory.create_circuit_builder(embedding_dim, sentence_length)
            
            # Generate parameters
            n_target_qubits = builder.n_target_qubits
            ansatz_dim = builder.ansatz_dim
            
            param_v_5d = get_params(n_target_qubits, ansatz_dim)
            param_k_5d = get_params(n_target_qubits, ansatz_dim)
            
            print(f"   Generated params shapes: V={param_v_5d.shape}, K={param_k_5d.shape}")
            
            # Create and run quantum circuit
            loss = builder.create_generalized_circuit(
                psi=states_calculated,
                U=U,
                Z=Z,
                params_v=param_v_5d,
                params_k=param_k_5d,
                num_layers=num_layers
            )
            
            print(f"   ✅ Loss calculated: {loss:.6f}")
            return loss
            
        except Exception as e:
            print(f"   ❌ Error in quantum circuit test: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')  # Return high loss on error



if __name__ == "__main__":
        # Test del sistema generalizzato
    print("🧪 Testing Generalized Quantum Circuit Architecture")
    print("="*60)
        
        # Test diverse embedding dimensions
    test_configs = [4, 8, 16, 32]
        
    for embedding_dim in test_configs:
        print(f"\n📊 Testing embedding dimension: {embedding_dim}")
            
            # Test multiple sentences
        for sentence_idx in range(0, 2):
            print(f"{'='*60}")
            print(f"Sentence {sentence_idx}:")
                
            loss = test_quantum_circuit_loss(embedding_dim, sentence_idx)
            print(f"Final loss: {loss:.6f}")
        
    print("\n✅ All tests completed!")
