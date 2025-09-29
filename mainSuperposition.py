"""
Main script for quantum superposition-based language modeling.

This script processes sentences, creates quantum circuits, and optimizes parameters
for next-word prediction using quantum superposition states.
"""

# Standard library imports
import numpy as np

# Local imports
from encoding import Encoding
from hamiltonian import HamiltonianEvolution
from quantum_utils import get_unitary_from_tk, get_params
from quantum_circuits import get_circuit_function
from optimization import optimize_parameters
from visualization import load_parameters, save_parameters
from config import OPTIMIZATION_CONFIG, TRAINING_SENTENCES, DEFAULT_SENTENCES
from quantum_circuits import get_circuit_function
from optimization import optimize_parameters
from visualization import load_parameters, save_parameters
from config import OPTIMIZATION_CONFIG, TRAINING_SENTENCES, DEFAULT_SENTENCES

# Global Hamiltonian evolution instance for consistency
_hamiltonian_evolution = None

def get_hamiltonian_evolution():
    """Get or create global Hamiltonian evolution instance"""
    global _hamiltonian_evolution
    if _hamiltonian_evolution is None:
        _hamiltonian_evolution = HamiltonianEvolution()
        print("Created new global Hamiltonian evolution instance")
    return _hamiltonian_evolution

def verify_hamiltonian_consistency():
    """Verify that the Hamiltonian evolution is consistent"""
    evolution = get_hamiltonian_evolution()
    if hasattr(evolution, 'H'):
        H_hash = hash(evolution.H.data.tobytes())
        print(f"✅ Hamiltonian hash: {H_hash}")
        print(f"✅ Hamiltonian shape: {evolution.H.shape}")
        return H_hash
    else:
        print("❌ No Hamiltonian found")
        return None

def reset_hamiltonian_evolution():
    """Reset the global Hamiltonian evolution instance"""
    global _hamiltonian_evolution
    _hamiltonian_evolution = None
    print("Reset global Hamiltonian evolution instance")


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

            print(f"  U_dagger shape={U_dagger.shape}, unitarity check={np.allclose(U_dagger.conj().T @ U_dagger, np.eye(U_dagger.shape[0]))}")
            print(f"  Z_dagger shape={Z_dagger.shape}, unitarity check={np.allclose(Z_dagger.conj().T @ Z_dagger, np.eye(Z_dagger.shape[0]))}")
            print(f"  unitary_psi shape={unitary_psi.shape}, unitarity check={np.allclose(unitary_psi.conj().T @ unitary_psi, np.eye(unitary_psi.shape[0]))}")

        except Exception as e:
            print(f"[ERROR] Unitary calculation failed: {e}")
            continue

        # Append to lists
        U.append(U_dagger)
        Z.append(Z_dagger)
        states_calculated.append(unitary_psi)

        print("  [OK] Triplet saved in U, Z, states_calculated")
    
    return states_calculated, U, Z


def generate_hamiltonian_states(num_words, embedding_dim=4):
    """
    Generate sequential quantum states using Hamiltonian evolution.
    
    Args:
        num_words (int): Number of sequential states to generate
        embedding_dim (int): Dimension of each state
        
    Returns:
        list: List of normalized quantum state vectors
    """
    print(f"Generating {num_words} sequential states using Hamiltonian evolution...")
    
    # Use global Hamiltonian evolution instance for consistency
    evolution = get_hamiltonian_evolution()
    
    # Check if Hamiltonian dimension matches required dimension
    current_H_dim = evolution.H.shape[0] if hasattr(evolution, 'H') else 0
    print(f"Current Hamiltonian dimension: {current_H_dim}, Required: {embedding_dim}")
    
    # Generate sequential states
    states = evolution.generate_sequential_states(num_words, state_dim=embedding_dim)
    
    # Verify Hamiltonian consistency
    new_H_dim = evolution.H.shape[0]
    if current_H_dim != 0 and current_H_dim != new_H_dim:
        print(f"⚠️  WARNING: Hamiltonian dimension changed from {current_H_dim} to {new_H_dim}")
    else:
        print(f"✅ Hamiltonian consistency maintained (dim={new_H_dim})")
    
    # Convert to list of individual state vectors
    state_list = []
    for i, state in enumerate(states):
        # Ensure state is properly normalized
        normalized_state = state / np.linalg.norm(state)
        state_list.append(normalized_state)
        print(f"  State {i}: norm = {np.linalg.norm(normalized_state):.6f}")
    
    return state_list


def train_on_hamiltonian_states(num_words, config, best_params=None):
    """
    Train the quantum model using Hamiltonian-generated sequential states.
    
    Args:
        num_words (int): Number of sequential states to generate and use
        config (dict): Configuration dictionary
        best_params (array): Previous best parameters (optional)
        
    Returns:
        array: Optimized parameters
    """
    print(f"\n{'='*60}")
    print("HAMILTONIAN-BASED TRAINING PHASE")
    print(f"{'='*60}")
    
    # Generate sequential states using Hamiltonian evolution
    states = generate_hamiltonian_states(num_words, config['embedding_dim'])
    
    print(f"\n{'='*60}")
    print(f"Processing {len(states)} Hamiltonian-generated states")
    print(f"{'='*60}")
    
    # Process the sequential states
    states_calculated, U, Z = process_sentence_states(states)
    
    if len(states_calculated) > 0:
        print(f"Successfully processed {len(states_calculated)} state transitions")
        best_params = optimize_parameters(
            config['max_hours'], 
            config['num_iterations'], 
            config['num_layers'], 
            states_calculated, 
            U, 
            Z, 
            best_params, 
            dim=config['embedding_dim']
        )
    else:
        print("No valid states calculated from Hamiltonian evolution, skipping optimization.")
    
    return best_params
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
            U_dagger = get_unitary_from_tk(x.data).conj().T
            Z_dagger = get_unitary_from_tk(z.data).conj().T
            unitary_psi = get_unitary_from_tk(psi)

            print(f"  U_dagger shape={U_dagger.shape}, unitarity check={np.allclose(U_dagger.conj().T @ U_dagger, np.eye(U_dagger.shape[0]))}")
            print(f"  Z_dagger shape={Z_dagger.shape}, unitarity check={np.allclose(Z_dagger.conj().T @ Z_dagger, np.eye(Z_dagger.shape[0]))}")
            print(f"  unitary_psi shape={unitary_psi.shape}, unitarity check={np.allclose(unitary_psi.conj().T @ unitary_psi, np.eye(unitary_psi.shape[0]))}")

        except Exception as e:
            print(f"[ERROR] Unitary calculation failed: {e}")
            continue

        # Append to lists
        U.append(U_dagger)
        Z.append(Z_dagger)
        states_calculated.append(unitary_psi)

        print("  [OK] Triplet saved in U, Z, states_calculated")
    
    return states_calculated, U, Z


def train_on_sentences(sentences, config, best_params=None):
    """
    Train the quantum model on a list of sentences.
    
    Args:
        sentences (list): List of sentences to train on
        config (dict): Configuration dictionary
        best_params (array): Previous best parameters (optional)
        
    Returns:
        array: Optimized parameters
    """
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    enc = Encoding(sentences, embeddingDim=config['embedding_dim'])
    
    for sentence_idx, sentence in enumerate(sentences):
        print(f"\n{'='*60}")
        print(f"Processing sentence {sentence_idx + 1}/{len(sentences)}: '{sentence}'")
        print(f"{'='*60}")
        
        states = enc.stateVectors[sentence_idx]
        states_calculated, U, Z = process_sentence_states(states)
        
        if len(states_calculated) > 0:
            best_params = optimize_parameters(
                config['max_hours'], 
                config['num_iterations'], 
                config['num_layers'], 
                states_calculated, 
                U, 
                Z, 
                best_params, 
                dim=config['embedding_dim']
            )
        else:
            print("No valid states calculated for this sentence, skipping optimization.")
    
    return best_params


def evaluate_on_hamiltonian_states(num_states, best_params, config):
    """
    Evaluate the trained model on Hamiltonian-generated states.
    
    Args:
        num_states (int): Number of states to generate for evaluation
        best_params (array): Trained parameters
        config (dict): Configuration dictionary
        
    Returns:
        float: Average loss on test states
    """
    if best_params is None:
        print("No trained parameters available for evaluation.")
        return None
    
    print(f"\n{'='*60}")
    print("EVALUATION PHASE - HAMILTONIAN STATES")
    print(f"{'='*60}")
    
    # Generate evaluation states using the same Hamiltonian
    eval_states = generate_hamiltonian_states(num_states, config['embedding_dim'])
    
    # Process states to get unitaries (same as training)
    states_calculated, U, Z = process_sentence_states(eval_states)
    
    if len(states_calculated) == 0:
        print("No valid states for evaluation.")
        return None
    
    print(f"Evaluating {len(states_calculated)} state transitions...")
    
    # Use the same circuit evaluation logic as in optimization
    from quantum_circuits import get_circuit_function
    from quantum_utils import get_params
    
    # Use the same logic as optimization: number of transitions directly
    num_words = len(states_calculated)
    print(f"Using circuit function for {num_words} words (same as optimization)")
    
    try:
        circuit_func = get_circuit_function(num_words)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    param_shape = get_params(2, config['num_layers']).shape
    n_params = int(np.prod(param_shape))
    
    # Extract V and K parameters
    params_flat = np.array(best_params).flatten()
    V = params_flat[:n_params].reshape(param_shape)
    K = params_flat[n_params:2*n_params].reshape(param_shape)
    
    total_loss = 0.0
    valid_evaluations = 0
    
    for i, (state_calc, U_val, Z_val) in enumerate(zip(states_calculated, U, Z)):
        try:
            circuit = circuit_func(config['num_layers'])
            
            # Calculate loss for this state transition
            result = circuit(V, K, state_calc, U_val, Z_val)
            if isinstance(result, (int, float, complex)):
                loss_val = abs(result)**2
            else:
                loss_val = np.real(np.trace(result @ result.conj().T))
            
            total_loss += loss_val
            valid_evaluations += 1
            print(f"  State transition {i+1}: loss = {loss_val:.6f}")
            
        except Exception as e:
            print(f"  State transition {i+1}: evaluation failed - {e}")
    
    if valid_evaluations > 0:
        avg_loss = total_loss / valid_evaluations
        print(f"\nAverage evaluation loss: {avg_loss:.6f}")
        return avg_loss
    else:
        print("No valid evaluations completed.")
        return None


def evaluate_on_sentences(sentences, best_params, config):
    """
    Evaluate the trained model on test sentences.
    
    Args:
        sentences (list): List of sentences to evaluate
        best_params (array): Trained parameters
        config (dict): Configuration dictionary
        
    Returns:
        float: Average loss on test sentences
    """
    if best_params is None:
        print("No trained parameters available for evaluation.")
        return None
    
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print(f"{'='*60}")
    
    enc = Encoding(sentences, embeddingDim=config['embedding_dim'])
    param_shape = get_params(2, config['num_layers']).shape
    n_params = int(np.prod(param_shape))
    
    # Extract V and K parameters
    params_flat = np.array(best_params).flatten()
    V = params_flat[:n_params].reshape(param_shape)
    K = params_flat[n_params:2*n_params].reshape(param_shape)
    
    total_loss = 0
    valid_sentences = 0
    
    for sentence_idx, sentence in enumerate(sentences):
        print(f"\n{'='*60}")
        print(f"Evaluating sentence {sentence_idx + 1}/{len(sentences)}: '{sentence}'")
        print(f"{'='*60}")
        
        states = enc.stateVectors[sentence_idx]
        states_calculated, U, Z = process_sentence_states(states)
        
        if len(states_calculated) > 0:
            try:
                circuit_function = get_circuit_function(len(states_calculated))
                loss = circuit_function(states_calculated, U, Z, V, K, config['num_layers'], dim=config['embedding_dim'])
                total_loss += loss
                valid_sentences += 1
                print(f"Loss for sentence: {loss:.6f}")
            except Exception as e:
                print(f"Error evaluating sentence: {e}")
        else:
            print("No valid states for evaluation.")
    
    if valid_sentences > 0:
        avg_loss = total_loss / valid_sentences
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Average loss on {valid_sentences} sentences: {avg_loss:.6f}")
        print(f"Best parameters:")
        print(f"{'='*60}")
        print(best_params)
        return avg_loss
    else:
        print("No valid sentences for evaluation.")
        return None


def main():
    """
    Main execution function.
    """
    # Load configuration
    config = OPTIMIZATION_CONFIG.copy()
    
    print("Quantum Superposition Language Model")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Choose training mode
    print("\nTraining options:")
    print("1. Traditional sentence-based training")
    print("2. Hamiltonian-based sequential state training")
    
    try:
        choice = input("Choose training mode (1 or 2, default=1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received, defaulting to mode 1 (traditional training).")
        choice = "1"
    
    if choice == "2":
        print("Using Hamiltonian-based training mode.")
        use_hamiltonian = True
        try:
            num_words_input = input("Number of sequential states to generate (5 for 4 transitions, 9 for 8 transitions, default=5): ").strip()
            num_words = int(num_words_input) if num_words_input else 5
        except (EOFError, KeyboardInterrupt, ValueError):
            print("Invalid input, using default value of 5.")
            num_words = 5
        
        # Validate that num_words will produce supported transitions
        num_transitions = num_words - 1
        supported_transitions = [2, 4, 8, 16]
        if num_transitions not in supported_transitions:
            print(f"Warning: {num_transitions} transitions not supported. Supported: {supported_transitions}")
            # Find closest supported number
            closest = min(supported_transitions, key=lambda x: abs(x - num_transitions))
            num_words = closest + 1
            print(f"Adjusted to {num_words} states ({closest} transitions)")
    else:
        print("Using traditional sentence-based training mode.")
        use_hamiltonian = False
    
    # Try to load existing parameters
    best_params = load_parameters()
    if best_params is not None:
        print("Loaded parameters from file.")
    else:
        print("No existing parameters found, starting from scratch.")
    
    # Training phase
    if use_hamiltonian:
        best_params = train_on_hamiltonian_states(num_words, config, best_params)
    else:
        best_params = train_on_sentences(TRAINING_SENTENCES, config, best_params)
    
    # Save the trained parameters
    if best_params is not None:
        save_parameters(best_params)
        print("Training completed. Parameters saved.")
    else:
        print("Training failed to produce valid parameters.")
        return
    
    # Evaluation phase
    if use_hamiltonian:
        print("Evaluation with Hamiltonian states...")
        avg_loss = evaluate_on_hamiltonian_states(num_words, best_params, config)
    else:
        avg_loss = evaluate_on_sentences(DEFAULT_SENTENCES, best_params, config)
    
    if avg_loss is not None:
        print(f"\nFinal evaluation loss: {avg_loss:.6f}")
    else:
        print("Evaluation failed.")


if __name__ == "__main__":
    main()