#!/usr/bin/env python3
"""
Debug script per riprodurre il crash HPC
Test con 32 workers come su HPC
"""

import traceback
import sys
from multiprocessing import cpu_count
import time

# Forza 32 workers come su HPC
import os
os.environ['OMP_NUM_THREADS'] = '32'

def debug_step(step_name, func):
    """Wrapper per debugging con try-catch dettagliato"""
    try:
        print(f"🔍 DEBUG: Inizio {step_name}...")
        result = func()
        print(f"✅ DEBUG: {step_name} completato con successo")
        return result
    except Exception as e:
        print(f"❌ DEBUG: ERRORE in {step_name}")
        print(f"Tipo errore: {type(e).__name__}")
        print(f"Messaggio: {str(e)}")
        print("Traceback completo:")
        traceback.print_exc()
        raise

def test_imports():
    """Test importazione moduli"""
    print("Importazione config...")
    from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
    
    print("Importazione encoding...")
    from encoding import Encoding
    
    print("Importazione quantum_circuits...")
    from quantum_circuits import create_superposition_circuit, calculate_loss_and_gradient
    
    print("Importazione numpy...")
    import numpy as np
    
    print("Importazione multiprocessing...")
    from multiprocessing import Pool
    
    return True

def test_encoding_setup():
    """Test setup encoding"""
    from config import TRAINING_SENTENCES
    from encoding import Encoding
    
    # Usa solo la prima sentence per test
    first_sentence = TRAINING_SENTENCES[0]
    print(f"Prima sentence: {first_sentence}")
    
    encoding = Encoding(first_sentence)
    print(f"Encoding creato, n_qubits: {encoding.n_qubits}")
    
    return encoding, first_sentence

def test_worker_detection():
    """Test rilevamento workers"""
    from quantum_mpi_utils import get_hpc_workers
    
    workers = get_hpc_workers()
    print(f"Workers rilevati: {workers}")
    
    return workers

def test_first_sentence_processing():
    """Test elaborazione della prima sentence"""
    from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
    from encoding import Encoding
    from quantum_circuits import create_superposition_circuit, calculate_loss_and_gradient
    import numpy as np
    
    # Setup come nel main
    first_sentence = TRAINING_SENTENCES[0]
    encoding = Encoding(first_sentence)
    
    # Parametri iniziali
    n_params = 2 * encoding.n_qubits  # V e K parameters
    initial_params = np.random.uniform(0, 2*np.pi, n_params)
    
    print(f"Sentence: {first_sentence}")
    print(f"N_qubits: {encoding.n_qubits}")
    print(f"N_params: {n_params}")
    print(f"Params shape: {initial_params.shape}")
    
    # Prova a creare il circuito
    circuit = create_superposition_circuit(encoding, len(first_sentence.split()))
    print(f"Circuito creato con successo")
    
    # Prova calcolo loss (senza parallelizzazione prima)
    sentence_data = (first_sentence, encoding, len(first_sentence.split()))
    
    print("Test calcolo loss sequenziale...")
    loss, grad = calculate_loss_and_gradient(initial_params, sentence_data)
    print(f"Loss: {loss:.6f}")
    print(f"Gradient norm: {np.linalg.norm(grad):.6f}")
    
    return True

def main():
    """Test principale che replica l'esecuzione HPC"""
    print("🖥️ DEBUG CONFIGURAZIONE HPC")
    
    # Step 1: Test imports
    debug_step("Importazione moduli", test_imports)
    
    # Step 2: Test worker detection
    workers = debug_step("Rilevamento workers", test_worker_detection)
    print(f"📊 Workers rilevati: {workers}")
    
    # Step 3: Test encoding setup
    encoding, sentence = debug_step("Setup encoding", test_encoding_setup)
    
    # Step 4: Test prima sentence
    debug_step("Elaborazione prima sentence", test_first_sentence_processing)
    
    print("⚡ DEBUG: Tutti i test base superati!")
    
    # Step 5: Test con multiprocessing (quello che probabilmente crasha)
    print("🔥 Test multiprocessing con 32 workers...")
    
    try:
        from multiprocessing import Pool
        from quantum_circuits import calculate_loss_and_gradient
        from config import TRAINING_SENTENCES
        import numpy as np
        
        # Setup per test multiprocessing
        sentence = TRAINING_SENTENCES[0]
        n_params = 2 * encoding.n_qubits
        params = np.random.uniform(0, 2*np.pi, n_params)
        sentence_data = (sentence, encoding, len(sentence.split()))
        
        # Test con Pool di 32 workers
        with Pool(processes=32) as pool:
            print(f"Pool di 32 workers creato")
            
            # Test semplice - calcola loss 3 volte
            tasks = [(params, sentence_data) for _ in range(3)]
            
            print("Invio tasks al pool...")
            results = pool.starmap(calculate_loss_and_gradient, tasks)
            
            print(f"✅ Multiprocessing test completato! {len(results)} risultati")
            
    except Exception as e:
        print(f"❌ ERRORE nel multiprocessing test:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Messaggio: {str(e)}")
        traceback.print_exc()
        
        # Prova con meno workers
        print("🔄 Provo con 4 workers...")
        try:
            with Pool(processes=4) as pool:
                results = pool.starmap(calculate_loss_and_gradient, tasks)
                print(f"✅ Con 4 workers funziona: {len(results)} risultati")
        except Exception as e2:
            print(f"❌ Anche con 4 workers fallisce: {e2}")
    
    print("🎯 DEBUG completato!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 CRASH GENERALE:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Messaggio: {str(e)}")
        traceback.print_exc()
        sys.exit(1)