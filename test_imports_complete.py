#!/usr/bin/env python3
"""
Test completo delle importazioni per HPC debug
"""

def test_all_imports():
    """Test di tutte le importazioni necessarie"""
    print("🔍 TEST IMPORTAZIONI COMPLETE")
    print("="*50)
    
    try:
        print("1. Config...")
        from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
        print(f"   ✅ Config OK: {len(TRAINING_SENTENCES)} sentences")
        
        print("2. Encoding...")
        from encoding import Encoding
        print("   ✅ Encoding OK")
        
        print("3. Quantum circuits...")
        from quantum_circuits import get_circuit_function
        print("   ✅ Quantum circuits OK")
        
        print("4. Quantum MPI utils...")
        from quantum_mpi_utils import loss_and_grad_for_sentence
        print("   ✅ MPI utils OK")
        
        print("5. Main superposition...")
        from main_superposition import process_sentence_states
        print("   ✅ Main superposition OK")
        
        print("6. Numpy...")
        import numpy as np
        print("   ✅ Numpy OK")
        
        print("7. Multiprocessing...")
        from multiprocessing import Pool, cpu_count
        print("   ✅ Multiprocessing OK")
        
        print("\n🧪 TEST FUNZIONE BASE")
        print("="*30)
        
        # Test rapido della funzione
        sentence = TRAINING_SENTENCES[0]
        encoding = Encoding(sentence)
        n_params = 2 * encoding.n_qubits
        params = np.random.uniform(0, 2*np.pi, n_params)
        sentence_data = (sentence, encoding, len(sentence.split()))
        
        print(f"Sentence: '{sentence}'")
        print(f"Params: {n_params}")
        
        loss, grad = loss_and_grad_for_sentence(params, sentence_data)
        print(f"✅ Test funzione OK: Loss={loss:.6f}, Grad norm={np.linalg.norm(grad):.6f}")
        
        print("\n🎯 TUTTI I TEST PASSATI!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_imports()
    if success:
        print("\n🚀 IL CODICE È PRONTO PER HPC!")
    else:
        print("\n💥 PROBLEMA NELLE IMPORTAZIONI!")