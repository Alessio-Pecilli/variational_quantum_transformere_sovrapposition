#!/usr/bin/env python3
"""
Test semplificato solo per le importazioni critiche HPC
"""

def test_critical_imports():
    """Test solo delle importazioni che servono per HPC"""
    print("🎯 TEST IMPORTAZIONI CRITICHE PER HPC")
    print("="*50)
    
    try:
        # Test import base
        print("1. Numpy e multiprocessing...")
        import numpy as np
        from multiprocessing import Pool, cpu_count
        print("   ✅ OK")
        
        print("2. Config...")
        from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
        print(f"   ✅ OK - {len(TRAINING_SENTENCES)} sentences disponibili")
        
        print("3. Quantum MPI utils...")
        from quantum_mpi_utils import loss_and_grad_for_sentence
        print("   ✅ OK")
        
        print("4. Quantum circuits...")
        from quantum_circuits import get_circuit_function
        print("   ✅ OK")
        
        print("5. Main superposition...")
        from main_superposition import process_sentence_states
        print("   ✅ OK")
        
        print("\n🚀 TUTTE LE IMPORTAZIONI CRITICHE FUNZIONANO!")
        print("✅ Il codice debug dovrebbe funzionare su HPC")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_critical_imports()
    print(f"\nRisultato: {'✅ PRONTO' if success else '❌ PROBLEMI'}")