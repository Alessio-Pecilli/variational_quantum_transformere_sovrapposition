#!/usr/bin/env python3
"""
Quick fix per HPC - controlliamo cosa c'è in quantum_circuits.py
"""

def main():
    print("🔍 Controllo contenuto quantum_circuits.py...")
    
    try:
        import quantum_circuits
        print("✅ quantum_circuits importato con successo")
        
        # Vediamo cosa contiene
        functions = [attr for attr in dir(quantum_circuits) if not attr.startswith('_')]
        print(f"📋 Funzioni disponibili: {functions}")
        
        # Controllo specifico delle funzioni che servono
        needed_functions = ['create_superposition_circuit', 'calculate_loss_and_gradient']
        
        for func in needed_functions:
            if hasattr(quantum_circuits, func):
                print(f"✅ {func}: PRESENTE")
            else:
                print(f"❌ {func}: MANCANTE")
        
    except Exception as e:
        print(f"❌ Errore import: {e}")

if __name__ == "__main__":
    main()