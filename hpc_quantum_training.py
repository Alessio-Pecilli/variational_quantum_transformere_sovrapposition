#!/usr/bin/env python3
"""
HPC QUANTUM TRAINING - FILE UNICO PER ESECUZIONE

COMANDO PER HPC:
    python hpc_quantum_training.py

FEATURES:
- Auto-detect risorse HPC (SLURM/PBS/OMP)
- 100% parallelo sempre  
- Usa tutte le tue configurazioni
"""

import numpy as np
import sys
import os
import time
from multiprocessing import cpu_count

# Imports essenziali
from config import OPTIMIZATION_CONFIG, TRAINING_SENTENCES
from encoding import Encoding
from quantum_mpi_utils import loss_and_grad_for_sentence
from quantum_circuits import get_circuit_function
from main_superposition import process_sentence_states
from quantum_utils import get_params
from visualization import save_parameters


def get_hpc_workers():
    """Auto-detect workers per HPC"""
    omp = os.environ.get('OMP_NUM_THREADS')
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    pbs = os.environ.get('PBS_NP')
    
    if omp: return int(omp)
    if slurm: return int(slurm) 
    if pbs: return int(pbs)
    return cpu_count()


def main():
    """Training principale HPC"""
    
    print("🚀 HPC QUANTUM TRAINING")
    print("=" * 40)
    
    # Config
    config = OPTIMIZATION_CONFIG.copy()
    workers = get_hpc_workers()
    print(f"Workers: {workers}")
    
    # Dati
    sentences = TRAINING_SENTENCES.copy()
    enc = Encoding(sentences, embeddingDim=config['embedding_dim'])
    circuit_func = get_circuit_function()
    
    # Parametri iniziali
    params = get_params(
        num_layers=config['num_layers'],
        num_qubits=config['num_qubits']
    )
    
    print(f"Sentences: {len(sentences)}")
    print(f"Parametri: V={len(params['V'])}, K={len(params['K'])}")
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        
        print(f"\n--- EPOCH {epoch+1}/{config['epochs']} ---")
        
        for i, sentence in enumerate(sentences):
            print(f"Sentence {i+1}/{len(sentences)}: '{sentence}'")
            
            # States dalla tua funzione
            states = process_sentence_states(sentence, enc, circuit_func, params, config)
            
            # Loss e gradiente PARALLELI
            loss, grad = loss_and_grad_for_sentence(
                params=params,
                states=states,
                target_sentence=sentence,
                enc=enc,
                circuit_func=circuit_func,
                config=config,
                parallel=True  # SEMPRE PARALLELO
            )
            
            # Update parametri
            params['V'] = params['V'] - config['learning_rate'] * grad['V']
            params['K'] = params['K'] - config['learning_rate'] * grad['K']
            
            epoch_loss += loss
            print(f"Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / len(sentences)
        print(f"EPOCH {epoch+1} LOSS MEDIA: {avg_loss:.4f}")
    
    # Salva risultati finali
    save_parameters(params, f"final_hpc_params_epoch_{config['epochs']}")
    print("\n✅ TRAINING COMPLETATO!")
    print(f"Parametri salvati in: final_hpc_params_epoch_{config['epochs']}")


if __name__ == "__main__":
    main()