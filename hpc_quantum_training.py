#!/usr/bin/env python3
"""
HPC QUANTUM TRAINING - NO MPI, SOLO MULTIPROCESSING

COMANDO PER HPC:
    python hpc_quantum_training.py

FEATURES:
- NO MPI (evita problemi librerie HPC)
- Auto-detect risorse HPC (SLURM/PBS/OMP) 
- Parallelizzazione multiprocessing pura
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
    
    print(f"🔍 RILEVAMENTO RISORSE HPC:")
    print(f"  OMP_NUM_THREADS: {omp}")
    print(f"  SLURM_CPUS_PER_TASK: {slurm}")
    print(f"  PBS_NP: {pbs}")
    print(f"  CPU_COUNT: {cpu_count()}")
    
    if omp and int(omp) > 1: 
        workers = int(omp)
        source = "OMP_NUM_THREADS"
    elif slurm and int(slurm) > 1: 
        workers = int(slurm)
        source = "SLURM_CPUS_PER_TASK"
    elif pbs and int(pbs) > 1: 
        workers = int(pbs)
        source = "PBS_NP"
    else: 
        workers = cpu_count()
        source = "CPU_COUNT"
    
    print(f"  ✅ WORKERS SCELTI: {workers} (da {source})")
    return workers


def main():
    """Training principale HPC senza MPI"""
    
    print("🚀 HPC QUANTUM TRAINING - MULTIPROCESSING ONLY")
    print("=" * 60)
    
    # Config
    config = OPTIMIZATION_CONFIG.copy()
    workers = get_hpc_workers()
    
    # Dati di training
    sentences = TRAINING_SENTENCES.copy()
    print(f"\n📊 CONFIGURAZIONE:")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Workers: {workers}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    
    # Encoding
    enc = Encoding(sentences, embeddingDim=config['embedding_dim'])
    
    # Parametri iniziali (come nel codice esistente)
    param_shape = get_params(config['num_qubits'], config['num_layers']).shape
    n_params = int(np.prod(param_shape))
    num_params = 2 * n_params  # V and K parameters
    
    # Inizializzazione parametri (STESSO MODO del main_superposition_mpi.py)
    params = np.random.randn(num_params) * 0.1
    
    print(f"  Total parametri: {num_params}")
    print(f"  Parametri V: {n_params}")
    print(f"  Parametri K: {n_params}")
    print(f"  Param shape: {param_shape}")
    
    print(f"\n📝 FRASI DI TRAINING:")
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        print(f"  {i+1:2d}. '{sentence}' ({len(words)} parole)")
    
    # Training loop
    print(f"\n🎯 INIZIO TRAINING")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        print(f"\n--- EPOCH {epoch+1}/{config['epochs']} ---")
        
        for i, sentence in enumerate(sentences):
            sentence_start = time.time()
            print(f"\nSentence {i+1}/{len(sentences)}: '{sentence}'")
            
            # Calcola num_words per get_circuit_function
            words = sentence.split()
            num_words = len(words)
            
            # Circuit function con num_words corretto
            circuit_func = get_circuit_function(num_words)
            
            # States dalla tua funzione
            states = process_sentence_states(sentence, enc, circuit_func, params, config)
            
            # Loss e gradiente PARALLELI (sempre parallel=True)
            loss, grad = loss_and_grad_for_sentence(
                params=params,
                states=states,
                target_sentence=sentence,
                enc=enc,
                circuit_func=circuit_func,
                config=config,
                parallel=True  # SEMPRE PARALLELO
            )
            
            # Update parametri (gradient descent step come nel MPI code)
            params = params - config['learning_rate'] * grad
            
            epoch_loss += loss
            sentence_time = time.time() - sentence_start
            
            # Calcolo gradient norm
            grad_norm = np.linalg.norm(grad)
            
            print(f"  Loss: {loss:.6f}")
            print(f"  Grad norm: {grad_norm:.6f}")
            print(f"  Time: {sentence_time:.2f}s")
        
        # Statistiche epoch
        avg_loss = epoch_loss / len(sentences)
        epoch_time = time.time() - epoch_start
        
        print(f"\n🎯 EPOCH {epoch+1} COMPLETATA:")
        print(f"  Loss media: {avg_loss:.6f}")
        print(f"  Tempo epoch: {epoch_time:.2f}s")
        print(f"  Tempo per sentence: {epoch_time/len(sentences):.2f}s")
        
        # Salva parametri ogni epoch
        save_parameters(params, f"hpc_params_epoch_{epoch+1}")
    
    # Salva risultati finali
    final_file = f"final_hpc_params_epoch_{config['epochs']}"
    save_parameters(params, final_file)
    
    print(f"\n✅ TRAINING COMPLETATO!")
    print("=" * 60)
    print(f"🎯 RISULTATI FINALI:")
    print(f"  Epochs completate: {config['epochs']}")
    print(f"  Loss finale: {avg_loss:.6f}")
    print(f"  Workers utilizzati: {workers}")
    print(f"  Parametri salvati: {final_file}.json")
    print("=" * 60)


if __name__ == "__main__":
    main()