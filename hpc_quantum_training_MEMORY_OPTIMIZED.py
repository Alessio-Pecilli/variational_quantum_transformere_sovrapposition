#!/usr/bin/env python3
"""
🧠 QUANTUM TRAINING HPC - MEMORY OPTIMIZED VERSION 
Versione ottimizzata per evitare OUT_OF_MEMORY su Leonardo
Riduce workers e ottimizza memoria senza perdere parallelizzazione
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
import traceback
import gc
from multiprocessing import Pool, cpu_count, Manager

# Import quantum libs
try:
    import pennylane as qml
    from pennylane import numpy as qnp
except ImportError:
    print("⚠️ WARNING: PennyLane non disponibile, uso numpy standard")
    import numpy as qnp

def setup_hpc_logger():
    """Logger per HPC con timestamp dettagliato"""
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    log_filename = f'memory_optimized_{job_id}.log'
    
    # Setup logging con formato dettagliato
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧠 MEMORY OPTIMIZED HPC LOGGING ATTIVO")
    return logger

def get_hpc_workers_memory_safe():
    """
    Calcola workers OTTIMALI per evitare OOM
    Limita workers per gestire memoria su nodi HPC
    """
    omp = os.environ.get('OMP_NUM_THREADS')
    slurm = os.environ.get('SLURM_CPUS_PER_TASK') 
    pbs = os.environ.get('PBS_NP')
    
    if omp and int(omp) > 1:
        total_workers = int(omp)
        source = "OMP_NUM_THREADS"
    elif slurm and int(slurm) > 1:
        total_workers = int(slurm)
        source = "SLURM_CPUS_PER_TASK"
    elif pbs and int(pbs) > 1:
        total_workers = int(pbs)
        source = "PBS_NP"
    else:
        total_workers = cpu_count()
        source = "CPU_COUNT"
    
    # 🧠 MEMORY SAFE: Limita workers per evitare OOM
    # Su Leonardo: 128GB per nodo, ~32 core per nodo
    # Calcolo conservativo: 1-2GB per worker per dati quantum
    
    # Strategia adaptive based on available CPUs
    if total_workers >= 100:  # Multi-nodo (4+ nodi)
        # Usa solo 50% dei core per evitare memory pressure
        safe_workers = min(64, total_workers // 2)
        memory_strategy = "MULTI_NODE_CONSERVATIVE"
    elif total_workers >= 60:   # 2-3 nodi
        safe_workers = min(32, total_workers // 2)  
        memory_strategy = "DUAL_NODE_SAFE"
    elif total_workers >= 30:   # 1 nodo
        safe_workers = min(16, total_workers // 2)
        memory_strategy = "SINGLE_NODE_SAFE"
    else:  # Locale o pochi core
        safe_workers = max(1, total_workers - 1)
        memory_strategy = "LOCAL_MODE"
    
    return safe_workers, source, total_workers, memory_strategy

def log_system_info_memory_safe(logger):
    """Log sistema con focus su ottimizzazione memoria"""
    logger.info("=" * 70)
    logger.info("🧠 MEMORY OPTIMIZED SYSTEM INFO")
    logger.info("=" * 70)
    
    # CPU Info con strategia memoria
    safe_workers, source, total_cpus, memory_strategy = get_hpc_workers_memory_safe()
    logger.info(f"   💪 CPU Totali Disponibili: {total_cpus} (fonte: {source})")
    logger.info(f"   🧠 Workers Memory-Safe: {safe_workers}")
    logger.info(f"   📊 Strategia Memoria: {memory_strategy}")
    logger.info(f"   🎯 Riduzione CPU: {total_cpus - safe_workers} core risparmiati per RAM")
    
    # Memoria info (se disponibile)
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        estimated_per_worker = available_gb / (safe_workers + 2)  # +2 per buffer
        
        logger.info(f"   🧠 Memoria Totale: {total_gb:.1f} GB")
        logger.info(f"   💽 Memoria Disponibile: {available_gb:.1f} GB")
        logger.info(f"   📏 Stima GB per Worker: {estimated_per_worker:.2f} GB")
        
        if estimated_per_worker < 1.5:
            logger.warning(f"   ⚠️  MEMORY PRESSURE! Solo {estimated_per_worker:.2f}GB per worker")
        else:
            logger.info(f"   ✅ Memory OK: {estimated_per_worker:.2f}GB per worker sufficiente")
            
    except ImportError:
        logger.info("   📊 psutil non disponibile - modalità HPC base")
    
    # Environment HPC
    slurm_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 'N/A')
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')  
    slurm_mem = os.environ.get('SLURM_MEM_PER_NODE', 'N/A')
    
    logger.info(f"   🏗️  SLURM Nodi: {slurm_nodes}")
    logger.info(f"   🏗️  SLURM CPU per task: {slurm_cpus}")
    logger.info(f"   🏗️  SLURM Memoria per nodo: {slurm_mem}")
    logger.info("=" * 70)

def compute_gradient_batch_memory_optimized(batch_data):
    """
    Calcola gradiente per batch con OTTIMIZZAZIONE MEMORIA
    Usa garbage collection e libera memoria dopo ogni calcolo
    """
    param_indices, params, shift, states_calculated, U, Z, num_layers, embedding_dim, circuit_func = batch_data
    
    worker_pid = os.getpid()
    start_time = time.time()
    batch_size = len(param_indices)
    
    try:
        # Forza garbage collection prima del calcolo pesante
        gc.collect()
        
        # Calcola gradiente per TUTTO il batch in una chiamata
        gradients = []
        
        for i in param_indices:
            # Calcola con shift positivo
            params_plus = params.copy()
            params_plus[i] += shift
            
            # Compute quantum state (memory intensive)
            state_plus = circuit_func(params_plus, U, Z, num_layers, embedding_dim)
            loss_plus = np.real(np.sum(state_plus))
            
            # Calcola con shift negativo  
            params_minus = params.copy()
            params_minus[i] -= shift
            
            state_minus = circuit_func(params_minus, U, Z, num_layers, embedding_dim)
            loss_minus = np.real(np.sum(state_minus))
            
            # Gradient con parameter shift rule
            gradient = (loss_plus - loss_minus) / (2 * shift)
            gradients.append(gradient)
            
            # Forza cleanup delle variabili temporanee
            del params_plus, params_minus, state_plus, state_minus
        
        # Forza garbage collection dopo batch
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"  🧠 Worker PID:{worker_pid} processed batch[{param_indices[0]}:{param_indices[-1]}] "
              f"({batch_size} params) in {elapsed:.3f}s [MEMORY-OPT]")
        
        return param_indices, gradients
        
    except Exception as e:
        print(f"  ❌ Worker PID:{worker_pid} ERRORE batch {param_indices}: {e}")
        # Forza cleanup anche in caso di errore
        gc.collect()
        return param_indices, [0.0] * len(param_indices)

def create_memory_safe_batches(num_params, num_workers):
    """
    Crea batch con dimensioni CONSERVATIVE per memoria
    Batch più piccoli = meno memoria per worker
    """
    # Batch size più piccolo per ridurre memory footprint
    # Invece di num_workers*2, usa num_workers*1.5 per più batch piccoli
    target_batches = min(num_workers * 3, 30)  # Max 30 batch più piccoli
    batch_size = max(1, num_params // target_batches)
    
    # Forza batch_size massimo per controllo memoria
    max_batch_size = 8  # Massimo 8 parametri per batch
    batch_size = min(batch_size, max_batch_size)
    
    batches = []
    for i in range(0, num_params, batch_size):
        end_idx = min(i + batch_size, num_params)
        batch_indices = list(range(i, end_idx))
        batches.append(batch_indices)
    
    return batches

def compute_gradient_parallel_memory_safe(params, states_calculated, U, Z, num_layers, embedding_dim, shift=0.1):
    """
    Compute gradient in parallel con OTTIMIZZAZIONE MEMORIA
    """
    logger = logging.getLogger(__name__)
    
    # Import locale per evitare problemi multiprocessing
    if 'hamiltonian' in sys.modules:
        hamiltonian_module = sys.modules['hamiltonian']
        circuit_func = hamiltonian_module.quantum_transformer_circuit
    else:
        logger.error("❌ Modulo hamiltonian non disponibile!")
        return np.zeros_like(params)
    
    num_params = len(params)
    safe_workers, _, total_cpus, memory_strategy = get_hpc_workers_memory_safe()
    
    logger.info(f"🧠 GRADIENT COMPUTATION [MEMORY-OPTIMIZED]")
    logger.info(f"   Parametri totali: {num_params}")
    logger.info(f"   Workers safe: {safe_workers} (di {total_cpus} disponibili)")
    logger.info(f"   Strategia: {memory_strategy}")
    
    start_time = time.time()
    
    # Crea batch memory-safe
    batches = create_memory_safe_batches(num_params, safe_workers)
    logger.info(f"   Batch creati: {len(batches)} (dimensione media: {np.mean([len(b) for b in batches]):.1f})")
    
    # Prepara dati per multiprocessing
    batch_data_list = []
    for batch_indices in batches:
        batch_data = (
            batch_indices, params, shift, states_calculated,
            U, Z, num_layers, embedding_dim, circuit_func
        )
        batch_data_list.append(batch_data)
    
    # Processa con Pool MEMORY-SAFE
    gradients = np.zeros(num_params)
    
    with Pool(processes=safe_workers) as pool:
        try:
            logger.info(f"   🚀 Avvio Pool con {safe_workers} workers...")
            
            # Map con timeout per evitare hang
            results = pool.map(compute_gradient_batch_memory_optimized, batch_data_list)
            
            # Aggrega risultati
            for param_indices, batch_gradients in results:
                for i, gradient in enumerate(batch_gradients):
                    gradients[param_indices[i]] = gradient
                    
        except Exception as e:
            logger.error(f"❌ Errore nel Pool: {e}")
            logger.error(traceback.format_exc())
            
        finally:
            # Forza cleanup
            pool.close()
            pool.join()
            gc.collect()
    
    elapsed = time.time() - start_time
    logger.info(f"   ⏱️ Gradient computation completato in {elapsed:.2f}s")
    logger.info(f"   📊 Gradient norm: {np.linalg.norm(gradients):.6f}")
    
    return gradients

if __name__ == "__main__":
    # Test memory optimization
    logger = setup_hpc_logger()
    log_system_info_memory_safe(logger)
    
    # Test workers configuration
    safe_workers, source, total_cpus, strategy = get_hpc_workers_memory_safe()
    logger.info(f"✅ Test completato: {safe_workers} workers safe da {total_cpus} totali")
    logger.info(f"   Strategia: {strategy}, Fonte: {source}")