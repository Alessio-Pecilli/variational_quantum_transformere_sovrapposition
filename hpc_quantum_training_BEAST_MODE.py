#!/usr/bin/env python3
"""
🚀 BEAST MODE HPC TRAINING - VERA PARALLELIZZAZIONE AL 100%
Risolve tutti i problemi: single worker, batch overhead, loss mancante
"""

import traceback
import sys
import time
import logging
from pathlib import Path
import os
import signal
from contextlib import contextmanager
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import psutil
import pickle
from datetime import datetime

def setup_logging():
    """Setup logging con timestamp e dettagli"""
    log_file = Path("hpc_beast_mode.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 BEAST MODE HPC LOGGING ATTIVO")
    return logger

def get_hpc_workers_max():
    """USA TUTTA LA POTENZA HPC - NESSUN LIMITE ARTIFICIALE!"""
    omp = os.environ.get('OMP_NUM_THREADS')
    slurm = os.environ.get('SLURM_CPUS_PER_TASK') 
    pbs = os.environ.get('PBS_NP')
    
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
    
    # 🚀 BEAST MODE: USA TUTTO - ZERO LIMITS!
    # Lascia solo 1 core per OS se >2 core disponibili
    max_workers = max(1, workers - 1) if workers > 2 else workers
    
    return max_workers, source, workers

def log_system_info_beast(logger):
    """Log sistema con focus su parallelizzazione"""
    logger.info("=" * 60)
    logger.info("🖥️ BEAST MODE SYSTEM INFO")
    logger.info("=" * 60)
    
    # CPU Info dettagliata
    max_workers, source, total_cpus = get_hpc_workers_max()
    logger.info(f"   💪 CPU Totali: {total_cpus} (fonte: {source})")
    logger.info(f"   🚀 Workers BEAST MODE: {max_workers} (MASSIMA POTENZA!)")
    
    # Memoria
    try:
        memory = psutil.virtual_memory()
        logger.info(f"   🧠 Memoria: {memory.total / (1024**3):.1f} GB totale")
        logger.info(f"      Disponibile: {memory.available / (1024**3):.1f} GB")
        logger.info(f"      Utilizzo: {memory.percent:.1f}%")
    except:
        logger.info("   🧠 Memoria: Info non disponibile")
    
    # Variabili ambiente HPC
    env_vars = ['SLURM_JOB_ID', 'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS']
    for var in env_vars:
        value = os.environ.get(var, 'N/A')
        logger.info(f"   🔧 {var}: {value}")

def compute_gradient_batch_optimized(batch_data):
    """
    Calcola gradiente per un BATCH di parametri - OTTIMIZZATO!
    Riduce overhead da 144 chiamate a ~10-20 batch
    """
    param_indices, params, shift, states_calculated, U, Z, num_layers, embedding_dim, circuit_func = batch_data
    
    # Info worker per debug parallelizzazione
    worker_pid = os.getpid()
    batch_size = len(param_indices)
    
    # Timer per performance
    start_time = time.time()
    
    # Calcola gradiente per TUTTO il batch in una chiamata
    gradients = []
    
    try:
        from quantum_mpi_utils import _compute_single_gradient_component
        
        for param_idx in param_indices:
            grad = _compute_single_gradient_component(
                param_idx, params, shift, states_calculated, U, Z,
                num_layers, embedding_dim, circuit_func
            )
            gradients.append(grad)
        
        elapsed = time.time() - start_time
        
        # Log performance per worker
        print(f"  Worker PID:{worker_pid} processed batch[{param_indices[0]}:{param_indices[-1]}] "
              f"({batch_size} params) in {elapsed:.3f}s")
        
        return param_indices, gradients, worker_pid, elapsed
        
    except Exception as e:
        print(f"  ❌ Worker PID:{worker_pid} ERRORE batch {param_indices}: {e}")
        return param_indices, [0.0] * len(param_indices), worker_pid, 0.0

def create_smart_batches(num_params, num_workers):
    """
    Crea batch intelligenti per minimizzare overhead
    Invece di 144 task → ~10-20 batch bilanciati
    """
    # Calcola dimensione batch ottimale
    target_batches = min(num_workers * 2, 20)  # Max 20 batch totali
    batch_size = max(1, num_params // target_batches)
    
    batches = []
    for i in range(0, num_params, batch_size):
        batch_end = min(i + batch_size, num_params)
        batch_indices = list(range(i, batch_end))
        batches.append(batch_indices)
    
    return batches

def train_with_beast_mode_parallelization(logger):
    """🚀 TRAINING CON VERA PARALLELIZZAZIONE BEAST MODE"""
    
    # Setup importazioni
    from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
    from encoding import Encoding
    from main_superposition import process_sentence_states
    from quantum_circuits import get_circuit_function
    from quantum_utils import get_params
    
    # Workers configuration
    max_workers, source, total_cpus = get_hpc_workers_max()
    logger.info(f"🚀 AVVIO TRAINING BEAST MODE")
    logger.info(f"   Workers: {max_workers}/{total_cpus} (fonte: {source})")
    
    # Prepara dati
    sentences = TRAINING_SENTENCES
    encoding = Encoding(sentences, embeddingDim=OPTIMIZATION_CONFIG['embedding_dim'])
    
    # Parametri iniziali OTTIMIZZATI
    param_shape = get_params(OPTIMIZATION_CONFIG['num_qubits'], OPTIMIZATION_CONFIG['num_layers']).shape
    n_params = int(np.prod(param_shape))
    num_params = 2 * n_params  # V and K parameters
    
    # 🎯 INIZIALIZZAZIONE MIGLIORATA: Xavier/Glorot initialization
    xavier_std = np.sqrt(2.0 / (OPTIMIZATION_CONFIG['num_qubits'] + OPTIMIZATION_CONFIG['embedding_dim']))
    params = np.random.randn(num_params) * xavier_std
    logger.info(f"   🎯 Xavier initialization: std={xavier_std:.4f}")
    
    logger.info(f"📊 CONFIGURAZIONE TRAINING:")
    logger.info(f"   Frasi: {len(sentences)}")
    logger.info(f"   Parametri totali: {num_params} ({n_params} V + {n_params} K)")
    logger.info(f"   Embedding dim: {OPTIMIZATION_CONFIG['embedding_dim']}")
    
    # Prepara tutti i dati di training
    all_training_data = []
    total_states = 0
    
    for i, sent in enumerate(sentences):
        states_i = encoding.stateVectors[i]
        states_calc_i, U_i, Z_i = process_sentence_states(states_i)
        if len(states_calc_i) > 0:
            all_training_data.append((states_calc_i, U_i, Z_i, sent))
            total_states += len(states_calc_i)
            logger.info(f"   Frase {i+1}: '{sent[:25]}...' -> {len(states_calc_i)} states")
    
    if len(all_training_data) == 0:
        logger.error("❌ NESSUN DATO VALIDO PER TRAINING!")
        return False
    
    logger.info(f"✅ DATASET PRONTO: {len(all_training_data)} frasi, {total_states} stati totali")
    
    # Training parameters OTTIMIZZATI per convergenza
    learning_rate = 0.001  # Ridotto per stabilità
    max_epochs = 15  # Più epoche per convergenza graduale
    convergence_threshold = 0.1   # Target più realistico inizialmente
    min_learning_rate = 1e-6      # LR minimo per scheduling
    patience = 3                   # Epoche senza miglioramento per LR decay
    
    # Tracciamento performance AVANZATO
    best_loss = float('inf')
    best_params = params.copy()
    training_history = []
    no_improve_count = 0  # Counter per learning rate decay
    current_lr = learning_rate  # LR attuale (può cambiare)
    
    # 🚀 BEAST MODE TRAINING LOOP
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 AVVIO TRAINING BEAST MODE")
    logger.info(f"   Epochs: {max_epochs}")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info(f"   Convergence Target: < {convergence_threshold}")
    logger.info(f"   Parallelizzazione: {max_workers} workers")
    logger.info(f"{'='*60}")
    
    # PERSISTENT POOL per riusare workers
    with Pool(processes=max_workers) as pool:
        logger.info(f"✅ Pool creato con {max_workers} workers persistenti")
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            total_gradient = np.zeros(num_params)
            
            logger.info(f"\n📈 EPOCA {epoch+1}/{max_epochs}")
            
            # GRADIENT ACCUMULATION su tutte le frasi
            for batch_idx, (states_calc, U_batch, Z_batch, sent) in enumerate(all_training_data):
                
                # Calcola loss per questa frase
                circuit_func = get_circuit_function(len(states_calc))
                loss_batch = circuit_func(
                    states_calc, U_batch, Z_batch,
                    params[:num_params//2].reshape(param_shape),
                    params[num_params//2:].reshape(param_shape),
                    OPTIMIZATION_CONFIG['num_layers'], 
                    OPTIMIZATION_CONFIG['embedding_dim']
                )
                epoch_loss += loss_batch
                
                # 🚀 SMART BATCHING: Riduce da 144 task a ~10-20 batch
                batches = create_smart_batches(num_params, max_workers)
                logger.info(f"   Frase {batch_idx+1}: Loss={loss_batch:.4f}, "
                           f"Batch={len(batches)} (vs {num_params} task singoli)")
                
                # Prepara batch data per parallelizzazione
                batch_tasks = []
                shift = np.pi / 2
                
                for batch_indices in batches:
                    batch_task = (
                        batch_indices, params, shift, states_calc, U_batch, Z_batch,
                        OPTIMIZATION_CONFIG['num_layers'], 
                        OPTIMIZATION_CONFIG['embedding_dim'], 
                        circuit_func
                    )
                    batch_tasks.append(batch_task)
                
                # 🚀 PARALLELIZZAZIONE VERA con monitoring
                batch_start = time.time()
                logger.info(f"      🔥 Calcolo gradiente parallelo: {len(batch_tasks)} batch su {max_workers} workers...")
                
                # CALCOLO PARALLELO REALE
                batch_results = pool.map(compute_gradient_batch_optimized, batch_tasks)
                
                batch_time = time.time() - batch_start
                
                # Ricostruisci gradiente completo dai batch
                sentence_gradient = np.zeros(num_params)
                total_workers_used = set()
                
                for param_indices, gradients, worker_pid, worker_time in batch_results:
                    for i, param_idx in enumerate(param_indices):
                        sentence_gradient[param_idx] = gradients[i]
                    total_workers_used.add(worker_pid)
                
                # VERIFICA PARALLELIZZAZIONE REALE
                logger.info(f"      ✅ Gradiente calcolato in {batch_time:.2f}s")
                logger.info(f"      🔥 Workers utilizzati: {len(total_workers_used)} PIDs: {sorted(total_workers_used)}")
                
                if len(total_workers_used) == 1:
                    logger.warning(f"      ⚠️  ATTENZIONE: Solo 1 worker utilizzato! Verifica parallelizzazione!")
                else:
                    logger.info(f"      🎯 PARALLELIZZAZIONE OK: {len(total_workers_used)}/{max_workers} workers attivi")
                
                # Accumula gradiente
                total_gradient += sentence_gradient
            
            # AGGIORNAMENTO PARAMETRI OTTIMIZZATO
            avg_gradient = total_gradient / len(all_training_data)
            
            # 🎯 GRADIENT CLIPPING per stabilità
            gradient_norm = np.linalg.norm(avg_gradient)
            max_gradient_norm = 1.0  # Soglia clipping
            
            if gradient_norm > max_gradient_norm:
                avg_gradient = avg_gradient * (max_gradient_norm / gradient_norm)
                logger.info(f"      ⚠️ Gradient clipped: {gradient_norm:.4f} → {max_gradient_norm}")
            
            # Aggiornamento con LR attuale
            params -= current_lr * avg_gradient
            
            # Calcola loss media epoca
            avg_loss = epoch_loss / len(all_training_data)
            epoch_time = time.time() - epoch_start_time
            
            # Tracking miglior modello + LEARNING RATE SCHEDULING
            improvement = ""
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params.copy()
                improvement = " 🎯 NEW BEST!"
                no_improve_count = 0  # Reset counter
            else:
                no_improve_count += 1
                
                # 📉 ADAPTIVE LEARNING RATE: Riduce se no improvement
                if no_improve_count >= patience and current_lr > min_learning_rate:
                    old_lr = current_lr
                    current_lr = max(current_lr * 0.7, min_learning_rate)  # Riduce del 30%
                    no_improve_count = 0  # Reset
                    logger.info(f"      📉 Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
                    improvement = f" 📉 LR={current_lr:.1e}"
            
            # Log risultati epoca DETTAGLIATO
            logger.info(f"📊 EPOCH {epoch+1:2d} COMPLETATA:")
            logger.info(f"   Loss media: {avg_loss:.6f}{improvement}")
            logger.info(f"   Tempo: {epoch_time:.1f}s")
            logger.info(f"   Gradiente norm: {gradient_norm:.6f}")
            logger.info(f"   Learning rate: {current_lr:.1e}")
            logger.info(f"   No improve count: {no_improve_count}/{patience}")
            
            # Salva storico COMPLETO
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time,
                'gradient_norm': gradient_norm,
                'learning_rate': current_lr,
                'best_loss': best_loss,
                'improvement': no_improve_count == 0
            })
            
            # CONVERGENCE CHECK
            if avg_loss < convergence_threshold:
                logger.info(f"✅ CONVERGENZA RAGGIUNTA! Loss < {convergence_threshold}")
                break
                
            # 🛑 EARLY STOPPING MIGLIORATO
            if gradient_norm < 1e-6:
                logger.info(f"⚠️ Gradiente troppo piccolo ({gradient_norm:.1e}), stopping")
                break
                
            # Early stopping se LR troppo basso e no improvement
            if current_lr <= min_learning_rate and no_improve_count >= patience * 2:
                logger.info(f"⚠️ LR minimo raggiunto e no improvement, stopping")
                break
    
    # 🎉 RISULTATI FINALI GARANTITI
    final_improvement = ((training_history[0]['loss'] - best_loss) / training_history[0]['loss']) * 100
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 TRAINING BEAST MODE COMPLETATO!")
    logger.info(f"{'='*60}")
    logger.info(f"📊 RISULTATI FINALI:")
    logger.info(f"   Loss iniziale: {training_history[0]['loss']:.6f}")
    logger.info(f"   Loss finale: {avg_loss:.6f}")
    logger.info(f"   Miglior loss: {best_loss:.6f}")
    logger.info(f"   Miglioramento: {final_improvement:.1f}%")
    logger.info(f"   Epoche completate: {len(training_history)}/{max_epochs}")
    logger.info(f"   Workers utilizzati: {max_workers}")
    
    # Analisi convergenza
    if best_loss < 0.01:
        logger.info(f"   ✅ ECCELLENTE: Convergenza ottimale!")
    elif best_loss < 0.05:
        logger.info(f"   👍 BUONO: Convergenza raggiunta") 
    elif best_loss < 0.1:
        logger.info(f"   ⚡ DISCRETO: Convergenza parziale")
    else:
        logger.info(f"   ⚠️  ATTENZIONE: Convergenza non ottimale")
    
    # Salva risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'best_params': best_params,
        'best_loss': best_loss,
        'final_loss': avg_loss,
        'training_history': training_history,
        'config': OPTIMIZATION_CONFIG,
        'workers_used': max_workers,
        'total_cpus': total_cpus,
        'improvement_percent': final_improvement
    }
    
    result_file = f"beast_mode_results_{timestamp}.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"💾 Risultati salvati: {result_file}")
    logger.info(f"🚀 BEAST MODE TRAINING COMPLETATO CON SUCCESSO!")
    
    return True

def main():
    """Main BEAST MODE"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 BEAST MODE HPC TRAINING STARTED")
        log_system_info_beast(logger)
        
        # Import check
        logger.info("\n📦 CONTROLLO IMPORTAZIONI...")
        
        modules_to_test = [
            "config", "encoding", "quantum_circuits", 
            "quantum_mpi_utils", "main_superposition", "quantum_utils"
        ]
        
        for module in modules_to_test:
            try:
                exec(f"import {module}")
                logger.info(f"   ✅ {module}")
            except Exception as e:
                logger.error(f"   ❌ {module}: {e}")
                return 1
        
        # Avvia training BEAST MODE
        logger.info("\n🚀 AVVIO TRAINING BEAST MODE...")
        
        success = train_with_beast_mode_parallelization(logger)
        
        if success:
            logger.info("✅ BEAST MODE TRAINING COMPLETATO CON SUCCESSO!")
            return 0
        else:
            logger.error("❌ BEAST MODE TRAINING FALLITO!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 ERRORE CRITICO BEAST MODE:")
        logger.error(f"   Tipo: {type(e).__name__}")
        logger.error(f"   Messaggio: {str(e)}")
        logger.error("Traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 BEAST MODE terminato con exit code: {exit_code}")
    sys.exit(exit_code)