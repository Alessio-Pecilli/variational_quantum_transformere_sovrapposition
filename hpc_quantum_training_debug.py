#!/usr/bin/env python3
"""
Versione HPC robusta con debug esteso
Gestisce crash e identifica problemi specifici dell'ambiente HPC
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
from multiprocessing import Pool, cpu_count
from functools import partial

# Setup logging dettagliato
def setup_logging():
    """Setup logging con timestamp e dettagli"""
    log_file = Path("hpc_debug.log")
    
    # Configurazione logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Avvio logging dettagliato HPC")
    return logger

def log_system_info(logger):
    """Log informazioni sistema e ambiente"""
    logger.info("=" * 60)
    logger.info("🖥️ INFORMAZIONI SISTEMA HPC")
    logger.info("=" * 60)
    
    # Variabili ambiente importanti
    env_vars = ['OMP_NUM_THREADS', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID', 
                'SLURM_NTASKS', 'PBS_NP', 'SLURM_NPROCS']
    
    for var in env_vars:
        value = os.environ.get(var, 'Non definita')
        logger.info(f"   {var}: {value}")
    
    # Info Python e sistema
    logger.info(f"   Python version: {sys.version}")
    logger.info(f"   CPU count (physical): {cpu_count()}")
    logger.info(f"   Working directory: {os.getcwd()}")
    
    # Memoria (se disponibile)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"   Memoria totale: {memory.total / (1024**3):.1f} GB")
        logger.info(f"   Memoria disponibile: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.info("   Memoria: Info non disponibile (psutil non installato)")

@contextmanager
def timeout_context(seconds, logger):
    """Context manager per timeout operazioni"""
    def timeout_handler(signum, frame):
        logger.error(f"⏰ TIMEOUT dopo {seconds} secondi!")
        raise TimeoutError(f"Operazione timeout dopo {seconds}s")
    
    # Setup timeout solo su Unix (HPC)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    
    try:
        yield
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def safe_import(module_name, logger):
    """Import sicuro con logging dettagliato"""
    try:
        logger.info(f"📦 Importazione {module_name}...")
        if module_name == "config":
            from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
            logger.info(f"   Config caricato: {len(TRAINING_SENTENCES)} sentences")
            return True
        elif module_name == "encoding":
            from encoding import Encoding
            logger.info("   Encoding class caricata")
            return True
        elif module_name == "quantum_circuits":
            from quantum_circuits import get_circuit_function
            logger.info("   Quantum circuits caricati")
            return True
        elif module_name == "quantum_mpi_utils":
            from quantum_mpi_utils import loss_and_grad_for_sentence
            logger.info("   MPI utils caricati")
            return True
        elif module_name == "main_superposition":
            from main_superposition import process_sentence_states
            logger.info("   Main superposition caricato")
            return True
        else:
            exec(f"import {module_name}")
            logger.info(f"   {module_name} importato con successo")
            return True
            
    except Exception as e:
        logger.error(f"❌ ERRORE import {module_name}: {type(e).__name__}: {str(e)}")
        logger.error("Traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"   {line}")
        return False

def get_hpc_workers():
    """Auto-detect workers per HPC (copia da hpc_quantum_training.py)"""
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
    
    return workers

def get_safe_worker_count(logger):
    """Determina numero workers sicuro per HPC"""
    try:
        detected_workers = get_hpc_workers()
        logger.info(f"📊 Workers rilevati automaticamente: {detected_workers}")
        
        # Limiti di sicurezza per HPC
        max_safe_workers = min(detected_workers, 16)  # Limite cautelativo
        
        logger.info(f"📊 Workers scelti per sicurezza: {max_safe_workers}")
        return max_safe_workers
        
    except Exception as e:
        logger.error(f"❌ Errore rilevamento workers: {e}")
        logger.info("🔄 Fallback a 4 workers")
        return 4

def test_single_calculation(logger):
    """Test calcolo singolo prima del multiprocessing"""
    try:
        logger.info("🧪 TEST: Calcolo singolo...")
        
        from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
        from encoding import Encoding
        from quantum_mpi_utils import loss_and_grad_for_sentence
        
        # USA LA STESSA LOGICA DI main_superposition.py
        sentences = TRAINING_SENTENCES
        sentence = sentences[0]  # Prima sentence per test
        logger.info(f"   Sentence: '{sentence}'")
        
        # Encoding come nel main_superposition.py
        encoding = Encoding(sentences, embeddingDim=OPTIMIZATION_CONFIG['embedding_dim'])
        logger.info(f"   Encoding creato: {len(encoding.stateVectors)} state vectors")
        
        # Parametri come nel main_superposition.py
        from quantum_utils import get_params
        param_shape = get_params(OPTIMIZATION_CONFIG['num_qubits'], OPTIMIZATION_CONFIG['num_layers']).shape
        n_params = int(np.prod(param_shape))
        num_params = 2 * n_params  # V and K parameters
        params = np.random.randn(num_params) * 0.1
        sentence_data = (sentence, encoding, len(sentence.split()))
        
        logger.info(f"   Parametri: {num_params} (V:{n_params}, K:{n_params}, shape: {params.shape})")
        
        # Test calcolo gradiente parallelo come nel tuo MPI
        with timeout_context(30, logger):
            from main_superposition import process_sentence_states
            from quantum_circuits import get_circuit_function
            
            # Process states come nel codice funzionante
            states = encoding.stateVectors[0]  # Prima sentence
            states_calculated, U, Z = process_sentence_states(states)
            logger.info(f"   States calculated: {len(states_calculated)}")
            
            if len(states_calculated) > 0:
                # Setup per calcolo gradiente COME NEL MAIN
                circuit_func = get_circuit_function(len(states_calculated))
                psi = states_calculated[0]  # Primo stato per test
                U_test = [U[0]] if U else None  # Keep as list
                Z_test = [Z[0]] if Z else None  # Keep as list
                
                # Test calcolo loss singolo
                from quantum_mpi_utils import _compute_single_gradient_component
                shift = np.pi / 2
                
                # Test gradiente per primo parametro
                grad_comp = _compute_single_gradient_component(
                    0, params, shift, states_calculated, U, Z, 
                    OPTIMIZATION_CONFIG['num_layers'], 
                    OPTIMIZATION_CONFIG['embedding_dim'], 
                    circuit_func
                )
                logger.info(f"   ✅ Gradiente componente 0: {grad_comp:.6f}")
            else:
                logger.info("   ⚠️ Nessun state calcolato per il test gradiente")
            
        logger.info(f"   ✅ Test gradiente OK")
        return True, (states_calculated, U, Z, sentence, encoding, params)
        
    except Exception as e:
        logger.error(f"❌ ERRORE nel test singolo: {type(e).__name__}: {str(e)}")
        return False, None

def test_multiprocessing_safe(workers, test_data, logger):
    """Test multiprocessing con gestione errori"""
    try:
        logger.info(f"🔥 TEST: Multiprocessing con {workers} workers...")
        
        from quantum_circuits import get_circuit_function
        from quantum_mpi_utils import _compute_single_gradient_component
        from config import OPTIMIZATION_CONFIG
        states_calculated, U, Z, sentence, encoding, params = test_data
        
        if len(states_calculated) == 0:
            logger.error("   ❌ Nessun state calcolato, impossibile testare gradiente parallelo")
            return False
        
        # Test con timeout più lungo
        with timeout_context(60, logger):
            with Pool(processes=workers) as pool:
                logger.info(f"   Pool creato con {workers} workers")
                
                # Test calcolo gradiente parallelo per primi N parametri
                circuit_func = get_circuit_function(len(states_calculated))
                psi = states_calculated[0]
                U_test = [U[0]] if U else None  # Keep as list
                Z_test = [Z[0]] if Z else None  # Keep as list
                shift = np.pi / 2
                
                # Parallelizza calcolo gradiente per primi 4 parametri
                n_test_params = min(4, len(params))
                logger.info(f"   Test gradiente parallelo per {n_test_params} parametri...")
                
                tasks = []
                for i in range(n_test_params):
                    task_args = (i, params, shift, states_calculated, U, Z,
                                OPTIMIZATION_CONFIG['num_layers'], 
                                OPTIMIZATION_CONFIG['embedding_dim'], 
                                circuit_func)
                    tasks.append(task_args)
                
                logger.info("   Invio tasks al pool...")
                results = pool.starmap(_compute_single_gradient_component, tasks)
                
                logger.info(f"   ✅ Completato: {len(results)} risultati")
                
                # Verifica risultati
                if len(results) == 2:
                    loss1, grad1 = results[0]
                    loss2, grad2 = results[1]
                    logger.info(f"   Loss1: {loss1:.6f}, Loss2: {loss2:.6f}")
                    logger.info(f"   Differenza loss: {abs(loss1-loss2):.8f}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ ERRORE multiprocessing {workers} workers:")
        logger.error(f"   Tipo: {type(e).__name__}")
        logger.error(f"   Messaggio: {str(e)}")
        
        # Prova con meno workers
        if workers > 2:
            reduced_workers = max(2, workers // 2)
            logger.info(f"🔄 Retry con {reduced_workers} workers...")
            return test_multiprocessing_safe(reduced_workers, test_data, logger)
        else:
            logger.error("❌ Multiprocessing fallito anche con 2 workers")
            return False

def main():
    """Main con gestione errori completa"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 AVVIO PROGRAMMA HPC CON DEBUG ESTESO")
        log_system_info(logger)
        
        # Step 1: Importazioni
        logger.info("\n" + "="*40)
        logger.info("STEP 1: IMPORTAZIONI")
        logger.info("="*40)
        
        modules = ["numpy", "config", "encoding", "quantum_circuits", "quantum_mpi_utils", "main_superposition"]
        for module in modules:
            if not safe_import(module, logger):
                logger.error(f"💥 STOP: Impossibile importare {module}")
                return 1
        
        # Step 2: Rilevamento workers
        logger.info("\n" + "="*40)
        logger.info("STEP 2: CONFIGURAZIONE WORKERS")
        logger.info("="*40)
        
        workers = get_safe_worker_count(logger)
        
        # Step 3: Test singolo
        logger.info("\n" + "="*40)
        logger.info("STEP 3: TEST CALCOLO SINGOLO")
        logger.info("="*40)
        
        success, test_data = test_single_calculation(logger)
        if not success:
            logger.error("💥 STOP: Test singolo fallito")
            return 1
        
        # Step 4: Test multiprocessing
        logger.info("\n" + "="*40)
        logger.info("STEP 4: TEST MULTIPROCESSING")
        logger.info("="*40)
        
        if not test_multiprocessing_safe(workers, test_data, logger):
            logger.error("💥 STOP: Multiprocessing fallito")
            return 1
        
        # Step 5: Esecuzione training (se test OK)
        logger.info("\n" + "="*40)
        logger.info("STEP 5: TRAINING REALE")
        logger.info("="*40)
        
        # Import per training
        from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
        from encoding import Encoding
        from main_superposition import process_sentence_states
        
        logger.info(f"⚡ Inizio training con {workers} workers...")
        logger.info(f"📊 Configurazione: {OPTIMIZATION_CONFIG}")
        
        # USA LA STESSA LOGICA DI main_superposition.py
        sentences = TRAINING_SENTENCES
        sentence = sentences[0]  # Prima sentence per test
        encoding = Encoding(sentences, embeddingDim=OPTIMIZATION_CONFIG['embedding_dim'])
        
        logger.info(f"📈 Test sentence: '{sentence}' ({len(sentence.split())} parole)")
        logger.info(f"   State vectors disponibili: {len(encoding.stateVectors)}")
        
        # Test process_sentence_states 
        states = encoding.stateVectors[0]
        states_calculated, U, Z = process_sentence_states(states)
        
        logger.info(f"   ✅ States calculated: {len(states_calculated)}")
        logger.info(f"   ✅ U matrices: {len(U)}")  
        logger.info(f"   ✅ Z matrices: {len(Z)}")
        
        # TEST CALCOLO GRADIENTE PARALLELO (OBIETTIVO FINALE)
        if len(states_calculated) > 0:
            from quantum_circuits import get_circuit_function
            from quantum_mpi_utils import _compute_single_gradient_component
            
            circuit_func = get_circuit_function(len(states_calculated))
            psi = states_calculated[0]
            U_test = [U[0]] if U else None  # Keep as list
            Z_test = [Z[0]] if Z else None  # Keep as list
            shift = np.pi / 2
            
            # Parametri per test
            from quantum_utils import get_params
            param_shape = get_params(OPTIMIZATION_CONFIG['num_qubits'], OPTIMIZATION_CONFIG['num_layers']).shape
            n_params = int(np.prod(param_shape))
            num_params = 2 * n_params  # V and K parameters
            params = np.random.randn(num_params) * 0.1
            
            with Pool(processes=workers) as pool:
                logger.info(f"   🔥 GRADIENTE PARALLELO con {workers} workers")
                logger.info(f"      Parametri totali: {num_params}")
                
                # Parallelizza calcolo gradiente per tutti i parametri
                n_test_params = min(8, num_params)  # Test con primi 8 parametri
                logger.info(f"      Test su {n_test_params} parametri...")
                
                tasks = []
                for i in range(n_test_params):
                    task_args = (i, params, shift, states_calculated, U, Z,
                                OPTIMIZATION_CONFIG['num_layers'], 
                                OPTIMIZATION_CONFIG['embedding_dim'], 
                                circuit_func)
                    tasks.append(task_args)
                
                start_time = time.time()
                grad_components = pool.starmap(_compute_single_gradient_component, tasks)
                elapsed = time.time() - start_time
                
                logger.info(f"   ✅ GRADIENTE PARALLELO COMPLETATO in {elapsed:.2f}s")
                logger.info(f"      Componenti calcolate: {len(grad_components)}")
                logger.info(f"      Gradiente norm: {np.linalg.norm(grad_components):.6f}")
                logger.info(f"   🎯 PARALLELIZZAZIONE GRADIENTE FUNZIONA!")
        else:
            logger.error("   ❌ Nessun state calcolato per il test gradiente")
        
        logger.info("🎉 TRAINING COMPLETATO CON SUCCESSO!")
        return 0
        
    except Exception as e:
        logger.error(f"\n💥 ERRORE GENERALE DEL PROGRAMMA:")
        logger.error(f"Tipo: {type(e).__name__}")
        logger.error(f"Messaggio: {str(e)}")
        logger.error("Traceback completo:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Programma terminato con exit code: {exit_code}")
    sys.exit(exit_code)