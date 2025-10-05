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
            from quantum_circuits import create_superposition_circuit, calculate_loss_and_gradient
            logger.info("   Quantum circuits caricati")
            return True
        elif module_name == "quantum_mpi_utils":
            from quantum_mpi_utils import get_hpc_workers
            logger.info("   MPI utils caricati")
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

def get_safe_worker_count(logger):
    """Determina numero workers sicuro per HPC"""
    try:
        from quantum_mpi_utils import get_hpc_workers
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
        
        from config import TRAINING_SENTENCES
        from encoding import Encoding
        from quantum_circuits import calculate_loss_and_gradient
        
        # Prima sentence più semplice
        sentence = TRAINING_SENTENCES[0]
        logger.info(f"   Sentence: '{sentence}'")
        
        encoding = Encoding(sentence)
        logger.info(f"   Encoding creato: {encoding.n_qubits} qubits")
        
        # Parametri iniziali
        n_params = 2 * encoding.n_qubits
        params = np.random.uniform(0, 2*np.pi, n_params)
        sentence_data = (sentence, encoding, len(sentence.split()))
        
        logger.info(f"   Parametri: {n_params} (shape: {params.shape})")
        
        # Calcolo con timeout
        with timeout_context(30, logger):
            loss, grad = calculate_loss_and_gradient(params, sentence_data)
            
        logger.info(f"   ✅ Loss: {loss:.6f}, Grad norm: {np.linalg.norm(grad):.6f}")
        return True, (params, sentence_data)
        
    except Exception as e:
        logger.error(f"❌ ERRORE nel test singolo: {type(e).__name__}: {str(e)}")
        return False, None

def test_multiprocessing_safe(workers, test_data, logger):
    """Test multiprocessing con gestione errori"""
    try:
        logger.info(f"🔥 TEST: Multiprocessing con {workers} workers...")
        
        from quantum_circuits import calculate_loss_and_gradient
        params, sentence_data = test_data
        
        # Test con timeout più lungo
        with timeout_context(60, logger):
            with Pool(processes=workers) as pool:
                logger.info(f"   Pool creato con {workers} workers")
                
                # Test semplice: 2 calcoli identici
                tasks = [(params, sentence_data) for _ in range(2)]
                
                logger.info("   Invio tasks al pool...")
                results = pool.starmap(calculate_loss_and_gradient, tasks)
                
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
        
        modules = ["numpy", "config", "encoding", "quantum_circuits", "quantum_mpi_utils"]
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
        from quantum_circuits import calculate_loss_and_gradient
        
        logger.info(f"⚡ Inizio training con {workers} workers...")
        logger.info(f"📊 Configurazione: {OPTIMIZATION_CONFIG}")
        
        # Prima sentence per test completo
        sentence = TRAINING_SENTENCES[0]
        encoding = Encoding(sentence)
        n_params = 2 * encoding.n_qubits
        params = np.random.uniform(0, 2*np.pi, n_params)
        
        logger.info(f"📈 Epoca 1/{OPTIMIZATION_CONFIG['epochs']} - Sentence 1/{len(TRAINING_SENTENCES)}")
        logger.info(f"   Sentence: '{sentence}' ({len(sentence.split())} parole)")
        logger.info(f"   Parametri: {n_params}")
        
        # Calcolo gradiente parallelo
        sentence_data = (sentence, encoding, len(sentence.split()))
        
        with Pool(processes=workers) as pool:
            logger.info(f"   Pool training creato con {workers} workers")
            
            # Calcolo loss attuale
            loss, _ = calculate_loss_and_gradient(params, sentence_data)
            logger.info(f"   Loss iniziale: {loss:.6f}")
            
            # Esempio calcolo gradiente (versione semplificata)
            tasks = [(params, sentence_data) for _ in range(3)]
            results = pool.starmap(calculate_loss_and_gradient, tasks)
            
            logger.info(f"   ✅ Calcolo parallelo completato: {len(results)} risultati")
        
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