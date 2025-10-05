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
import numpy as np
from multiprocessing import cpu_count
from datetime import datetime
from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 🔧 SETUP LOGGING
# ============================================================

def setup_logging():
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


# ============================================================
# 🧠 CONFIGURAZIONE SISTEMA HPC
# ============================================================

def get_hpc_workers_max():
    """Determina il numero massimo di worker HPC"""
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

    max_workers = max(1, workers - 1) if workers > 2 else workers
    return max_workers, source, workers


# ============================================================
# 🚀 TRAINING PARALLELIZZATO
# ============================================================

def train_with_beast_mode_parallelization(logger):
    """Esegue l'intera pipeline HPC con ottimizzazione parallela"""

    from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
    from encoding import Encoding
    from main_superposition import process_sentence_states
    from optimization import optimize_parameters_parallel
    from quantum_utils import get_params

    max_workers, source, total_cpus = get_hpc_workers_max()
    logger.info(f"🚀 AVVIO TRAINING BEAST MODE")
    logger.info(f"   Workers: {max_workers}/{total_cpus} (fonte: {source})")

    # ------------------------------------------------------------
    # ENCODING DELLE FRASI
    # ------------------------------------------------------------
    sentences = TRAINING_SENTENCES
    encoding = Encoding(sentences, embeddingDim=OPTIMIZATION_CONFIG['embedding_dim'])

    logger.info(f"📊 ENCODING COMPLETATO - {len(sentences)} frasi totali")

    # ------------------------------------------------------------
    # PARAMETRI INIZIALI
    # ------------------------------------------------------------
    param_shape = get_params(
        OPTIMIZATION_CONFIG['num_qubits'],
        OPTIMIZATION_CONFIG['num_layers']
    ).shape

    n_params = int(np.prod(param_shape))
    num_params = 2 * n_params

    xavier_std = np.sqrt(
        2.0 / (OPTIMIZATION_CONFIG['num_qubits'] + OPTIMIZATION_CONFIG['embedding_dim'])
    )
    params = np.random.randn(num_params) * xavier_std

    logger.info(f"🎯 Xavier initialization completata (std={xavier_std:.4f})")

    # ------------------------------------------------------------
    # LOOP SULLE FRASI
    # ------------------------------------------------------------
    best_params = None

    for sentence_idx, sentence in enumerate(sentences):
        print(f"\n{'='*60}")
        print(f"Processing sentence {sentence_idx + 1}/{len(sentences)}: '{sentence}'")
        print(f"{'='*60}")

        # Stati quantistici
        states = encoding.stateVectors[sentence_idx]
        states_calculated, U, Z = process_sentence_states(states)

        # Se abbiamo stati validi, parte l’ottimizzazione
        if len(states_calculated) > 0:
            print(f"🚀 Avvio ottimizzazione parallela per la frase {sentence_idx + 1}")

            best_params = optimize_parameters_parallel(
                OPTIMIZATION_CONFIG['max_hours'],
                OPTIMIZATION_CONFIG['num_iterations'],
                OPTIMIZATION_CONFIG['num_layers'],
                states_calculated,
                U,
                Z,
                best_params,
                dim=OPTIMIZATION_CONFIG['embedding_dim'],
                opt_maxiter=OPTIMIZATION_CONFIG['opt_maxiter'],
                opt_maxfev=OPTIMIZATION_CONFIG['opt_maxfev']
            )
        else:
            print("⚠️ Nessuno stato valido, salto ottimizzazione per questa frase.")

    # ------------------------------------------------------------
    # SALVATAGGIO RISULTATI
    # ------------------------------------------------------------
    if best_params is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"beast_mode_params_{timestamp}.npy"
        np.save(output_file, best_params)
        logger.info(f"💾 Parametri ottimizzati salvati in {output_file}")
    else:
        logger.error("❌ Nessun parametro ottimizzato disponibile")


# ============================================================
# 🧩 MAIN
# ============================================================

def main():
    logger = setup_logging()

    try:
        logger.info("🚀 BEAST MODE HPC TRAINING STARTED")

        # Info sistema
        max_workers, source, total_cpus = get_hpc_workers_max()
        logger.info(f"💪 Workers: {max_workers}/{total_cpus} ({source})")

        # Avvio training
        train_with_beast_mode_parallelization(logger)

        logger.info("✅ TRAINING BEAST MODE COMPLETATO CON SUCCESSO")
        return 0

    except Exception as e:
        logger.error("💥 ERRORE CRITICO DURANTE IL TRAINING")
        logger.error(f"Tipo: {type(e).__name__}")
        logger.error(f"Messaggio: {str(e)}")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    exit_code = main()
    print(f"\n🏁 BEAST MODE terminato con exit code: {exit_code}")
    sys.exit(exit_code)
    