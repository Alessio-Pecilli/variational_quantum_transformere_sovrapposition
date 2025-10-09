#!/usr/bin/env python3
"""
🚀 BEAST MODE HPC TRAINING - VERA PARALLELIZZAZIONE AL 100%
Risolve tutti i problemi: single worker, batch overhead, loss mancante
"""
from scipy.optimize import minimize

import traceback
from functools import partial
import sys
from multiprocessing import Pool
import time
from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder
from quantum_circuits import get_circuit_function
import logging
from pathlib import Path
import os
import numpy as np
from multiprocessing import cpu_count
from datetime import datetime
from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ============================================================
# 🔧 SETUP LOGGING
# ============================================================

def setup_logging():
    log_file = Path("hpc_beast_mode.log")

    # Forza UTF-8 su tutti i flussi
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("🚀 BEAST MODE HPC LOGGING ATTIVO (UTF-8)")
    return logger

# ============================================================
# 🧠 CONFIGURAZIONE SISTEMA HPC
# ============================================================

def cobyla_objective(y, low, high, sentences, encoding, OPTIMIZATION_CONFIG, max_workers, process_single_sentence):
    # 1️⃣ y sono i parametri scalati in [-1, 1]
    # li riportiamo nello spazio "naturale" [-pi, pi]
    params_native = descale_from_unit(y, low, high)

    # 2️⃣ calcoliamo la loss media su tutte le frasi (in parallelo)
    loss = process_all_sentences(
        params_native,
        sentences,
        encoding,
        OPTIMIZATION_CONFIG,
        max_workers,
        process_single_sentence
    )
    print(f"[COBYLA] Loss attuale: {loss:.6f}")


    # 3️⃣ COBYLA cerca di minimizzare il valore che ritorniamo
    return loss

def process_all_sentences(params_native, sentences, encoding, OPTIMIZATION_CONFIG, max_workers, process_single_sentence):
    """
    Calcola la loss media su tutte le frasi, in parallelo.
    Ogni processo esegue process_single_sentence su una frase.
    
    params_native : np.ndarray
        Parametri correnti (nel dominio [-pi, pi]) proposti da COBYLA.
    sentences : list
        Lista di frasi da ottimizzare.
    encoding : Encoding
        Oggetto di codifica condiviso.
    OPTIMIZATION_CONFIG : dict
        Config globale del training.
    max_workers : int
        Numero massimo di processi paralleli.
    process_single_sentence : callable
        Funzione che elabora una singola frase e restituisce la loss.
    """

    # crea i task per ogni frase
    tasks = [
        (idx, sentence, encoding, params_native, OPTIMIZATION_CONFIG)
        for idx, sentence in enumerate(sentences)
    ]

    # lancia i worker in parallelo (uno per frase)
    with Pool(max_workers) as pool:
        losses = pool.map(process_single_sentence, tasks)

    # ritorna la loss media come float
    return float(np.mean(losses))

def get_param_bounds(n_params):
    low  = -np.pi * np.ones(n_params)
    high =  np.pi * np.ones(n_params)
    return low, high

def scale_to_unit(x, low, high):
    return 2.0 * (x - low) / (high - low) - 1.0  # x->[ -1, 1 ]

def descale_from_unit(y, low, high):
    return low + (y + 1.0) * 0.5 * (high - low)  # [ -1,1 ]->x

# --- costruisci vincoli box per COBYLA (g(y)>=0) ---
def box_constraints_for_unit(y_dim):
    cons = []
    for i in range(y_dim):
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 - y[i]})  # y[i] <= 1
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 + y[i]})  # y[i] >= -1
    return cons
# === FUNZIONE OBIETTIVO PICKLABLE (chiamata da COBYLA) ===

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

from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder
from optimization import get_params
from main_serial import process_sentence_states

def process_single_sentence(args):
    import os
    import numpy as np
    from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder
    from main_serial import process_sentence_states

    sentence_idx, sentence, encoding, params, OPTIMIZATION_CONFIG = args

    print(f"[Worker {os.getpid()}] INIZIO elaborazione frase {sentence_idx+1}")

    # Ottieni gli stati quantistici della frase
    states = encoding.stateVectors[sentence_idx]
    states_calculated, U, Z = process_sentence_states(states)

    sentence_length = len(sentence) - 1

    # 🔹 Split dei parametri globali (V e K)
    half = len(params) // 2
    params_v = params[:half]
    params_k = params[half:]

    # 🔹 Reshape per compatibilità con AnsatzBuilder
    num_layers = OPTIMIZATION_CONFIG['num_layers']
    params_shape = get_params(
        OPTIMIZATION_CONFIG['num_qubits'],
        num_layers
    ).shape
    params_v = np.reshape(params_v, params_shape)
    params_k = np.reshape(params_k, params_shape)

    # 🔹 Costruzione e calcolo loss
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=OPTIMIZATION_CONFIG['embedding_dim'],
        sentence_length=sentence_length
    )

    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )

    print(f"[Worker {os.getpid()}] ✅ Loss frase {sentence_idx+1}: {loss:.6f}")
    return loss


def train_with_beast_mode_parallelization(logger):
    """Esegue l'intera pipeline HPC con ottimizzazione parallela (1 processo = 1 frase)."""

    from multiprocessing import Pool
    from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
    from encoding import Encoding
    from main_serial import process_sentence_states
    from optimization import optimize_parameters_parallel
    from quantum_utils import get_params
    import numpy as np
    from datetime import datetime

    # ------------------------------------------------------------
    # SETUP BASE
    # ------------------------------------------------------------
    max_workers, source, total_cpus = get_hpc_workers_max()
    # Limita alla parallelizzazione minima per evitare sovraccarico
    max_workers = min(max_workers, 2)  # Massimo 2 worker
    logger.info(f"🚀 AVVIO TRAINING BEAST MODE")
    logger.info(f"   Workers: {max_workers}/{total_cpus} (fonte: {source})")

    # ------------------------------------------------------------
    # ENCODING DELLE FRASI
    # ------------------------------------------------------------
    sentences = TRAINING_SENTENCES[:max_workers]
    encoding = Encoding(sentences, embeddingDim=OPTIMIZATION_CONFIG['embedding_dim'])
    logger.info(f"📊 ENCODING COMPLETATO - {len(sentences)} frasi totali")

    # ------------------------------------------------------------
    # PARAMETRI INIZIALI (θ condivisi)
    # ------------------------------------------------------------
    print(f"[DEBUG TRAIN] OPTIMIZATION_CONFIG['num_qubits'] =", OPTIMIZATION_CONFIG['num_qubits'])
    param_shape = get_params(
        OPTIMIZATION_CONFIG['num_qubits'],
        OPTIMIZATION_CONFIG['num_layers'],
    ).shape

    n_params = int(np.prod(param_shape))
    num_params = 2 * n_params

    xavier_std = np.sqrt(
        2.0 / (OPTIMIZATION_CONFIG['num_qubits'] + OPTIMIZATION_CONFIG['embedding_dim'])
    )
    params = np.random.randn(num_params) * xavier_std

    logger.info(f"🎯 Xavier initialization completata (std={xavier_std:.4f})")
    logger.info(f"   Totale parametri: {num_params}")


    epochs = OPTIMIZATION_CONFIG.get('epochs', 5)  # numero di epoche globali
    logger.info(f"🧠 INIZIO TRAINING per {epochs} epoche")

    # === PARAMS / BOUNDS ===
    low, high = get_param_bounds(num_params)

    # === OBJ e constraints ===
    objective = partial(
        cobyla_objective,
        low=low,
        high=high,
        sentences=sentences,
        encoding=encoding,
        OPTIMIZATION_CONFIG=OPTIMIZATION_CONFIG,
        max_workers=max_workers,
        process_single_sentence=process_single_sentence,
    )
    constraints = box_constraints_for_unit(num_params)

    # === CONFIG ===
    RANDOM_RESTARTS = OPTIMIZATION_CONFIG.get('restarts', 6)
    MAXITER_PER_RUN = OPTIMIZATION_CONFIG.get('maxiter', 500)
    RHO_BEG = OPTIMIZATION_CONFIG.get('rhobeg', 0.25)
    TOL = OPTIMIZATION_CONFIG.get('tol', 1e-5)
    rng = np.random.default_rng(OPTIMIZATION_CONFIG.get('seed', 42))

    best_f, best_x = np.inf, None

    for r in range(RANDOM_RESTARTS):
        y0 = rng.uniform(-1.0, 1.0, size=num_params)

        res = minimize(
            fun=objective,
            x0=y0,
            method="COBYLA",
            constraints=constraints,
            options={"maxiter": MAXITER_PER_RUN, "rhobeg": RHO_BEG, "tol": TOL},
        )

        x_star = descale_from_unit(res.x, low, high)
        f_star = process_all_sentences(x_star, sentences, encoding, OPTIMIZATION_CONFIG, max_workers, process_single_sentence)

        logger.info(f"[COBYLA run {r+1}/{RANDOM_RESTARTS}] f*={f_star:.6f}, iters={res.nit}, success={res.success}")
        if f_star < best_f:
            best_f, best_x = f_star, x_star

    params = best_x
    logger.info(f"✅ COBYLA completato — best loss={best_f:.6f}")


    # ------------------------------------------------------------
    # SALVATAGGIO RISULTATI FINALI
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"beast_mode_final_params_{timestamp}.npy"
    np.save(output_file, params)
    logger.info(f"💾 Parametri finali salvati in {output_file}")
    logger.info("✅ TRAINING BEAST MODE COMPLETATO")

# ============================================================
# 🧩 MAIN
# ============================================================

def main():
    
    import logging, re
    logger = setup_logging()
# -- crea/ottieni logger principale --
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # -- formatter e handler standard --
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(console)

    # -- Filtro per rimuovere le emoji dai messaggi --
    class EmojiFilter(logging.Filter):
        def filter(self, record):
            record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg))
            return True

    for handler in logger.handlers:
        handler.addFilter(EmojiFilter())


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
    