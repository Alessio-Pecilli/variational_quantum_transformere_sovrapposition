#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpi4py import MPI
"""
try:
    from mpi4py import MPI
except Exception:
    import numpy as np

    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def Bcast(self, *a, **k): pass
        def Allreduce(self, *a, **k): pass
        def gather(self, x, root=0): return [x]
        def bcast(self, x, root=0): return x

    class FakeMPI:
        COMM_WORLD = FakeComm()
        # tipi fittizi per compatibilità
        INT = np.int32
        DOUBLE = np.float64
        SUM = None

    MPI = FakeMPI()
"""

import numpy as np
from pathlib import Path
from functools import partial
from datetime import datetime
import logging
import sys
import os
import traceback
from generalized_quantum_circuits import process_sentence_states
# Import del tuo progetto
from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
from encoding import Encoding
from optimization import get_params
from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder

from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Logging minimale: dettagliato su rank 0, ridotto sugli altri
# ---------------------------------------------------------------------
def setup_logging(rank: int):
    logger = logging.getLogger("mpi_beast")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if rank == 0:
            fh = logging.FileHandler("hpc_beast_mode_mpi.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

# ---------------------------------------------------------------------
# Utility: bounds e scaling
# ---------------------------------------------------------------------
def get_param_bounds(n_params):
    low  = -np.pi * np.ones(n_params, dtype=np.float64)
    high =  np.pi * np.ones(n_params, dtype=np.float64)
    return low, high

def scale_to_unit(x, low, high):
    return 2.0 * (x - low) / (high - low) - 1.0

def descale_from_unit(y, low, high):
    return low + (y + 1.0) * 0.5 * (high - low)

def box_constraints_for_unit(y_dim):
    cons = []
    for i in range(y_dim):
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 - y[i]})  # y[i] <= 1
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 + y[i]})  # y[i] >= -1
    return cons

# ---------------------------------------------------------------------
# Loss su UNA frase, dato theta nativo
# ---------------------------------------------------------------------
def loss_for_sentence(sentence_idx, sentence, encoding, params_native, cfg):
    # states, U, Z per la frase
    states = encoding.stateVectors[sentence_idx]
    states_calculated, U, Z = process_sentence_states(states)

    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape

    half = params_native.size // 2
    params_v = np.reshape(params_native[:half], params_shape)
    params_k = np.reshape(params_native[half:], params_shape)
    print("frase:", sentence , "è lunga", len(sentence.split()), "parole")
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=cfg['embedding_dim'],
        sentence_length=len(sentence.split())
    )

    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )
    return float(loss)

# ---------------------------------------------------------------------
# Servizio di valutazione distribuita:
# - rank 0 lancia COBYLA, ad ogni chiamata:
#   * Bcast("EVAL"), Bcast(y_scaled)
#   * ogni rank calcola sum_loss_local e count_local
#   * Allreduce su somme e conteggi
#   * rank 0 ritorna media globale
# - al termine Bcast("STOP")
# ---------------------------------------------------------------------
def distributed_objective_factory(comm, rank, size, sentences_split, encoding, low, high, cfg, logger):
    tagbuf = np.array([0], dtype=np.int32)  # 1=EVAL, 2=STOP

    # Pre-alloc per riduzioni
    send_buf = np.zeros(2, dtype=np.float64)  # [sum_loss_local, count_local]
    recv_buf = np.zeros(2, dtype=np.float64)

    # Lista locale di (global_idx, sentence)
    my_items = sentences_split[rank]

    def objective(y_scaled):
        # Solo rank 0 chiama questa funzione
        assert rank == 0

        # Broadcast tag=EVAL
        tagbuf[0] = 1
        comm.Bcast([tagbuf, MPI.INT], root=0)

        # Broadcast parametri scalati
        y_scaled = np.ascontiguousarray(y_scaled, dtype=np.float64)

        y_dim = np.array([y_scaled.size], dtype=np.int32)
        comm.Bcast([y_dim, MPI.INT], root=0)
        comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

        # Ogni rank: descala e calcola sum loss locale
        params_native = descale_from_unit(y_scaled, low, high)

        local_sum = 0.0
        local_cnt = 0.0
        for (global_idx, sentence) in my_items:
            try:
                # Attenzione: encoding locale è allineato con my_items
                # my_items contiene indici e frasi; l'indice per encoding è relativo al rank
                # quindi usiamo la posizione nella split
                # Costruiamo la posizione relativa:
                # Trova posizione relativa in my_items
                # Più efficiente: manteniamo un mapping locale idx->pos
                pass
            except Exception:
                # Nel caso, contiamo NaN come 0, ma non incrementiamo count
                continue

        # Implementazione efficiente: prepariamo encoding come array allineato
        # quindi rifacciamo un ciclo semplice
        local_sum = 0.0
        local_cnt = 0.0
        for local_pos, (global_idx, sentence) in enumerate(my_items):
            try:
                loss = loss_for_sentence(local_pos, sentence, encoding, params_native, cfg)
                if np.isfinite(loss):
                    local_sum += loss
                    local_cnt += 1.0
            except Exception as e:
                # Log solo su rank 0 per non inondare
                if rank == 0:
                    logger.warning(f"Errore loss su frase globale {global_idx}: {e}")

        # Allreduce su [sum, count]
        send_buf[0] = local_sum
        send_buf[1] = local_cnt
        comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)

        global_sum = recv_buf[0]
        global_cnt = recv_buf[1] if recv_buf[1] > 0 else 1.0
        global_mean = global_sum / global_cnt

        if rank == 0:
            logger.info(f"[COBYLA] Loss media corrente: {global_mean:.6f}  (aggiornata)")

        return float(global_mean)

    def stop_workers():
        if rank == 0:
            tagbuf[0] = 2
            comm.Bcast([tagbuf, MPI.INT], root=0)

    # Ritorniamo le due funzioni: objective per rank 0, e uno "worker_loop" per gli altri
    def worker_loop():
        # Rank != 0: rimane in ascolto dei broadcast
        local_items = my_items
        while True:
            comm.Bcast([tagbuf, MPI.INT], root=0)
            if tagbuf[0] == 2:
                break
            elif tagbuf[0] == 1:
                # EVAL: ricevi dimensione e vettore y_scaled
                y_dim = np.array([0], dtype=np.int32)
                comm.Bcast([y_dim, MPI.INT], root=0)
                y_scaled = np.empty(y_dim[0], dtype=np.float64)
                comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

                params_native = descale_from_unit(y_scaled, low, high)

                # Calcolo locale
                local_sum = 0.0
                local_cnt = 0.0
                for local_pos, (global_idx, sentence) in enumerate(local_items):
                    try:
                        loss = loss_for_sentence(local_pos, sentence, encoding, params_native, cfg)
                        if np.isfinite(loss):
                            local_sum += loss
                            local_cnt += 1.0
                    except Exception:
                        pass

                # Contribuisce alla riduzione
                send_buf[0] = local_sum
                send_buf[1] = local_cnt
                comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)
            else:
                # Tag sconosciuto, termina per sicurezza
                break

    return objective, stop_workers, worker_loop

# ---------------------------------------------------------------------
# Split frasi e preparazione encoding locale
# ---------------------------------------------------------------------
def make_splits_and_encodings(sentences, comm, rank, size, embedding_dim, logger):
    # Distribuisci indici globali equamente
    indices = np.arange(len(sentences), dtype=np.int64)
    chunks = np.array_split(indices, size)

    # Costruiamo la lista locale di (global_idx, sentence)
    local_ids = chunks[rank].tolist()
    local_items = [(int(gid), sentences[int(gid)]) for gid in local_ids]

    # L'encoding locale si costruisce con l'elenco di frasi locali
    local_sentences = [s for (_, s) in local_items]
    encoding_local = Encoding(local_sentences, embeddingDim=embedding_dim)

    if rank == 0:
        logger.info(f"Frasi totali: {len(sentences)}; size MPI: {size}")
    logger.info(f"[Rank {rank}] Frasi locali: {len(local_items)} -> {[s for _, s in local_items]}")

    return chunks, local_items, encoding_local

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(rank)
    cfg = OPTIMIZATION_CONFIG

    try:
        # Prepara split frasi e encoding locale
        sentences = TRAINING_SENTENCES
        chunks, my_items, encoding_local = make_splits_and_encodings(
            sentences, comm, rank, size, cfg['embedding_dim'], logger
        )

        # Parametrizzazione
        param_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        n_params_half = int(np.prod(param_shape))
        n_params = 2 * n_params_half

        low, high = get_param_bounds(n_params)

        # Inizializzazione su rank 0
        if rank == 0:
            rng = np.random.default_rng(cfg.get('seed', 42))
            y0 = rng.uniform(-1.0, 1.0, size=n_params)  # nello spazio [-1,1]
            logger.info(f"Parametri: tot={n_params}  shape_half={param_shape}")

        # Prepara objective distribuito
        # sentences_split è una lista di liste: per ogni rank, lista di (global_idx, sentence)
        sentences_split = [[] for _ in range(size)]
        # Broadcast strutture: costruiamo localmente la stessa struttura
        sentences_split[rank] = my_items
        # Raccogliamo su tutti (Allgather-like manuale)
        all_items = comm.gather(my_items, root=0)
        if rank == 0:
            sentences_split = all_items
        sentences_split = comm.bcast(sentences_split, root=0)

        objective, stop_workers, worker_loop = distributed_objective_factory(
            comm, rank, size, sentences_split, encoding_local, low, high, cfg, logger
        )

        # Rank non-zero: entra nel loop worker e resta in ascolto
        if rank != 0:
            worker_loop()
            return 0

        # Rank 0: ottimizzazione COBYLA con restarts
        # Rank 0: ottimizzazione COBYLA con restarts
        constraints = box_constraints_for_unit(n_params)
        RANDOM_RESTARTS = cfg.get('restarts', 3)

        # IMPORTANTE: COBYLA vuole almeno n_params+2 "fun eval" interne.
        # Qui alziamo maxiter per evitare il warning "Invalid MAXFUN".
        MAXITER_PER_RUN = max(cfg.get('maxiter', 300), n_params + 2)

        RHO_BEG = cfg.get('rhobeg', 0.25)
        TOL = cfg.get('tol', 1e-5)
        rng = np.random.default_rng(cfg.get('seed', 42))

        best_f, best_y = np.inf, None
        logger.info(
            f"Inizio ottimizzazione con COBYLA: restarts={RANDOM_RESTARTS}, "
            f"maxiter={MAXITER_PER_RUN} (>= n_params+2={n_params+2})"
        )

        for r in range(RANDOM_RESTARTS):
            if r == 0:
                y_start = y0
            else:
                y_start = rng.uniform(-1.0, 1.0, size=n_params)

            res = minimize(
                fun=objective,
                x0=y_start,  # <-- fix: NON usare y0 qui
                method="COBYLA",
                constraints=constraints,
                options={
                    "maxiter": MAXITER_PER_RUN,
                    "rhobeg": RHO_BEG,
                    "tol": TOL,
                    "disp": True,
                },
            )

            # Valuta loss media globale con i migliori y trovati
            f_star = objective(res.x)
            logger.info(f"[Run {r+1}/{RANDOM_RESTARTS}] f*={f_star:.6f}, iters={res.nit}, success={res.success}")

            if f_star < best_f:
                best_f = f_star
                best_y = res.x.copy()

        # Stop ai worker
        stop_workers()

        # Parametri finali in spazio nativo
        best_params_native = descale_from_unit(best_y, low, high)

        # Salvataggi
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f"theta_finali_native_{ts}.npy", best_params_native)
        with open(f"training_summary_{ts}.txt", "w", encoding="utf-8") as f:
            f.write(f"best_loss_mean={best_f:.8f}\n")
            f.write(f"n_params={n_params}\n")
            f.write(f"param_shape_half={param_shape}\n")
            f.write(f"restarts={RANDOM_RESTARTS}, maxiter={MAXITER_PER_RUN}, tol={TOL}\n")
            f.write(f"mpi_size={size}, sentences={len(sentences)}\n")

        logger.info(f"Ottimizzazione completata. Loss media finale: {best_f:.6f}")
        logger.info(f"Parametri salvati in theta_finali_native_{ts}.npy")

        return 0

    except Exception as e:
        if rank == 0:
            logger.error("Errore critico durante il training")
            logger.error(f"Tipo: {type(e).__name__}, Msg: {e}")
            for line in traceback.format_exc().splitlines():
                logger.error(line)
        # Prova a svegliare i worker per non bloccare
        try:
            tag = np.array([2], dtype=np.int32)
            MPI.COMM_WORLD.Bcast([tag, MPI.INT], root=0)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
