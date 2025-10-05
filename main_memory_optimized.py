#!/usr/bin/env python3
"""
🧠 QUANTUM TRAINING HPC - COMPLETE MEMORY OPTIMIZED VERSION
Versione completa ottimizzata per evitare OUT_OF_MEMORY su Leonardo
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
import traceback
import json

# Aggiungi path locale per import moduli
sys.path.append('.')

# Import moduli locali
try:
    import hamiltonian
    import encoding 
    import layer
except ImportError as e:
    print(f"❌ ERRORE import moduli locali: {e}")
    sys.exit(1)

# Import funzioni ottimizzate per memoria
from hpc_quantum_training_MEMORY_OPTIMIZED import (
    setup_hpc_logger,
    log_system_info_memory_safe, 
    get_hpc_workers_memory_safe,
    compute_gradient_parallel_memory_safe
)

def main_training_memory_optimized():
    """Training principale ottimizzato per memoria"""
    
    # Setup logging
    logger = setup_hpc_logger()
    logger.info("🧠 QUANTUM TRAINING MEMORY OPTIMIZED - AVVIO")
    logger.info("=" * 80)
    
    # Log info sistema
    log_system_info_memory_safe(logger)
    
    # ==================== CONFIGURAZIONE ====================
    OPTIMIZATION_CONFIG = {
        'num_qubits': 4,
        'num_layers': 2,  # Ridotto per memoria
        'embedding_dim': 4,
        'vocab_size': 50,
        'max_epochs': 20,  # Ridotto per test
        'learning_rate': 0.01,
        'min_learning_rate': 0.0001,
        'patience': 3,
        'convergence_threshold': 1e-6
    }
    
    logger.info("⚙️ CONFIGURAZIONE TRAINING:")
    for key, value in OPTIMIZATION_CONFIG.items():
        logger.info(f"   {key}: {value}")
    
    # ==================== DATI TRAINING ====================
    logger.info("\n📚 PREPARAZIONE DATI...")
    
    # Dataset semplificato per test memoria
    training_data = [
        "Il gatto mangia", 
        "Il cane corre",
        "Il sole splende",
        "La luna brilla",
        "Il vento soffia",
        "L'acqua scorre"
    ]
    
    logger.info(f"   Dataset: {len(training_data)} frasi")
    
    # Encoding semplificato per ridurre memoria
    vocab = {"<PAD>": 0}
    for sentence in training_data:
        for word in sentence.split():
            if word.lower() not in vocab:
                vocab[word.lower()] = len(vocab)
    
    logger.info(f"   Vocabolario: {len(vocab)} token")
    
    # Converti in sequenze numeriche
    def encode_sentence(sentence, max_len=4):  # Ridotto max_len per memoria
        tokens = sentence.split()[:max_len]
        encoded = [vocab.get(token.lower(), 0) for token in tokens]
        # Padding
        while len(encoded) < max_len:
            encoded.append(0)
        return encoded
    
    encoded_data = [encode_sentence(sent) for sent in training_data]
    logger.info(f"   Sequenze codificate: {len(encoded_data)} x {len(encoded_data[0])}")
    
    # ==================== INIZIALIZZAZIONE PARAMETRI ====================
    logger.info("\n🎯 INIZIALIZZAZIONE PARAMETRI...")
    
    # Calcola numero parametri RIDOTTO per memoria
    params_per_layer = OPTIMIZATION_CONFIG['num_qubits'] * 3  # 3 rotazioni per qubit
    total_params = params_per_layer * OPTIMIZATION_CONFIG['num_layers']
    
    logger.info(f"   Parametri per layer: {params_per_layer}")
    logger.info(f"   Parametri totali: {total_params}")
    
    # Inizializzazione Xavier RIDOTTA
    xavier_std = np.sqrt(2.0 / (OPTIMIZATION_CONFIG['num_qubits'] + OPTIMIZATION_CONFIG['embedding_dim']))
    params = np.random.normal(0, xavier_std, total_params)
    
    logger.info(f"   Xavier std: {xavier_std:.6f}")
    logger.info(f"   Params shape: {params.shape}")
    logger.info(f"   Params norm iniziale: {np.linalg.norm(params):.6f}")
    
    # ==================== SETUP QUANTUM ====================
    logger.info("\n🔬 SETUP CIRCUITI QUANTICI...")
    
    try:
        # Matrici unitarie per encoding (ridotte)
        U = encoding.create_encoding_matrix(OPTIMIZATION_CONFIG['vocab_size'], 
                                          OPTIMIZATION_CONFIG['embedding_dim'])
        Z = encoding.create_position_encoding(max_length=4,  # Ridotto
                                            embedding_dim=OPTIMIZATION_CONFIG['embedding_dim'])
        
        logger.info(f"   Matrice encoding U: {U.shape}")
        logger.info(f"   Matrice posizione Z: {Z.shape}")
        
    except Exception as e:
        logger.error(f"❌ Errore setup quantum: {e}")
        return False
    
    # ==================== TRAINING LOOP ====================
    logger.info("\n🚀 INIZIO TRAINING LOOP...")
    logger.info("=" * 80)
    
    # Variabili training
    best_loss = float('inf')
    best_params = params.copy()
    training_history = []
    no_improve_count = 0
    
    # Learning rate parameters
    current_lr = OPTIMIZATION_CONFIG['learning_rate']
    min_lr = OPTIMIZATION_CONFIG['min_learning_rate']
    patience = OPTIMIZATION_CONFIG['patience']
    
    # Info workers una volta
    safe_workers, source, total_cpus, memory_strategy = get_hpc_workers_memory_safe()
    logger.info(f"🧠 Training con {safe_workers} workers memory-safe (di {total_cpus} totali)")
    
    try:
        for epoch in range(OPTIMIZATION_CONFIG['max_epochs']):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            total_gradient = np.zeros_like(params)
            
            logger.info(f"\n📈 EPOCH {epoch+1}/{OPTIMIZATION_CONFIG['max_epochs']}")
            logger.info(f"   Current LR: {current_lr:.1e}")
            
            # Process ogni frase del dataset
            for sentence_idx, sentence_data in enumerate(encoded_data):
                sentence_start = time.time()
                
                logger.info(f"   Frase {sentence_idx+1}/{len(encoded_data)}: {training_data[sentence_idx]}")
                
                # FORWARD PASS con gestione memoria
                try:
                    # Calcolo stati quantici (memory intensive)
                    states_calc = []
                    for token_seq in [sentence_data]:  # Processa una alla volta
                        state = hamiltonian.quantum_transformer_circuit(
                            params, U, Z, 
                            OPTIMIZATION_CONFIG['num_layers'],
                            OPTIMIZATION_CONFIG['embedding_dim']
                        )
                        states_calc.append(state)
                    
                    # Calcolo loss
                    loss = np.mean([np.real(np.sum(state)) for state in states_calc])
                    epoch_loss += loss
                    
                    logger.info(f"      Loss: {loss:.6f}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Errore forward pass: {e}")
                    continue
                
                # BACKWARD PASS - Gradient computation MEMORY-OPTIMIZED
                try:
                    gradient_start = time.time()
                    
                    # USA LA VERSIONE OTTIMIZZATA PER MEMORIA
                    sentence_gradient = compute_gradient_parallel_memory_safe(
                        params, states_calc, U, Z,
                        OPTIMIZATION_CONFIG['num_layers'],
                        OPTIMIZATION_CONFIG['embedding_dim']
                    )
                    
                    gradient_time = time.time() - gradient_start
                    logger.info(f"      Gradient computed in {gradient_time:.2f}s [MEMORY-OPT]")
                    
                    total_gradient += sentence_gradient
                    
                except Exception as e:
                    logger.error(f"      ❌ Errore gradient computation: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                sentence_time = time.time() - sentence_start
                logger.info(f"      Tempo frase: {sentence_time:.2f}s")
            
            # AGGIORNAMENTO PARAMETRI
            if np.any(total_gradient != 0):
                avg_gradient = total_gradient / len(encoded_data)
                
                # Gradient clipping
                gradient_norm = np.linalg.norm(avg_gradient)
                max_gradient_norm = 1.0
                
                if gradient_norm > max_gradient_norm:
                    avg_gradient = avg_gradient * (max_gradient_norm / gradient_norm)
                    logger.info(f"   ⚠️ Gradient clipped: {gradient_norm:.4f} → {max_gradient_norm}")
                
                # Update parameters
                params -= current_lr * avg_gradient
                
                logger.info(f"   Gradient norm: {gradient_norm:.6f}")
            else:
                logger.warning("   ⚠️ Gradient nullo - parametri non aggiornati")
                gradient_norm = 0.0
            
            # EPOCH SUMMARY
            avg_loss = epoch_loss / len(encoded_data)
            epoch_time = time.time() - epoch_start_time
            
            # Best model tracking
            improvement = ""
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params.copy()
                improvement = " 🎯 NEW BEST!"
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # Learning rate decay
                if no_improve_count >= patience and current_lr > min_lr:
                    old_lr = current_lr
                    current_lr = max(current_lr * 0.7, min_lr)
                    no_improve_count = 0
                    logger.info(f"   📉 Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
                    improvement = f" 📉 LR={current_lr:.1e}"
            
            # Log epoch results
            logger.info(f"📊 EPOCH {epoch+1} COMPLETATA:")
            logger.info(f"   Loss media: {avg_loss:.6f}{improvement}")
            logger.info(f"   Tempo: {epoch_time:.1f}s")
            logger.info(f"   Gradient norm: {gradient_norm:.6f}")
            logger.info(f"   Learning rate: {current_lr:.1e}")
            logger.info(f"   No improve: {no_improve_count}/{patience}")
            
            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time,
                'gradient_norm': gradient_norm,
                'learning_rate': current_lr,
                'best_loss': best_loss,
                'improvement': no_improve_count == 0
            })
            
            # Convergence check
            if no_improve_count >= patience * 2:
                logger.info(f"   🔄 EARLY STOPPING: No improvement per {no_improve_count} epochte")
                break
                
            if avg_loss < OPTIMIZATION_CONFIG['convergence_threshold']:
                logger.info(f"   ✅ CONVERGENZA: Loss {avg_loss:.8f} < {OPTIMIZATION_CONFIG['convergence_threshold']}")
                break
    
    except Exception as e:
        logger.error(f"❌ Errore durante training: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # ==================== RISULTATI FINALI ====================
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAINING COMPLETATO!")
    logger.info("=" * 80)
    
    logger.info(f"Best loss: {best_loss:.8f}")
    logger.info(f"Epochte totali: {len(training_history)}")
    logger.info(f"Parametri finali norm: {np.linalg.norm(best_params):.6f}")
    
    # Salva risultati
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    results_file = f'memory_opt_results_{job_id}.json'
    
    results = {
        'config': OPTIMIZATION_CONFIG,
        'final_loss': float(best_loss),
        'epochs_completed': len(training_history),
        'training_history': training_history,
        'final_params_norm': float(np.linalg.norm(best_params)),
        'workers_used': safe_workers,
        'memory_strategy': memory_strategy,
        'completion_time': datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📁 Risultati salvati in: {results_file}")
    logger.info("✅ TRAINING MEMORY-OPTIMIZED COMPLETATO CON SUCCESSO!")
    
    return True

if __name__ == "__main__":
    success = main_training_memory_optimized()
    if success:
        print("✅ Training completato con successo")
        sys.exit(0)
    else:
        print("❌ Training fallito")
        sys.exit(1)