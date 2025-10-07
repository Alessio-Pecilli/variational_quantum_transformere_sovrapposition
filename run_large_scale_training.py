"""
Script Principale per Training Quantico su Larga Scala con Penn Treebank.
Esegue training completo con 700 frasi di training + 300 di test.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import psutil
import signal

# Aggiungi directory corrente al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrated_quantum_trainer import IntegratedQuantumTrainer
from config import OPTIMIZATION_CONFIG


def setup_signal_handlers(trainer):
    """Setup gestione segnali per interruzioni graceful."""
    
    def signal_handler(signum, frame):
        print(f"\n⚠️ Ricevuto segnale {signum}, fermando training gracefully...")
        if hasattr(trainer, 'stop_monitoring'):
            trainer.stop_monitoring.set()
        
        # Genera report parziale se possibile
        if hasattr(trainer, 'loss_history') and trainer.loss_history:
            try:
                print("📊 Generazione report parziale...")
                partial_report = trainer._generate_error_report("Training interrotto dall'utente")
                print(f"Report parziale salvato: {partial_report}")
            except Exception as e:
                print(f"Errore generazione report parziale: {e}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_system_requirements():
    """Controlla requisiti di sistema per training su larga scala."""
    
    print("🔍 Controllo requisiti di sistema...")
    
    # Check memoria
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    if memory_gb < 16:
        print(f"⚠️ WARNING: Memoria disponibile {memory_gb:.1f}GB < 16GB raccomandati")
        response = input("Continuare comunque? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ Memoria: {memory_gb:.1f}GB")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    if cpu_count < 8:
        print(f"⚠️ WARNING: {cpu_count} CPU cores < 8 raccomandati per parallelizzazione")
    else:
        print(f"✅ CPU cores: {cpu_count}")
    
    # Check spazio disco
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < 10:
        print(f"❌ ERRORE: Spazio disco {free_gb:.1f}GB < 10GB richiesti")
        sys.exit(1)
    else:
        print(f"✅ Spazio disco: {free_gb:.1f}GB liberi")
    
    print("✅ Requisiti di sistema soddisfatti\n")


def create_custom_config(args):
    """Crea configurazione personalizzata basata su argomenti."""
    
    config = OPTIMIZATION_CONFIG.copy()
    
    # Override da argomenti
    if args.embedding_dim:
        config["embedding_dim"] = args.embedding_dim
    
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    
    if args.num_layers:
        config["num_layers"] = args.num_layers
    
    # Configurazioni specifiche per larga scala
    if args.train_sentences >= 500:
        # Configurazione ottimizzata per training su larga scala
        config.update({
            "batch_size": min(64, args.train_sentences // 10),
            "gradient_clip_norm": 1.0,
            "regularization_lambda": 0.0001,
            "early_stopping_patience": 20,
            "checkpoint_frequency": 10
        })
        print("🚀 Configurazione ottimizzata per training su larga scala attivata")
    
    return config


def main():
    """Main training script per Penn Treebank su larga scala."""
    
    parser = argparse.ArgumentParser(description='Quantum Training su Penn Treebank - Larga Scala')
    
    # Dataset arguments
    parser.add_argument('--train-sentences', type=int, default=700,
                        help='Numero frasi di training (default: 700)')
    parser.add_argument('--test-sentences', type=int, default=300,
                        help='Numero frasi di test (default: 300)')
    
    # Training arguments
    parser.add_argument('--max-iterations', type=int, default=500,
                        help='Numero massimo iterazioni (default: 500)')
    parser.add_argument('--convergence-threshold', type=float, default=1e-6,
                        help='Threshold per convergenza (default: 1e-6)')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=None,
                        help='Dimensione embedding (default: da config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (default: da config)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Numero layer quantum (default: da config)')
    
    # System arguments
    parser.add_argument('--output-dir', type=str, default='quantum_training_results',
                        help='Directory output (default: quantum_training_results)')
    parser.add_argument('--skip-requirements-check', action='store_true',
                        help='Salta controllo requisiti sistema')
    parser.add_argument('--verbose', action='store_true',
                        help='Output verboso')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging level
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    
    # Banner
    print("="*80)
    print("🚀 QUANTUM TRAINING SU PENN TREEBANK - LARGA SCALA")
    print("="*80)
    print(f"📚 Target: {args.train_sentences} frasi training + {args.test_sentences} frasi test")
    print(f"🎯 Max iterazioni: {args.max_iterations}")
    print(f"📁 Output: {args.output_dir}")
    print(f"⚡ Parallelizzazione: Massima disponibile")
    print("="*80)
    print()
    
    # Check requisiti sistema
    if not args.skip_requirements_check:
        check_system_requirements()
    
    # Crea configurazione personalizzata
    custom_config = create_custom_config(args)
    
    # Salva configurazione per riferimento
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    config_path = output_dir / f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    print(f"💾 Configurazione salvata: {config_path}")
    
    # Inizializza trainer
    print("🔧 Inizializzazione trainer integrato...")
    trainer = IntegratedQuantumTrainer(output_dir=args.output_dir)
    
    # Override config
    trainer.config.update(custom_config)
    
    # Setup gestione segnali
    setup_signal_handlers(trainer)
    
    # Stima tempo di training
    estimated_time_hours = (args.train_sentences * args.max_iterations * 0.1) / 3600.0
    print(f"⏱️ Tempo stimato: ~{estimated_time_hours:.1f} ore")
    print()
    
    # Conferma utente per training su larga scala
    if args.train_sentences >= 500:
        print("⚠️ ATTENZIONE: Training su larga scala richiede risorse significative")
        print(f"   • {args.train_sentences + args.test_sentences} frasi totali")
        print(f"   • {args.max_iterations} iterazioni massime")
        print(f"   • Tempo stimato: ~{estimated_time_hours:.1f} ore")
        print()
        
        response = input("Procedere con il training? (y/n): ")
        if response.lower() != 'y':
            print("Training annullato dall'utente")
            sys.exit(0)
        print()
    
    # Esegui training
    try:
        print("🎯 AVVIO TRAINING SU LARGA SCALA")
        print("="*50)
        
        start_time = datetime.now()
        
        report_path = trainer.run_complete_training(
            train_sentences=args.train_sentences,
            test_sentences=args.test_sentences,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold
        )
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 3600.0
        
        print("="*50)
        print("🎉 TRAINING COMPLETATO CON SUCCESSO!")
        print("="*50)
        print(f"⏱️ Durata totale: {total_duration:.2f} ore")
        print(f"📊 Report completo: {report_path}")
        
        # Apri report in browser se possibile
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            print("🌐 Report aperto nel browser")
        except:
            print("💡 Apri manualmente il report HTML nel browser")
        
        print()
        print("✅ Training su larga scala completato con successo!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrotto dall'utente")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE TRAINING: {str(e)}")
        
        # Tenta di generare report di errore
        try:
            error_report = trainer._generate_error_report(str(e))
            print(f"📊 Report di errore generato: {error_report}")
        except:
            print("⚠️ Impossibile generare report di errore")
        
        import traceback
        if args.verbose:
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()