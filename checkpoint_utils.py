"""
Utility Script per Gestione Checkpoint e Resume Training.
Permette di elencare, riprendere e gestire checkpoint di training.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json

# Aggiungi directory corrente al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checkpoint_system import QuantumTrainingCheckpoint, ResumableTrainer
from integrated_quantum_trainer import IntegratedQuantumTrainer


def list_checkpoints_command(args):
    """Lista tutti i checkpoint disponibili."""
    
    checkpoint_manager = QuantumTrainingCheckpoint(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints(args.run_id)
    
    if not checkpoints:
        print("Nessun checkpoint trovato")
        return
    
    print("CHECKPOINT DISPONIBILI:")
    print("="*80)
    print(f"{'Run ID':<20} {'Iterazione':<12} {'Loss Corrente':<15} {'Loss Migliore':<15} {'Dimensione':<10} {'Data'}")
    print("-"*80)
    
    for cp in checkpoints:
        run_id = cp['run_id']
        iteration = cp['iteration']
        current_loss = f"{cp['current_loss']:.6f}" if cp['current_loss'] else "N/A"
        best_loss = f"{cp['best_loss']:.6f}" if cp['best_loss'] else "N/A"
        size_mb = f"{cp['size_mb']:.1f} MB"
        timestamp = cp['timestamp']
        
        print(f"{run_id:<20} {iteration:<12} {current_loss:<15} {best_loss:<15} {size_mb:<10} {timestamp}")
    
    print("-"*80)
    print(f"Totale: {len(checkpoints)} checkpoint")


def resume_training_command(args):
    """Riprende training da checkpoint."""
    
    checkpoint_manager = QuantumTrainingCheckpoint(args.checkpoint_dir)
    
    # Determina checkpoint da usare
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    elif args.run_id:
        checkpoint_path = checkpoint_manager.find_latest_checkpoint(args.run_id)
        if not checkpoint_path:
            print(f"❌ Nessun checkpoint trovato per run_id: {args.run_id}")
            return
        print(f"📂 Usando checkpoint più recente: {checkpoint_path}")
    else:
        print("❌ Specificare --checkpoint-path o --run-id")
        return
    
    # Verifica esistenza checkpoint
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint non trovato: {checkpoint_path}")
        return
    
    # Carica info checkpoint
    try:
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
        training_metadata = checkpoint_data["training_metadata"]
        
        print("📊 INFORMAZIONI CHECKPOINT:")
        print(f"   Run ID: {training_metadata['run_id']}")
        print(f"   Iterazione: {training_metadata['iteration']}")
        print(f"   Loss corrente: {training_metadata.get('current_loss', 'N/A')}")
        print(f"   Loss migliore: {training_metadata.get('best_loss', 'N/A')}")
        print(f"   Data: {training_metadata['timestamp']}")
        print()
        
    except Exception as e:
        print(f"❌ Errore caricamento checkpoint: {e}")
        return
    
    # Conferma utente
    if not args.yes:
        response = input("Procedere con il resume? (y/n): ")
        if response.lower() != 'y':
            print("Resume annullato")
            return
    
    # Inizializza trainer per resume
    print("🔧 Inizializzazione trainer per resume...")
    
    base_trainer = IntegratedQuantumTrainer(output_dir=args.output_dir)
    resumable_trainer = ResumableTrainer(
        base_trainer=base_trainer,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    # Riprendi training
    try:
        print("🔄 RESUME TRAINING")
        print("="*50)
        
        start_time = datetime.now()
        
        report_path = resumable_trainer.resume_from_checkpoint(checkpoint_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600.0
        
        print("="*50)
        print("🎉 RESUME COMPLETATO!")
        print(f"⏱️ Durata: {duration:.2f} ore")
        print(f"📊 Report: {report_path}")
        
    except Exception as e:
        print(f"❌ Errore durante resume: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()


def checkpoint_info_command(args):
    """Mostra informazioni dettagliate su un checkpoint."""
    
    checkpoint_manager = QuantumTrainingCheckpoint(args.checkpoint_dir)
    
    try:
        checkpoint_data = checkpoint_manager.load_checkpoint(args.checkpoint_path)
        
        metadata = checkpoint_data["metadata"]
        training_metadata = checkpoint_data["training_metadata"]
        config = checkpoint_data["config"]
        
        print("INFORMAZIONI DETTAGLIATE CHECKPOINT")
        print("="*60)
        
        print("\n📊 TRAINING:")
        print(f"   Run ID: {training_metadata['run_id']}")
        print(f"   Iterazione: {training_metadata['iteration']}")
        print(f"   Iterazioni totali: {training_metadata['total_iterations']}")
        print(f"   Loss corrente: {training_metadata.get('current_loss', 'N/A')}")
        print(f"   Loss migliore: {training_metadata.get('best_loss', 'N/A')}")
        print(f"   Forma parametri: {training_metadata['params_shape']}")
        
        print("\n⚙️ CONFIGURAZIONE:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print("\n📈 PROGRESS:")
        loss_history = checkpoint_data["loss_history"]
        if loss_history:
            print(f"   Loss iniziale: {loss_history[0]:.6f}")
            print(f"   Loss finale: {loss_history[-1]:.6f}")
            improvement = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
            print(f"   Miglioramento: {improvement:.2f}%")
        
        print("\n🕒 TIMESTAMP:")
        print(f"   Creato: {training_metadata['timestamp']}")
        
        # Dimensione checkpoint
        checkpoint_path = Path(args.checkpoint_path)
        size_mb = sum(f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()) / (1024*1024)
        print(f"\n💾 DIMENSIONE: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Errore lettura checkpoint: {e}")


def delete_checkpoint_command(args):
    """Elimina un checkpoint."""
    
    checkpoint_manager = QuantumTrainingCheckpoint(args.checkpoint_dir)
    
    if not Path(args.checkpoint_path).exists():
        print(f"❌ Checkpoint non trovato: {args.checkpoint_path}")
        return
    
    # Conferma eliminazione
    if not args.yes:
        print(f"⚠️ ATTENZIONE: Stai per eliminare il checkpoint:")
        print(f"   {args.checkpoint_path}")
        response = input("Confermi eliminazione? (y/n): ")
        if response.lower() != 'y':
            print("Eliminazione annullata")
            return
    
    try:
        checkpoint_manager.delete_checkpoint(args.checkpoint_path)
        print(f"✅ Checkpoint eliminato: {args.checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Errore eliminazione: {e}")


def cleanup_command(args):
    """Cleanup checkpoint vecchi."""
    
    checkpoint_manager = QuantumTrainingCheckpoint(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints(args.run_id)
    
    if not checkpoints:
        print("Nessun checkpoint trovato per cleanup")
        return
    
    # Ordina per data
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Identifica checkpoint da eliminare
    to_keep = checkpoints[:args.keep_latest]
    to_delete = checkpoints[args.keep_latest:]
    
    if not to_delete:
        print(f"Nessun checkpoint da eliminare (mantenendo ultimi {args.keep_latest})")
        return
    
    print(f"CLEANUP CHECKPOINT - Mantenendo ultimi {args.keep_latest}")
    print("="*50)
    print("Checkpoint da eliminare:")
    
    total_size_mb = 0
    for cp in to_delete:
        print(f"   {cp['run_id']} - iter {cp['iteration']} - {cp['size_mb']:.1f} MB")
        total_size_mb += cp['size_mb']
    
    print(f"\nSpazio da liberare: {total_size_mb:.1f} MB")
    
    # Conferma
    if not args.yes:
        response = input("Procedere con cleanup? (y/n): ")
        if response.lower() != 'y':
            print("Cleanup annullato")
            return
    
    # Elimina checkpoint
    deleted_count = 0
    for cp in to_delete:
        try:
            checkpoint_manager.delete_checkpoint(cp['path'])
            deleted_count += 1
        except Exception as e:
            print(f"Errore eliminazione {cp['path']}: {e}")
    
    print(f"✅ Cleanup completato: {deleted_count} checkpoint eliminati")


def main():
    """Main utility script."""
    
    parser = argparse.ArgumentParser(description='Gestione Checkpoint Training Quantico')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory checkpoint (default: checkpoints)')
    parser.add_argument('--verbose', action='store_true',
                        help='Output verboso')
    
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Lista checkpoint
    list_parser = subparsers.add_parser('list', help='Lista checkpoint')
    list_parser.add_argument('--run-id', type=str, help='Filtra per run ID')
    
    # Resume training
    resume_parser = subparsers.add_parser('resume', help='Riprendi training')
    resume_parser.add_argument('--checkpoint-path', type=str, help='Path checkpoint specifico')
    resume_parser.add_argument('--run-id', type=str, help='Run ID per ultimo checkpoint')
    resume_parser.add_argument('--output-dir', type=str, default='resumed_training',
                             help='Directory output (default: resumed_training)')
    resume_parser.add_argument('--checkpoint-frequency', type=int, default=10,
                             help='Frequenza checkpoint (default: 10)')
    resume_parser.add_argument('--yes', action='store_true', help='Salta conferme')
    
    # Info checkpoint
    info_parser = subparsers.add_parser('info', help='Info checkpoint')
    info_parser.add_argument('checkpoint_path', help='Path checkpoint')
    
    # Elimina checkpoint
    delete_parser = subparsers.add_parser('delete', help='Elimina checkpoint')
    delete_parser.add_argument('checkpoint_path', help='Path checkpoint')
    delete_parser.add_argument('--yes', action='store_true', help='Salta conferma')
    
    # Cleanup checkpoint
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup checkpoint vecchi')
    cleanup_parser.add_argument('--run-id', type=str, help='Run ID specifico')
    cleanup_parser.add_argument('--keep-latest', type=int, default=5,
                              help='Numero checkpoint da mantenere (default: 5)')
    cleanup_parser.add_argument('--yes', action='store_true', help='Salta conferma')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging se verbose
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Esegui comando
    if args.command == 'list':
        list_checkpoints_command(args)
    elif args.command == 'resume':
        resume_training_command(args)
    elif args.command == 'info':
        checkpoint_info_command(args)
    elif args.command == 'delete':
        delete_checkpoint_command(args)
    elif args.command == 'cleanup':
        cleanup_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()