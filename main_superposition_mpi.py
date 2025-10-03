"""
Distributed training for Quantum Transformer using MPI (Data Parallel).

This script implements synchronous data-parallel training following the DDP pattern:
- Each MPI rank processes a subset of sentences
- Gradients are averaged via Allreduce
- Rank 0 updates parameters and broadcasts to all ranks

Usage:
    srun -n 8 python main_superposition_mpi.py --epochs 50 --lr 0.01
"""

import argparse
import time
import numpy as np
from mpi4py import MPI

# Import project modules
from encoding import Encoding
from quantum_mpi_utils import (
    build_variational_ansatz,
    shard_dataset,
    loss_and_grad_for_sentence,
    save_checkpoint,
    load_latest_checkpoint,
    log_training_metrics
)
from quantum_circuits import get_circuit_function
from quantum_utils import get_params
from config import TRAINING_SENTENCES, OPTIMIZATION_CONFIG
from main_superposition import process_sentence_states


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed QTransformer Training with MPI')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of ansatz layers')
    parser.add_argument('--embedding-dim', type=int, default=4, help='Embedding dimension')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    return parser.parse_args()


def prepare_training_data(sentences, embedding_dim):
    """
    Prepare training data from sentences.
    
    Args:
        sentences (list): List of training sentences
        embedding_dim (int): Embedding dimension
        
    Returns:
        list: List of (psi, U, Z) tuples for each sentence
    """
    enc = Encoding(sentences, embeddingDim=embedding_dim)
    
    training_data = []
    for sentence_idx in range(len(sentences)):
        states = enc.stateVectors[sentence_idx]
        states_calculated, U, Z = process_sentence_states(states)
        
        if len(states_calculated) > 0:
            training_data.append((states_calculated, U, Z))
    
    return training_data


def main():
    """Main training loop with MPI."""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed (different for each rank for data diversity)
    np.random.seed(args.seed + rank)
    
    if rank == 0:
        print("="*80)
        print("DISTRIBUTED QUANTUM TRANSFORMER TRAINING WITH MPI")
        print("="*80)
        print(f"Configuration:")
        print(f"  MPI Ranks: {size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate: {args.lr}")
        print(f"  Num Layers: {args.num_layers}")
        print(f"  Embedding Dim: {args.embedding_dim}")
        print(f"  Random Seed: {args.seed}")
        print("="*80)
        print(f"\nTraining on {len(TRAINING_SENTENCES)} sentences:")
        for i, sent in enumerate(TRAINING_SENTENCES):
            print(f"  {i+1}. '{sent}'")
        print("="*80)
    
    # Prepare training data (all ranks do this to avoid communication overhead)
    if rank == 0:
        print("\nPreparing training data...")
    
    training_data = prepare_training_data(TRAINING_SENTENCES, args.embedding_dim)
    
    if rank == 0:
        print(f"✓ Prepared {len(training_data)} training samples")
    
    # Shard dataset across ranks
    local_data = shard_dataset(training_data, rank, size)
    
    if rank == 0:
        print(f"\nData sharding:")
        for r in range(size):
            shard = shard_dataset(training_data, r, size)
            print(f"  Rank {r}: {len(shard)} samples")
    
    # Initialize parameters
    if rank == 0:
        print("\nInitializing model...")
        
        # Build variational ansatz template
        n_qubits = 2
        param_shape = get_params(n_qubits, args.num_layers).shape
        n_params = int(np.prod(param_shape))
        num_params = 2 * n_params  # V and K parameters
        
        # Initialize parameters
        if args.resume:
            checkpoint = load_latest_checkpoint()
            if checkpoint is not None:
                params = checkpoint['params']
                start_epoch = checkpoint['epoch'] + 1
                lr = checkpoint['lr']
                print(f"✓ Resumed from checkpoint: epoch {checkpoint['epoch']}")
            else:
                params = np.random.randn(num_params) * 0.1
                start_epoch = 0
                lr = args.lr
                print("✓ No checkpoint found, starting fresh")
        else:
            params = np.random.randn(num_params) * 0.1
            start_epoch = 0
            lr = args.lr
            print("✓ Initialized random parameters")
        
        print(f"  Total parameters: {num_params}")
        print(f"  Starting epoch: {start_epoch}")
    else:
        params = None
        start_epoch = 0
        lr = args.lr
        n_qubits = 2
        num_params = None
    
    # Broadcast parameters and metadata to all ranks
    params = comm.bcast(params, root=0)
    start_epoch = comm.bcast(start_epoch, root=0)
    lr = comm.bcast(lr, root=0)
    
    # Get circuit function (same for all ranks)
    # Assuming all samples have same number of words for simplicity
    # If variable, this logic needs adjustment
    if len(local_data) > 0:
        num_words = len(local_data[0][0])  # Number of state transitions
        circuit_function = get_circuit_function(num_words)
    else:
        circuit_function = None
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Initialize local accumulators
        grad_local = np.zeros_like(params)
        loss_local_sum = 0.0
        count_local = 0
        
        # Process local data shard
        for sentence_data in local_data:
            if circuit_function is not None:
                loss_s, grad_s = loss_and_grad_for_sentence(
                    sentence_data, 
                    params,
                    None,  # qc_template not used in current implementation
                    None,  # theta_vec not used in current implementation
                    circuit_function,
                    args.num_layers,
                    args.embedding_dim
                )
                
                grad_local += grad_s
                loss_local_sum += loss_s
                count_local += 1
        
        # Allreduce gradients (sum across all ranks)
        grad_sum = np.zeros_like(grad_local)
        comm.Allreduce(grad_local, grad_sum, op=MPI.SUM)
        
        # Average gradients
        grad_mean = grad_sum / size
        
        # Allreduce loss and count
        loss_sum_global = comm.allreduce(loss_local_sum, op=MPI.SUM)
        count_global = comm.allreduce(count_local, op=MPI.SUM)
        
        # Calculate mean loss
        loss_mean = loss_sum_global / max(count_global, 1)
        
        # Update parameters (only rank 0)
        if rank == 0:
            # Gradient descent step
            params = params - lr * grad_mean
            
            # Calculate gradient norm for logging
            grad_norm = np.linalg.norm(grad_mean)
            
            # Save checkpoint periodically
            if (epoch + 1) % args.checkpoint_every == 0:
                checkpoint_path = save_checkpoint(params, epoch, lr, args.seed)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Broadcast updated parameters to all ranks
        params = comm.bcast(params, root=0)
        
        # Logging (only rank 0)
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            
            # Log to console
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {loss_mean:.6f} | "
                  f"Grad Norm: {grad_norm:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Log to CSV
            log_training_metrics(epoch, loss_mean, grad_norm, epoch_time)
    
    # Final checkpoint
    if rank == 0:
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        
        final_checkpoint = save_checkpoint(params, args.epochs - 1, lr, args.seed)
        print(f"✓ Final checkpoint saved: {final_checkpoint}")
        
        # Save final parameters in standard format
        from visualization import save_parameters
        save_parameters(params)
        print("✓ Final parameters saved to params_best.json")
        
        print(f"\nFinal Loss: {loss_mean:.6f}")
        print(f"Total Epochs: {args.epochs}")
        print("="*80)


if __name__ == "__main__":
    main()
