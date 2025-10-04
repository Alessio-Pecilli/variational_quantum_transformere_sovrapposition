#!/bin/bash
#SBATCH --job-name=qtransformer_mpi
#SBATCH --output=logs/qtransformer_%j.out
#SBATCH --error=logs/qtransformer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT_HERE

# ============================================================================
# SLURM Job Script for Distributed Quantum Transformer Training on Leonardo
# ============================================================================
# 
# This script runs MPI-based data-parallel training with:
#   - 1 node, 16 MPI ranks, 4 CPUs per rank (64 cores total - FULL NODE)
#   - Dynamically adapts to SLURM allocation
#   - Python 3.11.7 virtual environment
#   - Checkpoint/resume support
#   - Automatic logs directory creation
#
# Usage:
#   sbatch run_leonardo_mpi.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/qtransformer_<jobid>.out
# ============================================================================

# Print job information
echo "================================================================================"
echo "QUANTUM TRANSFORMER MPI TRAINING - LEONARDO HPC"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $(($SLURM_NTASKS * $SLURM_CPUS_PER_TASK))"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "================================================================================"

# Load required modules
echo "Loading Python module..."
module load python/3.11.7

# Activate virtual environment
echo "Activating virtual environment..."
source $WORK/venv_py311/bin/activate

# Verify Python and mpi4py installation
echo "Verifying installation..."
python --version
python -c "from mpi4py import MPI; print(f'mpi4py loaded: MPI version {MPI.Get_version()}')" || {
    echo "ERROR: mpi4py not found in virtual environment!"
    echo "Please install: pip install mpi4py"
    exit 1
}

# Set OpenMP threads (important for hybrid MPI+OpenMP)
# This automatically adapts to the number of CPUs allocated per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo "MPI Ranks: $SLURM_NTASKS (auto-detected from SLURM allocation)"

# Create logs directory if not exists
mkdir -p logs
mkdir -p checkpoints

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Print training configuration
echo "================================================================================"
echo "TRAINING CONFIGURATION"
echo "================================================================================"
EPOCHS=50
LR=0.01
SEED=42
NUM_LAYERS=1
EMBEDDING_DIM=4
CHECKPOINT_EVERY=10

echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Random Seed: $SEED"
echo "Num Layers: $NUM_LAYERS"
echo "Embedding Dim: $EMBEDDING_DIM"
echo "Checkpoint Every: $CHECKPOINT_EVERY epochs"
echo "================================================================================"

# Run MPI training
echo ""
echo "Starting MPI training..."
echo "Command: srun -n $SLURM_NTASKS python main_superposition_mpi.py --epochs $EPOCHS --lr $LR --seed $SEED --num-layers $NUM_LAYERS --embedding-dim $EMBEDDING_DIM --checkpoint-every $CHECKPOINT_EVERY"
echo ""

srun -n $SLURM_NTASKS python main_superposition_mpi.py \
    --epochs $EPOCHS \
    --lr $LR \
    --seed $SEED \
    --num-layers $NUM_LAYERS \
    --embedding-dim $EMBEDDING_DIM \
    --checkpoint-every $CHECKPOINT_EVERY

# Check exit status
EXIT_CODE=$?
echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "================================================================================"

# Print checkpoint information
if [ -d "checkpoints" ]; then
    echo ""
    echo "Checkpoints created:"
    ls -lh checkpoints/qtransformer_ckpt_*.npz 2>/dev/null || echo "  No checkpoints found"
fi

# Print log information
if [ -f "logs/train_log.csv" ]; then
    echo ""
    echo "Training log summary (last 5 epochs):"
    tail -n 6 logs/train_log.csv | column -t -s ","
fi

echo ""
echo "Job finished at: $(date)"
echo "================================================================================"

exit $EXIT_CODE
