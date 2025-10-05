#!/bin/bash
#SBATCH --job-name=quantum_memory_opt
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --output=memory_opt_%j.out
#SBATCH --error=memory_opt_%j.err

echo "🧠 QUANTUM TRAINING - MEMORY OPTIMIZED VERSION"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodi allocati: $SLURM_JOB_NUM_NODES"
echo "CPU per task: $SLURM_CPUS_PER_TASK"
echo "Memoria per nodo: 100GB (ridotta per safety)"
echo "=============================================="

# Carica moduli HPC
module load python/3.9.16--gcc--11.3.0
module load numpy/1.24.3--python--3.9.16--gcc--11.3.0

# Environment variables per ottimizzazione memoria
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Memory optimization flags
export PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

echo "🧠 Environment configurato per ottimizzazione memoria"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo ""

# Info sistema pre-execution
echo "📊 Sistema info:"
free -h
echo ""

# Esegui training quantico ottimizzato
echo "🚀 Avvio training quantico MEMORY-OPTIMIZED..."
time python3 main_memory_optimized.py

echo ""
echo "📊 Sistema info post-execution:"
free -h

echo "✅ Job completato: $(date)"