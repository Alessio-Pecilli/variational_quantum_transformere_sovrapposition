#!/bin/bash
#SBATCH --job-name=quantum_beast
#SBATCH --output=logs/quantum_beast_%j.out
#SBATCH --error=logs/quantum_beast_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_rosati
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --mail-user=ale.pecilli@stud.uniroma3.it
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== 🚀 JOB $SLURM_JOB_ID STARTED at $(date) on $(hostname) ==="

# ============================================================
# 🔧 Setup ambiente HPC Leonardo
# ============================================================
module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
source $WORK/venv_py311/bin/activate

# Disattiva thread multipli nei backend numerici
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ============================================================
# 🚀 Avvio del training quantistico
# ============================================================
cd $WORK/variational_quantum_transformere_sovrapposition || exit 1
mkdir -p logs

echo "📦 Environment ready. Starting training..."
srun python main_hpc.py

EXIT_CODE=$?
echo "🏁 Job terminato con exit code: $EXIT_CODE"
echo "=== FINE JOB $(date) ==="
