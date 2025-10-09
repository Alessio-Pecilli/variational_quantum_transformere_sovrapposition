#SBATCH --job-name=quantum_beast_mpi
#SBATCH --output=logs/quantum_beast_%j.out
#SBATCH --error=logs/quantum_beast_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_rosati
#SBATCH --nodes=4                 # = 4 nodi × 64 core = 256 processi
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --exclusive

module purge
module load python/3.11.7
source $WORK/venv_py311/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd $WORK/variational_quantum_transformere_sovrapposition || exit 1
mkdir -p logs

echo "🚀 AVVIO TRAINING MPI — $(date)"
srun python main_hpc.py
echo "🏁 TRAINING COMPLETATO — $(date)"
