#!/bin/bash
#SBATCH --job-name=quantum_beast
#SBATCH --output=quantum_beast_%j.log
#SBATCH --error=quantum_beast_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_rosati
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ale.pecilli@stud.uniroma3.it

echo "=== ⚡ QUANTUM BEAST JOB $SLURM_JOB_ID STARTED at $(date) ==="

module purge
module load python/3.11.7
source $WORK/venv_py311/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $WORK/variational_quantum_transformere_sovrapposition || exit 1
mkdir -p logs

# Log di debug in tempo reale
echo "🚀 Avvio training parallelo..."
python hpc_quantum_training_BEAST_MODE.py > logs/job_${SLURM_JOB_ID}.out 2>&1

EXIT_CODE=$?

echo "=== ⚙️ JOB TERMINATO con exit code: $EXIT_CODE ==="
echo "=== FINE: $(date) ==="

# Invia sempre email finale con log allegato
mail -s "[Leonardo HPC] Job $SLURM_JOB_ID terminato con codice $EXIT_CODE" \
     ale.pecilli@stud.uniroma3.it < logs/job_${SLURM_JOB_ID}.out
