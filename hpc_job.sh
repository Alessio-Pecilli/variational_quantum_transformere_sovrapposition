#!/bin/bash
#SBATCH --job-name=quantum_hpc
#SBATCH --output=quantum_training_%j.log
#SBATCH --error=quantum_training_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_rosati
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ale.pecilli@stud.uniroma3.it

echo "=== QUANTUM HPC JOB (NO MPI) $SLURM_JOB_ID STARTED at $(date) ==="

# Setup moduli
module purge
module load python/3.11.7
module load openmpi/4.1.6--gcc--12.2.0

# Attiva ambiente virtuale
source $WORK/venv_py311/bin/activate

# Variabili multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "🖥️ CONFIGURAZIONE HPC (MULTIPROCESSING):"
echo "  - Job ID: $SLURM_JOB_ID"
echo "  - Nodes: $SLURM_NNODES"
echo "  - Tasks: $SLURM_NTASKS"
echo "  - CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Directory progetto
cd $WORK/variational_quantum_transformere_sovrapposition || {
    echo "❌ ERRORE: Directory progetto non trovata!"
    exit 1
}

echo "📁 Working directory: $(pwd)"
mkdir -p logs checkpoints results

# Avvio training
echo "🚀 AVVIO TRAINING QUANTISTICO..."
python hpc_quantum_training.py > logs/job_${SLURM_JOB_ID}.out 2>&1
EXIT_CODE=$?

# Fine job
echo ""
echo "=== QUANTUM HPC JOB $SLURM_JOB_ID COMPLETED ==="
echo "📊 RISULTATI:"
echo "  - Exit code: $EXIT_CODE"
echo "  - Ended at: $(date)"
echo "  - Duration: $SECONDS seconds"

if [ $EXIT_CODE -eq 0 ]; then
    STATUS="✅ TRAINING COMPLETATO CON SUCCESSO"
else
    STATUS="❌ TRAINING FALLITO (exit code: $EXIT_CODE)"
fi

# Crea report temporaneo
REPORT="logs/report_${SLURM_JOB_ID}.txt"
{
    echo "=== RISULTATI JOB CINECA ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Data completamento: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "Stato: $STATUS"
    echo ""
    echo "=== CONTENUTO LOG COMPLETO ==="
    echo ""
    cat logs/job_${SLURM_JOB_ID}.out
} > "$REPORT"

# Invia email con il log completo
if command -v mailx >/dev/null 2>&1; then
    cat "$REPORT" | mailx -s "[CINECA JOB] $STATUS - ID $SLURM_JOB_ID" ale.pecilli@stud.uniroma3.it
    echo "📧 Email inviata a ale.pecilli@stud.uniroma3.it con i log del job."
else
    echo "⚠️ mailx non trovato: impossibile inviare l'email. Log salvato in $REPORT"
fi

exit $EXIT_CODE
