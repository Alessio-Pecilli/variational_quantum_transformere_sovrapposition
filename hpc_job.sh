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

# === Setup moduli ===
module purge
module load python/3.11.7
module load openmpi/4.1.6--gcc--12.2.0

# === Ambiente virtuale ===
source $WORK/venv_py311/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "🖥️ CONFIGURAZIONE HPC (MULTIPROCESSING):"
echo "  Job ID: $SLURM_JOB_ID"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"

cd $WORK/variational_quantum_transformere_sovrapposition || {
    echo "❌ ERRORE: Directory progetto non trovata!"
    exit 1
}

mkdir -p logs checkpoints results

# === Esecuzione Python ===
echo "🚀 AVVIO TRAINING QUANTISTICO CON DEBUG ESTESO..."
python hpc_quantum_training_debug.py > logs/job_${SLURM_JOB_ID}.out 2>&1
EXIT_CODE=$?

# === Copia log debug ===
if [ -f "hpc_debug.log" ]; then
    cp hpc_debug.log logs/debug_${SLURM_JOB_ID}.log
    echo "📋 Log di debug salvato in logs/debug_${SLURM_JOB_ID}.log"
fi

# === Fine job ===
echo ""
echo "=== QUANTUM HPC JOB $SLURM_JOB_ID COMPLETED ==="
echo "Exit code: $EXIT_CODE"
echo "Ended at: $(date)"
echo "Duration: $SECONDS sec"

# === Prepara report ===
REPORT="logs/report_${SLURM_JOB_ID}.txt"
{
    echo "=== JOB CINECA - RISULTATI COMPLETI ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Host: $(hostname)"
    echo "Data completamento: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "=== ULTIME 50 RIGHE DEL LOG PRINCIPALE ==="
    echo ""
    tail -n 50 logs/job_${SLURM_JOB_ID}.out
    echo ""
    
    # Aggiungi log debug se presente
    if [ -f "logs/debug_${SLURM_JOB_ID}.log" ]; then
        echo "=== LOG DEBUG DETTAGLIATO ==="
        echo ""
        cat logs/debug_${SLURM_JOB_ID}.log
        echo ""
    fi
    
    echo "=== FINE REPORT ==="
} > "$REPORT"

# === Invio mail con allegato ===
if command -v mailx >/dev/null 2>&1; then
    echo "Invio email dettagliata con log allegato..."
    mailx -s "[CINECA JOB] Risultato job $SLURM_JOB_ID (Exit $EXIT_CODE)" \
          -a "logs/job_${SLURM_JOB_ID}.out" \
          ale.pecilli@stud.uniroma3.it < "$REPORT"
else
    echo "⚠️ mailx non trovato. Log salvato in $REPORT"
fi

exit $EXIT_CODE
