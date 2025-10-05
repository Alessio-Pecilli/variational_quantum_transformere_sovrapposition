#!/bin/bash
#SBATCH --job-name=quantum_apocalisse
#SBATCH --output=quantum_apocalisse_%j.log
#SBATCH --error=quantum_apocalisse_%j.log
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_rosati
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ale.pecilli@stud.uniroma3.it

echo "=== 💀 QUANTUM APOCALISSE HPC JOB $SLURM_JOB_ID STARTED at $(date) 💀 ==="

# === Setup moduli ===
module purge
module load python/3.11.7
module load openmpi/4.1.6--gcc--12.2.0

# === Ambiente virtuale ===
source $WORK/venv_py311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "🔥🔥🔥 CONFIGURAZIONE HPC BEAST MODE - 8 NODI DEVASTANTI! 🔥🔥🔥"
echo "  Job ID: $SLURM_JOB_ID"
echo "  NODI: $SLURM_JOB_NUM_NODES"
echo "  TASKS: $SLURM_NTASKS"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  🚀 CORE TOTALI: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "  ⚡ Workers disponibili: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK - SLURM_NTASKS))"
echo "  💀 POTENZA DEVASTANTE: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK)) CORES REALI DI LEONARDO!"

cd $WORK/variational_quantum_transformere_sovrapposition || {
    echo "❌ ERRORE: Directory progetto non trovata!"
    exit 1
}

mkdir -p logs checkpoints results

# === Esecuzione Python ===
echo "🔥 AVVIO TRAINING QUANTISTICO DEVASTANTE - BEAST MODE HPC 🔥"
python hpc_quantum_training_BEAST_MODE.py > logs/job_${SLURM_JOB_ID}.out 2>&1
EXIT_CODE=$?

# === Copia log beast mode ===
if [ -f "hpc_beast_mode.log" ]; then
    cp hpc_beast_mode.log logs/beast_mode_${SLURM_JOB_ID}.log
    echo "📋 Log BEAST MODE salvato in logs/beast_mode_${SLURM_JOB_ID}.log"
fi

# === Copia risultati training se esistono ===
if ls beast_mode_results_*.pkl >/dev/null 2>&1; then
    cp beast_mode_results_*.pkl logs/
    echo "💾 Risultati training salvati in logs/"
fi

# === Fine job ===
echo ""
echo "=== 💀 QUANTUM APOCALISSE HPC JOB $SLURM_JOB_ID COMPLETED 💀 ==="
echo "Exit code: $EXIT_CODE"
echo "Ended at: $(date)"
echo "Duration: $SECONDS sec"

# === Prepara report COMPLETO per email ===
REPORT="logs/report_${SLURM_JOB_ID}.txt"
{
    echo "=== [APOCALISSE HPC REPORT] ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Host: $(hostname)"
    echo "Nodi utilizzati: $SLURM_JOB_NUM_NODES"
    echo "Cores totali: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
    echo "Data inizio: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "Durata: $SECONDS secondi"
    echo ""
    echo "=== LOG PRINCIPALE ==="
    cat logs/job_${SLURM_JOB_ID}.out 2>/dev/null || echo "Log non trovato"
    echo ""
    if [ -f "logs/beast_mode_${SLURM_JOB_ID}.log" ]; then
        echo "=== LOG BEAST MODE ==="
        cat logs/beast_mode_${SLURM_JOB_ID}.log
    fi
    echo ""
    echo "=== AMBIENTE ==="
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "Working directory: $(pwd)"
    echo "Python version: $(python --version 2>&1)"
    echo ""
    echo "=== FILES ==="
    ls -la logs/ 2>/dev/null || echo "Directory logs non trovata"
} > "$REPORT"

# === Invio email ===
echo "📧 Invio report via email..."
if command -v mailx >/dev/null 2>&1; then
    mailx -s "[APOCALISSE HPC] Job $SLURM_JOB_ID - Exit $EXIT_CODE" \
          ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null
elif command -v mail >/dev/null 2>&1; then
    mail -s "[APOCALISSE HPC] Job $SLURM_JOB_ID - Exit $EXIT_CODE" \
         ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null
else
    echo "⚠️ mailx/mail non disponibili: report salvato localmente in $REPORT"
fi

echo ""
echo "📋 Report salvato in: $REPORT"
echo "📏 Dimensione report: $(wc -l < "$REPORT" 2>/dev/null || echo 'N/A') righe"
echo ""
echo "💀 APOCALISSE COMPLETATA! LEONARDO HA BRUCIATO! 💀"

exit $EXIT_CODE
