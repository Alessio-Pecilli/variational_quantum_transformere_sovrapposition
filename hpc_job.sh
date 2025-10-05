#!/bin/bash
#SBATCH --jobecho "🚀 CONFIGURAZIONE HPC BEAST MODE ULTIMATE (112 CORES - NODO COMPLETO!):"
echo "  Job ID: $SLURM_JOB            mail -s "[💀 APOCALISSE HPC] Job $SLURM_JOB_ID - 32 NODI/3584 CORES/3552 WORKERS - Exit $EXIT_CODE" \
         ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null || \il -s "[💀 APOCALISSE HPC] Job $SLURM_JOB_ID - 32 NODI/3584 CORES/3552 WORKERS - Exit $EXIT_CODE" \
             ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null || \    mailx -s "[� APOCALISSE HPC] Job $SLURM_JOB_ID - 32 NODI/3584 CORES/3552 WORKERS - Exit $EXIT_CODE" \
          ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  Workers disponibili: $((SLURM_CPUS_PER_TASK - 1))"
echo "  🔥 POTENZA MASSIMA: TUTTI I CORE DEL NODO LEONARDO!"e=quantum_hpc
#SBATCH --output=quantum_training_%j.log
#SBATCH --error=quantum_training_%j.log
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=112
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

echo "🔥🔥🔥 CONFIGURAZIONE HPC BEAST MODE APOCALITTICO - 32 NODI! 🔥🔥🔥"
echo "  Job ID: $SLURM_JOB_ID"
echo "  NODI: $SLURM_JOB_NUM_NODES"
echo "  TASKS: $SLURM_NTASKS"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  🚀 CORE TOTALI: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "  ⚡ Workers disponibili: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK - SLURM_NTASKS))"
echo "  💀 POTENZA DISTRUTTIVA: 3,584 CORES DI LEONARDO!"

cd $WORK/variational_quantum_transformere_sovrapposition || {
    echo "❌ ERRORE: Directory progetto non trovata!"
    exit 1
}

mkdir -p logs checkpoints results

# === Esecuzione Python ===
echo "�💀💀 AVVIO TRAINING QUANTISTICO APOCALITTICO - 3,552 WORKERS PARALLELI! 💀💀💀"
python hpc_quantum_training_BEAST_MODE.py > logs/job_${SLURM_JOB_ID}.out 2>&1
EXIT_CODE=$?

# === Copia log beast mode ===
if [ -f "hpc_beast_mode.log" ]; then
    cp hpc_beast_mode.log logs/beast_mode_${SLURM_JOB_ID}.log
    echo "📋 Log BEAST MODE salvato in logs/beast_mode_${SLURM_JOB_ID}.log"
fi

# === Copia risultati training se esistono ===
if [ -f "beast_mode_results_"*.pkl ]; then
    cp beast_mode_results_*.pkl logs/
    echo "💾 Risultati training salvati in logs/"
fi

# === Fine job ===
echo ""
echo "=== QUANTUM HPC JOB $SLURM_JOB_ID COMPLETED ==="
echo "Exit code: $EXIT_CODE"
echo "Ended at: $(date)"
echo "Duration: $SECONDS sec"

# === Prepara report COMPLETO per email ===
REPORT="logs/report_${SLURM_JOB_ID}.txt"
{
    echo "=== 📧 JOB CINECA - RISULTATI COMPLETI 📧 ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Host: $(hostname)"
    echo "Data inizio: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "Durata: $SECONDS secondi"
    echo ""
    echo "=== 📋 LOG PRINCIPALE COMPLETO ==="
    echo ""
    cat logs/job_${SLURM_JOB_ID}.out
    echo ""
    
    # Aggiungi log beast mode se presente
    if [ -f "logs/beast_mode_${SLURM_JOB_ID}.log" ]; then
        echo "=== � LOG BEAST MODE (100% PARALLELIZZAZIONE) ==="
        echo ""
        cat logs/beast_mode_${SLURM_JOB_ID}.log
        echo ""
    else
        echo "⚠️  File BEAST MODE log non trovato: logs/beast_mode_${SLURM_JOB_ID}.log"
        echo ""
    fi
    
    # Aggiungi info ambiente
    echo "=== 🖥️  INFORMAZIONI AMBIENTE ==="
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "Working directory: $(pwd)"
    echo "Python version: $(python --version 2>&1)"
    echo ""
    
    echo "=== 📁 FILE GENERATI ==="
    ls -la logs/ 2>/dev/null || echo "Directory logs non trovata"
    echo ""
    
    echo "=== 🏁 FINE REPORT ==="
} > "$REPORT"

# === Mostra report completo nel log principale ===
echo ""
echo "=== REPORT COMPLETO JOB $SLURM_JOB_ID ==="
cat "$REPORT"
echo "=== FINE REPORT ==="
echo ""

# === Invio email GARANTITO ===
echo "📧 Preparazione invio email..."

    # Prova mailx (metodo principale)
if command -v mailx >/dev/null 2>&1; then
    echo "   Usando mailx..."
    mailx -s "[🚀 BEAST MODE HPC] Job $SLURM_JOB_ID - 100% Parallelizzazione - Exit $EXIT_CODE" \
          ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null    if [ $? -eq 0 ]; then
        echo "   ✅ Email inviata con mailx"
    else
        echo "   ❌ Errore mailx, provo mail..."
        # Fallback a mail
        mail -s "[� BEAST MODE ULTIMATE] Job $SLURM_JOB_ID - 112 CORES/111 WORKERS - Exit $EXIT_CODE" \
             ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null || \
        echo "   ❌ Anche mail fallito"
    fi
else
    # Prova mail diretto
    echo "   mailx non disponibile, provo mail..."
    mail -s "[� BEAST MODE ULTIMATE] Job $SLURM_JOB_ID - 112 CORES/111 WORKERS - Exit $EXIT_CODE" \
         ale.pecilli@stud.uniroma3.it < "$REPORT" 2>/dev/null || \
    echo "   ❌ mail non funziona"
fi

# Log finale
echo ""
echo "📋 Report completo salvato in: $REPORT"
echo "� Dimensione report: $(wc -l < "$REPORT") righe"
echo "📧 Se non ricevi email, controlla il file: $REPORT"

exit $EXIT_CODE
