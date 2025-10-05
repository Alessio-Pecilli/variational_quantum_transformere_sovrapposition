#!/bin/bash
#SBATCH --job-name=quantum_mem_opt
#SBATCH --output=quantum_mem_opt_%j.log
#SBATCH --error=quantum_mem_opt_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
# NOTE: nessuna direttiva --account per usare l'account di default dell'utente
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ale.pecilli@stud.uniroma3.it

echo "=== QUANTUM MEMORY-OPT JOB ${SLURM_JOB_ID:-local} START $(date) ==="

module purge
module load python/3.11.7 || true
module load openmpi/4.1.6--gcc--12.2.0 || true

if [ -n "$WORK" ] && [ -d "$WORK/venv_py311" ]; then
  source "$WORK/venv_py311/bin/activate"
fi

# Limiti per ridurre uso memoria
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MAX_WORKERS=64
export BATCH_SIZE=8

cd ${WORK:-$PWD}/variational_quantum_transformere_sovrapposition || { echo "Directory progetto non trovata"; exit 1; }
mkdir -p logs checkpoints results

echo "🚀 Avvio training MEMORY-OPT (logs/job_${SLURM_JOB_ID:-local}.out)"
# Passa variabili via env se lo script le legge
MAX_WORKERS=${MAX_WORKERS} BATCH_SIZE=${BATCH_SIZE} python -u main_memory_optimized.py > "logs/job_${SLURM_JOB_ID:-local}.out" 2>&1
EXIT_CODE=$?

if [ -f "hpc_beast_mode.log" ]; then cp hpc_beast_mode.log "logs/beast_mode_${SLURM_JOB_ID:-local}.log"; fi
shopt -s nullglob
if compgen -G "memory_opt_results_*.json" > /dev/null; then cp memory_opt_results_*.json logs/ 2>/dev/null; fi
shopt -u nullglob

REPORT="logs/report_memopt_${SLURM_JOB_ID:-local}.txt"
{
  echo "JobID: ${SLURM_JOB_ID:-local}"
  echo "Exit code: $EXIT_CODE"
  echo "MAX_WORKERS=$MAX_WORKERS BATCH_SIZE=$BATCH_SIZE"
  tail -n 200 "logs/job_${SLURM_JOB_ID:-local}.out" 2>/dev/null || echo "Log non trovato"
} > "$REPORT"

echo "Report salvato in: $REPORT"

exit $EXIT_CODE
#!/bin/bash
#SBATCH --job-name=quantum_memory_opt
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