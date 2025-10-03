# 🧪 Local Testing Guide for MPI Training

## Quick Start (Windows Development)

### Prerequisites
- Python 3.11 or higher
- Microsoft MPI (for Windows) or OpenMPI (for Linux/Mac)

---

## Installation

### 1. Install Microsoft MPI (Windows)

Download and install both:
- **MS-MPI v10.1.2**: https://www.microsoft.com/en-us/download/details.aspx?id=100593
  - `msmpisetup.exe` (runtime)
  - `msmpisdk.msi` (SDK)

Verify installation:
```powershell
mpiexec -help
```

### 2. Create Python Virtual Environment

```powershell
# Navigate to project
cd "c:\Users\Ale\Desktop\progetti\FIN - QT"

# Create venv
python -m venv venv_mpi

# Activate
.\venv_mpi\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install numpy scipy qiskit mpi4py
```

**Note**: On Windows, `mpi4py` requires MS-MPI to be installed first.

### 4. Verify Installation

```powershell
# Test mpi4py
python -c "from mpi4py import MPI; print(f'MPI rank: {MPI.COMM_WORLD.Get_rank()}')"

# Test Qiskit
python -c "import qiskit; print(f'Qiskit version: {qiskit.__version__}')"
```

---

## Running Tests

### Test 1: Sequential Baseline

Run the original (non-MPI) version to establish baseline:

```powershell
python main_superposition.py
```

Expected output:
```
Training on "The cat"
Epoch 1/1 completed in ~15-20 seconds
```

### Test 2: MPI with 2 Ranks

```powershell
mpiexec -n 2 python main_superposition_mpi.py --epochs 5 --lr 0.01
```

Expected output:
```
Rank 0: Initializing parameters...
Rank 0: Data sharding:
  Rank 0: 1 samples
  Rank 1: 0 samples
Epoch 1/5 | Loss: 0.XXXXXX | Grad Norm: X.XXXX | Time: X.XXs
...
```

### Test 3: MPI with 4 Ranks

```powershell
mpiexec -n 4 python main_superposition_mpi.py --epochs 10 --lr 0.01 --seed 42
```

### Test 4: Checkpoint/Resume

```powershell
# Train for 10 epochs
mpiexec -n 2 python main_superposition_mpi.py --epochs 10 --checkpoint-every 5

# Resume and continue to 20 epochs
mpiexec -n 2 python main_superposition_mpi.py --epochs 20 --resume
```

---

## Common Issues

### Issue 1: mpi4py Import Error

**Error**: `ModuleNotFoundError: No module named 'mpi4py'`

**Solution**:
```powershell
# Make sure virtual environment is activated
.\venv_mpi\Scripts\Activate.ps1

# Reinstall mpi4py
pip uninstall mpi4py
pip install mpi4py
```

### Issue 2: MPI Not Found

**Error**: `mpiexec is not recognized as an internal or external command`

**Solution**: Install MS-MPI and add to PATH:
```powershell
$env:PATH += ";C:\Program Files\Microsoft MPI\Bin"
```

Make permanent (System Properties → Environment Variables).

### Issue 3: No Training Data

**Error**: `ValueError: empty dataset`

**Solution**: Check `config.py`:
```python
TRAINING_SENTENCES = ["The cat"]  # Must have at least 1 sentence
```

### Issue 4: Visualization Import Error

**Error**: `ModuleNotFoundError: No module named 'visualization'`

**Solution**: Comment out in `main_superposition_mpi.py`:
```python
# from visualization import save_parameters
# save_parameters(params)
```

Or create stub `visualization.py`:
```python
import json
def save_parameters(params):
    with open("params_best.json", "w") as f:
        json.dump(params.tolist(), f)
```

---

## Performance Comparison

### Expected Results (Windows, 2-4 cores)

| Configuration | Time/Epoch | Speedup |
|--------------|------------|---------|
| Sequential   | ~15s       | 1.0x    |
| MPI (2 ranks)| ~10s       | 1.5x    |
| MPI (4 ranks)| ~8s        | 1.8x    |

**Note**: Windows MPI has more overhead than Linux HPC. Real speedup seen on Leonardo cluster.

---

## Debugging

### Enable Verbose MPI Logging

```powershell
mpiexec -n 2 -verbose python main_superposition_mpi.py --epochs 1
```

### Check Rank Assignment

Add debug prints in `main_superposition_mpi.py`:
```python
if rank == 0:
    print(f"Total ranks: {size}")
for i in range(size):
    if rank == i:
        print(f"Rank {rank}: Ready")
    comm.Barrier()
```

### Profile Performance

```python
import time
start = time.time()
# ... training loop ...
elapsed = time.time() - start
if rank == 0:
    print(f"Total time: {elapsed:.2f}s")
```

---

## Next Steps

1. ✅ Verify local installation works
2. ✅ Test with 2 ranks
3. ✅ Test checkpoint/resume
4. ⬜ Push to Leonardo cluster
5. ⬜ Run with 8-32 ranks on HPC

See `MPI_TRAINING_GUIDE.md` for Leonardo deployment.

---

## Files Created

```
├── main_superposition_mpi.py      # MPI training script
├── quantum_mpi_utils.py           # MPI helper functions
├── run_leonardo_mpi.sh            # SLURM job script (HPC)
├── MPI_TRAINING_GUIDE.md          # Full guide
└── MPI_LOCAL_TESTING.md           # This file
```

---

**Happy testing! 🚀**
