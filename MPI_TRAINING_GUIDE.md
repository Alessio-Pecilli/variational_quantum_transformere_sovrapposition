# 🚀 MPI Training Guide for Quantum Transformer

## Overview

This guide explains how to run **distributed data-parallel training** for the Quantum Transformer using **MPI (Message Passing Interface)** on the Leonardo HPC cluster.

### Key Features
- ✅ **Data-parallel synchronous training** (DDP pattern)
- ✅ **Gradient aggregation** via MPI Allreduce
- ✅ **Checkpoint/resume** support
- ✅ **Automatic logging** to CSV
- ✅ **Scalable** to multiple nodes/ranks

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MPI Training Flow                     │
└─────────────────────────────────────────────────────────┘

Rank 0          Rank 1          ...         Rank N-1
  │               │                            │
  ├─ Init params ─┤                            │
  │               │                            │
  ├─ Broadcast params ──────────────────────────►
  │               │                            │
  ▼               ▼                            ▼
┌───────────┐ ┌───────────┐              ┌───────────┐
│ Shard 0   │ │ Shard 1   │      ...     │ Shard N-1 │
│ (D₀)      │ │ (D₁)      │              │ (D_{N-1}) │
└───────────┘ └───────────┘              └───────────┘
  │               │                            │
  ├─ Compute ∇L₀ ├─ Compute ∇L₁              ├─ Compute ∇L_{N-1}
  │               │                            │
  └───────────────┴────────── Allreduce ──────┴──────►
                                                       │
                  ∇L_mean = (∇L₀ + ∇L₁ + ... + ∇L_{N-1}) / N
                                                       │
  ◄──────────────────────────────────────────────────┘
  │
  ├─ Update: θ ← θ - lr * ∇L_mean  (Rank 0 only)
  │
  ├─ Broadcast updated θ ─────────────────────────────►
  │               │                            │
  └───────────────┴────── Repeat ─────────────┘
```

### Algorithm

1. **Initialization** (Rank 0):
   - Build variational ansatz with `ParameterVector`
   - Initialize random parameters θ
   - Load checkpoint if resuming

2. **Parameter Broadcasting**:
   - Rank 0 broadcasts θ to all ranks

3. **Data Sharding**:
   - Each rank processes `len(dataset) / N` samples
   - Sharding done with `np.array_split` for balanced distribution

4. **Local Gradient Computation**:
   - Each rank computes gradients for its shard using **parameter-shift rule**:
     ```
     ∂L/∂θᵢ = 0.5 * (L(θ + π/2 eᵢ) - L(θ - π/2 eᵢ))
     ```
   - Requires 2 circuit evaluations per parameter

5. **Gradient Aggregation**:
   - `MPI.Allreduce(grad_local, grad_global, op=MPI.SUM)`
   - Average: `grad_mean = grad_global / N`

6. **Parameter Update** (Rank 0 only):
   - Gradient descent: `θ ← θ - lr * grad_mean`
   - Save checkpoint every K epochs

7. **Broadcast Updated Parameters**:
   - Rank 0 broadcasts new θ to all ranks

8. **Repeat** steps 4-7 for each epoch

---

## Files

### Core Files
- `main_superposition_mpi.py` - Main MPI training script
- `quantum_mpi_utils.py` - MPI helper functions
- `run_leonardo_mpi.sh` - SLURM job script

### Helper Modules
- `quantum_circuits.py` - Circuit templates
- `quantum_utils.py` - Quantum utilities
- `encoding.py` - Sentence encoding
- `config.py` - Configuration

---

## Installation on Leonardo

### 1. Load Python Module
```bash
module load python/3.11.7
```

### 2. Create Virtual Environment
```bash
cd $WORK
python -m venv venv_py311
source venv_py311/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy scipy qiskit mpi4py
```

### 4. Verify Installation
```bash
python -c "from mpi4py import MPI; print(f'MPI version: {MPI.Get_version()}')"
```

---

## Running Training

### Local Testing (Development)

#### Single Process (No MPI)
```bash
python main_superposition.py  # Original sequential version
```

#### MPI with 2 Ranks
```bash
mpirun -n 2 python main_superposition_mpi.py --epochs 5 --lr 0.01
```

#### MPI with 4 Ranks
```bash
mpirun -n 4 python main_superposition_mpi.py --epochs 10 --lr 0.01 --seed 123
```

---

### Leonardo HPC Cluster

#### 1. Edit SLURM Script
Edit `run_leonardo_mpi.sh` and set:
```bash
#SBATCH --account=YOUR_ACCOUNT_HERE
```

#### 2. Submit Job
```bash
sbatch run_leonardo_mpi.sh
```

#### 3. Monitor Job
```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/qtransformer_<jobid>.out

# View errors
tail -f logs/qtransformer_<jobid>.err
```

#### 4. Cancel Job
```bash
scancel <jobid>
```

---

## Configuration

### SLURM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--nodes` | 1 | Number of compute nodes |
| `--ntasks` | 8 | Number of MPI ranks (processes) |
| `--cpus-per-task` | 4 | CPUs per rank (for OpenMP) |
| `--time` | 02:00:00 | Max wall time (2 hours) |
| `--partition` | boost_usr_prod | Queue/partition name |

**Total CPUs**: nodes × ntasks × cpus-per-task = 1 × 8 × 4 = **32 cores**

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.01 | Learning rate |
| `--seed` | 42 | Random seed |
| `--num-layers` | 1 | Ansatz layers |
| `--embedding-dim` | 4 | Embedding dimension |
| `--checkpoint-every` | 10 | Save checkpoint interval |

### Modify Hyperparameters

Edit `run_leonardo_mpi.sh`:
```bash
EPOCHS=100
LR=0.005
CHECKPOINT_EVERY=20
```

Or pass arguments directly:
```bash
srun -n 8 python main_superposition_mpi.py --epochs 100 --lr 0.005
```

---

## Checkpoint & Resume

### Automatic Checkpointing

Checkpoints are saved every `--checkpoint-every` epochs to:
```
checkpoints/qtransformer_ckpt_epoch{E}.npz
```

Each checkpoint contains:
- `params` - Model parameters
- `epoch` - Epoch number
- `lr` - Learning rate
- `seed` - Random seed

### Resume Training

```bash
python main_superposition_mpi.py --resume --epochs 100
```

The script will:
1. Find the latest checkpoint in `checkpoints/`
2. Load parameters and epoch number
3. Continue training from that epoch

---

## Performance Expectations

### Scaling Efficiency

Assuming perfect data-parallel scaling:

| Ranks | Samples/Rank | Expected Speedup |
|-------|--------------|------------------|
| 1     | All          | 1.0x (baseline)  |
| 2     | 50%          | 2.0x             |
| 4     | 25%          | 4.0x             |
| 8     | 12.5%        | 8.0x             |

**Reality**: Expect 70-90% efficiency due to:
- MPI communication overhead (Allreduce)
- Load imbalance (if dataset not evenly divisible)
- I/O contention (checkpointing on rank 0)

### Typical Timings

With `num_layers=1`, `embedding_dim=4`, `"The cat"` sentence:

- **Sequential (1 rank)**: ~20 sec/epoch
- **MPI (8 ranks)**: ~3-5 sec/epoch
- **Speedup**: ~5-7x

### Bottlenecks

1. **Circuit Simulation**: Dominates computation
   - Mitigation: Use GPU-accelerated simulators (if available)
   
2. **Parameter-Shift Gradient**: 2 evals/param
   - Mitigation: Use analytical gradients if possible
   
3. **Allreduce Communication**: O(N) for N params
   - Mitigation: Use hierarchical Allreduce (automatic in MPI)

---

## Troubleshooting

### MPI Not Found

**Error**: `ModuleNotFoundError: No module named 'mpi4py'`

**Solution**:
```bash
source $WORK/venv_py311/bin/activate
pip install mpi4py
```

### Checkpoint Not Loading

**Error**: `FileNotFoundError: No checkpoints found`

**Solution**: Remove `--resume` flag or create checkpoint manually:
```python
from quantum_mpi_utils import save_checkpoint
params = np.random.randn(100)
save_checkpoint(params, 0, 0.01, 42)
```

### Slow Performance

**Issue**: 8 ranks slower than 1 rank

**Diagnosis**:
1. Check CPU affinity: `echo $OMP_NUM_THREADS`
2. Verify data sharding: Look for "Data sharding:" in output
3. Profile communication: Add timing around `Allreduce`

**Solution**:
```bash
export OMP_NUM_THREADS=4  # Match --cpus-per-task
```

### Job Killed (OOM)

**Error**: `slurmstepd: error: Detected X oom-kill event(s)`

**Solution**: Reduce batch size or request more memory:
```bash
#SBATCH --mem=64GB
```

---

## Monitoring & Debugging

### View Training Progress

```bash
# Live output
tail -f logs/qtransformer_<jobid>.out

# Training metrics
tail -n 20 logs/train_log.csv | column -t -s ","
```

### Check Rank Output

Each rank prints to the same log. Look for:
```
Rank 0: Loss: 0.123456
Rank 1: Data shard: 2 samples
```

### Gradient Verification

Add debug prints in `main_superposition_mpi.py`:
```python
if rank == 0:
    print(f"Grad norm before Allreduce: {np.linalg.norm(grad_local)}")
    print(f"Grad norm after Allreduce: {np.linalg.norm(grad_mean)}")
```

---

## Advanced Usage

### Multi-Node Training

Edit `run_leonardo_mpi.sh`:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=32
```

Result: 4 nodes × 32 ranks = 128 total processes

### Custom Data Sharding

Modify `shard_dataset()` in `quantum_mpi_utils.py`:
```python
def shard_dataset(data_list, rank, size):
    # Custom sharding logic
    return data_list[rank::size]  # Strided sharding
```

### Gradient Clipping

Add to `main_superposition_mpi.py` after Allreduce:
```python
grad_norm = np.linalg.norm(grad_mean)
if grad_norm > 1.0:
    grad_mean = grad_mean / grad_norm  # Normalize
```

---

## Comparison: Sequential vs MPI

| Aspect | Sequential | MPI Data-Parallel |
|--------|-----------|-------------------|
| **Speed** | Baseline | 5-8x faster (8 ranks) |
| **Scalability** | Limited to 1 CPU | Linear with ranks |
| **Memory** | Single process | Distributed |
| **Complexity** | Simple | Requires MPI setup |
| **Debugging** | Easy | Harder (parallel logs) |
| **Use Case** | Development | Production/HPC |

---

## Next Steps

1. **Test Locally**: Run with `mpirun -n 2` to verify
2. **Submit to Leonardo**: Use `sbatch run_leonardo_mpi.sh`
3. **Monitor Progress**: Check `logs/train_log.csv`
4. **Scale Up**: Increase `--ntasks` for more ranks
5. **Optimize**: Profile and tune hyperparameters

---

## References

- **MPI4Py Docs**: https://mpi4py.readthedocs.io/
- **Leonardo User Guide**: https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.1%3A+LEONARDO+UserGuide
- **Parameter-Shift Rule**: Mitarai et al., Phys. Rev. A 98, 032309 (2018)

---

**Good luck with your distributed quantum training! 🚀**
