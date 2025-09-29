# Quantum Superposition Language Model

A quantum machine learning project that uses quantum superposition states for next-word prediction in natural language processing.

## Project Structure

```
├── mainSuperposition.py          # Main script - entry point
├── config.py                     # Configuration settings
├── encoding.py                   # Word embedding and encoding utilities
├── layer.py                      # Quantum ansatz builder
├── quantum_utils.py              # Quantum utility functions
├── quantum_circuits.py           # Quantum circuit creation functions
├── optimization.py               # Parameter optimization algorithms
├── visualization.py              # Plotting and file I/O utilities
└── example_hamiltonian_evolution.py  # Example Hamiltonian evolution (standalone)
```

## Core Components

### 1. **mainSuperposition.py**
- Main execution script
- Orchestrates training and evaluation phases
- Processes sentences and creates quantum states

### 2. **encoding.py**
- Handles word embeddings using Word2Vec
- Applies positional encoding
- Normalizes embeddings to quantum state vectors

### 3. **quantum_circuits.py**
- Creates quantum circuits for different word counts (2, 4, 8, 16 words)
- Implements controlled unitary operations
- Calculates loss from quantum state probabilities

### 4. **optimization.py**
- Multi-phase optimization using Powell and L-BFGS-B algorithms
- Handles parameter saving and loss tracking
- Supports experimental circuits with additional parameters

### 5. **layer.py**
- Defines the `AnsatzBuilder` class for variational quantum circuits
- Creates parameterized quantum gates (RX, RY, RZ, CNOT)
- Supports multi-layer ansatz construction

### 6. **quantum_utils.py**
- Utility functions for quantum operations
- Unitary matrix generation from state vectors
- Parameter management and angle wrapping

### 7. **visualization.py**
- Loss plotting and visualization
- Parameter and results saving/loading
- Dataset utilities for PTB corpus

### 8. **config.py**
- Centralized configuration management
- Optimization settings and hyperparameters
- Default sentences and training data

## Key Features

- **Modular Architecture**: Clean separation of concerns
- **Multiple Circuit Types**: Support for 2, 4, 8, and 16-word processing
- **Advanced Optimization**: Multi-algorithm optimization with early stopping
- **Comprehensive Logging**: Loss tracking and visualization
- **Parameter Persistence**: Save/load trained parameters
- **Configuration Management**: Centralized settings

## Usage

```python
python mainSuperposition.py
```

## Dependencies

- qiskit
- numpy
- matplotlib
- scipy
- gensim
- datasets
- PIL

## Configuration

Edit `config.py` to modify:
- Number of optimization iterations
- Number of quantum layers
- Embedding dimensions
- Training sentences
- File paths

## Output Files

- `params_best.json`: Optimized parameters
- `loss_results.txt`: Loss values per iteration
- `loss_plot_Nlayers.png`: Loss evolution plots