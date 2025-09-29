"""
Configuration settings for quantum optimization experiments.
"""

# Optimization settings
OPTIMIZATION_CONFIG = {
    'num_iterations': 2,
    'num_layers': 2,
    'max_hours': 100,
    'embedding_dim': 4,
    'n_qubits': 2,
    'save_frequency': 50,  # Save parameters every N evaluations
    'log_frequency': 60,   # Log progress every N seconds
    'early_stop_threshold': 0.1,
    'numerical_epsilon': 1e-12
}

# Optimization algorithm settings
OPTIMIZER_CONFIG = {
    'powell': {
        'maxiter': 40,
        'maxfev': 60,
        'xtol': 1e-4,
        'ftol': 1e-4
    },
    'lbfgs': {
        'maxiter': 100,
        'maxfun': 100,
        'ftol': 1e-10,
        'maxcor': 20
    },
    'experimental_f': {
        'maxiter': 30,
        'maxfev': 50
    }
}

# Circuit settings
CIRCUIT_CONFIG = {
    'shots': 1024 * 2,
    'max_supported_words': 16
}

# File settings
FILE_CONFIG = {
    'params_filename': 'params_best.json',
    'loss_values_filename': 'loss_results.txt',
    'loss_plot_base': 'loss_plot',
    'circuit_image': 'quantum_attention_circuit.png'
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (12, 6),
    'dpi': 300,
    'grid_alpha': 0.3,
    'colors': {
        'average': 'blue',
        'best': 'green',
        'worst': 'red'
    }
}

# Dataset settings
DATASET_CONFIG = {
    'default_split': 'train',
    'max_sentences': 100,
    'min_words_per_sentence': 4,
    'dataset_name': 'ptb_text_only'
}

# Default test sentences
DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "I am great",
    "every day is",
    "come and play"
]

# Training sentences
TRAINING_SENTENCES = [
    "The quick brown fox jumps",
    "Every morning birds sing songs",
    "Children happily play in parks",
    "Beautiful flowers bloom in spring",
    "Smart students learn new things",
    "Cats sleep quietly on sofas",
    "Dogs bark loudly at strangers"
]