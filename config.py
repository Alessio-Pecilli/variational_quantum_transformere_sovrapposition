"""
Configuration settings for quantum optimization experiments.
"""

# Optimization settings
OPTIMIZATION_CONFIG = {
    'num_iterations': 150,     # almeno 100–200 per un training serio
    'num_layers': 5,
    'max_hours': 25,            # tempo limite in ore
    'embedding_dim': 16,  # Aumentato da 4 a 16 per vocabolario PTB (4,148 parole) - bilanciato
    'num_qubits': 4,
    'opt_maxiter': 300,        # più iterazioni interne per stabilità
    'opt_maxfev': 400,         # più valutazioni loss
    'epochs': 10,
    'learning_rate':  0.001 ,  # RIDOTTO: da 0.15 a 0.001 per stabilità
    'save_frequency': 50,
    'log_frequency': 60,
    'early_stop_threshold': 0.05,
    'numerical_epsilon': 1e-12
}

# Optimization algorithm settings
OPTIMIZER_CONFIG = {
    'powell': {
        'maxiter': 400,
        'maxfev': 60,
        'xtol': 1e-4,
        'ftol': 1e-4
    },
    'lbfgs': {
        'maxiter': 1000,
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
    "The quick brown",
    "every day is great right",
    "The quick brown fox jumps over the lazy dog",
    "come and play with us today in the sunny beautiful garden now please lets go outside together", 
]

# Training sentences
TRAINING_SENTENCES = [
    "The cat flies",  
    "The quick brown fox jumps",
    "Every morning birds sing songs",
    "Children happily play in parks",
    "Beautiful flowers bloom in spring",
    "Smart students learn new things",
    "Cats sleep quietly on sofas",
    "Dogs bark loudly at strangers"
]
