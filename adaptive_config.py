"""
Adaptive Configuration System per Quantum Circuits.
Calcola automaticamente configurazioni ottimali per embedding dimensions variabili.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class QuantumConfig:
    """Configurazione completa per quantum circuits."""
    
    # Dimensioni base
    embedding_dim: int
    vocab_size: int
    max_sentence_length: int
    
    # Quantum parameters
    n_target_qubits: int
    n_control_qubits: int
    n_total_qubits: int
    
    # Training parameters
    learning_rate: float
    num_layers: int
    batch_size: int
    
    # Ansatz parameters
    ansatz_dim: int
    n_v_params: int
    n_k_params: int
    total_quantum_params: int
    
    # Memory and performance
    estimated_memory_mb: float
    estimated_gates_per_sentence: int
    complexity_score: float
    is_feasible: bool
    
    # HPC optimization
    recommended_workers: int
    parallel_sentences: int
    
    def to_dict(self) -> Dict:
        """Converte in dizionario per serializzazione."""
        return {
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "max_sentence_length": self.max_sentence_length,
            "n_target_qubits": self.n_target_qubits,
            "n_control_qubits": self.n_control_qubits,
            "n_total_qubits": self.n_total_qubits,
            "learning_rate": self.learning_rate,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "ansatz_dim": self.ansatz_dim,
            "n_v_params": self.n_v_params,
            "n_k_params": self.n_k_params,
            "total_quantum_params": self.total_quantum_params,
            "estimated_memory_mb": self.estimated_memory_mb,
            "estimated_gates_per_sentence": self.estimated_gates_per_sentence,
            "complexity_score": self.complexity_score,
            "is_feasible": self.is_feasible,
            "recommended_workers": self.recommended_workers,
            "parallel_sentences": self.parallel_sentences
        }


class AdaptiveConfigCalculator:
    """
    Calcola configurazioni ottimali per quantum circuits con embedding dimensions variabili.
    """
    
    def __init__(self):
        # Limiti di sistema
        self.max_qubits = 15  # Limite pratico per simulazione
        self.max_memory_mb = 8000  # 8GB limite memoria
        self.max_gates_per_sentence = 10000  # Limite gates per performance
        
        # Configurazioni predefinite per diversi scenari
        self.preset_configs = {
            "small": {"max_qubits": 8, "max_memory_mb": 2000},
            "medium": {"max_qubits": 12, "max_memory_mb": 4000},
            "large": {"max_qubits": 15, "max_memory_mb": 8000},
            "hpc": {"max_qubits": 18, "max_memory_mb": 16000}
        }
    
    def calculate_optimal_config(self, vocab_size: int, max_sentence_length: int,
                               target_embedding_dim: Optional[int] = None,
                               preset: str = "medium",
                               hpc_cores: int = 8) -> QuantumConfig:
        """
        Calcola configurazione ottimale per parametri dati.
        
        Args:
            vocab_size: Dimensione vocabolario
            max_sentence_length: Lunghezza massima frasi
            target_embedding_dim: Embedding dimension desiderata (None = auto)
            preset: Preset di sistema ("small", "medium", "large", "hpc")
            hpc_cores: Numero core HPC disponibili
            
        Returns:
            QuantumConfig ottimizzata
        """
        
        # Applica preset limits
        if preset in self.preset_configs:
            limits = self.preset_configs[preset]
            max_qubits = limits["max_qubits"]
            max_memory = limits["max_memory_mb"]
        else:
            max_qubits = self.max_qubits
            max_memory = self.max_memory_mb
        
        # Determina embedding dimension ottimale
        if target_embedding_dim is None:
            embedding_dim = self._calculate_optimal_embedding_dim(vocab_size, max_sentence_length, max_qubits)
        else:
            embedding_dim = self._validate_embedding_dim(target_embedding_dim, max_qubits)
        
        # Calcola parametri quantum
        n_target_qubits = int(math.log2(embedding_dim))
        n_control_qubits = int(math.ceil(math.log2(max_sentence_length)))
        n_total_qubits = n_target_qubits + n_control_qubits
        
        # Verifica feasibility
        if n_total_qubits > max_qubits:
            # Riduce embedding o sentence length
            embedding_dim, max_sentence_length = self._reduce_to_fit(
                embedding_dim, max_sentence_length, max_qubits
            )
            n_target_qubits = int(math.log2(embedding_dim))
            n_control_qubits = int(math.ceil(math.log2(max_sentence_length)))
            n_total_qubits = n_target_qubits + n_control_qubits
        
        # Calcola parametri ansatz
        ansatz_dim = max(1, n_target_qubits // 2)
        n_v_params = ansatz_dim * 3  # Parametri per ansatz V (semplificato)
        n_k_params = ansatz_dim * 3  # Parametri per ansatz K
        
        # Parametri embedding
        embedding_params = vocab_size * embedding_dim
        
        # Parametri quantum per unitarie (stima)
        quantum_unitaries_params = max_sentence_length * embedding_dim * embedding_dim * 2  # U e Z
        
        total_quantum_params = embedding_params + quantum_unitaries_params + n_v_params + n_k_params
        
        # Stima memoria
        statevector_memory = (2**n_total_qubits * 16) / (1024**2)  # Complex128 in MB
        params_memory = (total_quantum_params * 8) / (1024**2)  # Float64 in MB
        estimated_memory_mb = statevector_memory + params_memory
        
        # Stima gates
        gates_per_unitary = embedding_dim * 2  # Stima semplificata
        estimated_gates_per_sentence = max_sentence_length * gates_per_unitary + n_total_qubits * 2
        
        # Complessità score
        complexity_score = n_total_qubits * max_sentence_length * math.log2(embedding_dim)
        
        # Feasibility check
        is_feasible = (
            n_total_qubits <= max_qubits and
            estimated_memory_mb <= max_memory and
            estimated_gates_per_sentence <= self.max_gates_per_sentence
        )
        
        # Ottimizzazioni training
        learning_rate = self._calculate_optimal_learning_rate(embedding_dim, vocab_size)
        num_layers = self._calculate_optimal_layers(n_target_qubits)
        batch_size = self._calculate_optimal_batch_size(estimated_memory_mb, max_sentence_length)
        
        # HPC optimization
        recommended_workers = min(hpc_cores, max(1, hpc_cores // 2))
        parallel_sentences = max(1, batch_size // recommended_workers)
        
        return QuantumConfig(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_sentence_length=max_sentence_length,
            n_target_qubits=n_target_qubits,
            n_control_qubits=n_control_qubits,
            n_total_qubits=n_total_qubits,
            learning_rate=learning_rate,
            num_layers=num_layers,
            batch_size=batch_size,
            ansatz_dim=ansatz_dim,
            n_v_params=n_v_params,
            n_k_params=n_k_params,
            total_quantum_params=total_quantum_params,
            estimated_memory_mb=estimated_memory_mb,
            estimated_gates_per_sentence=estimated_gates_per_sentence,
            complexity_score=complexity_score,
            is_feasible=is_feasible,
            recommended_workers=recommended_workers,
            parallel_sentences=parallel_sentences
        )
    
    def _calculate_optimal_embedding_dim(self, vocab_size: int, max_sentence_length: int, 
                                       max_qubits: int) -> int:
        """Calcola embedding dimension ottimale basata su vocab size e constraint."""
        
        # Constraint da max qubits disponibili
        max_control_qubits = int(math.ceil(math.log2(max_sentence_length)))
        max_target_qubits = max_qubits - max_control_qubits
        max_embedding = 2**max_target_qubits if max_target_qubits > 0 else 2
        
        # Euristica basata su vocab size
        if vocab_size <= 1000:
            optimal_embedding = 4    # 2 qubits
        elif vocab_size <= 5000:
            optimal_embedding = 16   # 4 qubits
        elif vocab_size <= 20000:
            optimal_embedding = 64   # 6 qubits
        else:
            optimal_embedding = 256  # 8 qubits
        
        # Applica constraint
        return min(optimal_embedding, max_embedding)
    
    def _validate_embedding_dim(self, embedding_dim: int, max_qubits: int) -> int:
        """Valida e aggiusta embedding dimension per constraint."""
        
        # Verifica che sia potenza di 2
        if embedding_dim <= 0 or (embedding_dim & (embedding_dim - 1)) != 0:
            # Trova potenza di 2 più vicina
            embedding_dim = 2**int(math.ceil(math.log2(embedding_dim)))
        
        # Verifica constraint qubits
        n_target_qubits = int(math.log2(embedding_dim))
        if n_target_qubits > max_qubits - 1:  # -1 per almeno 1 control qubit
            embedding_dim = 2**(max_qubits - 1)
        
        return max(4, embedding_dim)  # Minimo 4 (2 qubits)
    
    def _reduce_to_fit(self, embedding_dim: int, max_sentence_length: int, 
                      max_qubits: int) -> Tuple[int, int]:
        """Riduce parametri per rientrare nei limiti di qubits."""
        
        # Prova a ridurre embedding dim prima
        while embedding_dim >= 4:
            n_target = int(math.log2(embedding_dim))
            n_control = int(math.ceil(math.log2(max_sentence_length)))
            
            if n_target + n_control <= max_qubits:
                return embedding_dim, max_sentence_length
            
            embedding_dim = embedding_dim // 2
        
        # Se ancora troppo, riduce sentence length
        embedding_dim = 4  # Minimo
        n_target = 2
        
        while max_sentence_length > 3:
            n_control = int(math.ceil(math.log2(max_sentence_length)))
            
            if n_target + n_control <= max_qubits:
                return embedding_dim, max_sentence_length
            
            max_sentence_length = max_sentence_length // 2
        
        return embedding_dim, max(3, max_sentence_length)
    
    def _calculate_optimal_learning_rate(self, embedding_dim: int, vocab_size: int) -> float:
        """Calcola learning rate ottimale basato su embedding e vocab size."""
        
        # Base learning rate
        base_lr = 0.001
        
        # Adjustment per embedding dimension
        embedding_factor = math.sqrt(embedding_dim / 16.0)  # Normalizzato a 16
        
        # Adjustment per vocab size
        vocab_factor = math.sqrt(10000.0 / vocab_size)  # Normalizzato a 10K
        
        # Learning rate adattato
        adapted_lr = base_lr * embedding_factor * vocab_factor
        
        # Clamp a range ragionevole
        return max(0.0001, min(0.01, adapted_lr))
    
    def _calculate_optimal_layers(self, n_target_qubits: int) -> int:
        """Calcola numero ottimale di layer basato su qubits target."""
        
        if n_target_qubits <= 2:
            return 2
        elif n_target_qubits <= 4:
            return 3
        elif n_target_qubits <= 6:
            return 4
        else:
            return 5
    
    def _calculate_optimal_batch_size(self, estimated_memory_mb: float, 
                                    max_sentence_length: int) -> int:
        """Calcola batch size ottimale basato su memoria e sentence length."""
        
        # Base batch size
        if max_sentence_length <= 5:
            base_batch = 64
        elif max_sentence_length <= 10:
            base_batch = 32
        elif max_sentence_length <= 20:
            base_batch = 16
        else:
            base_batch = 8
        
        # Adjustment per memoria
        if estimated_memory_mb > 4000:
            memory_factor = 0.5
        elif estimated_memory_mb > 2000:
            memory_factor = 0.75
        else:
            memory_factor = 1.0
        
        adjusted_batch = int(base_batch * memory_factor)
        
        return max(1, min(128, adjusted_batch))
    
    def compare_configurations(self, configs: List[QuantumConfig]) -> Dict:
        """Compara multiple configurazioni e suggerisce la migliore."""
        
        if not configs:
            return {"error": "No configurations provided"}
        
        # Filtra solo configurazioni feasible
        feasible_configs = [c for c in configs if c.is_feasible]
        
        if not feasible_configs:
            return {
                "error": "No feasible configurations",
                "all_configs": [c.to_dict() for c in configs]
            }
        
        # Scoring system per ranking
        def score_config(config: QuantumConfig) -> float:
            score = 0.0
            
            # Premia embedding dimension più alta (capacità)
            score += math.log2(config.embedding_dim) * 10
            
            # Premia supporto sentence length più alta
            score += math.log2(config.max_sentence_length) * 5
            
            # Penalizza complessità eccessiva
            score -= config.complexity_score * 0.001
            
            # Penalizza uso memoria eccessivo
            memory_penalty = max(0, config.estimated_memory_mb - 2000) * 0.01
            score -= memory_penalty
            
            # Premia parallelizzazione migliore
            score += config.recommended_workers * 2
            
            return score
        
        # Rank configurazioni
        scored_configs = [(score_config(c), c) for c in feasible_configs]
        scored_configs.sort(reverse=True)  # Best score first
        
        best_config = scored_configs[0][1]
        
        comparison = {
            "best_config": best_config.to_dict(),
            "best_score": scored_configs[0][0],
            "all_feasible": len(feasible_configs),
            "total_compared": len(configs),
            "ranking": [
                {
                    "score": score,
                    "embedding_dim": config.embedding_dim,
                    "max_sentence_length": config.max_sentence_length,
                    "complexity": config.complexity_score,
                    "memory_mb": config.estimated_memory_mb
                }
                for score, config in scored_configs[:5]  # Top 5
            ]
        }
        
        return comparison
    
    def generate_config_recommendations(self, vocab_size: int, 
                                      sentence_lengths: List[int],
                                      available_memory_gb: float = 8.0,
                                      hpc_cores: int = 8) -> Dict:
        """
        Genera raccomandazioni complete per diversi scenari.
        
        Args:
            vocab_size: Dimensione vocabolario
            sentence_lengths: Lista lunghezze frasi supportate
            available_memory_gb: Memoria disponibile in GB
            hpc_cores: Core HPC disponibili
            
        Returns:
            Dict con raccomandazioni per diversi scenari
        """
        
        max_sentence_length = max(sentence_lengths)
        
        # Aggiorna limiti basati su memoria disponibile
        memory_mb = available_memory_gb * 1024
        
        # Genera configurazioni per diversi embedding dimensions
        test_embeddings = [4, 16, 64, 256]
        test_presets = ["small", "medium", "large"]
        
        all_configs = []
        
        for embedding_dim in test_embeddings:
            for preset in test_presets:
                try:
                    config = self.calculate_optimal_config(
                        vocab_size=vocab_size,
                        max_sentence_length=max_sentence_length,
                        target_embedding_dim=embedding_dim,
                        preset=preset,
                        hpc_cores=hpc_cores
                    )
                    
                    # Verifica memoria disponibile
                    if config.estimated_memory_mb <= memory_mb:
                        all_configs.append(config)
                        
                except Exception as e:
                    print(f"⚠️ Config failed: {embedding_dim}D {preset} - {e}")
        
        # Compara configurazioni
        comparison = self.compare_configurations(all_configs)
        
        # Aggiungi raccomandazioni specifiche
        recommendations = {
            "vocab_size": vocab_size,
            "sentence_lengths": sentence_lengths,
            "max_sentence_length": max_sentence_length,
            "available_memory_gb": available_memory_gb,
            "hpc_cores": hpc_cores,
            **comparison
        }
        
        # Raccomandazioni per diversi use case
        if comparison.get("best_config"):
            best = comparison["best_config"]
            
            recommendations["use_cases"] = {
                "development": {
                    "embedding_dim": min(16, best["embedding_dim"]),
                    "description": "Fast iteration, reduced complexity"
                },
                "production": {
                    "embedding_dim": best["embedding_dim"],
                    "description": "Optimal performance and capacity"
                },
                "research": {
                    "embedding_dim": min(64, best["embedding_dim"] * 2),
                    "description": "Maximum capacity for experiments"
                }
            }
        
        return recommendations


def save_config_to_file(config: QuantumConfig, filename: str):
    """Salva configurazione su file JSON."""
    
    with open(filename, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"💾 Configurazione salvata: {filename}")


def load_config_from_file(filename: str) -> QuantumConfig:
    """Carica configurazione da file JSON."""
    
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    return QuantumConfig(**config_dict)


if __name__ == "__main__":
    # Test del sistema di configurazione adattiva
    
    print("🔧 ADAPTIVE QUANTUM CONFIGURATION SYSTEM")
    print("="*60)
    
    calculator = AdaptiveConfigCalculator()
    
    # Test per PTB dataset
    print("\n📚 Penn Treebank Dataset Configuration:")
    
    ptb_recommendations = calculator.generate_config_recommendations(
        vocab_size=4148,
        sentence_lengths=[3, 5, 9, 17],
        available_memory_gb=8.0,
        hpc_cores=16
    )
    
    print(f"Best config for PTB:")
    best_config = ptb_recommendations.get("best_config")
    if best_config:
        print(f"   Embedding: {best_config['embedding_dim']}D ({best_config['n_target_qubits']} qubits)")
        print(f"   Max sentence: {best_config['max_sentence_length']} words ({best_config['n_control_qubits']} qubits)")
        print(f"   Total qubits: {best_config['n_total_qubits']}")
        print(f"   Memory: {best_config['estimated_memory_mb']:.1f} MB")
        print(f"   Learning rate: {best_config['learning_rate']}")
        print(f"   Batch size: {best_config['batch_size']}")
        print(f"   HPC workers: {best_config['recommended_workers']}")
        print(f"   Feasible: {best_config['is_feasible']}")
        
        # Salva config migliore
        best_quantum_config = QuantumConfig(**best_config)
        save_config_to_file(best_quantum_config, "ptb_optimal_config.json")
    
    # Test configurazioni diverse
    print(f"\n🔬 Testing different scenarios:")
    
    test_scenarios = [
        (1000, [3, 5], "Small vocab"),
        (10000, [5, 9, 17], "Medium vocab"),
        (50000, [3, 5, 9, 17, 25], "Large vocab")
    ]
    
    for vocab_size, sentence_lengths, description in test_scenarios:
        print(f"\n{description} ({vocab_size:,} words, max {max(sentence_lengths)} length):")
        
        config = calculator.calculate_optimal_config(
            vocab_size=vocab_size,
            max_sentence_length=max(sentence_lengths),
            hpc_cores=8
        )
        
        print(f"   Suggested: {config.embedding_dim}D embedding, {config.n_total_qubits} total qubits")
        print(f"   Memory: {config.estimated_memory_mb:.1f} MB, Feasible: {config.is_feasible}")
    
    print("\n✅ Configuration system test completed!")