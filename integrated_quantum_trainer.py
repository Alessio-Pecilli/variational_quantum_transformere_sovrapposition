"""
Trainer Integrato con Sistema di Reporting Completo.
Integra PTB dataset, HPC parallelization e comprehensive reporting.
"""

import numpy as np
import time
import threading
import queue
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import nostri moduli
from ptb_dataset import PTBDatasetLoader
from hpc_max_parallel import HPC_ResourceDetector, HPC_MaxParallelOptimizer
from comprehensive_reporter import ComprehensiveReporter, TrainingMetrics, create_training_metrics
from optimization import AdamOptimizer
from quantum_circuits import create_quantum_circuit_for_sentence
from config import OPTIMIZATION_CONFIG


class IntegratedQuantumTrainer:
    """
    Trainer completo che integra tutto: PTB dataset, HPC parallelization e reporting.
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "training_output"):
        self.config = OPTIMIZATION_CONFIG if config_path is None else self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.ptb_loader = PTBDatasetLoader(cache_dir=self.output_dir / "ptb_cache")
        self.hpc_detector = HPC_ResourceDetector()
        self.hpc_optimizer = HPC_MaxParallelOptimizer()
        self.reporter = ComprehensiveReporter(self.output_dir / "reports")
        
        # Training state
        self.run_id = f"quantum_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.end_time = None
        
        # Metrics collection
        self.loss_history = []
        self.param_history = []
        self.sentence_results = []
        self.hpc_stats = {}
        self.errors_log = []
        self.memory_warnings = 0
        self.timeout_issues = 0
        
        # Performance monitoring
        self.performance_monitor = None
        self.stop_monitoring = threading.Event()
        
        self.logger.info(f"🚀 Integrated Quantum Trainer initialized - Run ID: {self.run_id}")
    
    def _setup_logging(self):
        """Setup logging completo."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Logger principale
        self.logger = logging.getLogger(f"QuantumTrainer_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"{self.run_id}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica configurazione da file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_complete_training(self, train_sentences: int = 700, test_sentences: int = 300,
                            max_iterations: int = 200, convergence_threshold: float = 1e-6) -> str:
        """
        Esegue training completo con reporting automatico.
        
        Returns:
            Path del report HTML generato
        """
        
        self.start_time = datetime.now()
        self.logger.info(f"🎯 Avvio training completo - Target: {train_sentences} train + {test_sentences} test")
        
        try:
            # 1. Setup HPC e rilevamento risorse
            self._setup_hpc_environment()
            
            # 2. Caricamento e preparazione dataset
            dataset_info = self._prepare_dataset(train_sentences, test_sentences)
            
            # 3. Inizializza parametri quantici
            initial_params = self._initialize_quantum_parameters(dataset_info)
            
            # 4. Avvia monitoring performance
            self._start_performance_monitoring()
            
            # 5. Training principale
            final_params, optimization_results = self._run_training_loop(
                dataset_info, initial_params, max_iterations, convergence_threshold
            )
            
            # 6. Valutazione finale
            evaluation_results = self._run_final_evaluation(dataset_info, final_params)
            
            # 7. Stop monitoring
            self._stop_performance_monitoring()
            
            # 8. Genera report completo
            report_path = self._generate_final_report(dataset_info, optimization_results, evaluation_results)
            
            self.end_time = datetime.now()
            self.logger.info(f"✅ Training completo terminato in {(self.end_time - self.start_time).total_seconds()/3600:.2f} ore")
            
            return report_path
            
        except Exception as e:
            self.end_time = datetime.now()
            self.logger.error(f"❌ Errore durante training: {str(e)}")
            self.errors_log.append({"error": str(e), "traceback": traceback.format_exc(), "timestamp": datetime.now().isoformat()})
            
            # Genera report anche in caso di errore
            if hasattr(self, 'dataset_info'):
                return self._generate_error_report(str(e))
            else:
                raise
    
    def _setup_hpc_environment(self):
        """Setup ambiente HPC e rilevamento risorse."""
        self.logger.info("🔧 Setup ambiente HPC...")
        
        # Rileva risorse disponibili
        resources = self.hpc_detector.detect_environment()
        self.logger.info(f"Risorse rilevate: {resources}")
        
        # Ottimizza configurazione parallela
        optimization = self.hpc_optimizer.optimize_for_resources(resources)
        self.logger.info(f"Configurazione ottimizzata: {optimization}")
        
        # Salva stats HPC
        self.hpc_stats = {
            "max_workers": optimization["recommended_processes"],
            "cpu_cores_total": resources["cpu_cores"],
            "memory_total_gb": resources["memory_gb"],
            "scheduler_type": resources.get("scheduler", "none"),
            "numa_nodes": resources.get("numa_nodes", 1),
            "avg_workers_utilized": 0.0,  # Sarà aggiornato durante training
            "cpu_cores_used": optimization["recommended_processes"],
            "memory_peak_gb": 0.0  # Sarà aggiornato dal monitor
        }
    
    def _prepare_dataset(self, train_sentences: int, test_sentences: int) -> Dict:
        """Prepara dataset PTB con constraints quantici."""
        self.logger.info(f"📚 Preparazione dataset PTB: {train_sentences} train + {test_sentences} test")
        
        # Carica dataset
        train_data, test_data, vocab_info = self.ptb_loader.load_ptb_for_quantum_training(
            train_size=train_sentences,
            test_size=test_sentences,
            quantum_length_constraints=True
        )
        
        # Analizza distribuzione lunghezze
        length_distribution = {}
        all_sentences = train_data + test_data
        for sentence in all_sentences:
            length = len(sentence)
            length_distribution[length] = length_distribution.get(length, 0) + 1
        
        dataset_info = {
            "train_data": train_data,
            "test_data": test_data,
            "vocab_info": vocab_info,
            "total_sentences": len(all_sentences),
            "train_sentences": len(train_data),
            "test_sentences": len(test_data),
            "vocab_size": len(vocab_info["word_to_idx"]),
            "length_distribution": length_distribution,
            "max_sentence_length": max(len(s) for s in all_sentences),
            "min_sentence_length": min(len(s) for s in all_sentences)
        }
        
        self.dataset_info = dataset_info  # Per error reporting
        
        self.logger.info(f"Dataset preparato: {dataset_info['total_sentences']} frasi, vocabolario: {dataset_info['vocab_size']}")
        self.logger.info(f"Distribuzione lunghezze: {dataset_info['length_distribution']}")
        
        return dataset_info
    
    def _initialize_quantum_parameters(self, dataset_info: Dict) -> np.ndarray:
        """Inizializza parametri quantici basati su dataset."""
        vocab_size = dataset_info["vocab_size"]
        embedding_dim = self.config["embedding_dim"]
        num_qubits = max(2, int(np.ceil(np.log2(dataset_info["max_sentence_length"]))))
        num_layers = self.config.get("num_layers", 3)
        
        # Calcola numero totale parametri
        # Embedding: vocab_size × embedding_dim
        # Quantum circuit: num_qubits × num_layers × gates_per_layer
        embedding_params = vocab_size * embedding_dim
        gates_per_layer = num_qubits * 3  # RY, RZ, CZ gates
        quantum_params = num_qubits * num_layers * gates_per_layer
        
        total_params = embedding_params + quantum_params
        
        # Inizializza parametri
        initial_params = np.random.normal(0, 0.1, total_params)
        
        # Aggiorna config con parametri calcolati
        self.config.update({
            "num_qubits": num_qubits,
            "num_layers": num_layers,
            "total_parameters": total_params,
            "embedding_params": embedding_params,
            "quantum_params": quantum_params
        })
        
        self.logger.info(f"Parametri quantici inizializzati: {total_params} totali ({embedding_params} embedding + {quantum_params} quantum)")
        
        return initial_params
    
    def _start_performance_monitoring(self):
        """Avvia monitoring performance in background."""
        def monitor_performance():
            peak_memory = 0.0
            worker_utilization_samples = []
            
            while not self.stop_monitoring.wait(timeout=5.0):
                try:
                    # Memory monitoring
                    memory_info = psutil.virtual_memory()
                    current_memory_gb = memory_info.used / (1024**3)
                    peak_memory = max(peak_memory, current_memory_gb)
                    
                    # Memory warning check
                    if memory_info.percent > 85:
                        self.memory_warnings += 1
                        self.logger.warning(f"⚠️ High memory usage: {memory_info.percent:.1f}%")
                    
                    # CPU monitoring (worker utilization approssimata)
                    cpu_percent = psutil.cpu_percent(interval=1)
                    estimated_workers = (cpu_percent / 100.0) * self.hpc_stats["max_workers"]
                    worker_utilization_samples.append(estimated_workers)
                    
                except Exception as e:
                    self.logger.warning(f"Performance monitoring error: {e}")
            
            # Update final stats
            self.hpc_stats["memory_peak_gb"] = peak_memory
            if worker_utilization_samples:
                self.hpc_stats["avg_workers_utilized"] = np.mean(worker_utilization_samples)
        
        self.performance_monitor = threading.Thread(target=monitor_performance, daemon=True)
        self.performance_monitor.start()
        self.logger.info("📊 Performance monitoring attivato")
    
    def _stop_performance_monitoring(self):
        """Ferma monitoring performance."""
        if self.performance_monitor and self.performance_monitor.is_alive():
            self.stop_monitoring.set()
            self.performance_monitor.join(timeout=10)
            self.logger.info("📊 Performance monitoring fermato")
    
    def _run_training_loop(self, dataset_info: Dict, initial_params: np.ndarray, 
                          max_iterations: int, convergence_threshold: float) -> Tuple[np.ndarray, Dict]:
        """Training loop principale con ottimizzazione Adam."""
        
        self.logger.info(f"🎯 Avvio training loop: {max_iterations} iterazioni max, threshold: {convergence_threshold}")
        
        # Setup ottimizzatore
        optimizer = AdamOptimizer(
            learning_rate=self.config["learning_rate"],
            beta1=self.config.get("beta1", 0.9),
            beta2=self.config.get("beta2", 0.999)
        )
        
        # Parametri training
        params = initial_params.copy()
        train_data = dataset_info["train_data"]
        vocab_info = dataset_info["vocab_info"]
        
        # Metriche training
        best_loss = float('inf')
        convergence_achieved = False
        iteration_times = []
        loss_variance_samples = []
        gradient_norms = []
        
        # Training loop
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            try:
                # Calcola loss e gradienti su batch sentences
                batch_loss, gradients = self._compute_batch_loss_and_gradients(
                    params, train_data, vocab_info, batch_size=min(32, len(train_data))
                )
                
                # Update parametri
                params = optimizer.update_parameters(params, gradients)
                
                # Logging metriche
                self.loss_history.append(batch_loss)
                self.param_history.append(params.copy())
                
                # Calcola gradient norm per stabilità
                grad_norm = np.linalg.norm(gradients)
                gradient_norms.append(grad_norm)
                
                # Check convergenza
                if batch_loss < best_loss:
                    best_loss = batch_loss
                
                # Check convergenza (ultimi 10% iterazioni)
                if iteration > max_iterations * 0.9:
                    loss_variance_samples.append(batch_loss)
                
                # Check convergenza per gradient norm
                if grad_norm < convergence_threshold:
                    convergence_achieved = True
                    self.logger.info(f"✅ Convergenza raggiunta all'iterazione {iteration+1}")
                    break
                
                # Timing
                iteration_time = time.time() - iteration_start
                iteration_times.append(iteration_time)
                
                # Log periodico
                if (iteration + 1) % 10 == 0:
                    self.logger.info(
                        f"Iter {iteration+1:3d}/{max_iterations}: "
                        f"Loss={batch_loss:.6f}, "
                        f"Best={best_loss:.6f}, "
                        f"GradNorm={grad_norm:.6f}, "
                        f"Time={iteration_time:.2f}s"
                    )
                
            except Exception as e:
                self.errors_log.append({
                    "iteration": iteration,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                self.logger.error(f"Errore iterazione {iteration}: {e}")
                continue
        
        # Calcola risultati finali
        final_loss = self.loss_history[-1] if self.loss_history else float('inf')
        initial_loss = self.loss_history[0] if self.loss_history else 0.0
        loss_improvement_percent = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0
        
        # Stabilità metriche
        loss_variance_last_10_percent = np.var(loss_variance_samples) if loss_variance_samples else 0.0
        gradient_stability_score = 1.0 / (1.0 + np.std(gradient_norms)) if gradient_norms else 0.0
        
        # Performance metriche
        avg_time_per_iteration = np.mean(iteration_times) if iteration_times else 0.0
        total_sentences_processed = len(train_data) * len(self.loss_history)
        total_training_time = sum(iteration_times)
        sentences_per_hour = (total_sentences_processed / (total_training_time / 3600.0)) if total_training_time > 0 else 0.0
        
        optimization_results = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "loss_improvement_percent": loss_improvement_percent,
            "convergence_achieved": convergence_achieved,
            "iterations_total": len(self.loss_history),
            "avg_time_per_iteration": avg_time_per_iteration,
            "sentences_per_hour": sentences_per_hour,
            "final_param_norm": np.linalg.norm(params),
            "final_gradient_norm": gradient_norms[-1] if gradient_norms else 0.0,
            "parameter_evolution": [np.linalg.norm(p) for p in self.param_history],
            "loss_variance_last_10_percent": loss_variance_last_10_percent,
            "gradient_stability_score": gradient_stability_score,
            "errors_encountered": len(self.errors_log),
            "memory_warnings": self.memory_warnings,
            "timeout_issues": self.timeout_issues
        }
        
        self.logger.info(f"🏁 Training completato: {len(self.loss_history)} iterazioni, Loss finale: {final_loss:.6f}")
        
        return params, optimization_results
    
    def _compute_batch_loss_and_gradients(self, params: np.ndarray, sentences: List[List[str]], 
                                         vocab_info: Dict, batch_size: int = 32) -> Tuple[float, np.ndarray]:
        """Calcola loss e gradienti su batch di frasi."""
        
        # Seleziona batch random
        batch_indices = np.random.choice(len(sentences), size=min(batch_size, len(sentences)), replace=False)
        batch_sentences = [sentences[i] for i in batch_indices]
        
        # Processa frasi in parallelo quando possibile
        if len(batch_sentences) >= 4 and self.hpc_stats["max_workers"] > 1:
            return self._compute_batch_parallel(params, batch_sentences, vocab_info)
        else:
            return self._compute_batch_sequential(params, batch_sentences, vocab_info)
    
    def _compute_batch_sequential(self, params: np.ndarray, sentences: List[List[str]], 
                                 vocab_info: Dict) -> Tuple[float, np.ndarray]:
        """Calcola batch sequenzialmente."""
        total_loss = 0.0
        total_gradients = np.zeros_like(params)
        processed_sentences = 0
        
        for sentence in sentences:
            try:
                sentence_start = time.time()
                
                # Calcola loss per questa frase
                loss, gradients = self._compute_sentence_loss_and_gradients(sentence, params, vocab_info)
                
                total_loss += loss
                total_gradients += gradients
                processed_sentences += 1
                
                # Salva risultato frase
                sentence_time = time.time() - sentence_start
                self.sentence_results.append({
                    "sentence_id": len(self.sentence_results),
                    "sentence_length": len(sentence),
                    "loss": loss,
                    "time": sentence_time,
                    "iteration": len(self.loss_history)
                })
                
            except Exception as e:
                self.logger.warning(f"Errore processing frase {sentence[:3]}...: {e}")
                self.errors_log.append({"sentence": sentence, "error": str(e)})
        
        # Media
        avg_loss = total_loss / max(processed_sentences, 1)
        avg_gradients = total_gradients / max(processed_sentences, 1)
        
        return avg_loss, avg_gradients
    
    def _compute_batch_parallel(self, params: np.ndarray, sentences: List[List[str]], 
                               vocab_info: Dict) -> Tuple[float, np.ndarray]:
        """Calcola batch in parallelo."""
        
        max_workers = min(self.hpc_stats["max_workers"], len(sentences))
        
        def process_sentence_wrapper(args):
            sentence, params, vocab_info = args
            sentence_start = time.time()
            try:
                loss, gradients = self._compute_sentence_loss_and_gradients(sentence, params, vocab_info)
                sentence_time = time.time() - sentence_start
                return {
                    "success": True,
                    "loss": loss,
                    "gradients": gradients,
                    "sentence_length": len(sentence),
                    "time": sentence_time
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "sentence": sentence
                }
        
        # Processa in parallelo
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(sentence, params, vocab_info) for sentence in sentences]
            results = list(executor.map(process_sentence_wrapper, tasks))
        
        # Aggrega risultati
        total_loss = 0.0
        total_gradients = np.zeros_like(params)
        processed_sentences = 0
        
        for i, result in enumerate(results):
            if result["success"]:
                total_loss += result["loss"]
                total_gradients += result["gradients"]
                processed_sentences += 1
                
                # Salva risultato
                self.sentence_results.append({
                    "sentence_id": len(self.sentence_results),
                    "sentence_length": result["sentence_length"],
                    "loss": result["loss"],
                    "time": result["time"],
                    "iteration": len(self.loss_history),
                    "parallel": True
                })
            else:
                self.errors_log.append({
                    "sentence": result["sentence"],
                    "error": result["error"],
                    "iteration": len(self.loss_history)
                })
        
        # Media
        avg_loss = total_loss / max(processed_sentences, 1)
        avg_gradients = total_gradients / max(processed_sentences, 1)
        
        return avg_loss, avg_gradients
    
    def _compute_sentence_loss_and_gradients(self, sentence: List[str], params: np.ndarray, 
                                           vocab_info: Dict) -> Tuple[float, np.ndarray]:
        """Calcola loss e gradienti per una frase (simulato per ora)."""
        
        # Questa è una versione semplificata per il testing
        # Nel codice reale useremo i quantum circuits
        
        sentence_length = len(sentence)
        
        # Simula calcolo quantico
        # Loss basata su lunghezza frase e parametri
        loss = 0.1 / (1.0 + np.linalg.norm(params) * 0.001) + 0.01 * sentence_length
        
        # Gradiente simulato
        gradients = np.random.normal(0, 0.001, len(params))
        gradients += params * 0.0001  # Regularization term
        
        return loss, gradients
    
    def _run_final_evaluation(self, dataset_info: Dict, final_params: np.ndarray) -> Dict:
        """Valutazione finale su test set."""
        
        self.logger.info("🔍 Valutazione finale su test set...")
        
        test_data = dataset_info["test_data"]
        vocab_info = dataset_info["vocab_info"]
        
        # Valuta ogni lunghezza frase separatamente
        loss_by_sentence_length = {}
        
        for sentence in test_data:
            length = len(sentence)
            
            if length not in loss_by_sentence_length:
                loss_by_sentence_length[length] = {
                    "losses": [],
                    "times": []
                }
            
            # Calcola loss per questa frase
            start_time = time.time()
            loss, _ = self._compute_sentence_loss_and_gradients(sentence, final_params, vocab_info)
            eval_time = time.time() - start_time
            
            loss_by_sentence_length[length]["losses"].append(loss)
            loss_by_sentence_length[length]["times"].append(eval_time)
        
        # Calcola statistiche per lunghezza
        final_loss_by_length = {}
        for length, data in loss_by_sentence_length.items():
            losses = data["losses"]
            times = data["times"]
            
            avg_loss = np.mean(losses)
            best_loss = np.min(losses)
            worst_loss = np.max(losses)
            avg_time = np.mean(times)
            
            # Calcola miglioramento rispetto a loss iniziale (approssimato)
            initial_loss_estimate = 0.15  # valore tipico iniziale
            improvement_percent = ((initial_loss_estimate - avg_loss) / initial_loss_estimate * 100)
            
            final_loss_by_length[length] = {
                "avg": avg_loss,
                "best": best_loss,
                "worst": worst_loss,
                "improvement_percent": improvement_percent,
                "avg_time": avg_time,
                "count": len(losses)
            }
        
        # Calcola metriche globali test
        all_test_losses = [loss for data in loss_by_sentence_length.values() for loss in data["losses"]]
        all_test_times = [time for data in loss_by_sentence_length.values() for time in data["times"]]
        
        evaluation_results = {
            "loss_by_sentence_length": final_loss_by_length,
            "test_avg_loss": np.mean(all_test_losses),
            "test_best_loss": np.min(all_test_losses),
            "test_worst_loss": np.max(all_test_losses),
            "avg_time_per_sentence": np.mean(all_test_times),
            "total_test_sentences": len(test_data)
        }
        
        self.logger.info(f"Valutazione completata: Loss media test = {evaluation_results['test_avg_loss']:.6f}")
        
        return evaluation_results
    
    def _generate_final_report(self, dataset_info: Dict, optimization_results: Dict, 
                              evaluation_results: Dict) -> str:
        """Genera report finale completo."""
        
        self.logger.info("📊 Generazione report finale...")
        
        # Crea TrainingMetrics
        metrics = create_training_metrics(
            run_id=self.run_id,
            start_time=self.start_time,
            end_time=self.end_time or datetime.now(),
            dataset_info=dataset_info,
            config=self.config,
            results={**optimization_results, **evaluation_results},
            hpc_stats=self.hpc_stats
        )
        
        # Genera report completo
        report_path = self.reporter.generate_comprehensive_report(
            metrics=metrics,
            loss_history=self.loss_history,
            param_history=self.param_history,
            sentence_results=self.sentence_results
        )
        
        self.logger.info(f"✅ Report finale generato: {report_path}")
        
        return report_path
    
    def _generate_error_report(self, error_message: str) -> str:
        """Genera report di errore."""
        
        self.logger.info("❌ Generazione report di errore...")
        
        error_report_path = self.output_dir / "reports" / f"error_report_{self.run_id}.txt"
        error_report_path.parent.mkdir(exist_ok=True)
        
        with open(error_report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
QUANTUM TRAINING ERROR REPORT
Run ID: {self.run_id}
Timestamp: {datetime.now().isoformat()}
Error: {error_message}

Errors Log:
{json.dumps(self.errors_log, indent=2)}

HPC Stats:
{json.dumps(self.hpc_stats, indent=2)}

Config:
{json.dumps(self.config, indent=2)}
            """)
        
        return str(error_report_path)


def main():
    """Test del trainer integrato."""
    
    # Crea trainer
    trainer = IntegratedQuantumTrainer(output_dir="quantum_training_results")
    
    # Esegui training completo
    try:
        report_path = trainer.run_complete_training(
            train_sentences=100,  # Test con numeri piccoli
            test_sentences=50,
            max_iterations=50,
            convergence_threshold=1e-4
        )
        
        print(f"\n🎉 Training completato con successo!")
        print(f"📊 Report disponibile: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Errore durante training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()