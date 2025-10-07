"""
Sistema di Checkpoint e Resume per Training Quantico su Larga Scala.
Permette di salvare e riprendere training interrotti.
"""

import numpy as np
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import hashlib
import shutil


class QuantumTrainingCheckpoint:
    """
    Gestisce checkpoint completi per training quantico.
    Salva stato completo per resume sicuro.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Metadata checkpoint
        self.checkpoint_metadata = {
            "version": "1.0",
            "created_by": "QuantumTrainingCheckpoint",
            "python_version": ".".join(map(str, __import__('sys').version_info[:3]))
        }
    
    def save_checkpoint(self, run_id: str, iteration: int, 
                       params: np.ndarray, optimizer_state: Dict,
                       loss_history: List[float], metrics: Dict,
                       config: Dict, additional_data: Optional[Dict] = None) -> str:
        """
        Salva checkpoint completo dello stato training.
        
        Returns:
            Path del checkpoint salvato
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{run_id}_iter_{iteration:04d}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info(f"💾 Salvataggio checkpoint: {checkpoint_name}")
        
        try:
            # 1. Salva parametri (formato numpy)
            params_path = checkpoint_path / "parameters.npy"
            np.save(params_path, params)
            
            # 2. Salva stato ottimizzatore
            optimizer_path = checkpoint_path / "optimizer_state.pkl"
            with open(optimizer_path, 'wb') as f:
                pickle.dump(optimizer_state, f)
            
            # 3. Salva loss history
            loss_path = checkpoint_path / "loss_history.npy"
            np.save(loss_path, np.array(loss_history))
            
            # 4. Salva metriche e config (JSON)
            metadata = {
                "checkpoint_metadata": self.checkpoint_metadata,
                "training_metadata": {
                    "run_id": run_id,
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "total_iterations": len(loss_history),
                    "current_loss": loss_history[-1] if loss_history else None,
                    "best_loss": min(loss_history) if loss_history else None,
                    "params_shape": list(params.shape),
                    "params_checksum": self._compute_checksum(params)
                },
                "config": config,
                "metrics": metrics,
                "additional_data": additional_data or {}
            }
            
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # 5. Crea file di controllo integrità
            integrity_path = checkpoint_path / "integrity.txt"
            integrity_info = self._compute_checkpoint_integrity(checkpoint_path)
            with open(integrity_path, 'w') as f:
                f.write(integrity_info)
            
            # 6. Crea link simbolico al checkpoint più recente
            latest_link = self.checkpoint_dir / f"{run_id}_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            try:
                latest_link.symlink_to(checkpoint_name, target_is_directory=True)
            except OSError:
                # Fallback per sistemi che non supportano symlink
                latest_file = self.checkpoint_dir / f"{run_id}_latest.txt"
                with open(latest_file, 'w') as f:
                    f.write(str(checkpoint_path))
            
            self.logger.info(f"✅ Checkpoint salvato: {checkpoint_path}")
            
            # 7. Cleanup checkpoint vecchi se necessario
            self._cleanup_old_checkpoints(run_id, max_keep=5)
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"❌ Errore salvataggio checkpoint: {e}")
            # Cleanup parziale in caso di errore
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Carica checkpoint completo.
        
        Returns:
            Dict con tutti i dati del checkpoint
        """
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        
        self.logger.info(f"📂 Caricamento checkpoint: {checkpoint_path}")
        
        try:
            # 1. Verifica integrità
            if not self._verify_checkpoint_integrity(checkpoint_path):
                self.logger.warning("⚠️ Checkpoint potrebbe essere corrotto, procedendo comunque...")
            
            # 2. Carica metadata
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 3. Carica parametri
            params_path = checkpoint_path / "parameters.npy"
            params = np.load(params_path)
            
            # 4. Verifica checksum parametri
            params_checksum = self._compute_checksum(params)
            expected_checksum = metadata["training_metadata"]["params_checksum"]
            
            if params_checksum != expected_checksum:
                self.logger.warning("⚠️ Checksum parametri non corrisponde!")
            
            # 5. Carica stato ottimizzatore
            optimizer_path = checkpoint_path / "optimizer_state.pkl"
            with open(optimizer_path, 'rb') as f:
                optimizer_state = pickle.load(f)
            
            # 6. Carica loss history
            loss_path = checkpoint_path / "loss_history.npy"
            loss_history = np.load(loss_path).tolist()
            
            checkpoint_data = {
                "metadata": metadata,
                "params": params,
                "optimizer_state": optimizer_state,
                "loss_history": loss_history,
                "config": metadata["config"],
                "metrics": metadata["metrics"],
                "additional_data": metadata.get("additional_data", {}),
                "training_metadata": metadata["training_metadata"]
            }
            
            self.logger.info(f"✅ Checkpoint caricato: iterazione {metadata['training_metadata']['iteration']}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento checkpoint: {e}")
            raise
    
    def find_latest_checkpoint(self, run_id: str) -> Optional[str]:
        """Trova l'ultimo checkpoint per un run_id."""
        
        # Prova prima il link simbolico
        latest_link = self.checkpoint_dir / f"{run_id}_latest"
        
        if latest_link.is_symlink() and latest_link.exists():
            return str(latest_link.resolve())
        
        # Fallback: cerca file latest
        latest_file = self.checkpoint_dir / f"{run_id}_latest.txt"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                checkpoint_path = f.read().strip()
                if Path(checkpoint_path).exists():
                    return checkpoint_path
        
        # Fallback: cerca checkpoint più recente per pattern
        checkpoints = list(self.checkpoint_dir.glob(f"{run_id}_iter_*"))
        
        if not checkpoints:
            return None
        
        # Ordina per timestamp (dal nome)
        def extract_timestamp(path):
            try:
                parts = path.name.split("_")
                if len(parts) >= 4:
                    return parts[3] + parts[4]  # timestamp
                return ""
            except:
                return ""
        
        latest = max(checkpoints, key=extract_timestamp)
        return str(latest)
    
    def list_checkpoints(self, run_id: Optional[str] = None) -> List[Dict]:
        """Lista tutti i checkpoint disponibili."""
        
        pattern = f"{run_id}_iter_*" if run_id else "*_iter_*"
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.glob(pattern):
            if checkpoint_path.is_dir():
                try:
                    metadata_path = checkpoint_path / "metadata.json"
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoint_info = {
                        "path": str(checkpoint_path),
                        "run_id": metadata["training_metadata"]["run_id"],
                        "iteration": metadata["training_metadata"]["iteration"],
                        "timestamp": metadata["training_metadata"]["timestamp"],
                        "current_loss": metadata["training_metadata"].get("current_loss"),
                        "best_loss": metadata["training_metadata"].get("best_loss"),
                        "size_mb": self._get_checkpoint_size(checkpoint_path)
                    }
                    
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    self.logger.warning(f"Errore lettura checkpoint {checkpoint_path}: {e}")
        
        # Ordina per timestamp
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: str):
        """Elimina un checkpoint."""
        
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            self.logger.info(f"🗑️ Checkpoint eliminato: {checkpoint_path}")
        else:
            self.logger.warning(f"Checkpoint non trovato: {checkpoint_path}")
    
    def _compute_checksum(self, data: np.ndarray) -> str:
        """Calcola checksum per verifica integrità."""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def _compute_checkpoint_integrity(self, checkpoint_path: Path) -> str:
        """Calcola informazioni di integrità per checkpoint."""
        
        integrity_info = []
        integrity_info.append(f"Checkpoint: {checkpoint_path.name}")
        integrity_info.append(f"Created: {datetime.now().isoformat()}")
        
        # Hash dei file principali
        for file_name in ["parameters.npy", "optimizer_state.pkl", "loss_history.npy", "metadata.json"]:
            file_path = checkpoint_path / file_name
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    file_size = file_path.stat().st_size
                    integrity_info.append(f"{file_name}: {file_hash} ({file_size} bytes)")
        
        return "\n".join(integrity_info)
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """Verifica integrità checkpoint."""
        
        integrity_path = checkpoint_path / "integrity.txt"
        
        if not integrity_path.exists():
            return False
        
        try:
            # Controlla che tutti i file principali esistano
            required_files = ["parameters.npy", "optimizer_state.pkl", "loss_history.npy", "metadata.json"]
            
            for file_name in required_files:
                if not (checkpoint_path / file_name).exists():
                    self.logger.warning(f"File mancante: {file_name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Errore verifica integrità: {e}")
            return False
    
    def _cleanup_old_checkpoints(self, run_id: str, max_keep: int = 5):
        """Elimina checkpoint vecchi mantenendo solo i più recenti."""
        
        checkpoints = [cp for cp in self.list_checkpoints(run_id) 
                      if cp["run_id"] == run_id]
        
        if len(checkpoints) <= max_keep:
            return
        
        # Ordina per timestamp e mantieni solo i più recenti
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        to_delete = checkpoints[max_keep:]
        
        for checkpoint in to_delete:
            try:
                self.delete_checkpoint(checkpoint["path"])
                self.logger.info(f"🧹 Rimosso checkpoint vecchio: {checkpoint['path']}")
            except Exception as e:
                self.logger.warning(f"Errore rimozione checkpoint {checkpoint['path']}: {e}")
    
    def _get_checkpoint_size(self, checkpoint_path: Path) -> float:
        """Calcola dimensione checkpoint in MB."""
        
        total_size = 0
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # MB


class ResumableTrainer:
    """
    Wrapper per IntegratedQuantumTrainer che aggiunge capacità di resume.
    """
    
    def __init__(self, base_trainer, checkpoint_frequency: int = 10):
        self.trainer = base_trainer
        self.checkpoint_manager = QuantumTrainingCheckpoint()
        self.checkpoint_frequency = checkpoint_frequency
        
        self.logger = logging.getLogger(__name__)
        
        # Override del training loop per aggiungere checkpoint
        self.original_run_training_loop = self.trainer._run_training_loop
        self.trainer._run_training_loop = self._resumable_training_loop
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> str:
        """Riprende training da checkpoint."""
        
        self.logger.info(f"🔄 Resume training da checkpoint: {checkpoint_path}")
        
        # Carica checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Ripristina stato trainer
        self.trainer.run_id = checkpoint_data["training_metadata"]["run_id"]
        self.trainer.config.update(checkpoint_data["config"])
        self.trainer.loss_history = checkpoint_data["loss_history"]
        self.trainer.param_history = []  # Reset per evitare duplicati
        
        # Continua training
        dataset_info = checkpoint_data["additional_data"].get("dataset_info")
        if not dataset_info:
            raise ValueError("Dataset info non trovato nel checkpoint")
        
        # Riprendi da iterazione successiva
        start_iteration = checkpoint_data["training_metadata"]["iteration"] + 1
        max_iterations = checkpoint_data["config"].get("max_iterations", 500)
        convergence_threshold = checkpoint_data["config"].get("convergence_threshold", 1e-6)
        
        self.logger.info(f"Riprendendo da iterazione {start_iteration}/{max_iterations}")
        
        # Continua training loop
        final_params, optimization_results = self._resume_training_loop(
            dataset_info, 
            checkpoint_data["params"],
            checkpoint_data["optimizer_state"],
            start_iteration,
            max_iterations, 
            convergence_threshold
        )
        
        # Genera report finale
        evaluation_results = self.trainer._run_final_evaluation(dataset_info, final_params)
        report_path = self.trainer._generate_final_report(dataset_info, optimization_results, evaluation_results)
        
        return report_path
    
    def _resumable_training_loop(self, dataset_info: Dict, initial_params: np.ndarray, 
                                max_iterations: int, convergence_threshold: float) -> Tuple[np.ndarray, Dict]:
        """Training loop con checkpoint automatici."""
        
        return self._resume_training_loop(
            dataset_info, initial_params, {}, 0, max_iterations, convergence_threshold
        )
    
    def _resume_training_loop(self, dataset_info: Dict, initial_params: np.ndarray,
                             optimizer_state: Dict, start_iteration: int,
                             max_iterations: int, convergence_threshold: float) -> Tuple[np.ndarray, Dict]:
        """Training loop che può essere ripreso da qualsiasi iterazione."""
        
        # Configura optimizer con stato
        from optimization import AdamOptimizer
        optimizer = AdamOptimizer(
            learning_rate=self.trainer.config["learning_rate"],
            beta1=self.trainer.config.get("beta1", 0.9),
            beta2=self.trainer.config.get("beta2", 0.999)
        )
        
        if optimizer_state:
            optimizer.load_state(optimizer_state)
        
        # Parametri training
        params = initial_params.copy()
        train_data = dataset_info["train_data"]
        vocab_info = dataset_info["vocab_info"]
        
        # Training loop con checkpoint
        for iteration in range(start_iteration, max_iterations):
            iteration_start = time.time()
            
            try:
                # Calcola loss e gradienti
                batch_loss, gradients = self.trainer._compute_batch_loss_and_gradients(
                    params, train_data, vocab_info, batch_size=min(32, len(train_data))
                )
                
                # Update parametri
                params = optimizer.update_parameters(params, gradients)
                
                # Logging metriche
                self.trainer.loss_history.append(batch_loss)
                self.trainer.param_history.append(params.copy())
                
                # Check convergenza
                grad_norm = np.linalg.norm(gradients)
                if grad_norm < convergence_threshold:
                    self.logger.info(f"✅ Convergenza raggiunta all'iterazione {iteration+1}")
                    break
                
                # Checkpoint automatico
                if (iteration + 1) % self.checkpoint_frequency == 0:
                    self._save_training_checkpoint(
                        iteration, params, optimizer.get_state(), dataset_info
                    )
                
                # Log periodico
                if (iteration + 1) % 10 == 0:
                    iteration_time = time.time() - iteration_start
                    self.logger.info(
                        f"Iter {iteration+1:3d}/{max_iterations}: "
                        f"Loss={batch_loss:.6f}, "
                        f"GradNorm={grad_norm:.6f}, "
                        f"Time={iteration_time:.2f}s"
                    )
                
            except Exception as e:
                self.logger.error(f"Errore iterazione {iteration}: {e}")
                
                # Salva checkpoint di emergenza
                self._save_emergency_checkpoint(iteration, params, optimizer.get_state(), dataset_info, str(e))
                raise
        
        # Checkpoint finale
        final_iteration = len(self.trainer.loss_history) - 1
        self._save_training_checkpoint(
            final_iteration, params, optimizer.get_state(), dataset_info, is_final=True
        )
        
        # Calcola risultati (usa la logica originale)
        return self.original_run_training_loop(dataset_info, initial_params, max_iterations, convergence_threshold)
    
    def _save_training_checkpoint(self, iteration: int, params: np.ndarray, 
                                 optimizer_state: Dict, dataset_info: Dict, is_final: bool = False):
        """Salva checkpoint durante training."""
        
        try:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                run_id=self.trainer.run_id,
                iteration=iteration,
                params=params,
                optimizer_state=optimizer_state,
                loss_history=self.trainer.loss_history,
                metrics=self.trainer.hpc_stats,
                config=self.trainer.config,
                additional_data={"dataset_info": dataset_info, "is_final": is_final}
            )
            
            if is_final:
                self.logger.info(f"💾 Checkpoint finale salvato: {checkpoint_path}")
            else:
                self.logger.info(f"💾 Checkpoint automatico salvato (iter {iteration})")
                
        except Exception as e:
            self.logger.error(f"Errore salvataggio checkpoint: {e}")
    
    def _save_emergency_checkpoint(self, iteration: int, params: np.ndarray, 
                                  optimizer_state: Dict, dataset_info: Dict, error_msg: str):
        """Salva checkpoint di emergenza in caso di errore."""
        
        try:
            emergency_path = self.checkpoint_manager.save_checkpoint(
                run_id=f"{self.trainer.run_id}_EMERGENCY",
                iteration=iteration,
                params=params,
                optimizer_state=optimizer_state,
                loss_history=self.trainer.loss_history,
                metrics=self.trainer.hpc_stats,
                config=self.trainer.config,
                additional_data={
                    "dataset_info": dataset_info,
                    "error": error_msg,
                    "is_emergency": True
                }
            )
            
            self.logger.info(f"🚨 Checkpoint di emergenza salvato: {emergency_path}")
            
        except Exception as e:
            self.logger.error(f"Errore salvataggio checkpoint di emergenza: {e}")


if __name__ == "__main__":
    # Test del sistema checkpoint
    checkpoint_manager = QuantumTrainingCheckpoint("test_checkpoints")
    
    # Simula dati di training
    test_params = np.random.random(100)
    test_optimizer_state = {"m": np.random.random(100), "v": np.random.random(100), "t": 10}
    test_loss_history = [0.5 - i*0.001 for i in range(50)]
    test_config = {"learning_rate": 0.001, "embedding_dim": 16}
    test_metrics = {"accuracy": 0.85}
    
    # Salva checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        run_id="test_run",
        iteration=49,
        params=test_params,
        optimizer_state=test_optimizer_state,
        loss_history=test_loss_history,
        metrics=test_metrics,
        config=test_config
    )
    
    print(f"Checkpoint salvato: {checkpoint_path}")
    
    # Carica checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)
    print(f"Checkpoint caricato: iterazione {loaded_data['training_metadata']['iteration']}")
    
    # Lista checkpoint
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Checkpoint disponibili: {len(checkpoints)}")
    
    print("✅ Test sistema checkpoint completato")