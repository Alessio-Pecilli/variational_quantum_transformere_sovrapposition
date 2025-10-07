"""
Massima Parallelizzazione HPC per Training Quantico su Larga Scala.
Sistema ottimizzato per utilizzare al 100% le risorse di calcolo disponibili.
"""

import os
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import psutil
import time
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from threading import Lock


@dataclass
class HPC_ParallelConfig:
    """Configurazione avanzata per parallelizzazione HPC."""
    # Hardware detection
    auto_detect_resources: bool = True
    force_cpu_count: int = None
    force_memory_gb: float = None
    
    # Parallelization strategy
    sentence_batch_size: int = 4        # Frasi processate in parallelo
    gradient_batch_size: int = 8        # Parametri per batch gradiente
    max_worker_processes: int = None    # Auto-detect da ambiente HPC
    max_worker_threads: int = None      # Thread per processo
    
    # Memory optimization
    memory_limit_per_process_gb: float = 2.0
    enable_memory_monitoring: bool = True
    gc_frequency: int = 100             # Garbage collection ogni N iterazioni
    
    # Performance tuning
    cpu_affinity_enabled: bool = True
    numa_awareness: bool = True
    process_priority: str = "high"      # normal, high, realtime
    
    # Fault tolerance
    max_retries_per_batch: int = 3
    timeout_per_batch_seconds: int = 3600
    enable_checkpointing: bool = True


class HPC_ResourceDetector:
    """Rilevamento intelligente delle risorse HPC."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def detect_hpc_environment(self) -> Dict[str, Any]:
        """Rileva automaticamente l'ambiente HPC e le risorse disponibili."""
        
        resources = {}
        
        # 1. Detect HPC scheduler
        scheduler_info = self._detect_scheduler()
        resources['scheduler'] = scheduler_info
        
        # 2. CPU resources
        cpu_info = self._detect_cpu_resources()
        resources['cpu'] = cpu_info
        
        # 3. Memory resources
        memory_info = self._detect_memory_resources()
        resources['memory'] = memory_info
        
        # 4. NUMA topology
        numa_info = self._detect_numa_topology()
        resources['numa'] = numa_info
        
        # 5. Optimal worker configuration
        worker_config = self._calculate_optimal_workers(cpu_info, memory_info)
        resources['workers'] = worker_config
        
        self._log_detected_resources(resources)
        return resources
    
    def _detect_scheduler(self) -> Dict[str, str]:
        """Detect HPC scheduler (SLURM, PBS, LSF)."""
        scheduler_info = {'type': 'unknown', 'details': {}}
        
        # SLURM detection
        slurm_vars = {
            'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
            'SLURM_CPUS_PER_TASK': os.environ.get('SLURM_CPUS_PER_TASK'),
            'SLURM_MEM_PER_NODE': os.environ.get('SLURM_MEM_PER_NODE'), 
            'SLURM_NNODES': os.environ.get('SLURM_NNODES'),
            'SLURM_NTASKS': os.environ.get('SLURM_NTASKS'),
            'SLURM_PARTITION': os.environ.get('SLURM_PARTITION')
        }
        
        if any(slurm_vars.values()):
            scheduler_info = {'type': 'SLURM', 'details': slurm_vars}
        
        # PBS/Torque detection
        pbs_vars = {
            'PBS_JOBID': os.environ.get('PBS_JOBID'),
            'PBS_NP': os.environ.get('PBS_NP'),
            'PBS_QUEUE': os.environ.get('PBS_QUEUE')
        }
        
        if any(pbs_vars.values()):
            scheduler_info = {'type': 'PBS', 'details': pbs_vars}
        
        # OMP detection
        omp_threads = os.environ.get('OMP_NUM_THREADS')
        if omp_threads:
            scheduler_info['omp_threads'] = int(omp_threads)
        
        return scheduler_info
    
    def _detect_cpu_resources(self) -> Dict[str, Any]:
        """Detect CPU resources and capabilities."""
        
        cpu_info = {
            'logical_cores': psutil.cpu_count(logical=True),
            'physical_cores': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq(),
            'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Additional CPU features
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                cpu_info['has_avx'] = 'avx' in cpuinfo
                cpu_info['has_sse4'] = 'sse4' in cpuinfo
                cpu_info['architecture'] = 'x86_64' if 'x86_64' in cpuinfo else 'unknown'
        except:
            pass
        
        return cpu_info
    
    def _detect_memory_resources(self) -> Dict[str, Any]:
        """Detect memory resources."""
        
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent_used': mem.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3)
        }
        
        return memory_info
    
    def _detect_numa_topology(self) -> Dict[str, Any]:
        """Detect NUMA topology if available."""
        numa_info = {'available': False, 'nodes': []}
        
        try:
            # Try to detect NUMA nodes
            numa_nodes = []
            if os.path.exists('/sys/devices/system/node'):
                for node_dir in os.listdir('/sys/devices/system/node'):
                    if node_dir.startswith('node'):
                        node_id = int(node_dir[4:])
                        numa_nodes.append(node_id)
            
            if numa_nodes:
                numa_info = {'available': True, 'nodes': sorted(numa_nodes)}
                
        except:
            pass
        
        return numa_info
    
    def _calculate_optimal_workers(self, cpu_info: Dict, memory_info: Dict) -> Dict[str, int]:
        """Calculate optimal number of workers based on resources."""
        
        # Available resources
        available_cores = cpu_info['logical_cores']
        available_memory_gb = memory_info['available_gb']
        
        # Memory constraint: 2GB per process minimum
        memory_limited_processes = int(available_memory_gb // 2.0)
        
        # CPU constraint: Reserve 1 core for system
        cpu_limited_processes = max(1, available_cores - 1)
        
        # Take minimum (most restrictive constraint)
        max_processes = min(memory_limited_processes, cpu_limited_processes)
        
        # Thread configuration
        threads_per_process = max(1, available_cores // max_processes)
        
        return {
            'max_processes': max_processes,
            'threads_per_process': threads_per_process,
            'total_workers': max_processes * threads_per_process,
            'memory_per_process_gb': available_memory_gb / max_processes,
            'constraint': 'memory' if memory_limited_processes < cpu_limited_processes else 'cpu'
        }
    
    def _log_detected_resources(self, resources: Dict):
        """Log detected resources comprehensively."""
        self.logger.info("🔍 HPC RESOURCE DETECTION COMPLETE")
        self.logger.info(f"   Scheduler: {resources['scheduler']['type']}")
        self.logger.info(f"   Physical cores: {resources['cpu']['physical_cores']}")
        self.logger.info(f"   Logical cores: {resources['cpu']['logical_cores']}")
        self.logger.info(f"   Available memory: {resources['memory']['available_gb']:.1f} GB")
        self.logger.info(f"   NUMA nodes: {len(resources['numa']['nodes']) if resources['numa']['available'] else 0}")
        self.logger.info(f"   Optimal processes: {resources['workers']['max_processes']}")
        self.logger.info(f"   Total workers: {resources['workers']['total_workers']}")
        self.logger.info(f"   Limiting factor: {resources['workers']['constraint']}")


class HPC_MaxParallelOptimizer:
    """
    Ottimizzatore con parallelizzazione massima per HPC.
    Multi-livello: sentence batches + gradient computation + NUMA awareness.
    """
    
    def __init__(self, config: HPC_ParallelConfig = None, logger=None):
        self.config = config or HPC_ParallelConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Detect resources
        detector = HPC_ResourceDetector(self.logger)
        self.resources = detector.detect_hpc_environment()
        
        # Configure workers
        self._setup_worker_configuration()
        
        # Performance monitoring
        self.performance_stats = {
            'sentences_processed': 0,
            'total_time': 0.0,
            'avg_time_per_sentence': 0.0,
            'cpu_utilization': [],
            'memory_usage': []
        }
        
        # Thread safety
        self.stats_lock = Lock()
    
    def _setup_worker_configuration(self):
        """Setup optimal worker configuration."""
        
        if self.config.auto_detect_resources:
            # Use detected optimal configuration
            self.max_processes = self.resources['workers']['max_processes']
            self.threads_per_process = self.resources['workers']['threads_per_process']
        else:
            # Use manual configuration
            self.max_processes = self.config.max_worker_processes or mp.cpu_count()
            self.threads_per_process = self.config.max_worker_threads or 2
        
        # Apply memory constraints
        available_memory = self.resources['memory']['available_gb']
        max_memory_processes = int(available_memory // self.config.memory_limit_per_process_gb)
        self.max_processes = min(self.max_processes, max_memory_processes)
        
        self.logger.info(f"🔧 WORKER CONFIGURATION")
        self.logger.info(f"   Max processes: {self.max_processes}")
        self.logger.info(f"   Threads per process: {self.threads_per_process}")
        self.logger.info(f"   Memory per process: {self.config.memory_limit_per_process_gb} GB")
    
    def optimize_sentences_parallel(self, sentences_data: List[Dict], 
                                   optimization_func: callable,
                                   initial_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Optimize parameters across multiple sentences with maximum parallelization.
        
        Args:
            sentences_data: List of sentence data dictionaries
            optimization_func: Function to optimize single sentence
            initial_params: Initial parameters
            
        Returns:
            Tuple of (optimized_parameters, performance_stats)
        """
        
        self.logger.info(f"🚀 STARTING MAXIMUM PARALLEL OPTIMIZATION")
        self.logger.info(f"   Sentences: {len(sentences_data)}")
        self.logger.info(f"   Processes: {self.max_processes}")
        self.logger.info(f"   Batch size: {self.config.sentence_batch_size}")
        
        start_time = time.time()
        current_params = initial_params.copy()
        best_params = current_params.copy()
        best_loss = float('inf')
        
        # Create sentence batches
        sentence_batches = self._create_sentence_batches(sentences_data)
        
        # Process batches
        for batch_idx, batch in enumerate(sentence_batches):
            
            self.logger.info(f"📦 Processing batch {batch_idx + 1}/{len(sentence_batches)}")
            batch_start = time.time()
            
            # Process batch with maximum parallelization
            batch_results = self._process_sentence_batch_parallel(
                batch, optimization_func, current_params
            )
            
            # Aggregate results and update parameters
            batch_params, batch_loss = self._aggregate_batch_results(batch_results)
            
            if batch_loss < best_loss:
                best_loss = batch_loss
                best_params = batch_params.copy()
                current_params = batch_params.copy()
                
            batch_time = time.time() - batch_start
            self.logger.info(f"   Batch {batch_idx + 1} complete: loss={batch_loss:.6f}, time={batch_time:.2f}s")
            
            # Memory cleanup
            if self.config.enable_memory_monitoring and batch_idx % self.config.gc_frequency == 0:
                self._cleanup_memory()
        
        total_time = time.time() - start_time
        
        # Update performance stats
        with self.stats_lock:
            self.performance_stats['sentences_processed'] = len(sentences_data)
            self.performance_stats['total_time'] = total_time
            self.performance_stats['avg_time_per_sentence'] = total_time / len(sentences_data)
        
        self.logger.info(f"✅ PARALLEL OPTIMIZATION COMPLETE")
        self.logger.info(f"   Total time: {total_time:.2f}s")
        self.logger.info(f"   Avg per sentence: {total_time/len(sentences_data):.3f}s")
        self.logger.info(f"   Best loss: {best_loss:.6f}")
        
        return best_params, self.performance_stats
    
    def _create_sentence_batches(self, sentences_data: List[Dict]) -> List[List[Dict]]:
        """Create optimally sized batches of sentences."""
        
        batch_size = self.config.sentence_batch_size
        batches = []
        
        for i in range(0, len(sentences_data), batch_size):
            batch = sentences_data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_sentence_batch_parallel(self, batch: List[Dict], 
                                       optimization_func: callable,
                                       params: np.ndarray) -> List[Tuple]:
        """
        Process a batch of sentences with maximum parallelization.
        Uses ProcessPoolExecutor for CPU-bound optimization tasks.
        """
        
        results = []
        
        # Create partial function with fixed parameters
        optimize_sentence = partial(self._optimize_single_sentence_wrapper,
                                  optimization_func=optimization_func,
                                  params=params)
        
        # Use ProcessPoolExecutor for CPU-intensive quantum optimization
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            
            # Submit all sentences in batch
            future_to_sentence = {
                executor.submit(optimize_sentence, sentence_data): sentence_data
                for sentence_data in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sentence, 
                                     timeout=self.config.timeout_per_batch_seconds):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Sentence optimization failed: {e}")
                    # Add fallback result
                    results.append((params.copy(), float('inf')))
        
        return results
    
    def _optimize_single_sentence_wrapper(self, sentence_data: Dict,
                                        optimization_func: callable,
                                        params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Wrapper for single sentence optimization in multiprocessing context.
        """
        try:
            # Set CPU affinity if enabled
            if self.config.cpu_affinity_enabled:
                self._set_cpu_affinity()
            
            # Run optimization
            optimized_params, loss = optimization_func(sentence_data, params)
            
            return optimized_params, loss
            
        except Exception as e:
            # Return original parameters on failure
            return params.copy(), float('inf')
    
    def _aggregate_batch_results(self, batch_results: List[Tuple]) -> Tuple[np.ndarray, float]:
        """
        Aggregate optimization results from a batch of sentences.
        Uses weighted averaging based on loss values.
        """
        
        if not batch_results:
            return np.zeros_like(batch_results[0][0]), float('inf')
        
        # Extract parameters and losses
        params_list = [result[0] for result in batch_results]
        losses = [result[1] for result in batch_results]
        
        # Find best result
        best_idx = np.argmin(losses)
        best_params = params_list[best_idx]
        best_loss = losses[best_idx]
        
        # For now, return best single result
        # Could implement weighted averaging here
        
        return best_params, best_loss
    
    def _set_cpu_affinity(self):
        """Set CPU affinity for current process."""
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Get available CPUs
            available_cpus = list(range(psutil.cpu_count()))
            
            # Set affinity to all available CPUs
            current_process.cpu_affinity(available_cpus)
            
        except Exception as e:
            # Silently ignore affinity errors
            pass
    
    def _cleanup_memory(self):
        """Perform memory cleanup and monitoring."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Monitor memory usage
        memory_info = psutil.virtual_memory()
        
        with self.stats_lock:
            self.performance_stats['memory_usage'].append(memory_info.percent)
        
        if memory_info.percent > 90:
            self.logger.warning(f"⚠️  High memory usage: {memory_info.percent:.1f}%")


def create_hpc_max_parallel_optimizer(logger=None) -> HPC_MaxParallelOptimizer:
    """
    Create HPC optimizer with maximum parallelization configuration.
    """
    
    config = HPC_ParallelConfig(
        auto_detect_resources=True,
        sentence_batch_size=8,          # Larger batches for efficiency
        gradient_batch_size=16,         # More gradient parallelization
        memory_limit_per_process_gb=3.0, # Higher memory limit
        enable_memory_monitoring=True,
        cpu_affinity_enabled=True,
        numa_awareness=True,
        process_priority="high",
        max_retries_per_batch=3,
        timeout_per_batch_seconds=7200,  # 2 hours per batch
        enable_checkpointing=True
    )
    
    return HPC_MaxParallelOptimizer(config, logger)


if __name__ == "__main__":
    # Test resource detection
    logging.basicConfig(level=logging.INFO)
    detector = HPC_ResourceDetector()
    resources = detector.detect_hpc_environment()
    
    # Test optimizer creation
    optimizer = create_hpc_max_parallel_optimizer()
    print(f"Max processes: {optimizer.max_processes}")
    print(f"Threads per process: {optimizer.threads_per_process}")