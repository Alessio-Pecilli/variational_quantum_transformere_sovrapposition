"""
Sistema di Reporting Completo per Training Quantico su Larga Scala.
Genera report dettagliati, comprensibili e utili per analisi dei risultati.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class TrainingMetrics:
    """Metriche di training con tutte le informazioni utili."""
    # Identificazione run
    run_id: str
    timestamp_start: str
    timestamp_end: str
    total_duration_hours: float
    
    # Dataset info
    total_sentences: int
    train_sentences: int
    test_sentences: int
    vocab_size: int
    sentence_length_distribution: Dict[int, int]
    
    # Parametri training
    embedding_dim: int
    num_qubits: int
    num_layers: int
    learning_rate: float
    total_parameters: int
    
    # Performance HPC
    max_workers: int
    avg_workers_utilized: float
    cpu_cores_used: int
    memory_peak_gb: float
    scheduler_type: str
    
    # Risultati ottimizzazione
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_improvement_percent: float
    convergence_achieved: bool
    iterations_total: int
    
    # Statistiche per lunghezza frase
    loss_by_sentence_length: Dict[int, Dict[str, float]]
    
    # Performance temporali
    avg_time_per_sentence: float
    avg_time_per_iteration: float
    sentences_per_hour: float
    
    # Parametri finali
    final_param_norm: float
    final_gradient_norm: float
    parameter_evolution: List[float]
    
    # Qualità convergenza
    loss_variance_last_10_percent: float
    gradient_stability_score: float
    
    # Errori e problemi
    errors_encountered: int
    memory_warnings: int
    timeout_issues: int


class ComprehensiveReporter:
    """
    Sistema di reporting completo per training quantico.
    Genera report dettagliati, grafici e analisi comprensibili.
    """
    
    def __init__(self, output_dir: str = "reports", logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Configurazione plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(self, metrics: TrainingMetrics, 
                                    loss_history: List[float],
                                    param_history: List[np.ndarray],
                                    sentence_results: List[Dict]) -> str:
        """
        Genera un report completo con tutte le informazioni utili.
        
        Returns:
            Path del report HTML generato
        """
        
        self.logger.info("📊 Generazione report completo...")
        
        # 1. Report HTML principale
        html_path = self._generate_html_report(metrics, loss_history, sentence_results)
        
        # 2. Grafici dettagliati
        self._generate_comprehensive_plots(metrics, loss_history, param_history, sentence_results)
        
        # 3. File CSV con dati
        self._export_data_csv(metrics, loss_history, sentence_results)
        
        # 4. JSON per elaborazioni successive
        self._export_json_summary(metrics, sentence_results)
        
        # 5. Report testuale per log/email
        self._generate_text_summary(metrics)
        
        self.logger.info(f"✅ Report completo generato: {html_path}")
        return str(html_path)
    
    def _generate_html_report(self, metrics: TrainingMetrics, 
                            loss_history: List[float], 
                            sentence_results: List[Dict]) -> Path:
        """Genera report HTML completo e comprensibile."""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Quantico Report - {metrics.run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               margin: 40px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                     padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                        gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                       border-left: 4px solid #007bff; }}
        .metric-title {{ font-weight: 600; color: #495057; margin-bottom: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-subtitle {{ font-size: 14px; color: #6c757d; margin-top: 4px; }}
        .section {{ margin: 40px 0; }}
        .section-title {{ font-size: 24px; font-weight: bold; color: #343a40; 
                         border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 20px; }}
        .status-success {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; 
                        border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(45deg, #007bff, #28a745); 
                         transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Quantum Training Report</h1>
            <h2>{metrics.run_id}</h2>
            <p><strong>Completato:</strong> {metrics.timestamp_end}</p>
            <p><strong>Durata:</strong> {metrics.total_duration_hours:.2f} ore</p>
        </div>

        <!-- RISULTATI PRINCIPALI -->
        <div class="section">
            <h2 class="section-title">📈 Risultati Principali</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Loss Finale</div>
                    <div class="metric-value">{metrics.final_loss:.6f}</div>
                    <div class="metric-subtitle">Miglioramento: {metrics.loss_improvement_percent:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Convergenza</div>
                    <div class="metric-value {"status-success" if metrics.convergence_achieved else "status-warning"}">
                        {"✅ Raggiunta" if metrics.convergence_achieved else "⚠️ Parziale"}
                    </div>
                    <div class="metric-subtitle">Iterazioni: {metrics.iterations_total}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Performance</div>
                    <div class="metric-value">{metrics.sentences_per_hour:.1f}</div>
                    <div class="metric-subtitle">Frasi/ora processate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Efficienza HPC</div>
                    <div class="metric-value">{metrics.avg_workers_utilized:.1f}/{metrics.max_workers}</div>
                    <div class="metric-subtitle">Worker utilizzati in media</div>
                </div>
            </div>
        </div>

        <!-- DATASET E CONFIGURAZIONE -->
        <div class="section">
            <h2 class="section-title">📚 Dataset e Configurazione</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Dataset</div>
                    <div class="metric-value">{metrics.total_sentences}</div>
                    <div class="metric-subtitle">{metrics.train_sentences} train + {metrics.test_sentences} test</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Vocabolario</div>
                    <div class="metric-value">{metrics.vocab_size:,}</div>
                    <div class="metric-subtitle">Parole uniche</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Parametri Quantici</div>
                    <div class="metric-value">{metrics.total_parameters:,}</div>
                    <div class="metric-subtitle">{metrics.num_qubits}Q × {metrics.num_layers}L × {metrics.embedding_dim}D</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Learning Rate</div>
                    <div class="metric-value">{metrics.learning_rate}</div>
                    <div class="metric-subtitle">Adam Optimizer</div>
                </div>
            </div>
        </div>

        <!-- PERFORMANCE HPC -->
        <div class="section">
            <h2 class="section-title">⚡ Performance HPC</h2>
            <table class="table">
                <tr><th>Metrica</th><th>Valore</th><th>Dettaglio</th></tr>
                <tr><td>Scheduler</td><td>{metrics.scheduler_type}</td><td>Sistema HPC rilevato</td></tr>
                <tr><td>CPU Cores</td><td>{metrics.cpu_cores_used}</td><td>Core utilizzati</td></tr>
                <tr><td>Memory Peak</td><td>{metrics.memory_peak_gb:.1f} GB</td><td>Picco utilizzo memoria</td></tr>
                <tr><td>Tempo/Frase</td><td>{metrics.avg_time_per_sentence:.3f} sec</td><td>Media elaborazione</td></tr>
                <tr><td>Tempo/Iterazione</td><td>{metrics.avg_time_per_iteration:.2f} sec</td><td>Media ottimizzazione</td></tr>
            </table>
        </div>

        <!-- ANALISI PER LUNGHEZZA FRASE -->
        <div class="section">
            <h2 class="section-title">📏 Analisi per Lunghezza Frase</h2>
            <table class="table">
                <tr><th>Lunghezza</th><th>Frasi</th><th>Loss Media</th><th>Loss Migliore</th><th>Miglioramento</th></tr>
        """
        
        # Tabella per lunghezza frasi
        for length in sorted(metrics.sentence_length_distribution.keys()):
            count = metrics.sentence_length_distribution[length]
            if length in metrics.loss_by_sentence_length:
                loss_data = metrics.loss_by_sentence_length[length]
                avg_loss = loss_data.get('avg', 0.0)
                best_loss = loss_data.get('best', 0.0)
                improvement = loss_data.get('improvement_percent', 0.0)
                
                html_content += f"""
                <tr>
                    <td>{length} parole</td>
                    <td>{count}</td>
                    <td>{avg_loss:.6f}</td>
                    <td>{best_loss:.6f}</td>
                    <td class="{"status-success" if improvement > 10 else "status-warning"}">{improvement:.1f}%</td>
                </tr>
                """
        
        html_content += f"""
            </table>
        </div>

        <!-- QUALITÀ CONVERGENZA -->
        <div class="section">
            <h2 class="section-title">🎯 Qualità Convergenza</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Stabilità Loss</div>
                    <div class="metric-value">{metrics.loss_variance_last_10_percent:.8f}</div>
                    <div class="metric-subtitle">Varianza ultimo 10%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Stabilità Gradienti</div>
                    <div class="metric-value">{metrics.gradient_stability_score:.3f}</div>
                    <div class="metric-subtitle">Score stabilità (0-1)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Parametri Norm</div>
                    <div class="metric-value">{metrics.final_param_norm:.4f}</div>
                    <div class="metric-subtitle">Norma finale parametri</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Gradient Norm</div>
                    <div class="metric-value">{metrics.final_gradient_norm:.6f}</div>
                    <div class="metric-subtitle">Norma finale gradienti</div>
                </div>
            </div>
        </div>

        <!-- PROBLEMI E AVVISI -->
        <div class="section">
            <h2 class="section-title">⚠️ Problemi Riscontrati</h2>
            <table class="table">
                <tr><th>Tipo</th><th>Occorrenze</th><th>Status</th></tr>
                <tr><td>Errori</td><td>{metrics.errors_encountered}</td>
                    <td class="{"status-success" if metrics.errors_encountered == 0 else "status-error"}">
                        {"✅ Nessuno" if metrics.errors_encountered == 0 else "❌ Presenti"}
                    </td></tr>
                <tr><td>Warning Memoria</td><td>{metrics.memory_warnings}</td>
                    <td class="{"status-success" if metrics.memory_warnings == 0 else "status-warning"}">
                        {"✅ Nessuno" if metrics.memory_warnings == 0 else "⚠️ Presenti"}
                    </td></tr>
                <tr><td>Timeout</td><td>{metrics.timeout_issues}</td>
                    <td class="{"status-success" if metrics.timeout_issues == 0 else "status-warning"}">
                        {"✅ Nessuno" if metrics.timeout_issues == 0 else "⚠️ Presenti"}
                    </td></tr>
            </table>
        </div>

        <!-- RACCOMANDAZIONI -->
        <div class="section">
            <h2 class="section-title">💡 Raccomandazioni</h2>
        """
        
        # Generiamo raccomandazioni intelligenti
        recommendations = self._generate_recommendations(metrics, loss_history)
        html_content += "<ul>"
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        html_content += "</ul>"
        
        html_content += """
        </div>

        <div class="section">
            <p style="text-align: center; color: #6c757d; margin-top: 40px;">
                Report generato automaticamente dal Sistema di Quantum Training
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        # Salva HTML
        html_path = self.output_dir / f"report_{metrics.run_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_comprehensive_plots(self, metrics: TrainingMetrics, 
                                    loss_history: List[float],
                                    param_history: List[np.ndarray],
                                    sentence_results: List[Dict]):
        """Genera grafici dettagliati e comprensibili."""
        
        # 1. Loss Evolution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(loss_history, color='blue', linewidth=2)
        plt.title('Evolution Loss Durante Training', fontsize=14, weight='bold')
        plt.xlabel('Iterazione')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        
        # 2. Loss per lunghezza frase
        plt.subplot(2, 3, 2)
        lengths = list(metrics.sentence_length_distribution.keys())
        counts = list(metrics.sentence_length_distribution.values())
        plt.bar(lengths, counts, alpha=0.7, color='green')
        plt.title('Distribuzione Lunghezza Frasi', fontsize=14, weight='bold')
        plt.xlabel('Lunghezza (parole)')
        plt.ylabel('Numero Frasi')
        plt.grid(True, alpha=0.3)
        
        # 3. Parameter evolution
        if param_history:
            plt.subplot(2, 3, 3)
            param_norms = [np.linalg.norm(params) for params in param_history]
            plt.plot(param_norms, color='red', linewidth=2)
            plt.title('Evoluzione Norma Parametri', fontsize=14, weight='bold')
            plt.xlabel('Iterazione')
            plt.ylabel('||θ||')
            plt.grid(True, alpha=0.3)
        
        # 4. Loss miglioramento per lunghezza
        plt.subplot(2, 3, 4)
        if metrics.loss_by_sentence_length:
            lengths = list(metrics.loss_by_sentence_length.keys())
            improvements = [metrics.loss_by_sentence_length[l].get('improvement_percent', 0) 
                          for l in lengths]
            plt.bar(lengths, improvements, alpha=0.7, color='orange')
            plt.title('Miglioramento Loss per Lunghezza', fontsize=14, weight='bold')
            plt.xlabel('Lunghezza (parole)')
            plt.ylabel('Miglioramento (%)')
            plt.grid(True, alpha=0.3)
        
        # 5. Convergenza stabilità
        plt.subplot(2, 3, 5)
        if len(loss_history) > 10:
            # Moving average
            window = max(1, len(loss_history) // 20)
            moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
            plt.plot(loss_history, alpha=0.3, label='Loss Raw')
            plt.plot(range(window-1, len(loss_history)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Avg (w={window})')
            plt.title('Stabilità Convergenza', fontsize=14, weight='bold')
            plt.xlabel('Iterazione')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Performance timeline
        plt.subplot(2, 3, 6)
        if sentence_results:
            times = [result.get('time', 0) for result in sentence_results]
            plt.plot(times, color='purple', linewidth=2, marker='o', markersize=3)
            plt.title('Tempo per Frase nel Tempo', fontsize=14, weight='bold')
            plt.xlabel('Frase #')
            plt.ylabel('Tempo (sec)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"plots_{metrics.run_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Grafici salvati: {plot_path}")
    
    def _generate_recommendations(self, metrics: TrainingMetrics, loss_history: List[float]) -> List[str]:
        """Genera raccomandazioni intelligenti basate sui risultati."""
        
        recommendations = []
        
        # Analisi convergenza
        if metrics.convergence_achieved:
            recommendations.append("✅ <strong>Convergenza ottimale</strong>: Il training è terminato con successo")
        else:
            recommendations.append("⚠️ <strong>Convergenza parziale</strong>: Considera di aumentare le iterazioni o ridurre il learning rate")
        
        # Analisi loss improvement
        if metrics.loss_improvement_percent > 50:
            recommendations.append("🎯 <strong>Eccellente miglioramento</strong>: La loss è migliorata significativamente")
        elif metrics.loss_improvement_percent > 20:
            recommendations.append("✅ <strong>Buon miglioramento</strong>: Progresso soddisfacente della loss")
        else:
            recommendations.append("🔄 <strong>Miglioramento limitato</strong>: Considera di aumentare embedding_dim o learning_rate")
        
        # Analisi performance HPC
        utilization_ratio = metrics.avg_workers_utilized / metrics.max_workers
        if utilization_ratio > 0.8:
            recommendations.append("⚡ <strong>Ottima parallelizzazione</strong>: Utilizzo efficiente delle risorse HPC")
        else:
            recommendations.append(f"🔧 <strong>Parallelizzazione sottoutilizzata</strong>: Solo {utilization_ratio:.1%} worker utilizzati, considera batch più grandi")
        
        # Analisi memoria
        if metrics.memory_peak_gb < 50:  # Assumendo cluster con >64GB
            recommendations.append("💾 <strong>Memoria disponibile</strong>: Puoi aumentare embedding_dim o batch_size")
        elif metrics.memory_warnings > 0:
            recommendations.append("⚠️ <strong>Pressione memoria</strong>: Riduci embedding_dim o implementa gradient checkpointing")
        
        # Analisi stabilità
        if metrics.gradient_stability_score > 0.8:
            recommendations.append("🎯 <strong>Gradienti stabili</strong>: Ottimizzazione ben configurata")
        else:
            recommendations.append("🔧 <strong>Gradienti instabili</strong>: Considera gradient clipping più aggressivo o learning rate adattivo")
        
        # Raccomandazioni specifiche per lunghezze
        if metrics.loss_by_sentence_length:
            best_length = min(metrics.loss_by_sentence_length.keys(), 
                            key=lambda k: metrics.loss_by_sentence_length[k].get('best', float('inf')))
            recommendations.append(f"📏 <strong>Performance per lunghezza</strong>: Frasi da {best_length} parole convergono meglio, considera dataset bilanciato")
        
        # Raccomandazioni future
        if metrics.sentences_per_hour < 10:
            recommendations.append("⏱️ <strong>Velocità bassa</strong>: Ottimizza il codice quantico o usa circuiti più semplici")
        
        recommendations.append("🚀 <strong>Prossimi step</strong>: Testa su dataset di validazione e implementa early stopping adattivo")
        
        return recommendations
    
    def _export_data_csv(self, metrics: TrainingMetrics, loss_history: List[float], sentence_results: List[Dict]):
        """Esporta dati in formato CSV per analisi successive."""
        
        # CSV loss history
        loss_df = pd.DataFrame({
            'iteration': range(len(loss_history)),
            'loss': loss_history
        })
        loss_csv = self.output_dir / f"loss_history_{metrics.run_id}.csv"
        loss_df.to_csv(loss_csv, index=False)
        
        # CSV sentence results
        if sentence_results:
            sentence_df = pd.DataFrame(sentence_results)
            sentence_csv = self.output_dir / f"sentence_results_{metrics.run_id}.csv"
            sentence_df.to_csv(sentence_csv, index=False)
        
        self.logger.info(f"💾 Dati CSV esportati: {loss_csv}")
    
    def _export_json_summary(self, metrics: TrainingMetrics, sentence_results: List[Dict]):
        """Esporta summary in JSON per elaborazioni automatiche."""
        
        summary = {
            'metrics': asdict(metrics),
            'sentence_count': len(sentence_results),
            'timestamp': datetime.now().isoformat(),
        }
        
        json_path = self.output_dir / f"summary_{metrics.run_id}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📄 Summary JSON: {json_path}")
    
    def _generate_text_summary(self, metrics: TrainingMetrics):
        """Genera summary testuale per log/email."""
        
        summary_text = f"""
🚀 QUANTUM TRAINING REPORT - {metrics.run_id}
{'='*60}

📊 RISULTATI PRINCIPALI:
• Loss: {metrics.initial_loss:.6f} → {metrics.final_loss:.6f} ({metrics.loss_improvement_percent:.1f}% improvement)
• Convergenza: {"✅ Raggiunta" if metrics.convergence_achieved else "⚠️ Parziale"} in {metrics.iterations_total} iterazioni
• Performance: {metrics.sentences_per_hour:.1f} frasi/ora
• Durata: {metrics.total_duration_hours:.2f} ore

📚 DATASET:
• {metrics.total_sentences} frasi totali ({metrics.train_sentences} train + {metrics.test_sentences} test)
• Vocabolario: {metrics.vocab_size:,} parole
• Parametri quantici: {metrics.total_parameters:,} ({metrics.num_qubits}Q × {metrics.num_layers}L × {metrics.embedding_dim}D)

⚡ HPC PERFORMANCE:
• Workers: {metrics.avg_workers_utilized:.1f}/{metrics.max_workers} ({metrics.avg_workers_utilized/metrics.max_workers:.1%} utilizzo)
• Memory peak: {metrics.memory_peak_gb:.1f} GB
• Scheduler: {metrics.scheduler_type}

🎯 QUALITÀ:
• Parametri norm: {metrics.final_param_norm:.4f}
• Gradienti norm: {metrics.final_gradient_norm:.6f}
• Stabilità score: {metrics.gradient_stability_score:.3f}

⚠️ PROBLEMI:
• Errori: {metrics.errors_encountered}
• Memory warnings: {metrics.memory_warnings}
• Timeout: {metrics.timeout_issues}

{'='*60}
Report completo disponibile in HTML
        """
        
        text_path = self.output_dir / f"summary_{metrics.run_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Log anche in console
        print(summary_text)
        
        self.logger.info(f"📝 Summary testuale: {text_path}")


def create_training_metrics(run_id: str, start_time: datetime, end_time: datetime,
                          dataset_info: Dict, config: Dict, results: Dict,
                          hpc_stats: Dict) -> TrainingMetrics:
    """
    Helper per creare TrainingMetrics da dati di training.
    """
    
    duration = (end_time - start_time).total_seconds() / 3600.0
    
    return TrainingMetrics(
        run_id=run_id,
        timestamp_start=start_time.isoformat(),
        timestamp_end=end_time.isoformat(),
        total_duration_hours=duration,
        
        # Dataset
        total_sentences=dataset_info.get('total_sentences', 0),
        train_sentences=dataset_info.get('train_sentences', 0),
        test_sentences=dataset_info.get('test_sentences', 0),
        vocab_size=dataset_info.get('vocab_size', 0),
        sentence_length_distribution=dataset_info.get('length_distribution', {}),
        
        # Config
        embedding_dim=config.get('embedding_dim', 0),
        num_qubits=config.get('num_qubits', 0),
        num_layers=config.get('num_layers', 0),
        learning_rate=config.get('learning_rate', 0.0),
        total_parameters=config.get('total_parameters', 0),
        
        # HPC
        max_workers=hpc_stats.get('max_workers', 0),
        avg_workers_utilized=hpc_stats.get('avg_workers_utilized', 0.0),
        cpu_cores_used=hpc_stats.get('cpu_cores_used', 0),
        memory_peak_gb=hpc_stats.get('memory_peak_gb', 0.0),
        scheduler_type=hpc_stats.get('scheduler_type', 'unknown'),
        
        # Results
        initial_loss=results.get('initial_loss', 0.0),
        final_loss=results.get('final_loss', 0.0),
        best_loss=results.get('best_loss', 0.0),
        loss_improvement_percent=results.get('loss_improvement_percent', 0.0),
        convergence_achieved=results.get('convergence_achieved', False),
        iterations_total=results.get('iterations_total', 0),
        loss_by_sentence_length=results.get('loss_by_sentence_length', {}),
        avg_time_per_sentence=results.get('avg_time_per_sentence', 0.0),
        avg_time_per_iteration=results.get('avg_time_per_iteration', 0.0),
        sentences_per_hour=results.get('sentences_per_hour', 0.0),
        final_param_norm=results.get('final_param_norm', 0.0),
        final_gradient_norm=results.get('final_gradient_norm', 0.0),
        parameter_evolution=results.get('parameter_evolution', []),
        loss_variance_last_10_percent=results.get('loss_variance_last_10_percent', 0.0),
        gradient_stability_score=results.get('gradient_stability_score', 0.0),
        errors_encountered=results.get('errors_encountered', 0),
        memory_warnings=results.get('memory_warnings', 0),
        timeout_issues=results.get('timeout_issues', 0)
    )


if __name__ == "__main__":
    # Test del sistema di reporting
    import tempfile
    
    # Crea metriche di test
    test_metrics = TrainingMetrics(
        run_id="test_run_001",
        timestamp_start="2025-10-05T20:00:00",
        timestamp_end="2025-10-05T22:30:00",
        total_duration_hours=2.5,
        total_sentences=1000,
        train_sentences=700,
        test_sentences=300,
        vocab_size=4148,
        sentence_length_distribution={3: 200, 5: 316, 9: 599, 17: 883},
        embedding_dim=16,
        num_qubits=2,
        num_layers=3,
        learning_rate=0.001,
        total_parameters=576,
        max_workers=32,
        avg_workers_utilized=28.5,
        cpu_cores_used=32,
        memory_peak_gb=45.2,
        scheduler_type="SLURM",
        initial_loss=0.156789,
        final_loss=0.072345,
        best_loss=0.071234,
        loss_improvement_percent=53.8,
        convergence_achieved=True,
        iterations_total=150,
        loss_by_sentence_length={
            3: {"avg": 0.08, "best": 0.075, "improvement_percent": 45.2},
            5: {"avg": 0.072, "best": 0.071, "improvement_percent": 55.1},
            9: {"avg": 0.074, "best": 0.072, "improvement_percent": 52.3},
            17: {"avg": 0.076, "best": 0.073, "improvement_percent": 48.9}
        },
        avg_time_per_sentence=3.2,
        avg_time_per_iteration=45.6,
        sentences_per_hour=280.5,
        final_param_norm=0.1456,
        final_gradient_norm=0.000123,
        parameter_evolution=[0.1, 0.12, 0.135, 0.1456],
        loss_variance_last_10_percent=0.0000234,
        gradient_stability_score=0.89,
        errors_encountered=0,
        memory_warnings=2,
        timeout_issues=0
    )
    
    # Test reporter
    reporter = ComprehensiveReporter("test_reports")
    loss_history = [0.156789 - i*0.0005 for i in range(150)]
    param_history = []
    sentence_results = [{"sentence_id": i, "time": 3.2 + np.random.normal(0, 0.5)} for i in range(1000)]
    
    report_path = reporter.generate_comprehensive_report(
        test_metrics, loss_history, param_history, sentence_results
    )
    
    print(f"Report di test generato: {report_path}")