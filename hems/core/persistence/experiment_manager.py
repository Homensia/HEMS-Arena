"""
Experiment Manager Module
Handles experiment persistence, metadata tracking, and result organization.
"""

import json
import pickle
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np


class ExperimentManager:
    """
    Experiment manager for persistent storage and metadata tracking.
    """
    
    def __init__(self, experiment_dir: Path, experiment_id: str, logger):
        """
        Initialize experiment manager.
        
        Args:
            experiment_dir: Experiment directory
            experiment_id: Unique experiment identifier
            logger: Logger instance
        """
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.logger = logger
        
        # Create storage structure
        self._create_storage_structure()
        
        # Initialize metadata
        self.experiment_metadata = {
            'experiment_id': experiment_id,
            'creation_time': datetime.now().isoformat(),
            'status': 'initialized',
            'components': {
                'training': False,
                'testing': False,
                'evaluation': False,
                'visualization': False
            },
            'metrics_summary': {},
            'file_registry': {}
        }
        
        # Save initial metadata
        self._save_metadata()
    
    def _create_storage_structure(self):
        """Create hierarchical storage structure."""
        structure = {
            'config': self.experiment_dir / 'config',
            'models': self.experiment_dir / 'models',
            'results': self.experiment_dir / 'results',
            'plots': self.experiment_dir / 'plots',
            'logs': self.experiment_dir / 'logs'
        }
        
        # Create subdirectories
        for name, path in structure.items():
            path.mkdir(parents=True, exist_ok=True)
            
            # Create component-specific subdirectories
            if name == 'results':
                (path / 'training').mkdir(exist_ok=True)
                (path / 'testing').mkdir(exist_ok=True)
                (path / 'evaluation').mkdir(exist_ok=True)
            elif name == 'plots':
                (path / 'training_analytics').mkdir(exist_ok=True)
                (path / 'performance_comparison').mkdir(exist_ok=True)
                (path / 'cross_dataset_analysis').mkdir(exist_ok=True)
                (path / 'research_dashboard').mkdir(exist_ok=True)
            elif name == 'logs':
                (path / 'training').mkdir(exist_ok=True)
                (path / 'testing').mkdir(exist_ok=True)
                (path / 'evaluation').mkdir(exist_ok=True)
        
        self.storage_structure = structure
        self.logger.info(f"Created storage structure for experiment {self.experiment_id}")
    
    def save_experiment_results(self, training_results: Dict[str, Any], 
                              testing_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any]):
        """
        Save complete experiment results.
        
        Args:
            training_results: Training phase results
            testing_results: Testing phase results
            evaluation_results: Evaluation phase results
        """
        self.logger.info("Saving experiment results...")
        
        try:
            # Save training results
            self._save_training_results(training_results)
            self.experiment_metadata['components']['training'] = True
            
            # Save testing results
            self._save_testing_results(testing_results)
            self.experiment_metadata['components']['testing'] = True
            
            # Save evaluation results
            self._save_evaluation_results(evaluation_results)
            self.experiment_metadata['components']['evaluation'] = True
            
            # Update experiment metadata
            self._update_experiment_summary(training_results, testing_results, evaluation_results)
            
            # Generate experiment report
            self._generate_experiment_report(training_results, testing_results, evaluation_results)
            
            # Update status
            self.experiment_metadata['status'] = 'completed'
            self.experiment_metadata['completion_time'] = datetime.now().isoformat()
            
            # Save final metadata
            self._save_metadata()
            
            self.logger.info("Experiment results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save experiment results: {str(e)}")
            self.experiment_metadata['status'] = 'failed'
            self.experiment_metadata['error'] = str(e)
            self._save_metadata()
            raise
    
    def _save_training_results(self, training_results: Dict[str, Any]):
        """Save training results with proper organization."""
        training_dir = self.storage_structure['results'] / 'training'
        
        # Save main training results
        training_file = training_dir / 'training_results.json'
        self._save_json_results(training_results, training_file)
        
        # Save individual agent analytics
        for agent_name, agent_results in training_results.get('agent_results', {}).items():
            if agent_results.get('status') == 'completed':
                # Save agent-specific analytics
                agent_file = training_dir / f'{agent_name}_analytics.json'
                analytics = agent_results.get('training_analytics', {})
                self._save_json_results(analytics, agent_file)
                
                # Register file
                self.experiment_metadata['file_registry'][f'training_{agent_name}'] = str(agent_file)
        
        # Save training summary
        summary_file = training_dir / 'training_summary.json'
        training_summary = training_results.get('training_summary', {})
        self._save_json_results(training_summary, summary_file)
        
        self.experiment_metadata['file_registry']['training_main'] = str(training_file)
        self.experiment_metadata['file_registry']['training_summary'] = str(summary_file)
    
    def _save_testing_results(self, testing_results: Dict[str, Any]):
        """Save testing results with proper organization."""
        testing_dir = self.storage_structure['results'] / 'testing'
        
        # Save main testing results
        testing_file = testing_dir / 'testing_results.json'
        self._save_json_results(testing_results, testing_file)
        
        # Save individual agent testing results
        for agent_name, agent_results in testing_results.get('agent_results', {}).items():
            if agent_results.get('status') == 'completed':
                # Save general testing results
                general_file = testing_dir / f'{agent_name}_general_testing.json'
                general_results = agent_results.get('general_testing', {})
                self._save_json_results(general_results, general_file)
                
                # Save specific testing results
                specific_file = testing_dir / f'{agent_name}_specific_testing.json'
                specific_results = agent_results.get('specific_testing', {})
                self._save_json_results(specific_results, specific_file)
                
                # Save robustness metrics
                robustness_file = testing_dir / f'{agent_name}_robustness.json'
                robustness_metrics = agent_results.get('robustness_metrics', {})
                self._save_json_results(robustness_metrics, robustness_file)
                
                # Register files
                self.experiment_metadata['file_registry'][f'testing_general_{agent_name}'] = str(general_file)
                self.experiment_metadata['file_registry'][f'testing_specific_{agent_name}'] = str(specific_file)
                self.experiment_metadata['file_registry'][f'robustness_{agent_name}'] = str(robustness_file)
        
        # Save testing summary
        summary_file = testing_dir / 'testing_summary.json'
        testing_summary = testing_results.get('testing_summary', {})
        self._save_json_results(testing_summary, summary_file)
        
        self.experiment_metadata['file_registry']['testing_main'] = str(testing_file)
        self.experiment_metadata['file_registry']['testing_summary'] = str(summary_file)
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Save evaluation results with proper organization."""
        evaluation_dir = self.storage_structure['results'] / 'evaluation'
        
        # Save main evaluation results (convert MetricResult objects)
        evaluation_serializable = self._make_evaluation_serializable(evaluation_results)
        evaluation_file = evaluation_dir / 'evaluation_results.json'
        self._save_json_results(evaluation_serializable, evaluation_file)
        
        # Save individual agent metrics
        agent_metrics = evaluation_results.get('agent_metrics', {})
        for agent_name, metrics in agent_metrics.items():
            agent_metrics_file = evaluation_dir / f'{agent_name}_metrics.json'
            agent_metrics_serializable = {}
            
            for metric_name, metric_result in metrics.items():
                agent_metrics_serializable[metric_name] = {
                    'value': metric_result.value,
                    'unit': metric_result.unit,
                    'description': metric_result.description,
                    'category': metric_result.category
                }
            
            self._save_json_results(agent_metrics_serializable, agent_metrics_file)
            self.experiment_metadata['file_registry'][f'metrics_{agent_name}'] = str(agent_metrics_file)
        
        # Save comparative analysis
        comparative_file = evaluation_dir / 'comparative_analysis.json'
        comparative_analysis = evaluation_results.get('comparative_analysis', {})
        self._save_json_results(comparative_analysis, comparative_file)
        
        # Save statistical analysis
        statistical_file = evaluation_dir / 'statistical_analysis.json'
        statistical_analysis = evaluation_results.get('statistical_analysis', {})
        self._save_json_results(statistical_analysis, statistical_file)
        
        self.experiment_metadata['file_registry']['evaluation_main'] = str(evaluation_file)
        self.experiment_metadata['file_registry']['comparative_analysis'] = str(comparative_file)
        self.experiment_metadata['file_registry']['statistical_analysis'] = str(statistical_file)
    
    def _make_evaluation_serializable(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert evaluation results to JSON-serializable format."""
        serializable = {}
        
        for key, value in evaluation_results.items():
            if key == 'agent_metrics':
                serializable[key] = {}
                for agent_name, metrics in value.items():
                    serializable[key][agent_name] = {}
                    for metric_name, metric_result in metrics.items():
                        serializable[key][agent_name][metric_name] = {
                            'value': metric_result.value,
                            'unit': metric_result.unit,
                            'description': metric_result.description,
                            'category': metric_result.category
                        }
            else:
                serializable[key] = value
        
        return serializable
    
    def save_eda_results(self, eda_results: Dict[str, Any]):
        """Save EDA results."""
        eda_file = self.storage_structure['results'] / 'eda_results.json'
        self._save_json_results(eda_results, eda_file)
        self.experiment_metadata['file_registry']['eda'] = str(eda_file)
        self.experiment_metadata['components']['eda'] = True
        self._save_metadata()
    
    def save_error_results(self, error_results: Dict[str, Any]):
        """Save error results when pipeline fails."""
        error_file = self.storage_structure['results'] / 'error_results.json'
        self._save_json_results(error_results, error_file)
        self.experiment_metadata['file_registry']['error'] = str(error_file)
        self.experiment_metadata['status'] = 'failed'
        self.experiment_metadata['error_time'] = datetime.now().isoformat()
        self._save_metadata()
    
    def _save_json_results(self, results: Dict[str, Any], file_path: Path):
        """Save results to JSON file with proper handling of numpy types."""
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _update_experiment_summary(self, training_results: Dict[str, Any], 
                                 testing_results: Dict[str, Any],
                                 evaluation_results: Dict[str, Any]):
        """Update experiment metadata with summary information."""
        summary = {}
        
        # Training summary
        training_summary = training_results.get('training_summary', {})
        summary['training'] = {
            'total_agents': training_summary.get('total_agents', 0),
            'successful_agents': training_summary.get('successful_agents', 0),
            'failed_agents': training_summary.get('failed_agents', 0),
            'best_performing_agent': training_summary.get('performance_ranking', [None])[0]
        }
        
        # Testing summary
        testing_summary = testing_results.get('testing_summary', {})
        summary['testing'] = {
            'total_agents_tested': testing_summary.get('total_agents_tested', 0),
            'successful_tests': testing_summary.get('successful_tests', 0),
            'failed_tests': testing_summary.get('failed_tests', 0),
            'most_robust_agent': testing_summary.get('robustness_ranking', [None])[0] if testing_summary.get('robustness_ranking') else None
        }
        
        # Evaluation summary
        agent_rankings = evaluation_results.get('agent_rankings', [])
        summary['evaluation'] = {
            'overall_best_agent': agent_rankings[0] if agent_rankings else None,
            'total_metrics_calculated': len(evaluation_results.get('metric_categories', {})),
            'agent_rankings': agent_rankings
        }
        
        self.experiment_metadata['metrics_summary'] = summary
    
    def _generate_experiment_report(self, training_results: Dict[str, Any], 
                                  testing_results: Dict[str, Any],
                                  evaluation_results: Dict[str, Any]):
        """Generate comprehensive experiment report."""
        report_content = []
        
        # Header
        report_content.append("="*80)
        report_content.append("HEMS ENHANCED RESEARCH PIPELINE - EXPERIMENT REPORT")
        report_content.append("="*80)
        report_content.append(f"Experiment ID: {self.experiment_id}")
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("EXECUTIVE SUMMARY")
        report_content.append("-" * 50)
        
        agent_rankings = evaluation_results.get('agent_rankings', [])
        if agent_rankings:
            report_content.append(f"Best Overall Agent: {agent_rankings[0]}")
            report_content.append(f"Performance Ranking: {', '.join(agent_rankings)}")
        
        training_summary = training_results.get('training_summary', {})
        report_content.append(f"Training Success Rate: {training_summary.get('successful_agents', 0)}/{training_summary.get('total_agents', 0)}")
        
        testing_summary = testing_results.get('testing_summary', {})
        report_content.append(f"Testing Success Rate: {testing_summary.get('successful_tests', 0)}/{testing_summary.get('total_agents_tested', 0)}")
        report_content.append("")
        
        # Training Results
        report_content.append("TRAINING RESULTS")
        report_content.append("-" * 50)
        
        convergence_summary = training_summary.get('convergence_summary', {})
        for agent_name, conv_data in convergence_summary.items():
            status = "Converged" if conv_data.get('converged', False) else "Not Converged"
            conv_episode = conv_data.get('convergence_episode', 'N/A')
            best_perf = conv_data.get('best_performance', 0)
            training_time = conv_data.get('training_time', 0)
            
            report_content.append(f"{agent_name}:")
            report_content.append(f"  Status: {status}")
            report_content.append(f"  Convergence Episode: {conv_episode}")
            report_content.append(f"  Best Performance: {best_perf:.4f}")
            report_content.append(f"  Training Time: {training_time:.2f}s")
            report_content.append("")
        
        # Testing Results
        report_content.append("TESTING RESULTS")
        report_content.append("-" * 50)
        
        robustness_ranking = testing_summary.get('robustness_ranking', [])
        generalization_ranking = testing_summary.get('generalization_ranking', [])
        
        report_content.append(f"Robustness Ranking: {', '.join(robustness_ranking)}")
        report_content.append(f"Generalization Ranking: {', '.join(generalization_ranking)}")
        report_content.append("")
        
        # Evaluation Results
        report_content.append("EVALUATION RESULTS")
        report_content.append("-" * 50)
        
        comparative_analysis = evaluation_results.get('comparative_analysis', {})
        best_in_category = comparative_analysis.get('best_in_category', {})
        
        for category, best_agent in best_in_category.items():
            category_name = category.replace('_', ' ').title()
            report_content.append(f"Best in {category_name}: {best_agent}")
        
        report_content.append("")
        
        # File Registry
        report_content.append("GENERATED FILES")
        report_content.append("-" * 50)
        
        for file_type, file_path in self.experiment_metadata['file_registry'].items():
            report_content.append(f"{file_type}: {file_path}")
        
        report_content.append("")
        report_content.append("="*80)
        
        # Save report
        report_file = self.experiment_dir / 'experiment_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_content))
        
        self.experiment_metadata['file_registry']['experiment_report'] = str(report_file)
        
        self.logger.info(f"Experiment report generated: {report_file}")
    
    def _save_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.experiment_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get current experiment information."""
        return self.experiment_metadata.copy()
    
    def load_experiment_results(self, component: str = 'all') -> Dict[str, Any]:
        """
        Load experiment results from storage.
        
        Args:
            component: Which component to load ('training', 'testing', 'evaluation', 'all')
            
        Returns:
            Loaded results
        """
        results = {}
        
        if component in ['training', 'all']:
            training_file = self.storage_structure['results'] / 'training' / 'training_results.json'
            if training_file.exists():
                with open(training_file, 'r') as f:
                    results['training'] = json.load(f)
        
        if component in ['testing', 'all']:
            testing_file = self.storage_structure['results'] / 'testing' / 'testing_results.json'
            if testing_file.exists():
                with open(testing_file, 'r') as f:
                    results['testing'] = json.load(f)
        
        if component in ['evaluation', 'all']:
            evaluation_file = self.storage_structure['results'] / 'evaluation' / 'evaluation_results.json'
            if evaluation_file.exists():
                with open(evaluation_file, 'r') as f:
                    results['evaluation'] = json.load(f)
        
        return results
    
    def archive_experiment(self, archive_location: Optional[Path] = None) -> str:
        """
        Archive the complete experiment.
        
        Args:
            archive_location: Where to create the archive (optional)
            
        Returns:
            Path to the archive file
        """
        if archive_location is None:
            archive_location = self.experiment_dir.parent
        
        archive_name = f"{self.experiment_id}_archive.zip"
        archive_path = archive_location / archive_name
        
        # Create zip archive
        shutil.make_archive(
            str(archive_path.with_suffix('')), 
            'zip', 
            str(self.experiment_dir)
        )
        
        self.logger.info(f"Experiment archived to: {archive_path}")
        return str(archive_path)
    
    def cleanup_temp_files(self):
        """Clean up temporary files while preserving important results."""
        # Remove temporary model checkpoints (keep only best models)
        checkpoint_dir = self.storage_structure['models'] / 'checkpoints'
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            self.logger.info("Cleaned up temporary checkpoint files")
        
        # Clean up large log files if needed
        logs_dir = self.storage_structure['logs']
        for log_file in logs_dir.rglob('*.log'):
            if log_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                log_file.unlink()
                self.logger.info(f"Cleaned up large log file: {log_file}")
    
    @staticmethod
    def list_experiments(experiments_root: Path) -> List[Dict[str, Any]]:
        """
        List all experiments in the experiments root directory.
        
        Args:
            experiments_root: Root directory containing experiments
            
        Returns:
            List of experiment information dictionaries
        """
        experiments = []
        
        if not experiments_root.exists():
            return experiments
        
        for exp_dir in experiments_root.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / 'experiment_metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        experiments.append(metadata)
                    except Exception:
                        # Skip corrupted metadata files
                        pass
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.get('creation_time', ''), reverse=True)
        return experiments