"""
Enhanced HEMS Runner Module
Orchestrates comprehensive research pipeline including training, testing, evaluation, and visualization.
"""

import os
import time
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json

from hems.environments import HEMSEnvironment 
from hems.core.config import SimulationConfig
from hems.agents.legacy_adapter import create_agent, AGENT_REGISTRY
from hems.analysis.eda import HEMSDataAnalyzer
from hems.visualization.plots import HEMSVisualizer
from hems.utils.utils import setup_logger, save_config, calculate_cost_savings, ProgressTracker

# Import new enhanced modules
from hems.core.training.enhanced_trainer import EnhancedTrainer
from hems.core.testing.cross_dataset_tester import CrossDatasetTester
from hems.core.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from hems.core.visualization.research_visualizer import ResearchVisualizer
from hems.core.persistence.experiment_manager import ExperimentManager


class HEMSRunner:
    """
    Enhanced HEMS runner that orchestrates comprehensive research pipeline.
    Maintains full backward compatibility while adding research-grade capabilities.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize enhanced HEMS runner.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.logger = setup_logger('HEMSRunner', config.output_dir)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique experiment ID
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self.output_dir / f"{self.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment manager
        self.env_manager = HEMSEnvironment(config)
        
        # Initialize enhanced components
        self._initialize_research_components()
        
        # Save configuration
        save_config(config, self.experiment_dir)
        
        self.logger.info(f"Enhanced HEMS Runner initialized with experiment ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def _initialize_research_components(self):
        """Initialize all research pipeline components."""
        # Enhanced trainer
        self.trainer = EnhancedTrainer(
            config=self.config,
            env_manager=self.env_manager,
            experiment_dir=self.experiment_dir,
            logger=self.logger
        )
        
        # Cross-dataset tester
        self.tester = CrossDatasetTester(
            config=self.config,
            env_manager=self.env_manager,
            experiment_dir=self.experiment_dir,
            logger=self.logger
        )
        
        # Comprehensive evaluator
        self.evaluator = ComprehensiveEvaluator(
            config=self.config,
            experiment_dir=self.experiment_dir,
            logger=self.logger
        )
        
        # Research visualizer (enhanced plots)
        self.research_visualizer = ResearchVisualizer(
            experiment_dir=self.experiment_dir,
            logger=self.logger
        )
        
        # Legacy visualizer (maintain compatibility)
        self.visualizer = HEMSVisualizer(self.experiment_dir / 'plots')
        
        # Experiment manager for persistence
        self.experiment_manager = ExperimentManager(
            experiment_dir=self.experiment_dir,
            experiment_id=self.experiment_id,
            logger=self.logger
        )
    
    def run_eda(self) -> Dict[str, Any]:
        """
        Run exploratory data analysis.
        
        Returns:
            EDA results dictionary
        """
        self.logger.info("Starting Exploratory Data Analysis...")
        
        analyzer = HEMSDataAnalyzer(self.config)
        eda_results = analyzer.run_full_analysis()
        
        # Save EDA results
        self.experiment_manager.save_eda_results(eda_results)
        
        self.logger.info("EDA completed successfully")
        return eda_results
    
    def run_comparison(self) -> Dict[str, Any]:
        """
        Run enhanced research pipeline with comprehensive agent comparison.
        
        Returns:
            Complete results dictionary with all phases
        """
        self.logger.info(f"Starting Enhanced Research Pipeline - Experiment: {self.experiment_id}")
        
        start_time = time.time()
        
        try:
            # Phase 1: Enhanced Training
            self.logger.info("Phase 1: Enhanced Training with Analytics")
            training_results = self._execute_training_phase()
            
            #Phase 2: Cross-Dataset Testing
            #self.logger.info("Phase 2: Cross-Dataset Testing")
            #testing_results = self._execute_testing_phase(training_results)

            testing_results = {
            'agent_results': {},
            'testing_summary': {},
            'general_testing_scenarios': [],
            'specific_testing_scenarios': []
}
            
            # Phase 3: Comprehensive Evaluation
            self.logger.info("Phase 3: Comprehensive Evaluation")
            evaluation_results = self._execute_evaluation_phase(training_results, testing_results)
            
            # Phase 4: Advanced Visualization
            self.logger.info("Phase 4: Advanced Visualization")
            visualization_results = self._execute_visualization_phase(
                training_results, testing_results, evaluation_results
            )
            
            # Phase 5: Results Persistence
            self.logger.info("Phase 5: Results Persistence")
            self._execute_persistence_phase(
                training_results, testing_results, evaluation_results
            )
            
            # Compile final results
            final_results = self._compile_final_results(
                training_results, testing_results, evaluation_results, visualization_results
            )
            
            execution_time = time.time() - start_time
            final_results['execution_metadata'] = {
                'experiment_id': self.experiment_id,
                'start_time': start_time,
                'end_time': time.time(),
                'execution_time_seconds': execution_time,
                'status': 'completed',
                'output_directory': str(self.experiment_dir)
            }
            
            self.logger.info(f"Enhanced Research Pipeline completed in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            error_results = {
                'status': 'failed',
                'error': str(e),
                'experiment_id': self.experiment_id,
                'execution_time': time.time() - start_time
            }
            self.experiment_manager.save_error_results(error_results)
            raise
    
    def _execute_training_phase(self) -> Dict[str, Any]:
        """Execute enhanced training phase with analytics and model persistence."""
        return self.trainer.run_enhanced_training()
    
    def _execute_testing_phase(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cross-dataset testing phase."""
        return self.tester.run_cross_dataset_testing(training_results)
    
    def _execute_evaluation_phase(self, training_results: Dict[str, Any], 
                                 testing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive evaluation phase."""
        return self.evaluator.run_comprehensive_evaluation(training_results, testing_results)
    
    def _execute_visualization_phase(self, training_results: Dict[str, Any], 
                                   testing_results: Dict[str, Any],
                                   evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced visualization phase."""
        return self.research_visualizer.generate_research_visualizations(
            training_results, testing_results, evaluation_results
        )
    
    def _execute_persistence_phase(self, training_results: Dict[str, Any], 
                                 testing_results: Dict[str, Any],
                                 evaluation_results: Dict[str, Any]):
        """Execute experiment persistence phase."""
        self.experiment_manager.save_experiment_results(
            training_results, testing_results, evaluation_results
        )
    
    def _compile_final_results(self, training_results: Dict[str, Any], 
                             testing_results: Dict[str, Any],
                             evaluation_results: Dict[str, Any],
                             visualization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all results into final comprehensive dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'training_results': training_results,
            'testing_results': testing_results,
            'evaluation_results': evaluation_results,
            'visualization_results': visualization_results,
            'summary': self._generate_experiment_summary(
                training_results, testing_results, evaluation_results
            )
        }
    
    def _generate_experiment_summary(self, training_results: Dict[str, Any], 
                                   testing_results: Dict[str, Any],
                                   evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level experiment summary."""
        summary = {
            'best_agent_overall': None,
            'performance_ranking': [],
            'key_insights': [],
            'training_convergence': {},
            'generalization_performance': {},
            'top_metrics': {}
        }
        
        # Extract best performing agent
        if evaluation_results and 'agent_rankings' in evaluation_results:
            summary['best_agent_overall'] = evaluation_results['agent_rankings'][0]
            summary['performance_ranking'] = evaluation_results['agent_rankings']
        
        # Training convergence summary
        for agent_name, agent_results in training_results.get('agent_results', {}).items():
            if 'training_analytics' in agent_results:
                analytics = agent_results['training_analytics']
                summary['training_convergence'][agent_name] = {
                    'converged': analytics.get('convergence_episode', 0) > 0,
                    'convergence_episode': analytics.get('convergence_episode', 'Not converged'),
                    'best_performance': analytics.get('best_performance', 0),
                    'training_time': analytics.get('training_time', 0)
                }
        
        # Generalization performance
        if testing_results:
            for agent_name in testing_results.get('agent_results', {}):
                general_test = testing_results['agent_results'][agent_name].get('general_testing', {})
                specific_test = testing_results['agent_results'][agent_name].get('specific_testing', {})
                
                summary['generalization_performance'][agent_name] = {
                    'general_test_score': general_test.get('performance_consistency', 0),
                    'specific_test_score': specific_test.get('performance_consistency', 0),
                    'robustness_index': testing_results['agent_results'][agent_name].get('robustness_index', 0)
                }
        
        return summary
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive experiment summary."""
        print("\n" + "="*80)
        print(f"HEMS Enhanced Research Pipeline Results - Experiment: {self.experiment_id}")
        print("="*80)
        
        summary = results.get('summary', {})
        
        # Best agent
        if summary.get('best_agent_overall'):
            print(f"\n🏆 Best Overall Agent: {summary['best_agent_overall']}")
        
        # Performance ranking
        if summary.get('performance_ranking'):
            print(f"\n📊 Performance Ranking:")
            for i, agent in enumerate(summary['performance_ranking'], 1):
                print(f"  {i}. {agent}")
        
        # Training summary
        print(f"\n🎯 Training Performance:")
        for agent_name, conv_data in summary.get('training_convergence', {}).items():
            status = "✅ Converged" if conv_data['converged'] else "❌ Not Converged"
            print(f"  {agent_name}: {status} | Best: {conv_data['best_performance']:.2f} | Time: {conv_data['training_time']:.1f}s")
        
        # Testing summary
        print(f"\n🧪 Generalization Performance:")
        for agent_name, gen_data in summary.get('generalization_performance', {}).items():
            print(f"  {agent_name}: General Test: {gen_data['general_test_score']:.3f} | Specific Test: {gen_data['specific_test_score']:.3f}")
        
        # Execution metadata
        if 'execution_metadata' in results:
            meta = results['execution_metadata']
            print(f"\n⏱️  Execution Time: {meta['execution_time_seconds']:.2f} seconds")
            print(f"📁 Results Directory: {meta['output_directory']}")
        
        print("\n" + "="*80)