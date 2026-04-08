"""
Cross-Dataset Testing Module
Handles both general testing (diverse scenarios) and specific testing (same type, different data).
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import copy

from hems.agents.legacy_adapter import create_agent


class CrossDatasetTester:
    """
    Cross-dataset testing module for evaluating model generalization.
    Implements both general testing and specific testing strategies.
    """
    
    def __init__(self, config, env_manager, experiment_dir: Path, logger):
        """
        Initialize cross-dataset tester.
        
        Args:
            config: Simulation configuration
            env_manager: Environment manager
            experiment_dir: Experiment directory
            logger: Logger instance
        """
        self.config = config
        self.env_manager = env_manager
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        # Create testing directories
        self.testing_logs_dir = experiment_dir / 'logs' / 'testing'
        self.testing_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Testing configuration
        self.test_episodes = getattr(config, 'test_episodes', 5)
        
    def run_cross_dataset_testing(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive cross-dataset testing.
        
        Args:
            training_results: Results from training phase
            
        Returns:
            Testing results with general and specific testing
        """
        self.logger.info("Starting cross-dataset testing phase...")
        
        testing_results = {
            'agent_results': {},
            'testing_summary': {},
            'general_testing_scenarios': [],
            'specific_testing_scenarios': []
        }
        
        # Generate test scenarios
        general_scenarios = self._generate_general_test_scenarios()
        specific_scenarios = self._generate_specific_test_scenarios()
        
        testing_results['general_testing_scenarios'] = general_scenarios
        testing_results['specific_testing_scenarios'] = specific_scenarios
        
        # Test each trained agent
        for agent_name, agent_training_data in training_results.get('agent_results', {}).items():
            if agent_training_data.get('status') != 'completed':
                continue
                
            self.logger.info(f"Testing agent: {agent_name}")
            
            try:
                # Load best model
                best_model_path = training_results['best_models'].get(agent_name)
                if not best_model_path:
                    self.logger.warning(f"No best model found for {agent_name}")
                    continue
                
                agent = self._load_model(best_model_path)
                
                # Run general testing
                general_results = self._run_general_testing(agent, agent_name, general_scenarios)
                
                # Run specific testing
                specific_results = self._run_specific_testing(agent, agent_name, specific_scenarios)
                
                # Calculate robustness metrics
                robustness_metrics = self._calculate_robustness_metrics(
                    general_results, specific_results
                )
                
                testing_results['agent_results'][agent_name] = {
                    'status': 'completed',
                    'general_testing': general_results,
                    'specific_testing': specific_results,
                    'robustness_metrics': robustness_metrics
                }
                
            except Exception as e:
                self.logger.error(f"Testing failed for agent {agent_name}: {str(e)}")
                testing_results['agent_results'][agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate testing summary
        testing_results['testing_summary'] = self._generate_testing_summary(testing_results)
        
        self.logger.info("Cross-dataset testing phase completed")
        return testing_results
    
    def _load_model(self, model_path: str):
        """Load saved model from path."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['agent_state']
    
    def _generate_general_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate diverse test scenarios for general testing.
        
        Returns:
            List of test scenario configurations
        """
        scenarios = []
        
        # Different buildings (if multiple available)
        available_buildings = getattr(self.config, 'available_buildings', ['Building_1', 'Building_2', 'Building_3'])
        training_building = getattr(self.config, 'primary_building', 'Building_1')
        
        test_buildings = [b for b in available_buildings if b != training_building][:2]
        
        for building in test_buildings:
            scenarios.append({
                'name': f'different_building_{building}',
                'buildings': [building],
                'days': self.config.simulation_days,
                'tariff': self.config.tariff_type,
                'description': f'Test on different building: {building}'
            })
        
        # Different tariff structures
        tariff_options = ['hp_hc', 'tempo', 'standard']
        current_tariff = self.config.tariff_type
        
        for tariff in tariff_options:
            if tariff != current_tariff:
                scenarios.append({
                    'name': f'different_tariff_{tariff}',
                    'buildings': [getattr(self.config, 'primary_building', 'Building_1')],
                    'days': self.config.simulation_days,
                    'tariff': tariff,
                    'description': f'Test with different tariff: {tariff}'
                })
        
        # Extended time period
        extended_days = min(self.config.simulation_days * 2, 90)
        scenarios.append({
            'name': 'extended_period',
            'buildings': [getattr(self.config, 'primary_building', 'Building_1')],
            'days': extended_days,
            'tariff': self.config.tariff_type,
            'description': f'Extended testing period: {extended_days} days'
        })
        
        # Seasonal variation (if applicable)
        if hasattr(self.config, 'start_month'):
            different_season = (self.config.start_month + 6) % 12
            scenarios.append({
                'name': 'seasonal_variation',
                'buildings': [getattr(self.config, 'primary_building', 'Building_1')],
                'days': self.config.simulation_days,
                'tariff': self.config.tariff_type,
                'start_month': different_season,
                'description': f'Different season: month {different_season}'
            })
        
        return scenarios
    
    def _generate_specific_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate specific test scenarios (same type as training, different data).
        
        Returns:
            List of specific test scenario configurations
        """
        scenarios = []
        
        # Same building type, different time period
        base_scenario = {
            'buildings': [getattr(self.config, 'primary_building', 'Building_1')],
            'days': self.config.simulation_days,
            'tariff': self.config.tariff_type
        }
        
        # Different time periods for the same building
        time_shifts = [30, 60, 90]  # days to shift
        
        for shift in time_shifts:
            scenarios.append({
                'name': f'time_shift_{shift}days',
                'buildings': base_scenario['buildings'],
                'days': base_scenario['days'],
                'tariff': base_scenario['tariff'],
                'time_shift': shift,
                'description': f'Same conditions, {shift} days later'
            })
        
        # Same conditions, different random seed
        for seed in [42, 123, 456]:
            scenarios.append({
                'name': f'different_seed_{seed}',
                'buildings': base_scenario['buildings'],
                'days': base_scenario['days'],
                'tariff': base_scenario['tariff'],
                'random_seed': seed,
                'description': f'Same conditions, random seed {seed}'
            })
        
        return scenarios
    
    def _run_general_testing(self, agent, agent_name: str, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run general testing across diverse scenarios.
        
        Args:
            agent: Trained agent
            agent_name: Name of the agent
            scenarios: List of test scenarios
            
        Returns:
            General testing results
        """
        self.logger.info(f"Running general testing for {agent_name}")
        
        results = {
            'scenario_results': {},
            'performance_consistency': 0,
            'generalization_score': 0,
            'scenario_performances': []
        }
        
        performances = []
        
        for scenario in scenarios:
            try:
                # Create test environment for this scenario
                test_env = self._create_test_environment(scenario)
                
                # Run multiple test episodes
                scenario_performances = []
                
                for episode in range(self.test_episodes):
                    episode_performance = self._run_test_episode(agent, test_env)
                    scenario_performances.append(episode_performance)
                
                # Calculate scenario metrics
                mean_performance = np.mean(scenario_performances)
                std_performance = np.std(scenario_performances)
                
                results['scenario_results'][scenario['name']] = {
                    'mean_performance': mean_performance,
                    'std_performance': std_performance,
                    'individual_performances': scenario_performances,
                    'scenario_config': scenario
                }
                
                performances.append(mean_performance)
                
            except Exception as e:
                self.logger.error(f"General testing failed for scenario {scenario['name']}: {str(e)}")
                results['scenario_results'][scenario['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate overall metrics
        if performances:
            results['performance_consistency'] = 1.0 / (1.0 + np.std(performances))
            results['generalization_score'] = np.mean(performances)
            results['scenario_performances'] = performances
        
        return results
    
    def _run_specific_testing(self, agent, agent_name: str, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run specific testing (same type as training, different data).
        
        Args:
            agent: Trained agent
            agent_name: Name of the agent
            scenarios: List of specific test scenarios
            
        Returns:
            Specific testing results
        """
        self.logger.info(f"Running specific testing for {agent_name}")
        
        results = {
            'scenario_results': {},
            'performance_consistency': 0,
            'adaptation_score': 0,
            'scenario_performances': []
        }
        
        performances = []
        
        for scenario in scenarios:
            try:
                # Create test environment for this scenario
                test_env = self._create_test_environment(scenario)
                
                # Run multiple test episodes
                scenario_performances = []
                
                for episode in range(self.test_episodes):
                    episode_performance = self._run_test_episode(agent, test_env)
                    scenario_performances.append(episode_performance)
                
                # Calculate scenario metrics
                mean_performance = np.mean(scenario_performances)
                std_performance = np.std(scenario_performances)
                
                results['scenario_results'][scenario['name']] = {
                    'mean_performance': mean_performance,
                    'std_performance': std_performance,
                    'individual_performances': scenario_performances,
                    'scenario_config': scenario
                }
                
                performances.append(mean_performance)
                
            except Exception as e:
                self.logger.error(f"Specific testing failed for scenario {scenario['name']}: {str(e)}")
                results['scenario_results'][scenario['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate overall metrics
        if performances:
            results['performance_consistency'] = 1.0 / (1.0 + np.std(performances))
            results['adaptation_score'] = np.mean(performances)
            results['scenario_performances'] = performances
        
        return results
    
    def _create_test_environment(self, scenario: Dict[str, Any]):
        """Create test environment based on scenario configuration."""
        # Create a copy of the original config
        test_config = copy.deepcopy(self.config)
        
        # Update config with scenario parameters
        if 'buildings' in scenario:
            test_config.primary_building = scenario['buildings'][0]
        if 'days' in scenario:
            test_config.simulation_days = scenario['days']
        if 'tariff' in scenario:
            test_config.tariff_type = scenario['tariff']
        if 'random_seed' in scenario:
            test_config.random_seed = scenario['random_seed']
        if 'start_month' in scenario:
            test_config.start_month = scenario['start_month']
        
        # Create environment with modified config
        return self.env_manager.get_evaluation_environment(test_config)
    
    def _run_test_episode(self, agent, test_env) -> float:
        """
        Run a single test episode.
        
        Args:
            agent: Agent to test
            test_env: Test environment
            
        Returns:
            Episode performance (total reward)
        """
        # Reset environment - handle both old and new gym formats
        reset_result = test_env.reset()
        if isinstance(reset_result, tuple):
            observations, info = reset_result
        else:
            observations = reset_result
            info = {}
        
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 1000  # Safety limit
        
        while not done and step_count < max_steps:
            try:
                # Agent action (deterministic for testing)
                actions = agent.act(observations, deterministic=True)
                
                # Ensure actions are in correct format for different agents
                if hasattr(agent, 'name') and 'rbc' in agent.name.lower():
                    # Handle RBC action format
                    if isinstance(actions, list):
                        actions = np.array(actions, dtype=np.float32)
                
                # Environment step - handle both old and new gym formats
                step_result = test_env.step(actions)
                
                if len(step_result) == 5:
                    # New gym format: (obs, reward, terminated, truncated, info)
                    observations, rewards, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old gym format: (obs, reward, done, info)
                    observations, rewards, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step result format: {len(step_result)} values")
                
                # Accumulate reward
                if isinstance(rewards, (list, np.ndarray)):
                    total_reward += sum(rewards)
                else:
                    total_reward += float(rewards)
                
                step_count += 1
                
            except Exception as step_error:
                # Log error but continue or break depending on severity
                self.logger.warning(f"Test step error: {step_error}")
                break
        
        return total_reward
        
    def _calculate_robustness_metrics(self, general_results: Dict[str, Any], 
                                     specific_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate robustness metrics from testing results.
        
        Args:
            general_results: General testing results
            specific_results: Specific testing results
            
        Returns:
            Robustness metrics
        """
        metrics = {
            'robustness_index': 0,
            'adaptation_speed': 0,
            'performance_variance': 0,
            'consistency_score': 0
        }
        
        # Get all performances
        all_performances = []
        
        # General testing performances
        general_perfs = general_results.get('scenario_performances', [])
        specific_perfs = specific_results.get('scenario_performances', [])
        
        all_performances.extend(general_perfs)
        all_performances.extend(specific_perfs)
        
        if all_performances:
            # Robustness index (inverse of coefficient of variation)
            mean_perf = np.mean(all_performances)
            std_perf = np.std(all_performances)
            
            if mean_perf != 0:
                cv = std_perf / abs(mean_perf)
                metrics['robustness_index'] = 1.0 / (1.0 + cv)
            
            # Performance variance
            metrics['performance_variance'] = float(std_perf)
            
            # Consistency score (specific vs general)
            if general_perfs and specific_perfs:
                general_mean = np.mean(general_perfs)
                specific_mean = np.mean(specific_perfs)
                
                if general_mean != 0:
                    consistency = 1.0 - abs(specific_mean - general_mean) / abs(general_mean)
                    metrics['consistency_score'] = max(0, consistency)
        
        return metrics
    
    def _generate_testing_summary(self, testing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive testing summary."""
        summary = {
            'total_agents_tested': len(testing_results['agent_results']),
            'successful_tests': 0,
            'failed_tests': 0,
            'robustness_ranking': [],
            'generalization_ranking': [],
            'testing_insights': {}
        }
        
        agent_robustness = []
        agent_generalization = []
        
        for agent_name, results in testing_results['agent_results'].items():
            if results.get('status') == 'completed':
                summary['successful_tests'] += 1
                
                # Extract metrics
                robustness = results.get('robustness_metrics', {}).get('robustness_index', 0)
                generalization = results.get('general_testing', {}).get('generalization_score', 0)
                
                agent_robustness.append((agent_name, robustness))
                agent_generalization.append((agent_name, generalization))
                
                # Testing insights
                general_consistency = results.get('general_testing', {}).get('performance_consistency', 0)
                specific_consistency = results.get('specific_testing', {}).get('performance_consistency', 0)
                
                summary['testing_insights'][agent_name] = {
                    'general_consistency': general_consistency,
                    'specific_consistency': specific_consistency,
                    'robustness_index': robustness,
                    'better_at': 'general' if general_consistency > specific_consistency else 'specific'
                }
            else:
                summary['failed_tests'] += 1
        
        # Sort rankings
        agent_robustness.sort(key=lambda x: x[1], reverse=True)
        agent_generalization.sort(key=lambda x: x[1], reverse=True)
        
        summary['robustness_ranking'] = [agent[0] for agent in agent_robustness]
        summary['generalization_ranking'] = [agent[0] for agent in agent_generalization]
        
        return summary