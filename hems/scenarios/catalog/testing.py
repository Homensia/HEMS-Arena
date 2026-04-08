"""
hems/scenarios/catalog/testing.py
Testing and development scenarios for validation and debugging.
"""

from ..base import BaseScenario
from ..config import ScenarioConfig


class TestingQuickValidation(BaseScenario):
    """
    Quick validation test for development and CI/CD.
    
    Fast execution for basic functionality testing.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_quick",
            description="Quick validation test for development",
            category="testing",
            scientific_purpose="Rapid algorithm validation and debugging",
            
            # Dataset and environment - use dummy for speed
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters - minimal for speed
            building_count=1,
            simulation_days=7,
            
            # Agent evaluation - basic set
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=20,  # Minimal training
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,  # CPU for compatibility
            random_seed=42,
            
            # Analysis
            perform_eda=False,  # Skip for speed
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="All algorithms complete without errors",
            metrics_focus=["functionality", "basic_performance", "error_handling"],
            
            experiment_name="test_quick"
        )


class TestingRobustnessCheck(BaseScenario):
    """
    Robustness testing with edge cases and stress conditions.
    
    Tests algorithm stability under challenging conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_robustness",
            description="Test algorithm robustness with extreme conditions",
            category="testing",
            scientific_purpose="Validate algorithm stability under stress conditions",
            
            # Dataset and environment - synthetic for control
            dataset_type="synthetic",
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=1,
            simulation_days=14,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=50,
            
            # Environment settings - complex tariff for stress testing
            tariff_type="tempo",
            use_gpu=False,
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Algorithms handle edge cases gracefully",
            metrics_focus=["stability", "error_handling", "convergence", "boundary_conditions"],
            
            experiment_name="test_robustness"
        )


class TestingPerformanceProfile(BaseScenario):
    """
    Performance profiling and computational benchmarking.
    
    Measures execution time, memory usage, and computational efficiency.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_performance",
            description="Profile algorithm computational performance",
            category="testing",
            scientific_purpose="Optimize computational efficiency and identify bottlenecks",
            
            # Dataset and environment - dummy for controlled timing
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters
            building_count=2,
            simulation_days=30,
            
            # Agent evaluation - focus on computational intensive algorithms
            agents_to_evaluate=["dqn", "sac"],
            train_episodes=100,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # Test GPU performance
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Identify performance bottlenecks and optimization opportunities",
            metrics_focus=["training_time", "inference_time", "memory_usage", "gpu_utilization"],
            
            experiment_name="test_performance"
        )


class TestingIntegrationTest(BaseScenario):
    """
    Full integration test across all components.
    
    Tests end-to-end functionality with realistic but controlled parameters.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_integration",
            description="Full integration test across all system components",
            category="testing",
            scientific_purpose="Validate end-to-end system functionality and component integration",
            
            # Dataset and environment - realistic but controlled
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=21,  # 3 weeks for realistic testing
            
            # Agent evaluation - all agent types
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=100,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis - full pipeline test
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="All components work together seamlessly",
            metrics_focus=["integration", "end_to_end", "component_compatibility", "data_flow"],
            
            experiment_name="test_integration"
        )


class TestingDataConsistency(BaseScenario):
    """
    Data consistency and validation testing.
    
    Ensures data integrity across different dataset types and environments.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_data_consistency",
            description="Validate data consistency across different dataset types",
            category="testing",
            scientific_purpose="Ensure data integrity and consistency in different environments",
            
            # Dataset and environment - test both types
            dataset_type="synthetic",
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=1,
            simulation_days=14,
            
            # Agent evaluation - simple for focus on data
            agents_to_evaluate=["baseline", "rbc"],
            train_episodes=20,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Important for data validation
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Data consistency across all components",
            metrics_focus=["data_integrity", "consistency", "validation", "format_compatibility"],
            
            experiment_name="test_data_consistency"
        )


class TestingErrorHandling(BaseScenario):
    """
    Error handling and recovery testing.
    
    Tests system behavior under various error conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_error_handling",
            description="Test system error handling and recovery mechanisms",
            category="testing",
            scientific_purpose="Validate robust error handling under various failure conditions",
            
            # Dataset and environment
            dataset_type="dummy",  # Controlled environment for error testing
            environment_type="dummy",
            
            # Simulation parameters
            building_count=1,
            simulation_days=7,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "dqn"],
            train_episodes=30,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Graceful error handling and meaningful error messages",
            metrics_focus=["error_handling", "recovery", "stability", "diagnostics"],
            
            experiment_name="test_error_handling"
        )


class TestingScalabilityLimit(BaseScenario):
    """
    Test scalability limits and resource constraints.
    
    Identifies system limits under increasing load.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_scalability_limit",
            description="Test system behavior at scalability limits",
            category="testing",
            scientific_purpose="Identify performance limits and resource constraints",
            
            # Dataset and environment
            dataset_type="dummy",  # Controlled for scalability testing
            environment_type="dummy",
            
            # Simulation parameters - push limits
            building_count=10,  # High building count
            simulation_days=30,
            
            # Agent evaluation - computationally intensive
            agents_to_evaluate=["baseline", "dqn"],
            train_episodes=50,  # Moderate to avoid excessive time
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # Use GPU for high-scale testing
            random_seed=42,
            
            # Analysis
            perform_eda=False,  # Skip to focus on scalability
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Identify scalability bottlenecks and resource limits",
            metrics_focus=["scalability", "resource_usage", "performance_degradation", "memory_limits"],
            
            experiment_name="test_scalability_limit"
        )


class TestingRandomization(BaseScenario):
    """
    Test reproducibility and random seed behavior.
    
    Ensures consistent results with same random seeds.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="test_randomization",
            description="Test reproducibility and random seed consistency",
            category="testing",
            scientific_purpose="Validate reproducible results and proper randomization",
            
            # Dataset and environment
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters
            building_count=1,
            simulation_days=14,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=50,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=123,  # Fixed seed for reproducibility testing
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Consistent results across runs with same seed",
            metrics_focus=["reproducibility", "randomization", "seed_consistency", "determinism"],
            
            experiment_name="test_randomization"
        )


# Development scenarios for algorithm development and tuning

class DevelopmentAlgorithmTesting(BaseScenario):
    """
    Development scenario for new algorithms.
    
    Provides controlled environment for testing new algorithm implementations.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="dev_algorithm_test",
            description="Test new algorithm implementations",
            category="development",
            scientific_purpose="Validate new algorithm implementations and compare with baselines",
            
            # Dataset and environment - simple for development
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters
            building_count=1,
            simulation_days=14,
            
            # Agent evaluation - include new algorithm slot
            agents_to_evaluate=["baseline", "rbc", "dqn"],  # Add new algorithm here
            train_episodes=50,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,  # CPU for development
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="New algorithm shows promising initial results",
            metrics_focus=["basic_functionality", "learning_curve", "comparative_performance"],
            
            experiment_name="dev_algorithm"
        )


class DevelopmentRewardTuning(BaseScenario):
    """
    Reward function development and parameter tuning.
    
    Focused testing for reward function optimization.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="dev_reward_tuning",
            description="Test and tune reward function parameters",
            category="development",
            scientific_purpose="Optimize reward function design for better learning",
            
            # Dataset and environment
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters
            building_count=1,
            simulation_days=21,
            
            # Agent evaluation - focus on RL algorithm
            agents_to_evaluate=["dqn"],  # Single RL algorithm for focused testing
            train_episodes=100,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=42,
            
            # Custom reward for tuning
            reward_config={
                'alpha_import_hp': 0.5,  # Tune these values
                'alpha_peak': 0.02,
                'alpha_pv_base': 0.15,
                'alpha_pv_soc': 0.30,
                'alpha_soc': 0.08,
                'soc_lo': 0.25,
                'soc_hi': 0.65,
            },
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Improved learning stability and performance",
            metrics_focus=["reward_components", "learning_stability", "convergence_speed"],
            
            experiment_name="dev_reward_tuning"
        )


class DevelopmentHyperparameterSearch(BaseScenario):
    """
    Hyperparameter optimization scenario.
    
    Systematic testing of algorithm hyperparameters.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="dev_hyperparameter_search",
            description="Systematic hyperparameter optimization",
            category="development",
            scientific_purpose="Find optimal hyperparameters for RL algorithms",
            
            # Dataset and environment
            dataset_type="synthetic",
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=1,
            simulation_days=21,
            
            # Agent evaluation
            agents_to_evaluate=["dqn"],  # Focus on one algorithm
            train_episodes=150,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # GPU for hyperparameter search
            random_seed=42,
            
            # Custom DQN configuration for tuning
            dqn_config={
                'action_bins': 31,
                'buffer_size': 200_000,  # Tune these values
                'batch_size': 64,        # Different batch size
                'learning_rate': 1e-4,   # Different learning rate
                'gamma': 0.95,           # Different discount factor
                'epsilon_start': 0.8,
                'epsilon_end': 0.05,
                'epsilon_decay_steps': 100_000,
                'target_update_freq': 500,
                'train_start_steps': 500,
                'hidden_layers': [128, 128]  # Different architecture
            },
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="Identify optimal hyperparameter configurations",
            metrics_focus=["learning_rate_impact", "architecture_effects", "training_stability"],
            
            experiment_name="dev_hyperparameter"
        )


class DevelopmentEnvironmentTesting(BaseScenario):
    """
    Environment and wrapper testing scenario.
    
    Tests different environment configurations and wrappers.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="dev_environment_test",
            description="Test different environment configurations and wrappers",
            category="development",
            scientific_purpose="Validate environment components and wrapper functionality",
            
            # Dataset and environment - test both types
            dataset_type="dummy",
            environment_type="dummy",
            
            # Simulation parameters
            building_count=2,
            simulation_days=14,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=50,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Test EDA with different environments
            save_plots=True,
            
            # Scientific parameters
            expected_outcome="All environment types function correctly",
            metrics_focus=["environment_compatibility", "wrapper_functionality", "data_consistency"],
            
            experiment_name="dev_environment"
        )