"""
hems/scenarios/catalog/benchmark.py
Scientific benchmark scenarios for agent performance evaluation.
"""

from ..base import BaseScenario
from ..config import ScenarioConfig


class BenchmarkBasicScenario(BaseScenario):
    """
    Basic benchmark scenario for fundamental agent comparison.
    
    This scenario establishes baseline performance metrics across
    different algorithm types under standard conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_basic",
            description="Basic agent benchmarking with standard conditions",
            category="benchmark",
            scientific_purpose="Establish baseline performance comparison across different algorithms",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=30,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=False,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Advanced RL algorithms (DQN, SAC) outperform rule-based approaches in cost optimization",
            expected_outcome="5-15% cost reduction compared to baseline",
            metrics_focus=["cost", "peak_demand", "battery_utilization", "pv_self_consumption"],
            
            experiment_name="benchmark_basic"
        )


class BenchmarkTariffComparison(BaseScenario):
    """
    Benchmark agent performance across different electricity tariff structures.
    
    Evaluates algorithm robustness and adaptability to varying price signals.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_tariff_comparison",
            description="Compare agent performance across different electricity tariffs",
            category="benchmark",
            scientific_purpose="Evaluate algorithm robustness across different pricing schemes",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=45,  # Longer to capture tariff effects
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=250,
            
            # Environment settings - will be run with different tariffs
            tariff_type="tempo",  # Complex tariff for testing adaptability
            use_gpu=True,  # More complex scenario
            random_seed=42,
            
            # Analysis
            perform_eda=False,  # Focus on performance comparison
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Advanced algorithms adapt better to complex pricing structures",
            expected_outcome="Higher savings differential in complex tariffs (Tempo vs HP/HC)",
            metrics_focus=["cost", "price_responsiveness", "load_shifting", "adaptability"],
            
            experiment_name="benchmark_tariff"
        )


class BenchmarkScalabilityTest(BaseScenario):
    """
    Test algorithm scalability with increasing number of buildings.
    
    Evaluates computational efficiency and performance scaling.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_scalability",
            description="Evaluate algorithm performance with increasing number of buildings",
            category="benchmark",
            scientific_purpose="Assess computational and performance scalability",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=5,  # Can be varied: 1, 2, 5, 10
            simulation_days=30,
            
            # Agent evaluation - focus on scalable algorithms
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=300,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # Required for larger scales
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Centralized algorithms maintain efficiency with more buildings",
            expected_outcome="Linear scaling of benefits with building count",
            metrics_focus=["cost", "computation_time", "peak_demand", "scalability"],
            
            experiment_name="benchmark_scalability"
        )


class BenchmarkLongTermPerformance(BaseScenario):
    """
    Long-term performance evaluation over full year.
    
    Tests seasonal adaptation and algorithm stability over extended periods.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_longterm",
            description="Long-term algorithm performance over full year",
            category="benchmark",
            scientific_purpose="Evaluate seasonal adaptation and long-term stability",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=3,
            simulation_days=365,  # Full year
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=500,  # More training for stability
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # Long simulation needs GPU
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Important for seasonal analysis
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL agents maintain performance advantages over seasonal variations",
            expected_outcome="Consistent 10%+ savings across all seasons",
            metrics_focus=["cost", "seasonal_adaptation", "battery_health", "stability"],
            
            experiment_name="benchmark_longterm"
        )


class BenchmarkRobustnessTest(BaseScenario):
    """
    Robustness testing under various conditions.
    
    Tests algorithm performance under edge cases and stress conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_robustness",
            description="Test algorithm robustness under various stress conditions",
            category="benchmark",
            scientific_purpose="Evaluate algorithm stability under edge cases and varying conditions",
            
            # Dataset and environment
            dataset_type="synthetic",  # More control over conditions
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=3,
            simulation_days=60,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Environment settings - complex conditions
            tariff_type="tempo",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Robust algorithms maintain performance under varying conditions",
            expected_outcome="Stable performance across different scenarios",
            metrics_focus=["stability", "robustness", "error_handling", "performance_variance"],
            
            experiment_name="benchmark_robustness"
        )


class BenchmarkComputationalEfficiency(BaseScenario):
    """
    Computational efficiency benchmark.
    
    Measures training time, inference time, and resource usage.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_computational",
            description="Measure computational efficiency and resource usage",
            category="benchmark",
            scientific_purpose="Evaluate computational costs and efficiency trade-offs",
            
            # Dataset and environment
            dataset_type="dummy",  # Controlled environment for timing
            environment_type="dummy",
            
            # Simulation parameters
            building_count=2,
            simulation_days=30,
            
            # Agent evaluation - focus on computational aspects
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=100,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,  # Test both GPU and CPU
            random_seed=42,
            
            # Analysis
            perform_eda=False,  # Focus on performance metrics
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL algorithms provide performance benefits worth computational cost",
            expected_outcome="Identify optimal efficiency-performance trade-offs",
            metrics_focus=["training_time", "inference_time", "memory_usage", "cost_per_computation"],
            
            experiment_name="benchmark_computational"
        )


class BenchmarkDataEfficiency(BaseScenario):
    """
    Data efficiency benchmark - learning with limited data.
    
    Tests how quickly algorithms learn and adapt with minimal training.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_data_efficiency",
            description="Evaluate learning efficiency with limited training data",
            category="benchmark",
            scientific_purpose="Assess sample efficiency and quick adaptation capabilities",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=30,
            
            # Agent evaluation - limited training
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=50,  # Limited training episodes
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=False,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Some algorithms learn faster with limited data",
            expected_outcome="Identify most sample-efficient algorithms",
            metrics_focus=["sample_efficiency", "learning_curve", "early_performance", "convergence_speed"],
            
            experiment_name="benchmark_data_efficiency"
        )


class BenchmarkMultiBuildingTypes(BaseScenario):
    """
    Benchmark performance across different building configurations.
    
    Tests algorithm generalization across diverse building types.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="benchmark_multi_building",
            description="Test performance across different building configurations",
            category="benchmark",
            scientific_purpose="Evaluate algorithm generalization across diverse building types",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=4,  # Multiple diverse buildings
            simulation_days=45,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=250,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Important for building analysis
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Algorithms generalize well across different building types",
            expected_outcome="Consistent performance improvements across building diversity",
            metrics_focus=["generalization", "building_diversity", "cost", "adaptability"],
            
            experiment_name="benchmark_multi_building"
        )