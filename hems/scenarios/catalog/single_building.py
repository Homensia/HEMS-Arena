"""
hems/scenarios/catalog/single_building.py
Specialized scenarios for single building research under different conditions.
"""

from ..base import BaseScenario
from ..config import ScenarioConfig


class SingleBuildingBaseline(BaseScenario):
    """
    Single building baseline performance under standard conditions.
    
    Perfect for establishing baseline performance and algorithm comparison
    on a single building without multi-building complexity.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_baseline",
            description="Single building baseline performance evaluation",
            category="research",
            scientific_purpose="Establish single building baseline performance for algorithm comparison",
            
            # Single building focus
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            building_count=1,  # 🎯 SINGLE BUILDING
            building_id="Building_1",  # Specific building for consistency
            
            # Standard simulation
            simulation_days=30,
            
            # Complete agent comparison
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Standard conditions
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Full analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="Single building allows clearer algorithm performance comparison",
            expected_outcome="Clear performance ranking without multi-building interference",
            metrics_focus=["cost", "battery_utilization", "pv_self_consumption", "peak_demand"],
            
            experiment_name="single_baseline"
        )


class SingleBuildingTariffSensitivity(BaseScenario):
    """
    Single building performance across different tariff structures.
    
    Tests how algorithms adapt to different pricing signals on a single building.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_tariff_sensitivity",
            description="Single building response to different electricity tariffs",
            category="research",
            scientific_purpose="Analyze algorithm adaptation to price signals in single building context",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Extended simulation for tariff analysis
            simulation_days=60,
            
            # Focus on adaptive algorithms
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=250,
            
            # Complex tariff for sensitivity testing
            tariff_type="tempo",  # Variable pricing
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="RL algorithms show better price responsiveness than rule-based",
            expected_outcome="Higher price elasticity and better peak shifting with RL",
            metrics_focus=["price_responsiveness", "load_shifting", "cost", "peak_coincidence"],
            
            experiment_name="single_tariff_sensitivity"
        )


class SingleBuildingSeasonalAnalysis(BaseScenario):
    """
    Single building performance across different seasons.
    
    Studies seasonal adaptation and learning transfer within one building.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_seasonal",
            description="Single building seasonal adaptation and performance variation",
            category="research",
            scientific_purpose="Study seasonal learning and adaptation patterns in single building",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Full seasonal cycle
            simulation_days=120,  # 4 months across seasons
            
            # Algorithm comparison
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=300,  # More training for seasonal learning
            
            # Standard tariff
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Important for seasonal analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="RL algorithms adapt better to seasonal energy pattern changes",
            expected_outcome="Improved performance evolution and pattern recognition over seasons",
            metrics_focus=["seasonal_adaptation", "learning_curve", "pattern_recognition", "performance_stability"],
            
            experiment_name="single_seasonal"
        )


class SingleBuildingBatteryOptimization(BaseScenario):
    """
    Single building focused on battery optimization strategies.
    
    Deep dive into battery management with single building complexity.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_battery_focus",
            description="Single building battery optimization and health management",
            category="research",
            scientific_purpose="Optimize battery usage patterns and study degradation trade-offs",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Extended period for battery analysis
            simulation_days=90,  # Quarter year
            
            # Algorithm comparison
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=300,
            
            # Standard conditions
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Battery-focused reward configuration
            reward_config={
                'alpha_import_hp': 0.4,   # Reduced cost focus
                'alpha_peak': 0.01,
                'alpha_pv_base': 0.10,
                'alpha_pv_soc': 0.25,
                'alpha_soc': 0.20,       # Strong SoC management focus
                'soc_lo': 0.20,          # Strict SoC bounds for health
                'soc_hi': 0.80,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="Optimal battery management balances cost savings with battery health",
            expected_outcome="Identify SoC strategies that maximize long-term value",
            metrics_focus=["battery_cycles", "soc_distribution", "battery_health", "cost", "degradation_proxy"],
            
            experiment_name="single_battery_focus"
        )


class SingleBuildingPVMaximization(BaseScenario):
    """
    Single building PV self-consumption maximization.
    
    Focuses on maximizing solar energy utilization in single building.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_pv_max",
            description="Single building PV self-consumption maximization",
            category="research",
            scientific_purpose="Maximize renewable energy utilization and minimize grid interaction",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Focus on summer months for PV
            simulation_days=60,
            
            # Algorithm comparison
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=250,
            
            # Standard conditions
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # PV-focused reward configuration
            reward_config={
                'alpha_import_hp': 0.3,   # Reduced import focus
                'alpha_peak': 0.01,
                'alpha_pv_base': 0.35,    # Strong PV incentive
                'alpha_pv_soc': 0.45,     # Very strong PV+battery incentive
                'alpha_soc': 0.05,
                'soc_lo': 0.25,
                'soc_hi': 0.70,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="Battery scheduling significantly improves PV self-consumption",
            expected_outcome="20%+ improvement in self-consumption rate vs baseline",
            metrics_focus=["self_consumption_rate", "export_reduction", "renewable_utilization", "grid_independence"],
            
            experiment_name="single_pv_max"
        )


class SingleBuildingDataEfficiency(BaseScenario):
    """
    Single building learning efficiency with limited data.
    
    Tests how quickly algorithms learn with minimal training data.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_data_efficiency",
            description="Single building learning efficiency with limited training data",
            category="research",
            scientific_purpose="Evaluate sample efficiency and quick adaptation in single building",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Standard simulation
            simulation_days=30,
            
            # Algorithm comparison with limited training
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=50,  # 🎯 LIMITED TRAINING for efficiency testing
            
            # Standard conditions
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=False,  # Focus on learning curves
            save_plots=True,
            
            # Research parameters
            hypothesis="Some algorithms achieve good performance with minimal training data",
            expected_outcome="Identify most sample-efficient algorithms for single building",
            metrics_focus=["sample_efficiency", "learning_speed", "early_performance", "convergence_rate"],
            
            experiment_name="single_data_efficiency"
        )


class SingleBuildingRobustness(BaseScenario):
    """
    Single building robustness under uncertainty.
    
    Tests algorithm stability under various uncertainty conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_robustness",
            description="Single building robustness under uncertainty and varying conditions",
            category="research",
            scientific_purpose="Test algorithm stability and robustness in single building context",
            
            # Single building setup - use synthetic for controlled uncertainty
            dataset_type="synthetic",
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            building_count=1,
            
            # Extended testing period
            simulation_days=45,
            
            # Algorithm comparison
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Variable conditions for robustness testing
            tariff_type="tempo",  # Variable pricing adds uncertainty
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="RL algorithms handle uncertainty better than deterministic approaches",
            expected_outcome="More stable performance under varying conditions",
            metrics_focus=["robustness", "performance_variance", "adaptability", "stability"],
            
            experiment_name="single_robustness"
        )


class SingleBuildingQuickDevelopment(BaseScenario):
    """
    Quick single building scenario for algorithm development.
    
    Fast iteration for testing new algorithms and ideas.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_quick_dev",
            description="Quick single building scenario for development and testing",
            category="development",
            scientific_purpose="Rapid algorithm testing and development in single building context",
            
            # Single building setup - dummy for speed
            dataset_type="dummy",
            environment_type="dummy",
            building_count=1,
            
            # Quick testing
            simulation_days=14,
            
            # Basic agent set
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=50,  # Quick training
            
            # Simple conditions
            tariff_type="hp_hc",
            use_gpu=False,  # CPU for compatibility
            random_seed=42,
            
            # Minimal analysis for speed
            perform_eda=False,
            save_plots=True,
            
            # Development parameters
            expected_outcome="Quick validation of algorithm functionality",
            metrics_focus=["functionality", "basic_performance", "development_speed"],
            
            experiment_name="single_quick_dev"
        )


class SingleBuildingLongTermStability(BaseScenario):
    """
    Single building long-term performance and stability.
    
    Tests algorithm performance over extended periods.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="single_longterm",
            description="Single building long-term performance and stability analysis",
            category="research",
            scientific_purpose="Evaluate long-term algorithm stability and performance drift",
            
            # Single building setup
            dataset_type="original",
            environment_type="citylearn",
            building_count=1,
            building_id="Building_1",
            
            # Long-term simulation
            simulation_days=365,  # Full year
            
            # Algorithm comparison
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=500,  # Extensive training
            
            # Standard conditions
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Comprehensive analysis
            perform_eda=True,
            save_plots=True,
            
            # Research parameters
            hypothesis="RL algorithms maintain stable performance over long periods",
            expected_outcome="Consistent performance without degradation over full year",
            metrics_focus=["long_term_stability", "performance_drift", "seasonal_consistency", "learning_retention"],
            
            experiment_name="single_longterm"
        )