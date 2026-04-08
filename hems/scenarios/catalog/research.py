"""
hems/scenarios/catalog/research.py
Scientific research scenarios for advanced studies and hypothesis testing.
"""

from ..base import BaseScenario
from ..config import ScenarioConfig


class ResearchSyntheticVsReal(BaseScenario):
    """
    Compare algorithm performance on synthetic vs real datasets.
    
    Validates synthetic data generation for research purposes.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_synthetic_vs_real",
            description="Compare algorithm performance on synthetic vs original datasets",
            category="research",
            scientific_purpose="Validate synthetic data generator for research purposes",
            
            # Dataset and environment - will run both synthetic and real
            dataset_type="synthetic",
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=30,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Important for data comparison
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Synthetic data produces similar performance patterns as real data",
            expected_outcome="<5% difference in relative performance rankings",
            metrics_focus=["cost", "peak_demand", "performance_correlation", "data_validity"],
            
            experiment_name="research_synthetic_validation"
        )


class ResearchBatteryHealthImpact(BaseScenario):
    """
    Study battery health vs cost optimization trade-offs.
    
    Analyzes the impact of aggressive battery usage on long-term economics.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_battery_health",
            description="Analyze trade-offs between cost optimization and battery health",
            category="research",
            scientific_purpose="Quantify battery degradation impact on long-term economics",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=1,  # Focus on single building for detailed analysis
            simulation_days=90,  # Quarter year for battery analysis
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=300,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Custom reward configuration focused on battery health
            reward_config={
                'alpha_import_hp': 0.4,  # Reduced cost focus
                'alpha_peak': 0.01,
                'alpha_pv_base': 0.10,
                'alpha_pv_soc': 0.25,
                'alpha_soc': 0.15,  # Increased SoC management focus
                'soc_lo': 0.20,      # Stricter SoC bounds
                'soc_hi': 0.80,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Aggressive charging strategies reduce long-term profitability due to battery degradation",
            expected_outcome="Identify optimal SoC management strategies balancing cost and health",
            metrics_focus=["cost", "battery_cycles", "soc_distribution", "charging_patterns", "degradation_proxy"],
            
            experiment_name="research_battery_health"
        )


class ResearchPVSelfConsumption(BaseScenario):
    """
    Optimize PV self-consumption strategies.
    
    Develops strategies for maximizing renewable energy utilization.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_pv_optimization",
            description="Maximize PV self-consumption while minimizing costs",
            category="research",
            scientific_purpose="Develop strategies for maximizing renewable energy utilization",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=60,  # Focus on summer months for PV
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=250,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Custom reward configuration for PV optimization
            reward_config={
                'alpha_import_hp': 0.3,  # Reduced import cost focus
                'alpha_peak': 0.01,
                'alpha_pv_base': 0.30,   # Strong PV incentive
                'alpha_pv_soc': 0.50,    # Very strong PV+battery incentive
                'alpha_soc': 0.05,
                'soc_lo': 0.30,
                'soc_hi': 0.60,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Battery scheduling can significantly improve PV utilization rates",
            expected_outcome="20%+ improvement in self-consumption rates compared to baseline",
            metrics_focus=["self_consumption_rate", "export_reduction", "cost", "grid_interaction", "renewable_utilization"],
            
            experiment_name="research_pv_selfconsumption"
        )


class ResearchMultiObjectiveOptimization(BaseScenario):
    """
    Multi-objective optimization research.
    
    Balances multiple competing objectives in energy management.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_multiobjective",
            description="Balance multiple objectives: cost, peak demand, grid stability",
            category="research",
            scientific_purpose="Develop Pareto-optimal energy management strategies",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=3,
            simulation_days=45,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=250,
            
            # Environment settings
            tariff_type="tempo",  # Complex pricing for multi-objective analysis
            use_gpu=True,
            random_seed=42,
            
            # Balanced reward configuration
            reward_config={
                'alpha_import_hp': 0.4,  # Balanced cost focus
                'alpha_peak': 0.05,      # Increased peak penalty
                'alpha_pv_base': 0.15,
                'alpha_pv_soc': 0.30,
                'alpha_soc': 0.08,      # Balanced SoC management
                'soc_lo': 0.25,
                'soc_hi': 0.70,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL agents can balance competing objectives better than rule-based approaches",
            expected_outcome="Pareto-optimal solutions across cost-peak-stability spectrum",
            metrics_focus=["cost", "peak_demand", "grid_stability", "load_factor", "pareto_efficiency"],
            
            experiment_name="research_multiobjective"
        )


class ResearchSeasonalAdaptation(BaseScenario):
    """
    Study seasonal adaptation and learning transfer.
    
    Analyzes how algorithms adapt to seasonal variations in energy patterns.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_seasonal_adaptation",
            description="Study algorithm adaptation to seasonal energy pattern variations",
            category="research",
            scientific_purpose="Analyze learning transfer and adaptation across different seasons",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=120,  # Four months across seasons
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=300,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,  # Important for seasonal analysis
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL algorithms adapt faster to seasonal changes than rule-based systems",
            expected_outcome="Improved performance as algorithms learn seasonal patterns",
            metrics_focus=["seasonal_adaptation", "learning_transfer", "performance_evolution", "pattern_recognition"],
            
            experiment_name="research_seasonal_adaptation"
        )


class ResearchGridStabilityImpact(BaseScenario):
    """
    Study impact of energy management on grid stability.
    
    Analyzes how distributed energy management affects grid-level metrics.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_grid_stability",
            description="Analyze energy management impact on grid stability and load balancing",
            category="research",
            scientific_purpose="Quantify grid-level benefits of coordinated energy management",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=5,  # Multiple buildings for grid analysis
            simulation_days=60,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn"],
            train_episodes=200,
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Grid-focused reward configuration
            reward_config={
                'alpha_import_hp': 0.3,
                'alpha_peak': 0.15,      # Strong peak reduction focus
                'alpha_pv_base': 0.10,
                'alpha_pv_soc': 0.25,
                'alpha_soc': 0.05,
                'soc_lo': 0.30,
                'soc_hi': 0.60,
            },
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="Coordinated energy management reduces grid stress and improves stability",
            expected_outcome="Reduced peak demand and improved load factor at district level",
            metrics_focus=["grid_stability", "peak_coincidence", "load_factor", "ramping", "voltage_support"],
            
            experiment_name="research_grid_stability"
        )


class ResearchTransferLearning(BaseScenario):
    """
    Study transfer learning between different building types.
    
    Tests how well algorithms trained on one building type perform on others.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_transfer_learning",
            description="Study knowledge transfer between different building types and conditions",
            category="research",
            scientific_purpose="Evaluate algorithm generalization and transfer learning capabilities",
            
            # Dataset and environment
            dataset_type="original",
            dataset_name="citylearn_challenge_2022_phase_all",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=1,  # Focus on single building per run
            simulation_days=30,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=150,  # Moderate training
            
            # Environment settings
            tariff_type="hp_hc",
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL algorithms show positive transfer between similar building types",
            expected_outcome="Faster learning and better initial performance on similar buildings",
            metrics_focus=["transfer_efficiency", "generalization", "learning_acceleration", "domain_adaptation"],
            
            experiment_name="research_transfer_learning"
        )


class ResearchUncertaintyHandling(BaseScenario):
    """
    Study algorithm robustness to uncertainty in forecasts and parameters.
    
    Tests performance under imperfect information conditions.
    """
    
    def _create_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="research_uncertainty_handling",
            description="Test algorithm robustness to forecast uncertainty and parameter variations",
            category="research",
            scientific_purpose="Evaluate performance under uncertainty and imperfect information",
            
            # Dataset and environment
            dataset_type="synthetic",  # Better control over uncertainty
            synthetic_dataset_name="demo_1",
            environment_type="citylearn",
            
            # Simulation parameters
            building_count=2,
            simulation_days=45,
            
            # Agent evaluation
            agents_to_evaluate=["baseline", "rbc", "dqn", "sac"],
            train_episodes=200,
            
            # Environment settings
            tariff_type="tempo",  # Variable pricing adds uncertainty
            use_gpu=True,
            random_seed=42,
            
            # Analysis
            perform_eda=True,
            save_plots=True,
            
            # Scientific parameters
            hypothesis="RL algorithms handle uncertainty better than deterministic rule-based approaches",
            expected_outcome="More robust performance under varying conditions",
            metrics_focus=["robustness", "uncertainty_tolerance", "performance_variance", "adaptability"],
            
            experiment_name="research_uncertainty_handling"
        )