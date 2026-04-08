"""
hems/scenarios/__init__.py
Complete scenarios module.
"""

# Core scenario system
from .base import BaseScenario, ScenarioTemplate, ScenarioValidator
from .config import ScenarioConfig, save_scenario_config, load_scenario_config
from .registry import ScenarioRegistry, ScenarioCollection, StandardCollections
from .runner import ScenarioRunner, BatchScenarioRunner

# Import and auto-register all scenario catalogs
def _register_scenarios():
    """Auto-register all scenarios from catalogs."""
    
    # Import single building scenarios (🎯 FOR YOUR RESEARCH)
    try:
        from .catalog.single_building import (
            SingleBuildingBaseline,
            SingleBuildingTariffSensitivity,
            SingleBuildingSeasonalAnalysis,
            SingleBuildingBatteryOptimization,
            SingleBuildingPVMaximization,
            SingleBuildingDataEfficiency,
            SingleBuildingRobustness,
            SingleBuildingQuickDevelopment,
            SingleBuildingLongTermStability
        )
        
        # Register single building scenarios
        ScenarioRegistry.register('single_baseline', SingleBuildingBaseline)
        ScenarioRegistry.register('single_tariff_sensitivity', SingleBuildingTariffSensitivity)
        ScenarioRegistry.register('single_seasonal', SingleBuildingSeasonalAnalysis)
        ScenarioRegistry.register('single_battery_focus', SingleBuildingBatteryOptimization)
        ScenarioRegistry.register('single_pv_max', SingleBuildingPVMaximization)
        ScenarioRegistry.register('single_data_efficiency', SingleBuildingDataEfficiency)
        ScenarioRegistry.register('single_robustness', SingleBuildingRobustness)
        ScenarioRegistry.register('single_quick_dev', SingleBuildingQuickDevelopment)
        ScenarioRegistry.register('single_longterm', SingleBuildingLongTermStability)
        
    except ImportError as e:
        print(f"⚠️ Could not import single building scenarios: {e}")
    
    # Import benchmark scenarios
    try:
        from .catalog.benchmark import (
            BenchmarkBasicScenario,
            BenchmarkTariffComparison,
            BenchmarkScalabilityTest,
            BenchmarkLongTermPerformance,
            BenchmarkRobustnessTest,
            BenchmarkComputationalEfficiency,
            BenchmarkDataEfficiency,
            BenchmarkMultiBuildingTypes
        )
        
        # Register benchmark scenarios
        ScenarioRegistry.register('benchmark_basic', BenchmarkBasicScenario)
        ScenarioRegistry.register('benchmark_tariff_comparison', BenchmarkTariffComparison)
        ScenarioRegistry.register('benchmark_scalability', BenchmarkScalabilityTest)
        ScenarioRegistry.register('benchmark_longterm', BenchmarkLongTermPerformance)
        ScenarioRegistry.register('benchmark_robustness', BenchmarkRobustnessTest)
        ScenarioRegistry.register('benchmark_computational', BenchmarkComputationalEfficiency)
        ScenarioRegistry.register('benchmark_data_efficiency', BenchmarkDataEfficiency)
        ScenarioRegistry.register('benchmark_multi_building', BenchmarkMultiBuildingTypes)
        
    except ImportError as e:
        print(f"⚠️ Could not import benchmark scenarios: {e}")
    
    # Import research scenarios
    try:
        from .catalog.research import (
            ResearchSyntheticVsReal,
            ResearchBatteryHealthImpact,
            ResearchPVSelfConsumption,
            ResearchMultiObjectiveOptimization,
            ResearchSeasonalAdaptation,
            ResearchGridStabilityImpact,
            ResearchTransferLearning,
            ResearchUncertaintyHandling
        )
        
        # Register research scenarios
        ScenarioRegistry.register('research_synthetic_vs_real', ResearchSyntheticVsReal)
        ScenarioRegistry.register('research_battery_health', ResearchBatteryHealthImpact)
        ScenarioRegistry.register('research_pv_optimization', ResearchPVSelfConsumption)
        ScenarioRegistry.register('research_multiobjective', ResearchMultiObjectiveOptimization)
        ScenarioRegistry.register('research_seasonal_adaptation', ResearchSeasonalAdaptation)
        ScenarioRegistry.register('research_grid_stability', ResearchGridStabilityImpact)
        ScenarioRegistry.register('research_transfer_learning', ResearchTransferLearning)
        ScenarioRegistry.register('research_uncertainty_handling', ResearchUncertaintyHandling)
        
    except ImportError as e:
        print(f"⚠️ Could not import research scenarios: {e}")
    
    # Import testing scenarios
    try:
        from .catalog.testing import (
            TestingQuickValidation,
            TestingRobustnessCheck,
            TestingPerformanceProfile,
            TestingIntegrationTest,
            TestingDataConsistency,
            TestingErrorHandling,
            TestingScalabilityLimit,
            TestingRandomization,
            DevelopmentAlgorithmTesting,
            DevelopmentRewardTuning,
            DevelopmentHyperparameterSearch,
            DevelopmentEnvironmentTesting
        )
        
        # Register testing scenarios
        ScenarioRegistry.register('test_quick', TestingQuickValidation)
        ScenarioRegistry.register('test_robustness', TestingRobustnessCheck)
        ScenarioRegistry.register('test_performance', TestingPerformanceProfile)
        ScenarioRegistry.register('test_integration', TestingIntegrationTest)
        ScenarioRegistry.register('test_data_consistency', TestingDataConsistency)
        ScenarioRegistry.register('test_error_handling', TestingErrorHandling)
        ScenarioRegistry.register('test_scalability_limit', TestingScalabilityLimit)
        ScenarioRegistry.register('test_randomization', TestingRandomization)
        
        # Register development scenarios
        ScenarioRegistry.register('dev_algorithm_test', DevelopmentAlgorithmTesting)
        ScenarioRegistry.register('dev_reward_tuning', DevelopmentRewardTuning)
        ScenarioRegistry.register('dev_hyperparameter_search', DevelopmentHyperparameterSearch)
        ScenarioRegistry.register('dev_environment_test', DevelopmentEnvironmentTesting)
        
    except ImportError as e:
        print(f"⚠️ Could not import testing scenarios: {e}")


# Auto-register scenarios on import
_register_scenarios()

# Clean up namespace
del _register_scenarios

# Public API
__all__ = [
    # Core classes
    'BaseScenario',
    'ScenarioConfig',
    'ScenarioTemplate',
    'ScenarioValidator',
    
    # Registry and management
    'ScenarioRegistry',
    'ScenarioCollection',
    'StandardCollections',
    
    # Execution
    'ScenarioRunner',
    'BatchScenarioRunner',
    
    # Utilities
    'save_scenario_config',
    'load_scenario_config',
]

# Convenience functions
def list_scenarios(category=None):
    """List available scenarios."""
    return ScenarioRegistry.list_scenarios(category)

def get_scenario(name):
    """Get scenario by name."""
    return ScenarioRegistry.get_scenario(name)

def run_scenario(name, **overrides):
    """Run scenario with optional overrides."""
    runner = ScenarioRunner()
    return runner.execute_scenario_by_name(name, **overrides)

def create_custom_scenario(name, **config):
    """Create custom scenario."""
    return ScenarioRegistry.create_custom_scenario(name, **config)

# Single building research helpers
def list_single_building_scenarios():
    """List scenarios focused on single building research."""
    all_scenarios = ScenarioRegistry.list_scenarios()
    single_scenarios = {name: desc for name, desc in all_scenarios.items() 
                       if 'single' in name.lower()}
    return single_scenarios

def run_single_building_suite():
    """Run all single building research scenarios."""
    runner = ScenarioRunner()
    single_scenarios = [
        'single_baseline',
        'single_tariff_sensitivity', 
        'single_battery_focus',
        'single_pv_max',
        'single_data_efficiency'
    ]
    
    results = {}
    for scenario_name in single_scenarios:
        print(f"\n🏠 Running single building scenario: {scenario_name}")
        try:
            result = runner.execute_scenario_by_name(scenario_name)
            results[scenario_name] = result
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[scenario_name] = {'error': str(e)}
    
    return results

# Module info
def get_module_info():
    """Get information about the scenarios module."""
    stats = ScenarioRegistry.get_registry_stats()
    
    return {
        'name': 'HEMS Scenarios System',
        'version': '1.0.0',
        'description': 'Scientific scenario system for HEMS benchmarking and evaluation',
        'total_scenarios': stats['total_scenarios'],
        'categories': list(stats['categories'].keys()),
        'single_building_scenarios': len(list_single_building_scenarios()),
        'usage': 'python -m hems.scenarios --help'
    }