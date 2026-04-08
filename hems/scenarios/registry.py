"""
hems/scenarios/registry.py
Scenario registry and management system.
"""

from typing import Dict, List, Type, Optional, Any
from .base import BaseScenario
from .config import ScenarioConfig


class ScenarioRegistry:
    """
    Central registry for managing all HEMS scenarios.
    
    Provides discovery, creation, and validation of scenarios.
    """
    
    _scenarios: Dict[str, Type[BaseScenario]] = {}
    _categories: Dict[str, List[str]] = {}
    
    @classmethod
    def register(cls, name: str, scenario_class: Type[BaseScenario], 
                 category: str = None):
        """
        Register a scenario in the registry.
        
        Args:
            name: Unique scenario name
            scenario_class: Scenario class
            category: Optional category override
        """
        if not issubclass(scenario_class, BaseScenario):
            raise TypeError("Scenario class must inherit from BaseScenario")
        
        cls._scenarios[name] = scenario_class
        
        # Auto-detect category if not provided
        if category is None:
            try:
                temp_scenario = scenario_class()
                category = temp_scenario.config.category
            except Exception:
                category = 'unknown'
        
        # Update category mapping
        if category not in cls._categories:
            cls._categories[category] = []
        
        if name not in cls._categories[category]:
            cls._categories[category].append(name)
        
        print(f"✅ Registered scenario: {name} ({category})")
    
    @classmethod
    def get_scenario(cls, name: str) -> BaseScenario:
        """
        Get scenario instance by name.
        
        Args:
            name: Scenario name
            
        Returns:
            Scenario instance
            
        Raises:
            ValueError: If scenario not found
        """
        if name not in cls._scenarios:
            available = list(cls._scenarios.keys())
            raise ValueError(f"Unknown scenario: {name}. Available: {available}")
        
        return cls._scenarios[name]()
    
    @classmethod
    def list_scenarios(cls, category: Optional[str] = None) -> Dict[str, str]:
        """
        List available scenarios.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary mapping scenario names to descriptions
        """
        scenarios = {}
        
        for name, scenario_class in cls._scenarios.items():
            try:
                scenario = scenario_class()
                if category is None or scenario.config.category == category:
                    scenarios[name] = scenario.config.description
            except Exception as e:
                scenarios[name] = f"Error loading scenario: {e}"
        
        return scenarios
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """Get list of all scenario categories."""
        return sorted(cls._categories.keys())
    
    @classmethod
    def get_scenarios_by_category(cls, category: str) -> Dict[str, BaseScenario]:
        """
        Get all scenarios in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary of scenario name to scenario instance
        """
        scenarios = {}
        
        if category in cls._categories:
            for name in cls._categories[category]:
                try:
                    scenarios[name] = cls.get_scenario(name)
                except Exception as e:
                    print(f"⚠️ Error loading scenario {name}: {e}")
        
        return scenarios
    
    @classmethod
    def create_custom_scenario(cls, name: str, **kwargs) -> BaseScenario:
        """
        Create a custom scenario from parameters.
        
        Args:
            name: Scenario name
            **kwargs: ScenarioConfig parameters
            
        Returns:
            Custom scenario instance
        """
        # Set defaults for required fields
        defaults = {
            'description': f'Custom scenario: {name}',
            'category': 'custom',
            'scientific_purpose': 'Custom analysis'
        }
        
        # Merge with provided kwargs
        config_params = {**defaults, **kwargs, 'name': name}
        
        class CustomScenario(BaseScenario):
            def _create_config(self) -> ScenarioConfig:
                return ScenarioConfig(**config_params)
        
        return CustomScenario()
    
    @classmethod
    def search_scenarios(cls, query: str) -> Dict[str, str]:
        """
        Search scenarios by name, description, or scientific purpose.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of matching scenarios
        """
        query_lower = query.lower()
        matches = {}
        
        for name, scenario_class in cls._scenarios.items():
            try:
                scenario = scenario_class()
                config = scenario.config
                
                # Search in name, description, and scientific purpose
                searchable_text = f"{name} {config.description} {config.scientific_purpose}".lower()
                
                if query_lower in searchable_text:
                    matches[name] = config.description
            except Exception:
                continue
        
        return matches
    
    @classmethod
    def validate_scenario(cls, name: str) -> tuple[bool, List[str]]:
        """
        Validate a scenario configuration.
        
        Args:
            name: Scenario name
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        from .base import ScenarioValidator
        
        try:
            scenario = cls.get_scenario(name)
            return ScenarioValidator.validate_scenario_config(scenario.config)
        except Exception as e:
            return False, [f"Failed to load scenario: {e}"]
    
    @classmethod
    def get_scenario_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a scenario.
        
        Args:
            name: Scenario name
            
        Returns:
            Dictionary with scenario information
        """
        try:
            scenario = cls.get_scenario(name)
            config = scenario.config
            
            return {
                'name': config.name,
                'description': config.description,
                'category': config.category,
                'scientific_purpose': config.scientific_purpose,
                'building_count': config.building_count,
                'simulation_days': config.simulation_days,
                'agents': config.agents_to_evaluate,
                'train_episodes': config.train_episodes,
                'dataset_type': config.dataset_type,
                'environment_type': config.environment_type,
                'tariff_type': config.tariff_type,
                'use_gpu': config.use_gpu,
                'perform_eda': config.perform_eda,
                'hypothesis': config.hypothesis,
                'expected_outcome': config.expected_outcome,
                'metrics_focus': config.metrics_focus,
                'cli_command': scenario.get_cli_command()
            }
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def export_scenarios(cls, output_path: str, category: str = None):
        """
        Export scenario definitions to JSON file.
        
        Args:
            output_path: Output file path
            category: Optional category filter
        """
        import json
        
        scenarios_data = {}
        
        for name in cls._scenarios:
            try:
                scenario = cls.get_scenario(name)
                if category is None or scenario.config.category == category:
                    scenarios_data[name] = scenario.config.to_dict()
            except Exception as e:
                scenarios_data[name] = {'error': str(e)}
        
        with open(output_path, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
        
        print(f"📁 Exported {len(scenarios_data)} scenarios to: {output_path}")
    
    @classmethod
    def import_scenarios(cls, json_path: str):
        """
        Import scenario definitions from JSON file.
        
        Args:
            json_path: Path to JSON file
        """
        import json
        
        with open(json_path, 'r') as f:
            scenarios_data = json.load(f)
        
        for name, config_dict in scenarios_data.items():
            if 'error' not in config_dict:
                try:
                    custom_scenario = cls.create_custom_scenario(name, **config_dict)
                    cls._scenarios[name] = type(custom_scenario)
                    print(f"📥 Imported scenario: {name}")
                except Exception as e:
                    print(f"⚠️ Failed to import {name}: {e}")
    
    @classmethod
    def get_registry_stats(cls) -> Dict[str, Any]:
        """Get statistics about the scenario registry."""
        stats = {
            'total_scenarios': len(cls._scenarios),
            'categories': dict(cls._categories),
            'category_counts': {cat: len(scenarios) for cat, scenarios in cls._categories.items()},
            'scenario_list': list(cls._scenarios.keys())
        }
        
        return stats
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered scenarios (for testing)."""
        cls._scenarios.clear()
        cls._categories.clear()
        print("🧹 Cleared scenario registry")


class ScenarioCollection:
    """
    Collection of related scenarios for batch operations.
    
    Useful for running benchmark suites or related experiments.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize scenario collection.
        
        Args:
            name: Collection name
            description: Collection description
        """
        self.name = name
        self.description = description
        self.scenarios: List[BaseScenario] = []
    
    def add_scenario(self, scenario: BaseScenario):
        """Add scenario to collection."""
        self.scenarios.append(scenario)
        print(f"➕ Added scenario '{scenario.config.name}' to collection '{self.name}'")
    
    def add_scenario_by_name(self, scenario_name: str):
        """Add scenario to collection by name."""
        scenario = ScenarioRegistry.get_scenario(scenario_name)
        self.add_scenario(scenario)
    
    def remove_scenario(self, scenario_name: str):
        """Remove scenario from collection."""
        self.scenarios = [s for s in self.scenarios if s.config.name != scenario_name]
        print(f"➖ Removed scenario '{scenario_name}' from collection '{self.name}'")
    
    def list_scenarios(self) -> List[str]:
        """Get list of scenario names in collection."""
        return [s.config.name for s in self.scenarios]
    
    def run_collection(self) -> Dict[str, Any]:
        """
        Run all scenarios in the collection.
        
        Returns:
            Dictionary with results from all scenarios
        """
        from .runner import ScenarioRunner
        
        print(f"🚀 Running scenario collection: {self.name}")
        print(f"📋 Description: {self.description}")
        print(f"📊 Scenarios: {len(self.scenarios)}")
        
        runner = ScenarioRunner()
        results = {}
        
        for scenario in self.scenarios:
            print(f"\n{'='*60}")
            print(f"Running {scenario.config.name}...")
            
            try:
                result = runner.execute_scenario(scenario)
                results[scenario.config.name] = result
                print(f"✅ Completed: {scenario.config.name}")
            except Exception as e:
                print(f"❌ Failed: {scenario.config.name} - {e}")
                results[scenario.config.name] = {'error': str(e)}
        
        print(f"\n🏁 Collection '{self.name}' completed!")
        print(f"✅ Successful: {sum(1 for r in results.values() if 'error' not in r)}")
        print(f"❌ Failed: {sum(1 for r in results.values() if 'error' in r)}")
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get collection summary."""
        categories = {}
        for scenario in self.scenarios:
            cat = scenario.config.category
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            'name': self.name,
            'description': self.description,
            'scenario_count': len(self.scenarios),
            'categories': categories,
            'scenarios': [
                {
                    'name': s.config.name,
                    'category': s.config.category,
                    'description': s.config.description
                }
                for s in self.scenarios
            ]
        }


# Predefined scenario collections
class StandardCollections:
    """Predefined scenario collections for common use cases."""
    
    @staticmethod
    def benchmark_suite() -> ScenarioCollection:
        """Create standard benchmark suite."""
        collection = ScenarioCollection(
            name="benchmark_suite",
            description="Standard benchmark scenarios for agent evaluation"
        )
        
        benchmark_scenarios = [
            'benchmark_basic',
            'benchmark_tariff_comparison',
            'benchmark_scalability'
        ]
        
        for scenario_name in benchmark_scenarios:
            try:
                collection.add_scenario_by_name(scenario_name)
            except ValueError:
                print(f"⚠️ Benchmark scenario not found: {scenario_name}")
        
        return collection
    
    @staticmethod
    def quick_test_suite() -> ScenarioCollection:
        """Create quick test suite for development."""
        collection = ScenarioCollection(
            name="quick_test_suite",
            description="Quick test scenarios for development and validation"
        )
        
        test_scenarios = [
            'test_quick',
            'test_robustness'
        ]
        
        for scenario_name in test_scenarios:
            try:
                collection.add_scenario_by_name(scenario_name)
            except ValueError:
                print(f"⚠️ Test scenario not found: {scenario_name}")
        
        return collection
    
    @staticmethod
    def research_suite() -> ScenarioCollection:
        """Create research scenario suite."""
        collection = ScenarioCollection(
            name="research_suite",
            description="Research scenarios for scientific evaluation"
        )
        
        research_scenarios = [
            'research_battery_health',
            'research_pv_optimization',
            'research_multiobjective'
        ]
        
        for scenario_name in research_scenarios:
            try:
                collection.add_scenario_by_name(scenario_name)
            except ValueError:
                print(f"⚠️ Research scenario not found: {scenario_name}")
        
        return collection