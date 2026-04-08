"""
hems/scenarios/base.py
Base classes for scenario system with robust architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from .config import ScenarioConfig


class BaseScenario(ABC):
    """
    Abstract base class for all HEMS scenarios.
    
    Provides the foundation for creating scientific, reproducible scenarios
    that can be easily executed and compared.
    """
    
    def __init__(self):
        """Initialize base scenario."""
        self.config = self._create_config()
        self._validate_config()
    
    @abstractmethod
    def _create_config(self) -> ScenarioConfig:
        """
        Create scenario configuration.
        
        This method must be implemented by each scenario to define
        its specific parameters and scientific purpose.
        """
        pass
    
    def _validate_config(self):
        """Validate scenario configuration."""
        # Basic validation
        assert self.config.name, "Scenario must have a name"
        assert self.config.description, "Scenario must have a description"
        assert self.config.category, "Scenario must have a category"
        assert self.config.scientific_purpose, "Scenario must have a scientific purpose"
        
        # Parameter validation
        assert self.config.building_count > 0, "Building count must be positive"
        assert self.config.simulation_days > 0, "Simulation days must be positive"
        assert len(self.config.agents_to_evaluate) > 0, "Must evaluate at least one agent"
        assert self.config.train_episodes > 0, "Training episodes must be positive"
    
    def get_config(self) -> ScenarioConfig:
        """Get scenario configuration."""
        return self.config
    
    def get_simulation_config(self):
        """Get SimulationConfig for execution."""
        return self.config.to_simulation_config()
    
    def run(self) -> Dict[str, Any]:
        """
        Execute this scenario.
        
        Returns:
            Dictionary with execution results
        """
        from .runner import ScenarioRunner
        
        print(f"🚀 Executing Scenario: {self.config.name}")
        print(f"📋 Description: {self.config.description}")
        print(f"🔬 Scientific Purpose: {self.config.scientific_purpose}")
        print(f"📊 Category: {self.config.category}")
        
        runner = ScenarioRunner()
        return runner.execute_scenario(self)
    
    def get_cli_command(self) -> str:
        """Get equivalent CLI command for this scenario."""
        return self.config.get_cli_command()
    
    def info(self) -> str:
        """Get detailed scenario information."""
        info = f"""
🎯 Scenario: {self.config.name}
📝 Description: {self.config.description}
🔬 Scientific Purpose: {self.config.scientific_purpose}
📂 Category: {self.config.category}

📊 Configuration:
  - Dataset: {self.config.dataset_type}
  - Environment: {self.config.environment_type}
  - Buildings: {self.config.building_count}
  - Days: {self.config.simulation_days}
  - Agents: {', '.join(self.config.agents_to_evaluate)}
  - Training Episodes: {self.config.train_episodes}
  - Tariff: {self.config.tariff_type}
  - GPU: {self.config.use_gpu}
  - EDA: {self.config.perform_eda}

🎯 Target Metrics: {', '.join(self.config.metrics_focus)}
"""
        
        if self.config.hypothesis:
            info += f"💡 Hypothesis: {self.config.hypothesis}\n"
        
        if self.config.expected_outcome:
            info += f"🎯 Expected Outcome: {self.config.expected_outcome}\n"
        
        info += f"\n🖥️ Equivalent CLI Command:\n{self.get_cli_command()}"
        
        return info
    
    def save_config(self, output_path: str):
        """Save scenario configuration to file."""
        from .config import save_scenario_config
        save_scenario_config(self.config, output_path)
    
    def __str__(self) -> str:
        """String representation of scenario."""
        return f"Scenario(name='{self.config.name}', category='{self.config.category}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Scenario(name='{self.config.name}', "
                f"category='{self.config.category}', "
                f"buildings={self.config.building_count}, "
                f"days={self.config.simulation_days}, "
                f"agents={len(self.config.agents_to_evaluate)})")


class ScenarioTemplate:
    """
    Template for creating new scenarios easily.
    
    This class provides common scenario patterns and configurations.
    """
    
    @staticmethod
    def quick_test(name: str = "quick_test", 
                   agents: list = None,
                   buildings: int = 1,
                   days: int = 7) -> ScenarioConfig:
        """Create a quick test scenario template."""
        if agents is None:
            agents = ['baseline', 'rbc']
        
        return ScenarioConfig(
            name=name,
            description=f"Quick test scenario with {buildings} buildings for {days} days",
            category='testing',
            scientific_purpose='Quick validation and testing',
            
            dataset_type='dummy',
            environment_type='dummy',
            building_count=buildings,
            simulation_days=days,
            agents_to_evaluate=agents,
            train_episodes=20,
            
            perform_eda=False,
            save_plots=True,
            
            metrics_focus=['functionality', 'basic_performance']
        )
    
    @staticmethod
    def benchmark_base(name: str = "benchmark_base",
                       agents: list = None,
                       buildings: int = 2,
                       days: int = 30) -> ScenarioConfig:
        """Create a benchmark scenario template."""
        if agents is None:
            agents = ['baseline', 'rbc', 'dqn', 'sac']
        
        return ScenarioConfig(
            name=name,
            description=f"Benchmark scenario comparing {len(agents)} agents",
            category='benchmark',
            scientific_purpose='Comparative performance evaluation',
            
            dataset_type='original',
            environment_type='citylearn',
            building_count=buildings,
            simulation_days=days,
            agents_to_evaluate=agents,
            train_episodes=200,
            
            perform_eda=True,
            save_plots=True,
            tariff_type='hp_hc',
            
            hypothesis='Advanced algorithms outperform rule-based approaches',
            expected_outcome='5-15% cost reduction vs baseline',
            metrics_focus=['cost', 'peak_demand', 'battery_utilization']
        )
    
    @staticmethod
    def research_template(name: str,
                          scientific_purpose: str,
                          hypothesis: str = None,
                          expected_outcome: str = None,
                          **kwargs) -> ScenarioConfig:
        """Create a research scenario template."""
        default_config = {
            'description': f'Research scenario: {scientific_purpose}',
            'category': 'research',
            'scientific_purpose': scientific_purpose,
            
            'dataset_type': 'original',
            'environment_type': 'citylearn',
            'building_count': 2,
            'simulation_days': 30,
            'agents_to_evaluate': ['baseline', 'rbc', 'dqn'],
            'train_episodes': 200,
            
            'perform_eda': True,
            'save_plots': True,
            'tariff_type': 'hp_hc',
            
            'hypothesis': hypothesis,
            'expected_outcome': expected_outcome,
            'metrics_focus': ['cost', 'performance', 'efficiency']
        }
        
        # Update with user-provided parameters
        default_config.update(kwargs)
        default_config['name'] = name
        
        return ScenarioConfig(**default_config)
    
    @staticmethod
    def development_template(name: str,
                            purpose: str = "Algorithm development and testing",
                            **kwargs) -> ScenarioConfig:
        """Create a development scenario template."""
        default_config = {
            'description': f'Development scenario: {purpose}',
            'category': 'development',
            'scientific_purpose': purpose,
            
            'dataset_type': 'dummy',
            'environment_type': 'dummy',
            'building_count': 1,
            'simulation_days': 14,
            'agents_to_evaluate': ['baseline', 'dqn'],
            'train_episodes': 50,
            
            'perform_eda': False,
            'save_plots': True,
            'tariff_type': 'hp_hc',
            
            'metrics_focus': ['functionality', 'learning_curve']
        }
        
        default_config.update(kwargs)
        default_config['name'] = name
        
        return ScenarioConfig(**default_config)


class ScenarioValidator:
    """Validator for scenario configurations."""
    
    @staticmethod
    def validate_scenario_config(config: ScenarioConfig) -> tuple[bool, list[str]]:
        """
        Validate scenario configuration.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Required fields
        required_fields = ['name', 'description', 'category', 'scientific_purpose']
        for field in required_fields:
            if not getattr(config, field, None):
                issues.append(f"Missing required field: {field}")
        
        # Numeric validations
        if config.building_count <= 0:
            issues.append("building_count must be positive")
        elif config.building_count > 15:
            issues.append("building_count should not exceed 15 for performance")
        
        if config.simulation_days <= 0:
            issues.append("simulation_days must be positive")
        elif config.simulation_days > 365:
            issues.append("simulation_days should not exceed 365")
        
        if config.train_episodes <= 0:
            issues.append("train_episodes must be positive")
        
        # Agent validation
        valid_agents = ['baseline', 'rbc', 'dqn', 'tql', 'sac', 'custom']
        for agent in config.agents_to_evaluate:
            if agent not in valid_agents:
                issues.append(f"Unknown agent: {agent}")
        
        # Dataset validation
        valid_dataset_types = ['original', 'synthetic', 'dummy']
        if config.dataset_type not in valid_dataset_types:
            issues.append(f"Invalid dataset_type: {config.dataset_type}")
        
        # Environment validation
        valid_env_types = ['citylearn', 'dummy']
        if config.environment_type not in valid_env_types:
            issues.append(f"Invalid environment_type: {config.environment_type}")
        
        # Category validation
        valid_categories = ['benchmark', 'research', 'testing', 'development']
        if config.category not in valid_categories:
            issues.append(f"Invalid category: {config.category}")
        
        # Tariff validation
        valid_tariffs = ['hp_hc', 'tempo', 'standard', 'default']
        if config.tariff_type not in valid_tariffs:
            issues.append(f"Invalid tariff_type: {config.tariff_type}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_scenario_compatibility(config: ScenarioConfig) -> tuple[bool, list[str]]:
        """
        Validate scenario compatibility with current system.
        
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        
        # Performance warnings
        if config.building_count > 10 and not config.use_gpu:
            warnings.append("Consider enabling GPU for >10 buildings")
        
        if config.simulation_days > 180 and 'sac' in config.agents_to_evaluate:
            warnings.append("Long simulations with SAC may be slow")
        
        if config.train_episodes > 1000:
            warnings.append("High training episodes may take considerable time")
        
        # Dataset compatibility
        if config.dataset_type == 'synthetic' and config.environment_type != 'citylearn':
            warnings.append("Synthetic datasets work best with citylearn environment")
        
        if config.dataset_type == 'dummy' and config.perform_eda:
            warnings.append("EDA on dummy data may not be meaningful")
        
        # Agent compatibility
        if 'tql' in config.agents_to_evaluate and config.building_count > 5:
            warnings.append("TQL may not scale well with many buildings")
        
        is_compatible = len(warnings) == 0
        return is_compatible, warnings