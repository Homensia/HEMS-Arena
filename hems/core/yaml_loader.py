#==============================
#hems/core/benchmark_runner.py
#==============================

"""
YAML Configuration Loader for HEMS Benchmarks 
Loads and validates YAML/JSON benchmark configurations.

Key Features:
- Loads benchmark configs from YAML/JSON
- Validates configuration structure
- Checks building overlap between train/val/test
- Converts to SimulationConfig for backward compatibility
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class BuildingSelection:
    """Building selection configuration."""
    selection: str  # "manual", "random", "all", "different"
    ids: List[str] = field(default_factory=list)
    count: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    mode: str  # "sequential", "parallel", "both"
    episodes: int
    buildings: BuildingSelection
    save_frequency: int = 50
    save_best: bool = True
    save_final: bool = True
    early_stopping: Optional[Dict[str, Any]] = None


@dataclass
class ValidationConfig:
    """Validation configuration."""
    enabled: bool
    episodes: int = 20
    frequency: int = 50
    buildings: Optional[BuildingSelection] = None


@dataclass
class TestingConfig:
    """Testing configuration."""
    enabled: bool
    mode: str  # "normal", "general", "specific"
    episodes: int
    buildings: Optional[BuildingSelection] = None
    run_general_scenarios: bool = False
    run_specific_scenarios: bool = False


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""
    name: str
    description: str
    seed: int
    output_dir: str
    use_gpu: bool
    
    dataset: Dict[str, Any]
    environment: Dict[str, Any]
    tariff: Dict[str, Any]
    agents: List[Dict[str, Any]]
    training: TrainingConfig
    validation: ValidationConfig
    testing: TestingConfig
    reward: Dict[str, Any]
    analysis: Dict[str, Any]
    advanced: Dict[str, Any]
    
    # Metadata
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


# ============================================================================
# YAML Configuration Loader
# ============================================================================

class YAMLConfigLoader:
    """
    Loader for YAML/JSON benchmark configurations.
    Validates and converts to appropriate config formats.
    """
    
    def __init__(self):
        """Initialize YAML config loader."""
        self.config_cache = {}
        self.logger = logger
        
    def load(self, config_path: Union[str, Path]) -> BenchmarkConfig:
        """
        Load benchmark configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Parsed benchmark configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading configuration from: {config_path}")
        
        # Load file based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = self._load_yaml(config_path)
        elif config_path.suffix == '.json':
            config_dict = self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Parse and validate
        benchmark_config = self._parse_config(config_dict)
        self._validate_config(benchmark_config)
        
        self.logger.info(f"[OK] Configuration loaded: {benchmark_config.name}")
        return benchmark_config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file: {e}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> BenchmarkConfig:
        """
        Parse configuration dictionary into BenchmarkConfig.
        
        Args:
            config_dict: Raw configuration dictionary
            
        Returns:
            Structured BenchmarkConfig object
        """
        benchmark = config_dict.get('benchmark', {})
        
        # Parse building selections
        training_buildings = self._parse_building_selection(
            benchmark.get('training', {}).get('buildings', {})
        )
        
        validation_buildings = None
        if benchmark.get('validation', {}).get('enabled', False):
            validation_buildings = self._parse_building_selection(
                benchmark.get('validation', {}).get('buildings', {})
            )
        
        testing_buildings = None
        if benchmark.get('testing', {}).get('enabled', False):
            testing_buildings = self._parse_building_selection(
                benchmark.get('testing', {}).get('buildings', {})
            )
        
        # Create configurations
        training_cfg = TrainingConfig(
            mode=benchmark.get('training', {}).get('mode', 'sequential'),
            episodes=benchmark.get('training', {}).get('episodes', 100),
            buildings=training_buildings,
            save_frequency=benchmark.get('training', {}).get('save_frequency', 50),
            save_best=benchmark.get('training', {}).get('save_best', True),
            save_final=benchmark.get('training', {}).get('save_final', True),
            early_stopping=benchmark.get('training', {}).get('early_stopping')
        )
        
        validation_cfg = ValidationConfig(
            enabled=benchmark.get('validation', {}).get('enabled', False),
            episodes=benchmark.get('validation', {}).get('episodes', 20),
            frequency=benchmark.get('validation', {}).get('frequency', 50),
            buildings=validation_buildings
        )
        
        testing_cfg = TestingConfig(
            enabled=benchmark.get('testing', {}).get('enabled', True),
            mode=benchmark.get('testing', {}).get('mode', 'normal'),
            episodes=benchmark.get('testing', {}).get('episodes', 50),
            buildings=testing_buildings,
            run_general_scenarios=benchmark.get('testing', {}).get('run_general_scenarios', False),
            run_specific_scenarios=benchmark.get('testing', {}).get('run_specific_scenarios', False)
        )
        
        # Create benchmark config
        return BenchmarkConfig(
            name=benchmark.get('name', 'Unnamed Benchmark'),
            description=benchmark.get('description', ''),
            author=benchmark.get('author'),
            tags=benchmark.get('tags', []),
            seed=benchmark.get('seed', 42),
            output_dir=benchmark.get('output_dir', 'experiments/benchmark'),
            use_gpu=benchmark.get('use_gpu', False),
            dataset=benchmark.get('dataset', {}),
            environment=benchmark.get('environment', {}),
            tariff=benchmark.get('tariff', {}),
            agents=benchmark.get('agents', []),
            training=training_cfg,
            validation=validation_cfg,
            testing=testing_cfg,
            reward=benchmark.get('reward', {}),
            analysis=benchmark.get('analysis', {}),
            advanced=benchmark.get('advanced', {})
        )
    
    def _parse_building_selection(self, buildings_dict: Dict[str, Any]) -> BuildingSelection:
        """Parse building selection configuration."""
        return BuildingSelection(
            selection=buildings_dict.get('selection', 'manual'),
            ids=buildings_dict.get('ids', []),
            count=buildings_dict.get('count')
        )
    
    def _validate_config(self, config: BenchmarkConfig):
        """
        Validate benchmark configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Validate training mode
        valid_modes = ['sequential', 'round_robin', 'parallel', 'both']
        if config.training.mode not in valid_modes:
            errors.append(f"Invalid training mode: {config.training.mode}. Must be one of {valid_modes}")
        
        # Validate episodes
        if config.training.episodes <= 0:
            errors.append(f"Training episodes must be > 0, got {config.training.episodes}")
        
        if config.validation.enabled and config.validation.episodes <= 0:
            errors.append(f"Validation episodes must be > 0, got {config.validation.episodes}")
        
        if config.testing.enabled and config.testing.episodes <= 0:
            errors.append(f"Testing episodes must be > 0, got {config.testing.episodes}")
        
        # Validate building selection
        valid_selections = ['manual', 'random', 'all', 'different']
        if config.training.buildings.selection not in valid_selections:
            errors.append(f"Invalid building selection: {config.training.buildings.selection}")
        
        # Validate manual building selection
        if config.training.buildings.selection == 'manual':
            if not config.training.buildings.ids:
                errors.append("Manual building selection requires 'ids' list")
        
        # Validate agents
        if not config.agents:
            errors.append("At least one agent must be specified")
        
        enabled_agents = [a for a in config.agents if a.get('enabled', True)]
        if not enabled_agents:
            errors.append("At least one agent must be enabled")
        
        # Validate testing mode
        valid_test_modes = ['normal', 'general', 'specific']
        if config.testing.mode not in valid_test_modes:
            errors.append(f"Invalid testing mode: {config.testing.mode}")
        
        # Validate seed
        if config.seed < 0:
            errors.append(f"Seed must be >= 0, got {config.seed}")
        
        # Check building overlap
        overlap_analysis = self.validate_building_overlap(config)
        if overlap_analysis['has_overlap']:
            errors.append("Building overlap detected between train/validation/test sets:")
            if overlap_analysis['train_val_overlap']:
                errors.append(f"  Train/Val overlap: {overlap_analysis['train_val_overlap']}")
            if overlap_analysis['train_test_overlap']:
                errors.append(f"  Train/Test overlap: {overlap_analysis['train_test_overlap']}")
            if overlap_analysis['val_test_overlap']:
                errors.append(f"  Val/Test overlap: {overlap_analysis['val_test_overlap']}")
        
        # Raise if errors found
        if errors:
            error_msg = "\n".join(["Configuration validation failed:"] + errors)
            raise ValueError(error_msg)
        
        self.logger.info(f"[OK] Configuration validated successfully")
    
    def validate_building_overlap(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Check for building overlap between train/validation/test sets.
        
        Returns:
            Dictionary with overlap analysis
        """
        train_buildings = set(config.training.buildings.ids) if config.training.buildings.ids else set()
        
        val_buildings = set()
        if config.validation.enabled and config.validation.buildings:
            val_buildings = set(config.validation.buildings.ids) if config.validation.buildings.ids else set()
        
        test_buildings = set()
        if config.testing.enabled and config.testing.buildings:
            test_buildings = set(config.testing.buildings.ids) if config.testing.buildings.ids else set()
        
        # Check overlaps
        train_val_overlap = train_buildings & val_buildings
        train_test_overlap = train_buildings & test_buildings
        val_test_overlap = val_buildings & test_buildings
        
        has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)
        
        return {
            'has_overlap': has_overlap,
            'train_val_overlap': list(train_val_overlap),
            'train_test_overlap': list(train_test_overlap),
            'val_test_overlap': list(val_test_overlap),
            'train_buildings': list(train_buildings),
            'val_buildings': list(val_buildings),
            'test_buildings': list(test_buildings)
        }
    
    def to_simulation_config(self, benchmark_config: BenchmarkConfig):
        """
        Convert BenchmarkConfig to SimulationConfig for backward compatibility.
        
        Args:
            benchmark_config: Benchmark configuration
            
        Returns:
            SimulationConfig instance
            
        Note:
            This imports SimulationConfig locally to avoid circular dependencies
        """
        from hems.core.config import SimulationConfig
        
        # Extract agent names
        agent_names = [
            agent['name'] for agent in benchmark_config.agents 
            if agent.get('enabled', True)
        ]
        
        # Create base simulation config
        sim_config = SimulationConfig(
            # Basic settings
            building_count=len(benchmark_config.training.buildings.ids),
            simulation_days=benchmark_config.dataset.get('simulation_days', 30),
            dataset_name=benchmark_config.dataset.get('name', 'citylearn_challenge_2022_phase_all'),
            
            # Environment
            environment_type=benchmark_config.environment.get('type', 'citylearn'),
            
            # Agent settings
            agents_to_evaluate=agent_names,
            train_episodes=benchmark_config.training.episodes,
            
            # Hardware
            use_gpu=benchmark_config.use_gpu,
            
            # Seed
            random_seed=benchmark_config.seed,
            
            # Output
            output_dir=benchmark_config.output_dir,
            save_plots=benchmark_config.analysis.get('save_plots', True),
            perform_eda=benchmark_config.analysis.get('perform_eda', False),
            
            # Observations
            active_observations=benchmark_config.advanced.get('active_observations', [
                'hour', 'electricity_pricing', 'net_electricity_consumption',
                'electrical_storage_soc', 'solar_generation'
            ])
        )
        
        # Tariff settings (use defaults from SimulationConfig, don't override)
        # CityLearn will use its own pricing data
        
        # Add agent-specific configurations
        for agent in benchmark_config.agents:
            if not agent.get('enabled', True):
                continue
                
            agent_name = agent['name']
            agent_config = agent.get('config', {})
            
            # Set agent config in SimulationConfig
            config_attr = f"{agent_name}_config"
            if hasattr(sim_config, config_attr):
                # Merge with defaults
                default_config = getattr(sim_config, config_attr)
                merged_config = {**default_config, **agent_config}
                setattr(sim_config, config_attr, merged_config)
        
        return sim_config


# ============================================================================
# Helper Functions
# ============================================================================

def load_benchmark_config(config_path: Union[str, Path]) -> BenchmarkConfig:
    """
    Convenience function to load benchmark configuration.
    
    Args:
        config_path: Path to YAML/JSON config file
        
    Returns:
        Parsed benchmark configuration
    """
    loader = YAMLConfigLoader()
    return loader.load(config_path)


def validate_benchmark_config(config_path: Union[str, Path]) -> bool:
    """
    Validate benchmark configuration without full loading.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        load_benchmark_config(config_path)
        return True
    except Exception as e:
        logger.error(f"[FAIL] Configuration validation failed: {e}")
        raise


# ============================================================================
# Main - For Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"Loading config: {config_path}")
        
        try:
            config = load_benchmark_config(config_path)
            print(f"[OK] Successfully loaded: {config.name}")
            print(f"   Mode: {config.training.mode}")
            print(f"   Buildings: {config.training.buildings.ids}")
            print(f"   Agents: {[a['name'] for a in config.agents if a.get('enabled', True)]}")
            print(f"   Seed: {config.seed}")
        except Exception as e:
            print(f"[FAIL] Failed to load config: {e}")
            sys.exit(1)
    else:
        print("Usage: python yaml_loader.py <config_path>")