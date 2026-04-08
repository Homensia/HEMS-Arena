"""
Environment Factory for HEMS Framework
Provides unified interface for creating different types of environments.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .base import EnvironmentRegistry, BaseHEMSEnvironment


class EnvironmentFactory:
    """
    Factory for creating HEMS environments.
    
    This class provides a unified interface for creating different types of
    environments (CityLearn, dummy, custom) while handling configuration
    and validation automatically.
    """
    
    def __init__(self):
        """Initialize environment factory."""
        # Import environment implementations to register them
        try:
            from .citylearn.citylearn_wrapper import CityLearnEnvironmentManager
        except ImportError:
            pass
        
        try:
            from .dummy.dummy_env import DummyEnvironmentManager
        except ImportError:
            pass
        # Future imports will be added here automatically
    
    @staticmethod
    def create_environment(config) -> BaseHEMSEnvironment:
        """
        Create environment based on configuration.
        
        Args:
            config: Configuration object with environment settings
            
        Returns:
            Environment instance
            
        Raises:
            ValueError: If environment type not supported
            RuntimeError: If environment cannot be created
        """
        # Determine environment type
        env_type = EnvironmentFactory._determine_environment_type(config)
        
        # Create environment using registry
        try:
            environment = EnvironmentRegistry.create_environment(env_type, config)
            
            # Validate environment
            if not environment.validate_configuration():
                raise RuntimeError(f"Environment configuration validation failed for {env_type}")
            
            print(f"Successfully created {env_type} environment")
            return environment
            
        except Exception as e:
            print(f"Failed to create {env_type} environment: {e}")
            
            # Try fallback to dummy environment for development
            if env_type != 'dummy':
                print("Attempting fallback to dummy environment...")
                try:
                    return EnvironmentRegistry.create_environment('dummy', config)
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
            
            raise RuntimeError(f"Could not create any environment: {e}")
    
    @staticmethod
    def _determine_environment_type(config) -> str:
        """
        Determine environment type from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Environment type string
        """
        # Check if explicitly specified
        if hasattr(config, 'environment_type'):
            return config.environment_type
        
        # Check if dataset type implies environment type
        if hasattr(config, 'dataset_type'):
            if config.dataset_type in ['original', 'synthetic']:
                return 'citylearn'
            elif config.dataset_type == 'dummy':
                return 'dummy'
        
        # Check if dataset name implies environment type
        if hasattr(config, 'dataset_name'):
            if 'citylearn' in config.dataset_name.lower():
                return 'citylearn'
            elif 'dummy' in config.dataset_name.lower() or 'test' in config.dataset_name.lower():
                return 'dummy'
        
        # Default to CityLearn if available, otherwise dummy
        available_envs = EnvironmentRegistry.get_available_environments()
        if 'citylearn' in available_envs:
            return 'citylearn'
        elif 'dummy' in available_envs:
            return 'dummy'
        else:
            raise ValueError("No suitable environment type found")
    
    @staticmethod
    def list_available_environments() -> Dict[str, Dict[str, Any]]:
        """
        List all available environments with their information.
        
        Returns:
            Dictionary with environment information
        """
        return EnvironmentRegistry.list_environments()
    
    @staticmethod
    def get_supported_datasets(env_type: str = None) -> Dict[str, List[str]]:
        """
        Get supported datasets for environment types.
        
        Args:
            env_type: Specific environment type (optional)
            
        Returns:
            Dictionary mapping environment types to supported datasets
        """
        if env_type:
            # Get datasets for specific environment type
            try:
                env = EnvironmentRegistry.create_environment(env_type, None)
                return {env_type: env.get_supported_datasets()}
            except Exception:
                return {env_type: []}
        else:
            # Get datasets for all environment types
            datasets = {}
            for env_name in EnvironmentRegistry.get_available_environments():
                try:
                    env = EnvironmentRegistry.create_environment(env_name, None)
                    datasets[env_name] = env.get_supported_datasets()
                except Exception:
                    datasets[env_name] = []
            return datasets
    
    @staticmethod
    def validate_environment_config(config) -> Tuple[bool, List[str]]:
        """
        Validate environment configuration without creating environment.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check basic configuration attributes
            required_attrs = ['building_count', 'simulation_days']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    issues.append(f"Missing required configuration: {attr}")
            
            # Validate building count
            if hasattr(config, 'building_count'):
                if not isinstance(config.building_count, int) or config.building_count <= 0:
                    issues.append("building_count must be a positive integer")
                elif config.building_count > 15:
                    issues.append("building_count should not exceed 15 for performance reasons")
            
            # Validate simulation days
            if hasattr(config, 'simulation_days'):
                if not isinstance(config.simulation_days, int) or config.simulation_days <= 0:
                    issues.append("simulation_days must be a positive integer")
                elif config.simulation_days > 365:
                    issues.append("simulation_days should not exceed 365 for performance reasons")
            
            # Check environment type availability
            env_type = EnvironmentFactory._determine_environment_type(config)
            available_envs = EnvironmentRegistry.get_available_environments()
            if env_type not in available_envs:
                issues.append(f"Environment type '{env_type}' is not available")
            
            # Validate dataset
            if hasattr(config, 'dataset_name') and env_type in available_envs:
                try:
                    env = EnvironmentRegistry.create_environment(env_type, config)
                    supported_datasets = env.get_supported_datasets()
                    if config.dataset_name not in supported_datasets:
                        issues.append(f"Dataset '{config.dataset_name}' not supported by {env_type} environment")
                except Exception as e:
                    issues.append(f"Could not validate dataset: {e}")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class HEMSEnvironmentManager:
    """
    Enhanced HEMS Environment Manager with factory pattern.
    
    This replaces the original HEMSEnvironment class with a cleaner,
    more modular design that supports multiple environment types.
    """
    
    def __init__(self, config):
        """
        Initialize enhanced HEMS environment manager.
        
        Args:
            config: Enhanced SimulationConfig
        """
        self.config = config
        self.factory = EnvironmentFactory()
        
        # Create environment instance
        self.environment = self.factory.create_environment(config)
        
        print(f"HEMS Environment Manager initialized:")
        print(f"  Environment Type: {type(self.environment).__name__}")
        print(f"  Dataset: {config.get_active_dataset_name()}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        return self.environment.get_dataset_info()
    
    def select_buildings(self, exclude_buildings: List[str] = None) -> List[str]:
        """Select buildings based on configuration."""
        return self.environment.select_buildings(exclude_buildings)
    
    def select_simulation_period(self) -> Tuple[int, int]:
        """Select simulation period."""
        return self.environment.select_simulation_period()
    
    def create_base_environment(self, buildings: List[str], start_time: int, end_time: int):
        """Create base environment."""
        return self.environment.create_environment(buildings, start_time, end_time)
    
    def inject_tariff(self, env) -> np.ndarray:
        """Inject custom electricity tariff into environment."""
        return self.environment.inject_tariff(env)
    
    def create_environment_for_agent(self, agent_type: str, buildings: List[str], 
                                   start_time: int, end_time: int):
        """Create environment configured for specific agent type."""
        return self.environment.create_environment_for_agent(
            agent_type, buildings, start_time, end_time
        )
    
    def print_environment_info(self, env):
        """Print detailed environment information."""
        self.environment.print_environment_info(env)
    
    @staticmethod
    def list_available_environments():
        """List all available environment types."""
        return EnvironmentFactory.list_available_environments()
    
    @staticmethod
    def get_supported_datasets(env_type: str = None):
        """Get supported datasets for environment types."""
        return EnvironmentFactory.get_supported_datasets(env_type)


    def get_training_environment(self):
        """
        Get training environment (required by EnhancedTrainer).
        
        Returns:
            Training environment instance
        """
        # Select buildings and period for training
        buildings = self.select_buildings()
        start_time, end_time = self.select_simulation_period()
        
        # Create training environment
        training_env = self.create_base_environment(buildings, start_time, end_time)
        
        # Inject tariff if configured
        self.inject_tariff(training_env)
        
        return training_env    


    def get_evaluation_environment(self, config=None):
        """
        Get evaluation environment (required by testing modules).
        
        Args:
            config: Optional configuration override
            
        Returns:
            Evaluation environment instance
        """
        if config is not None:
            # Create new environment manager with different config
            eval_manager = HEMSEnvironmentManager(config)
            return eval_manager.get_training_environment()
        else:
            # Use same configuration as training
            return self.get_training_environment()    


# Backward compatibility alias
HEMSEnvironment = HEMSEnvironmentManager