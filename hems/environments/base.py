"""
Base classes for HEMS environment abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np


class BaseHEMSEnvironment(ABC):
    """
    Abstract base class for all HEMS environments.
    
    This defines the common interface that all HEMS environments must implement,
    regardless of their underlying simulation engine (CityLearn, custom, etc.).
    """
    
    def __init__(self, config):
        """
        Initialize environment.
        
        Args:
            config: Environment configuration object
        """
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def create_environment(self, buildings: List[str], start_time: int, end_time: int):
        """
        Create the actual simulation environment.
        
        Args:
            buildings: List of building identifiers
            start_time: Simulation start timestep
            end_time: Simulation end timestep
            
        Returns:
            Configured environment instance
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset/environment.
        
        Returns:
            Dictionary with dataset metadata
        """
        pass
    
    @abstractmethod
    def select_buildings(self, exclude_buildings: List[str] = None) -> List[str]:
        """
        Select buildings based on configuration.
        
        Args:
            exclude_buildings: Buildings to exclude from selection
            
        Returns:
            List of selected building identifiers
        """
        pass
    
    @abstractmethod
    def select_simulation_period(self) -> Tuple[int, int]:
        """
        Select simulation time period.
        
        Returns:
            Tuple of (start_time, end_time)
        """
        pass
    
    @abstractmethod
    def create_environment_for_agent(self, agent_type: str, buildings: List[str], 
                                   start_time: int, end_time: int):
        """
        Create environment configured for specific agent type.
        
        Args:
            agent_type: Type of agent (dqn, sac, tql, etc.)
            buildings: Selected buildings
            start_time: Start timestep
            end_time: End timestep
            
        Returns:
            Agent-specific environment
        """
        pass
    
    def validate_configuration(self) -> bool:
        """
        Validate environment configuration.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def print_environment_info(self, env):
        """
        Print detailed environment information.
        
        Args:
            env: Environment instance
        """
        dataset_info = self.get_dataset_info()
        
        print("\n" + "="*60)
        print("ENVIRONMENT INFORMATION")
        print("="*60)
        print(f"Environment Type: {self.name}")
        print(f"Dataset: {dataset_info.get('dataset_name', 'N/A')}")
        print(f"Buildings: {getattr(env, 'buildings', 'N/A')}")
        print(f"Time steps: {getattr(env, 'time_steps', 'N/A')}")
        print("="*60)


class BaseEnvironmentManager(ABC):
    """
    Abstract base class for environment managers.
    
    Environment managers handle the lifecycle and configuration of environments,
    including dataset loading, building selection, and environment creation.
    """
    
    def __init__(self, config):
        """
        Initialize environment manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this environment type is available.
        
        Returns:
            True if environment can be created
        """
        pass
    
    @abstractmethod
    def get_supported_datasets(self) -> List[str]:
        """
        Get list of supported datasets.
        
        Returns:
            List of dataset names
        """
        pass
    
    @abstractmethod
    def create_environment_instance(self) -> BaseHEMSEnvironment:
        """
        Create environment instance.
        
        Returns:
            Environment instance
        """
        pass


class EnvironmentRegistry:
    """Registry for environment types."""
    
    _environments = {}
    
    @classmethod
    def register(cls, name: str, manager_class: type):
        """
        Register an environment manager.
        
        Args:
            name: Environment type name
            manager_class: Manager class
        """
        cls._environments[name] = manager_class
    
    @classmethod
    def create_environment(cls, env_type: str, config) -> BaseHEMSEnvironment:
        """
        Create environment of specified type.
        
        Args:
            env_type: Environment type name
            config: Configuration object
            
        Returns:
            Environment instance
            
        Raises:
            ValueError: If environment type not registered
        """
        if env_type not in cls._environments:
            available = list(cls._environments.keys())
            raise ValueError(f"Unknown environment type: {env_type}. Available: {available}")
        
        manager_class = cls._environments[env_type]
        manager = manager_class(config)
        
        if not manager.is_available():
            raise RuntimeError(f"Environment type {env_type} is not available")
        
        return manager.create_environment_instance()
    
    @classmethod
    def get_available_environments(cls) -> List[str]:
        """Get list of available environment types."""
        available = []
        for name, manager_class in cls._environments.items():
            try:
                # Create dummy config to test availability
                manager = manager_class(None)
                if manager.is_available():
                    available.append(name)
            except Exception:
                pass
        return available
    
    @classmethod
    def list_environments(cls) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all environments."""
        info = {}
        for name, manager_class in cls._environments.items():
            try:
                manager = manager_class(None)
                info[name] = {
                    'available': manager.is_available(),
                    'supported_datasets': manager.get_supported_datasets() if manager.is_available() else [],
                    'description': manager_class.__doc__ or 'No description'
                }
            except Exception as e:
                info[name] = {
                    'available': False,
                    'error': str(e),
                    'description': manager_class.__doc__ or 'No description'
                }
        return info