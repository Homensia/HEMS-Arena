"""
Algorithm registry for easy algorithm creation and discovery.
"""

from typing import Dict, Type
from .base import BaseAlgorithm
from .baseline import BaselineAlgorithm
from .rule_based import RuleBasedAlgorithm
from .dqn import DQNAlgorithm
from .Memdqn import MemDQNAlgorithm
from .mp_ppo import MPPPOAlgorithm
from .mpc_forecast import MPCForecastAlgorithm
from .ambitious_engineers_algorithm import AmbitiousEngineersAlgorithm

# Import other algorithms conditionally
try:
    from .tabular_ql import TabularQLearningAlgorithm
    TQL_AVAILABLE = True
except ImportError:
    TQL_AVAILABLE = False

try:
    from .sac import SACAlgorithm
    SAC_AVAILABLE = True
except ImportError:
    SAC_AVAILABLE = False


# Algorithm registry mapping names to classes
ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    'baseline': BaselineAlgorithm,
    'rbc': RuleBasedAlgorithm, 
    'rule_based': RuleBasedAlgorithm,  # Alias
    'dqn': DQNAlgorithm,
    'Memdqn' : MemDQNAlgorithm,
    'mp_ppo': MPPPOAlgorithm,
    'mpc_forecast': MPCForecastAlgorithm,
    'ambitious_engineers': AmbitiousEngineersAlgorithm,
}

# Add optional algorithms if available
if TQL_AVAILABLE:
    ALGORITHM_REGISTRY['tql'] = TabularQLearningAlgorithm
    ALGORITHM_REGISTRY['tabular_ql'] = TabularQLearningAlgorithm  # Alias

if SAC_AVAILABLE:
    ALGORITHM_REGISTRY['sac'] = SACAlgorithm


def get_available_algorithms() -> Dict[str, str]:
    """
    Get list of available algorithms with descriptions.
    
    Returns:
        Dictionary mapping algorithm names to descriptions
    """
    descriptions = {}
    for name, algo_class in ALGORITHM_REGISTRY.items():
        # Create a dummy instance to get info (not ideal but works for now)
        try:
            # Skip creating actual instance, just use class info
            if hasattr(algo_class, 'get_info'):
                descriptions[name] = f"{algo_class.__name__}: {algo_class.__doc__ or 'No description'}"
            else:
                descriptions[name] = f"{algo_class.__name__}: {algo_class.__doc__ or 'No description'}"
        except Exception:
            descriptions[name] = f"{algo_class.__name__}: Available"
    
    return descriptions


def create_algorithm(algorithm_name: str, env, config: Dict[str, any]) -> BaseAlgorithm:
    """
    Create algorithm instance from registry.
    
    Args:
        algorithm_name: Name of algorithm to create
        env: CityLearn environment
        config: Algorithm configuration
        
    Returns:
        Instantiated algorithm
        
    Raises:
        ValueError: If algorithm name not found
        ImportError: If algorithm dependencies not available
    """
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")
    
    algorithm_class = ALGORITHM_REGISTRY[algorithm_name]
    
    try:
        return algorithm_class(env, config)
    except Exception as e:
        raise ImportError(f"Failed to create algorithm {algorithm_name}: {e}")


def register_algorithm(name: str, algorithm_class: Type[BaseAlgorithm]):
    """
    Register a new algorithm.
    
    Args:
        name: Algorithm name
        algorithm_class: Algorithm class (must inherit from BaseAlgorithm)
        
    Raises:
        TypeError: If algorithm_class doesn't inherit from BaseAlgorithm
    """
    if not issubclass(algorithm_class, BaseAlgorithm):
        raise TypeError(f"Algorithm class must inherit from BaseAlgorithm")
    
    ALGORITHM_REGISTRY[name] = algorithm_class
    print(f"Registered new algorithm: {name} -> {algorithm_class.__name__}")


# Utility functions for algorithm management
def list_algorithms() -> list:
    """Get list of all available algorithm names."""
    return list(ALGORITHM_REGISTRY.keys())


def is_algorithm_available(name: str) -> bool:
    """Check if algorithm is available."""
    return name in ALGORITHM_REGISTRY


def get_algorithm_info(name: str) -> Dict[str, any]:
    """
    Get information about a specific algorithm.
    
    Args:
        name: Algorithm name
        
    Returns:
        Algorithm information dictionary
        
    Raises:
        ValueError: If algorithm not found
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm {name} not found")
    
    algo_class = ALGORITHM_REGISTRY[name]
    return {
        'name': name,
        'class': algo_class.__name__,
        'description': algo_class.__doc__ or 'No description available',
        'module': algo_class.__module__
    }