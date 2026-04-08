"""
Strategy registry for easy strategy creation and discovery.
"""

from typing import Dict, Type
from .base import BaseStrategy
from .single_agent import SingleAgentStrategy

# Strategy registry mapping names to classes
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'single_agent': SingleAgentStrategy,
    'centralized': SingleAgentStrategy,  # Alias
    'single': SingleAgentStrategy,       # Alias
}


def get_available_strategies() -> Dict[str, str]:
    """
    Get list of available strategies with descriptions.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    descriptions = {}
    for name, strategy_class in STRATEGY_REGISTRY.items():
        descriptions[name] = f"{strategy_class.__name__}: {strategy_class.__doc__ or 'No description'}"
    
    return descriptions


def create_strategy(strategy_name: str, env, algorithm, reward_function, 
                   config: Dict[str, any]) -> BaseStrategy:
    """
    Create strategy instance from registry.
    
    Args:
        strategy_name: Name of strategy to create
        env: CityLearn environment
        algorithm: Algorithm instance
        reward_function: Reward function instance
        config: Strategy configuration
        
    Returns:
        Instantiated strategy
        
    Raises:
        ValueError: If strategy name not found
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    try:
        return strategy_class(env, algorithm, reward_function, config)
    except Exception as e:
        raise RuntimeError(f"Failed to create strategy {strategy_name}: {e}")


def register_strategy(name: str, strategy_class: Type[BaseStrategy]):
    """
    Register a new strategy.
    
    Args:
        name: Strategy name
        strategy_class: Strategy class (must inherit from BaseStrategy)
        
    Raises:
        TypeError: If strategy_class doesn't inherit from BaseStrategy
    """
    if not issubclass(strategy_class, BaseStrategy):
        raise TypeError(f"Strategy class must inherit from BaseStrategy")
    
    STRATEGY_REGISTRY[name] = strategy_class
    print(f"Registered new strategy: {name} -> {strategy_class.__name__}")


# Utility functions for strategy management
def list_strategies() -> list:
    """Get list of all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def is_strategy_available(name: str) -> bool:
    """Check if strategy is available."""
    return name in STRATEGY_REGISTRY


def get_strategy_info(name: str) -> Dict[str, any]:
    """
    Get information about a specific strategy.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy information dictionary
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy {name} not found")
    
    strategy_class = STRATEGY_REGISTRY[name]
    return {
        'name': name,
        'class': strategy_class.__name__,
        'description': strategy_class.__doc__ or 'No description available',
        'module': strategy_class.__module__
    }