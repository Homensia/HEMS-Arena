"""
HEMS Algorithms Module
Contains individual algorithm implementations for home energy management.
"""

from .base import BaseAlgorithm
from .baseline import BaselineAlgorithm
from .rule_based import RuleBasedAlgorithm
from .dqn import DQNAlgorithm
from .Memdqn import MemDQNAlgorithm
from .registry import (
    ALGORITHM_REGISTRY,
    create_algorithm,
    register_algorithm,
    get_available_algorithms,
    list_algorithms,
    is_algorithm_available,
    get_algorithm_info
)

# Conditionally import optional algorithms
try:
    from .tabular_ql import TabularQLearningAlgorithm
    __all_algorithms__ = ['BaselineAlgorithm', 'RuleBasedAlgorithm', 'DQNAlgorithm', 'TabularQLearningAlgorithm']
except ImportError:
    __all_algorithms__ = ['BaselineAlgorithm', 'RuleBasedAlgorithm', 'DQNAlgorithm']

try:
    from .sac import SACAlgorithm
    __all_algorithms__.append('SACAlgorithm')
except ImportError:
    pass

__all__ = [
    'BaseAlgorithm',
    'BaselineAlgorithm', 
    'RuleBasedAlgorithm',
    'DQNAlgorithm',
    'ALGORITHM_REGISTRY',
    'create_algorithm',
    'register_algorithm',
    'get_available_algorithms',
    'list_algorithms',
    'is_algorithm_available',
    'get_algorithm_info'
] + __all_algorithms__