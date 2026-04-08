"""
Base strategy class for HEMS deployment strategies.
Strategies define how algorithms are deployed and coordinated.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseStrategy(ABC):
    """
    Abstract base class for agent deployment strategies.
    
    A strategy defines how algorithms are deployed and coordinated:
    - Single agent: One centralized algorithm controls all buildings
    - Multi-agent: Multiple algorithms, one per building or group
    - Hierarchical: Layered control with coordination between levels
    """
    
    def __init__(self, env, algorithm, reward_function, config: Dict[str, Any]):
        """
        Initialize strategy.
        
        Args:
            env: CityLearn environment
            algorithm: Algorithm instance to deploy
            reward_function: Reward function to use
            config: Strategy-specific configuration
        """
        self.env = env
        self.algorithm = algorithm
        self.reward_function = reward_function
        self.config = config
        self.name = self.__class__.__name__
        
        # Strategy state
        self.is_training = False
        self.training_stats = {}
    
    @abstractmethod
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Execute strategy to get actions.
        
        Args:
            observations: List of building observations
            deterministic: Whether to act deterministically
            
        Returns:
            List of actions in the format expected by the environment
        """
        pass
    
    @abstractmethod
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute strategy for learning.
        
        Returns:
            Training statistics if applicable
        """
        pass
    
    def reset(self):
        """Reset strategy state."""
        if hasattr(self.algorithm, 'reset'):
            self.algorithm.reset()
    
    def set_training_mode(self, training: bool):
        """Set training mode for the strategy."""
        self.is_training = training
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        stats = self.training_stats.copy()
        
        # Add algorithm-specific stats if available
        if hasattr(self.algorithm, 'get_training_stats'):
            algo_stats = self.algorithm.get_training_stats()
            stats.update({f'algorithm_{k}': v for k, v in algo_stats.items()})
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information.
        
        Returns:
            Dictionary with strategy info
        """
        return {
            'name': self.name,
            'type': 'strategy',
            'algorithm': self.algorithm.name if hasattr(self.algorithm, 'name') else str(self.algorithm),
            'reward_function': self.reward_function.__class__.__name__,
            'is_training': self.is_training
        }