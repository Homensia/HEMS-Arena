"""
Base algorithm class for HEMS algorithms.
All algorithms inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseAlgorithm(ABC):
    """
    Abstract base class for all HEMS algorithms.
    
    An algorithm is the core AI method (DQN, SAC, RBC, etc.) that decides
    what actions to take given observations. It's separate from rewards and strategies.
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize algorithm.
        
        Args:
            env: CityLearn environment
            config: Algorithm-specific configuration
        """
        self.env = env
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Choose actions based on observations.
        
        Args:
            observations: List of building observations
            deterministic: Whether to act deterministically
            
        Returns:
            List of actions for each building (centralized format)
        """
        pass
    
    @abstractmethod
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Learn from experience (for trainable algorithms).
        
        Returns:
            Optional training statistics
        """
        pass
    
    def reset(self):
        """Reset algorithm state (if needed)."""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return {}
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get algorithm information.
        
        Returns:
            Dictionary with algorithm info
        """
        return {
            'name': self.name,
            'type': 'algorithm',
            'trainable': hasattr(self, 'learn') and self.learn != BaseAlgorithm.learn
        }


    def store_transition(self, obs, action, reward, next_obs, done, **kwargs):
        """
        Store a transition for learning (optional, algorithm-specific).
        
        Args:
            obs: Current observation
            action: Action taken  
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
            **kwargs: Additional algorithm-specific data
        """
        pass