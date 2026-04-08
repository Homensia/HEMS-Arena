"""
No reward function for optimization-based algorithms
Used when algorithm doesn't require reward signals (like MPC)
"""

from typing import List, Dict, Any
from .base import BaseRewardFunction



class NoReward(BaseRewardFunction):
    """
    Dummy reward function for algorithms that don't use rewards
    
    Used for optimization-based algorithms like MPC that optimize
    directly using forecasts and constraints, not reward signals.
    """
    
    def __init__(self, env_metadata: Dict[str, Any] = None, **kwargs):
        """
        Initialize no-reward function
        
        Args:
            env_metadata: Environment metadata (unused)
            **kwargs: Additional arguments (unused)
        """
        super().__init__()
        self.name = "NoReward"
    
    def calculate(self, observations: List[List[float]]) -> List[float]:
        """
        Return zero rewards for all buildings
        
        Args:
            observations: Building observations (unused)
            
        Returns:
            List of zeros (one per building)
        """
        num_buildings = len(observations) if observations else 1
        return [0.0] * num_buildings
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """Get reward summary statistics"""
        return {
            'reward_type': 'NoReward',
            'description': 'Dummy reward for optimization-based algorithms',
            'total_reward': 0.0,
            'components': {},
            'usage': 'MPC and other non-RL algorithms'
        }