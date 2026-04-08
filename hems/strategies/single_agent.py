"""
Single agent strategy - one centralized algorithm controls all buildings.
"""

from typing import List, Dict, Any, Optional
from .base import BaseStrategy


class SingleAgentStrategy(BaseStrategy):
    """
    Single agent strategy where one centralized algorithm controls all buildings.
    This is the current approach used in the original HEMS code.
    """
    
    def __init__(self, env, algorithm, reward_function, config: Dict[str, Any]):
        """
        Initialize single agent strategy.
        
        Args:
            env: CityLearn environment
            algorithm: Algorithm instance to use
            reward_function: Reward function instance
            config: Strategy configuration
        """
        super().__init__(env, algorithm, reward_function, config)
        
        # Single agent specific configuration
        self.centralized_control = config.get('centralized_control', True)
        
        print(f"Single Agent Strategy initialized:")
        print(f"  Algorithm: {self.algorithm.__class__.__name__}")
        print(f"  Reward: {self.reward_function.__class__.__name__}")
        print(f"  Buildings: {len(self.env.buildings)}")
        print(f"  Centralized control: {self.centralized_control}")
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Get actions from the single centralized algorithm.
        
        Args:
            observations: List of building observations
            deterministic: Whether to act deterministically
            
        Returns:
            Actions from the algorithm in centralized format
        """
        # Pass observations directly to algorithm
        # The algorithm handles the centralized control logic
        return self.algorithm.act(observations, deterministic)
    
    def learn(self, obs, actions, reward, next_obs, done) -> Optional[Dict[str, Any]]:
        """
        Learn from experience using the single algorithm.
        
        Args:
            obs: Current observations
            actions: Actions taken
            reward: Reward received  
            next_obs: Next observations
            done: Whether episode ended
            
        Returns:
            Training statistics if applicable
        """
        if not hasattr(self.algorithm, 'learn'):
            return None
        
        stats = self.algorithm.learn(obs, actions, reward, next_obs, done)
        
        if stats:
            self.training_stats.update(stats)
        
        return stats
    
    def learn_episode(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Learn from a complete episode (for episode-based algorithms like TQL).
        
        Returns:
            Training statistics if applicable
        """
        # Only call algorithm.learn() if it explicitly supports episode-based learning
        # Check for a specific method or attribute that indicates episode-based learning
        if hasattr(self.algorithm, 'learn_from_episode') or hasattr(self.algorithm, 'is_episode_based'):
            if hasattr(self.algorithm, 'learn_from_episode'):
                stats = self.algorithm.learn_from_episode(*args, **kwargs)
            else:
                stats = self.algorithm.learn()
            
            if stats:
                self.training_stats.update(stats)
            
            return stats
        
        # For step-based algorithms like DQN, do nothing here
        return None
    
    def learn_timesteps(self, total_timesteps: int) -> Optional[Dict[str, Any]]:
        """
        Learn for a specified number of timesteps (for algorithms like SAC).
        
        Args:
            total_timesteps: Number of timesteps to train
            
        Returns:
            Training statistics if applicable
        """
        # Check if algorithm supports timestep-based learning
        if hasattr(self.algorithm, 'learn') and hasattr(self.algorithm.learn, '__code__'):
            # Check if learn method accepts total_timesteps parameter
            code = self.algorithm.learn.__code__
            if 'total_timesteps' in code.co_varnames:
                stats = self.algorithm.learn(total_timesteps=total_timesteps)
                
                if stats:
                    self.training_stats.update(stats)
                
                return stats
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get single agent strategy information."""
        info = super().get_info()
        info.update({
            'description': 'Single centralized agent controlling all buildings',
            'centralized_control': self.centralized_control,
            'num_buildings': len(self.env.buildings),
            'algorithm_info': self.algorithm.get_info() if hasattr(self.algorithm, 'get_info') else {}
        })
        return info