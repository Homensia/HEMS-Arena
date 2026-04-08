"""
Main HEMS agent class that combines Algorithm + Reward + Strategy.
"""

from typing import List, Dict, Any, Optional


class HEMSAgent:
    """
    Main agent class that combines Algorithm + Reward + Strategy.
    
    This is the composed agent that represents:
    Agent = Algorithm + Reward Function + Strategy
    
    Where:
    - Algorithm: The AI method (DQN, SAC, RBC, etc.)
    - Reward Function: How to evaluate performance
    - Strategy: How the algorithm is deployed (single, multi-agent, etc.)
    """
    
    def __init__(self, algorithm, reward_function, strategy, name: Optional[str] = None):
        """
        Initialize HEMS agent.
        
        Args:
            algorithm: Algorithm instance (BaseAlgorithm)
            reward_function: Reward function instance (RewardFunction)
            strategy: Strategy instance (BaseStrategy)
            name: Optional agent name
        """
        self.algorithm = algorithm
        self.reward_function = reward_function
        self.strategy = strategy
        
        # Generate name if not provided
        if name is None:
            algo_name = getattr(algorithm, 'name', algorithm.__class__.__name__)
            reward_name = reward_function.__class__.__name__
            strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
            name = f"{algo_name}_{reward_name}_{strategy_name}"
        
        self.name = name
        
        # Agent state
        self.is_training = False
        self._episode_count = 0
        self._step_count = 0
        
        print(f"HEMS Agent '{self.name}' created:")
        print(f"  Algorithm: {self.algorithm.__class__.__name__}")
        print(f"  Reward: {self.reward_function.__class__.__name__}")
        print(f"  Strategy: {self.strategy.__class__.__name__}")
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Choose actions using the strategy.
        
        Args:
            observations: List of building observations
            deterministic: Whether to act deterministically
            
        Returns:
            Actions determined by the strategy
        """
        self._step_count += 1
        return self.strategy.act(observations, deterministic)
    
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Learn from experience using the strategy.
        
        Args:
            *args: Arguments passed to strategy's learn method
            **kwargs: Keyword arguments passed to strategy's learn method
            
        Returns:
            Training statistics if applicable
        """
        if not self.is_training:
            return None
        
        stats = self.strategy.learn(*args, **kwargs)
        return stats
    
    def learn_episode(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Learn from a complete episode (for episode-based algorithms).
        
        Returns:
            Training statistics if applicable
        """
        if not self.is_training:
            return None
        
        self._episode_count += 1
        
        # Check if strategy supports episode learning
        if hasattr(self.strategy, 'learn_episode'):
            return self.strategy.learn_episode(*args, **kwargs)
        
        return None
    
    def learn_timesteps(self, total_timesteps: int) -> Optional[Dict[str, Any]]:
        """
        Learn for a specified number of timesteps.
        
        Args:
            total_timesteps: Number of timesteps to train
            
        Returns:
            Training statistics if applicable
        """
        if not self.is_training:
            return None
        
        # Check if strategy supports timestep-based learning
        if hasattr(self.strategy, 'learn_timesteps'):
            return self.strategy.learn_timesteps(total_timesteps)
        
        return None
    
    def reset(self):
        """Reset agent state."""
        self.strategy.reset()
        self._step_count = 0
    
    def set_training_mode(self, training: bool):
        """
        Set training mode for the agent.
        
        Args:
            training: Whether the agent should be in training mode
        """
        self.is_training = training
        self.strategy.set_training_mode(training)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary with training statistics from all components
        """
        stats = {
            'agent_name': self.name,
            'episodes_completed': self._episode_count,
            'steps_taken': self._step_count,
            'is_training': self.is_training
        }
        
        # Get strategy stats (which include algorithm stats)
        strategy_stats = self.strategy.get_training_stats()
        stats.update(strategy_stats)
        
        # Add reward stats if available
        if hasattr(self.reward_function, 'get_reward_summary'):
            reward_stats = self.reward_function.get_reward_summary()
            stats.update({f'reward_{k}': v for k, v in reward_stats.items()})
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive agent information.
        
        Returns:
            Dictionary with agent information
        """
        info = {
            'name': self.name,
            'type': 'hems_agent',
            'is_training': self.is_training,
            'episodes_completed': self._episode_count,
            'steps_taken': self._step_count
        }
        
        # Add component info
        if hasattr(self.algorithm, 'get_info'):
            info['algorithm'] = self.algorithm.get_info()
        else:
            info['algorithm'] = {'name': self.algorithm.__class__.__name__}
        
        if hasattr(self.strategy, 'get_info'):
            info['strategy'] = self.strategy.get_info()
        else:
            info['strategy'] = {'name': self.strategy.__class__.__name__}
        
        info['reward_function'] = {'name': self.reward_function.__class__.__name__}
        
        return info
    
    def save_state(self, filepath: str):
        """
        Save agent state (if supported by components).
        
        Args:
            filepath: Path to save state
        """
        # This is a placeholder for future state saving functionality
        # Could save algorithm weights, strategy state, etc.
        raise NotImplementedError("State saving not yet implemented")
    
    def load_state(self, filepath: str):
        """
        Load agent state (if supported by components).
        
        Args:
            filepath: Path to load state from
        """
        # This is a placeholder for future state loading functionality
        raise NotImplementedError("State loading not yet implemented")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"HEMSAgent(name='{self.name}', algorithm={self.algorithm.__class__.__name__}, " \
               f"reward={self.reward_function.__class__.__name__}, strategy={self.strategy.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return self.__str__()