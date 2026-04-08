"""
Custom Stable Baselines3 wrapper for HEMS environments.
Fixes the space conversion issue for SAC algorithm.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Any, Dict, List, Tuple, Union


class CustomStableBaselines3Wrapper(gym.Wrapper):
    """
    Custom wrapper to convert HEMS multi-building environment to single-agent format
    compatible with Stable Baselines3.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Store original spaces for debugging
        self.original_observation_space = env.observation_space
        self.original_action_space = env.action_space
        
        # Convert observation space
        self.observation_space = self._flatten_observation_space(env.observation_space)
        
        # Convert action space  
        self.action_space = self._flatten_action_space(env.action_space)
        
        print(f"CustomStableBaselines3Wrapper initialized:")
        print(f"  Original obs space: {self.original_observation_space}")
        print(f"  Flattened obs space: {self.observation_space}")
        print(f"  Original action space: {self.original_action_space}")
        print(f"  Flattened action space: {self.action_space}")
    
    def _flatten_observation_space(self, obs_space) -> Box:
        """Convert list of Box spaces to single flattened Box space."""
        if isinstance(obs_space, list):
            # Calculate total dimension
            total_dim = 0
            low_bounds = []
            high_bounds = []
            
            for space in obs_space:
                if hasattr(space, 'shape') and hasattr(space, 'low') and hasattr(space, 'high'):
                    dim = int(np.prod(space.shape))
                    total_dim += dim
                    
                    # Flatten bounds
                    low_flat = np.array(space.low).flatten()
                    high_flat = np.array(space.high).flatten()
                    
                    low_bounds.extend(low_flat)
                    high_bounds.extend(high_flat)
            
            return Box(
                low=np.array(low_bounds, dtype=np.float32),
                high=np.array(high_bounds, dtype=np.float32),
                shape=(total_dim,),
                dtype=np.float32
            )
        elif hasattr(obs_space, 'shape'):
            # Already a single Box space
            return obs_space
        else:
            # Fallback
            return Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    
    def _flatten_action_space(self, action_space) -> Box:
        """Convert list of Box spaces to single flattened Box space."""
        if isinstance(action_space, list):
            # Calculate total dimension
            total_dim = 0
            low_bounds = []
            high_bounds = []
            
            for space in action_space:
                if hasattr(space, 'shape') and hasattr(space, 'low') and hasattr(space, 'high'):
                    dim = int(np.prod(space.shape))
                    total_dim += dim
                    
                    # Flatten bounds
                    low_flat = np.array(space.low).flatten()
                    high_flat = np.array(space.high).flatten()
                    
                    low_bounds.extend(low_flat)
                    high_bounds.extend(high_flat)
            
            return Box(
                low=np.array(low_bounds, dtype=np.float32),
                high=np.array(high_bounds, dtype=np.float32),
                shape=(total_dim,),
                dtype=np.float32
            )
        elif hasattr(action_space, 'shape'):
            # Already a single Box space
            return action_space
        else:
            # Fallback
            return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def _flatten_observation(self, obs) -> np.ndarray:
        """Convert list-based observation to flattened numpy array."""
        if isinstance(obs, list):
            # Flatten all observations
            flattened = []
            for building_obs in obs:
                if isinstance(building_obs, list):
                    flattened.extend(building_obs)
                elif isinstance(building_obs, np.ndarray):
                    flattened.extend(building_obs.flatten())
                else:
                    flattened.append(float(building_obs))
            return np.array(flattened, dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
        else:
            # Fallback - create dummy observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _unflatten_action(self, action) -> List[List[float]]:
        """Convert flattened action back to list format expected by HEMS."""
        if isinstance(action, np.ndarray):
            action = action.flatten()
        elif not isinstance(action, (list, tuple)):
            action = [action]
        
        # Convert to nested list format expected by HEMS
        # Assuming single building for now
        action_list = [float(a) for a in action]
        return [action_list]
    
    def reset(self, **kwargs):
        """Reset environment and flatten observation."""
        obs, info = self.env.reset(**kwargs)
        flattened_obs = self._flatten_observation(obs)
        return flattened_obs, info
    
    def step(self, action):
        """Step environment with unflattened action and return flattened observation."""
        # Convert flattened action back to HEMS format
        hems_action = self._unflatten_action(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(hems_action)
        
        # Flatten observation
        flattened_obs = self._flatten_observation(obs)
        
        return flattened_obs, reward, terminated, truncated, info

