#===================
#hems/core/adapters
#===================
"""
Adapters for CityLearn observation and action format conversion.
"""

import numpy as np
from typing import List, Any, Union


class ObservationAdapter:
    """
    Converts CityLearn observations to algorithm-expected format.
    
    CityLearn with central_agent=True returns observations in various formats
    depending on the number of buildings and CityLearn version. This adapter
    normalizes all formats to List[List[float]].
    """
    
    def __init__(self):
        self.conversion_count = 0
    
    def adapt(self, obs: Any) -> List[List[float]]:
        """
        Convert CityLearn observation to List[List[float]] format.
        
        Args:
            obs: Observation from CityLearn environment
            
        Returns:
            List of observation lists, one per building
        """
        self.conversion_count += 1
        
        # Case 0: Handle Gym API tuple (observation, info) from env.reset()
        if isinstance(obs, tuple):
            # New Gym API returns (observation, info)
            obs = obs[0]
        
        # Case 1: Already correct format - List[List[float]]
        if isinstance(obs, list) and len(obs) > 0:
            if isinstance(obs[0], (list, np.ndarray)):
                # Convert numpy arrays to lists if needed
                return [list(o) if isinstance(o, np.ndarray) else o for o in obs]
        
        # Case 2: Single flat array (one building)
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                return [obs.tolist()]
            elif obs.ndim == 2:
                return [obs[i].tolist() for i in range(obs.shape[0])]
        
        # Case 3: Single list (one building)
        if isinstance(obs, list) and len(obs) > 0:
            if isinstance(obs[0], (int, float, np.number)):
                return [obs]
        
        raise ValueError(
            f"Unsupported observation format: type={type(obs)}, "
            f"shape={getattr(obs, 'shape', 'N/A')}"
        )


class ActionAdapter:
    def __init__(self, central_agent: bool = True):
        self.conversion_count = 0
        self.central_agent = central_agent
        print(f"[ActionAdapter] Initialized with central_agent={central_agent}")
    
    def adapt(self, actions: List[List[float]]) -> Union[List[List[float]], np.ndarray]:
        if not isinstance(actions, list) or len(actions) == 0:
            raise ValueError(f"Invalid action format: {type(actions)}")
        
        self.conversion_count += 1
        
        if self.central_agent:
            # Centralized mode: keep as [[a1, a2, ..., aN]]
            return [[float(a) if not isinstance(a, list) else a for a in action] 
                    if isinstance(action, list) else [float(action)]
                    for action in actions]
        else:
            # Decentralized mode: convert [[a1, a2, ..., aN]] → [[a1], [a2], ..., [aN]]
            if len(actions) == 1 and isinstance(actions[0], list):
                return [[float(a)] for a in actions[0]]
            else:
                return [[float(a) if not isinstance(a, list) else float(a[0])] 
                        for a in actions]