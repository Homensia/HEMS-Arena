"""
Rule-based control algorithm implementing time-of-use strategy.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseAlgorithm


class RuleBasedAlgorithm(BaseAlgorithm):
    """
    Rule-based control algorithm implementing time-of-use strategy.
    Charges during low-price periods, discharges during high-price periods.
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize rule-based algorithm.
        
        Args:
            env: CityLearn environment
            config: Algorithm configuration
        """
        super().__init__(env, config)
        
        # Control parameters
        self.charge_level = config.get('charge_level', 0.7)
        self.discharge_level = config.get('discharge_level', 0.7)
        
        # Extract tariff parameters from environment config
        self.price_hp = getattr(env, 'price_hp', 0.22)
        self.price_hc = getattr(env, 'price_hc', 0.14)
        self.hc_hours = getattr(env, 'hc_hours', [23, 0, 1, 2, 3, 4, 5, 6])
        
        print(f"RBC Algorithm: charge_level={self.charge_level}, discharge_level={self.discharge_level}")
        print(f"HP/HC hours: HC={self.hc_hours}, HP=others")
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Implement rule-based control strategy.
        
        Args:
            observations: List of building observations
            deterministic: Whether to act deterministically (unused for RBC)
            
        Returns:
            List of actions based on time-of-use rules
        """
        if not observations or len(observations[0]) == 0:
            # Return actions for all buildings
            num_buildings = len(self.env.buildings)
            return [[0.0] * num_buildings]
    
        # Extract hour from first building's observations
        hour = int(round(observations[0][0])) % 24
    
        # Generate actions for ALL buildings (not just observations count)
        num_buildings = len(self.env.buildings)
        actions = []
    
        for building_idx in range(num_buildings):
            # Use available observation or default SoC
            if building_idx < len(observations) and len(observations[building_idx]) >= 4:
                current_soc = observations[building_idx][3]  # electrical_storage_soc
            else:
                current_soc = 0.5  # Default SoC if observation not available
        
            if hour in self.hc_hours:
                # Off-peak: charge battery (negative action)
                if current_soc < 0.9:  # Don't overcharge
                    action = -self.charge_level
                else:
                    action = 0.0
            else:
                # Peak: discharge battery (positive action)
                if current_soc > 0.1:  # Don't overdischarge
                     action = self.discharge_level
                else:
                    action = 0.0
        
            actions.append(action)
    
        return [actions]  # Return [[action1, action2]]
    
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        No learning for rule-based algorithm.
        
        Returns:
            None (no training stats)
        """
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get rule-based algorithm information."""
        info = super().get_info()
        info.update({
            'description': 'Rule-based time-of-use control strategy',
            'trainable': False,
            'parameters': {
                'charge_level': self.charge_level,
                'discharge_level': self.discharge_level,
                'hc_hours': self.hc_hours
            }
        })
        return info