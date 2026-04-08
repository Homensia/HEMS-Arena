"""
Baseline algorithm that takes no control actions.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseAlgorithm


class BaselineAlgorithm(BaseAlgorithm):
    """
    Baseline algorithm that takes no control actions (all actions = 0).
    This serves as a baseline comparison for other algorithms.
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize baseline algorithm.
        
        Args:
            env: CityLearn environment
            config: Algorithm configuration (unused for baseline)
        """
        super().__init__(env, config)
        
    def _zero_action(self) -> List[List[float]]:
        """Return correctly-shaped zero action for this environment."""
        space = getattr(self.env, "action_space", None)
        if space is None:
            return [[]]

        # Case A: single Box space (flat vector)
        if hasattr(space, "shape"):
            n = int(space.shape[0]) if space.shape is not None else 0
            return [[0.0] * n] if n > 0 else [[]]

        # Case B: list/tuple of per-building spaces
        if isinstance(space, (list, tuple)):
            per_building = []
            total = 0
            for s in space:
                m = int(getattr(s, "shape", [0])[0]) if getattr(s, "shape", None) else 0
                per_building.append([0.0] * m)
                total += m
            # Some wrappers expect [] when total actions == 0
            return [[]] if total == 0 else [per_building[0] if len(per_building) == 1 else per_building]

        # Fallback
        return [[]]
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """
        Take no action (return zeros).
        
        Args:
            observations: Building observations (unused)
            deterministic: Whether to act deterministically (unused)
            
        Returns:
            Zero actions for all buildings
        """
        return self._zero_action()
    
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        No learning for baseline algorithm.
        
        Returns:
            None (no training stats)
        """
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get baseline algorithm information."""
        info = super().get_info()
        info.update({
            'description': 'Baseline algorithm with no control actions',
            'trainable': False
        })
        return info