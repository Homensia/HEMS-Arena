"""

P2P Reward Function
Combines CustomRewardV5 logic + WearAndShortageMixin functionality in one self-contained class.
This reward is from the article "Realistic Peer-to-Peer Energy Trading Model for
Microgrids using Deep Reinforcement Learning" for Tianyi Chen and Shengrong Bu 

"""

import numpy as np
from typing import Any, Dict, List, Optional

try:
    from citylearn.reward_function import RewardFunction
except ImportError:
    from citylearn.reward import RewardFunction


class P2PReward(RewardFunction):
    """
    Self-contained P2P reward function that combines:
    1. Advanced multi-objective optimization (from CustomRewardV5)
    2. Battery wear and shortage penalties (from WearAndShortageMixin)
    3. All functionality directly implemented 
    
    Objectives:
    - Cost minimization (primary)
    - PV self-consumption maximization
    - Battery health protection (SoC management + wear minimization)
    - Peak demand reduction
    - Grid stability support
    - Demand shortage penalty
    """
    
    def __init__(self, env_metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize standalone P2P reward function.
        
        Args:
            env_metadata: Environment metadata
            **kwargs: All reward function parameters
        """
        safe_metadata = env_metadata or {'buildings': []}
        super().__init__(safe_metadata)
        
        # ===== COST-FOCUSED PARAMETERS (from CustomRewardV5) =====
        self.alpha_import_hp = float(kwargs.get('alpha_import_hp', 0.6))
        self.price_high_threshold = float(kwargs.get('price_high_threshold', 0.70))
        
        # Peak penalty (moderate)
        self.alpha_peak = float(kwargs.get('alpha_peak', 0.01))
        
        # PV self-consumption shaping (strong incentive)
        self.alpha_pv_base = float(kwargs.get('alpha_pv_base', 0.10))
        self.alpha_pv_soc = float(kwargs.get('alpha_pv_soc', 0.35))
        
        # Battery health (SoC management)
        self.alpha_soc = float(kwargs.get('alpha_soc', 0.05))
        self.soc_lo = float(kwargs.get('soc_lo', 0.30))
        self.soc_hi = float(kwargs.get('soc_hi', 0.60))
        
        # ===== P2P-SPECIFIC PARAMETERS (from mixin) =====
        self.alpha_wear = float(kwargs.get('alpha_wear', 0.0))
        self.alpha_unserved = float(kwargs.get('alpha_unserved', 0.0))
        
        # ===== INTERNAL STATE =====
        # Dynamic price tracking
        self.max_price_seen = 1e-6
        
        # SoC tracking for wear calculation
        self._soc_prev = {}  # Track per building
        
        # Reward component tracking for analysis
        self.reward_components = {
            'cost': [],
            'import_penalty': [],
            'peak_penalty': [],
            'pv_base': [],
            'pv_soc': [],
            'soc_band': [],
            'wear': [],
            'unserved': []
        }
    
    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        """
        Calculate comprehensive reward for current observations.
        
        Args:
            observations: List of building observations
            
        Returns:
            List of rewards per building
        """
        building_rewards = []
        
        for building_idx, obs in enumerate(observations):
            # Extract observation values
            net = float(obs.get('net_electricity_consumption', 0.0))
            price = float(obs.get('electricity_pricing', 0.0))
            soc = float(obs.get('electrical_storage_soc', 0.5))
            pv = float(obs.get('solar_generation', 0.0))
            
            # ===== CORE REWARD COMPONENTS (CustomRewardV5 logic) =====
            
            # 1. Cost component (primary objective)
            cost_reward = -net * price
            
            # 2. High-price import penalty
            import_penalty = 0.0
            if price > self.price_high_threshold and net > 0:
                import_penalty = -self.alpha_import_hp * net * price
            
            # 3. Peak demand penalty
            peak_penalty = -self.alpha_peak * max(0, net)**2
            
            # 4. PV self-consumption rewards
            pv_base_reward = 0.0
            pv_soc_reward = 0.0
            
            if pv > 0:
                # Base PV reward (encourage any PV usage)
                pv_base_reward = self.alpha_pv_base * min(pv, max(0, net))
                
                # SoC-dependent PV reward (encourage storage when available)
                if soc < 0.9:  # Battery has capacity
                    soc_factor = (0.9 - soc) / 0.9  # Higher reward for lower SoC
                    pv_soc_reward = self.alpha_pv_soc * pv * soc_factor
            
            # 5. SoC band reward (battery health protection)
            soc_band_reward = 0.0
            if self.soc_lo <= soc <= self.soc_hi:
                # Reward for keeping SoC in healthy range
                soc_band_reward = self.alpha_soc
            else:
                # Penalty for extreme SoC values
                if soc < self.soc_lo:
                    soc_band_reward = -self.alpha_soc * (self.soc_lo - soc)
                else:  # soc > self.soc_hi
                    soc_band_reward = -self.alpha_soc * (soc - self.soc_hi)
            
            # ===== P2P-SPECIFIC COMPONENTS (mixin logic) =====
            
            # 6. Battery wear penalty
            wear_penalty = self._calculate_wear_penalty(building_idx, soc)
            
            # 7. Unserved demand penalty (placeholder - can be extended)
            kwh_unserved = 0.0  # This would come from environment if available
            unserved_penalty = -self.alpha_unserved * max(0.0, kwh_unserved)
            
            # ===== COMBINE ALL COMPONENTS =====
            building_reward = (
                cost_reward + 
                import_penalty + 
                peak_penalty + 
                pv_base_reward + 
                pv_soc_reward + 
                soc_band_reward + 
                wear_penalty + 
                unserved_penalty
            )
            
            # Update tracking
            self.reward_components['cost'].append(cost_reward)
            self.reward_components['import_penalty'].append(import_penalty)
            self.reward_components['peak_penalty'].append(peak_penalty)
            self.reward_components['pv_base'].append(pv_base_reward)
            self.reward_components['pv_soc'].append(pv_soc_reward)
            self.reward_components['soc_band'].append(soc_band_reward)
            self.reward_components['wear'].append(wear_penalty)
            self.reward_components['unserved'].append(unserved_penalty)
            
            # Update dynamic price tracking
            self.max_price_seen = max(self.max_price_seen, price)
            
            building_rewards.append(float(building_reward))
        
        return building_rewards
    
    def _calculate_wear_penalty(self, building_idx: int, soc_now: float) -> float:
        """
        Calculate battery wear penalty based on SoC changes.
        Implements the wear and shortage mixin logic directly.
        
        Args:
            building_idx: Building index for tracking
            soc_now: Current state of charge
            
        Returns:
            Wear penalty (negative value)
        """
        # Get previous SoC for this building
        soc_prev = self._soc_prev.get(building_idx, soc_now)
        
        # Calculate SoC change (proxy for battery stress)
        d_soc = abs(float(soc_now) - float(soc_prev))
        
        # Update tracking
        self._soc_prev[building_idx] = soc_now
        
        # Return wear penalty (negative because it's a penalty)
        return -self.alpha_wear * d_soc
    
    def __call__(self, observations):
        """
        Compatibility method for different calling conventions.
        """
        if isinstance(observations, list) and len(observations) > 0:
            if isinstance(observations[0], dict):
                # Standard format: list of dicts
                return self.calculate(observations)
            elif isinstance(observations[0], list):
                # Convert list of lists to list of dicts (fallback)
                dict_obs = []
                for obs_list in observations:
                    if len(obs_list) >= 5:
                        # Assume standard format: [hour, price, net, soc, solar]
                        obs_dict = {
                            'hour': obs_list[0],
                            'electricity_pricing': obs_list[1],
                            'net_electricity_consumption': obs_list[2],
                            'electrical_storage_soc': obs_list[3],
                            'solar_generation': obs_list[4]
                        }
                        dict_obs.append(obs_dict)
                return self.calculate(dict_obs) if dict_obs else [0.0]
        
        # Fallback
        return [0.0]
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """
        Get summary of reward components for analysis.
        
        Returns:
            Dictionary with reward component statistics
        """
        summary = {}
        
        for component, values in self.reward_components.items():
            if values:
                summary[f'{component}_mean'] = np.mean(values)
                summary[f'{component}_std'] = np.std(values)
                summary[f'{component}_total'] = np.sum(values)
            else:
                summary[f'{component}_mean'] = 0.0
                summary[f'{component}_std'] = 0.0
                summary[f'{component}_total'] = 0.0
        
        # Overall statistics
        total_rewards = [sum(self.reward_components[comp]) for comp in self.reward_components]
        summary['total_reward'] = sum(total_rewards)
        summary['max_price_seen'] = self.max_price_seen
        
        return summary
    
    def reset(self):
        """Reset internal state for new episode."""
        self._soc_prev = {}
        self.reward_components = {
            'cost': [],
            'import_penalty': [],
            'peak_penalty': [],
            'pv_base': [],
            'pv_soc': [],
            'soc_band': [],
            'wear': [],
            'unserved': []
        }
        self.max_price_seen = 1e-6
