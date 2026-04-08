#================================
# hems/rewards/mp_ppo_reward.py
#================================
from __future__ import annotations
from typing import Dict, Sequence, Optional


class MPPPOReward:
    """
    MP-PPO reward implementing Equations 17-18.
    
    R(t) = R1(t) + R2(t)
    
    Eq. 17:  R1(t) = C^g_t * (E^{r,b}_t - E^{r,a}_t)
    Eq. 18:  R2(t) = R2a(t) + R2b(t)
             R2a(t) = (1-ζ) * C^d_t * max(0, H_{t-1} - H_t) * (1-α)
             R2b(t) = ζ * C^g_t * Σ_{e∈E^th}(E^r_t + e)
    
    Parameters:
        zeta: Balance factor (default 0.2 per paper)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        cfg = dict(config or {})
        cfg.update(kwargs or {})
        self.zeta = float(cfg.get("zeta", 0.2))
        self.prev_soc: Optional[float] = None

    def reset(self, soc0: float) -> None:
        self.prev_soc = float(soc0)

    def __call__(self, info: Dict, soc_t: float) -> float:
        """
        Args:
            info: Must contain:
                E_r_t: Agent residual grid energy (kWh)
                E_r_baseline_t: Baseline residual grid energy (kWh)
                C_g_t: Grid unit price ($/kWh or kg_CO2/kWh)
                C_d_t: Battery degradation cost ($/kWh)
                alpha: Efficiency loss factor (default 0.0)
                thermal_loads: List of thermal components (optional)
            soc_t: Current state of charge [0,1]
        
        Returns:
            Total reward R(t)
        """
        assert self.prev_soc is not None, "Call reset() before first step"

        E_r_t = float(info["E_r_t"])
        E_r_baseline_t = float(info["E_r_baseline_t"])
        C_g_t = float(info["C_g_t"])
        C_d_t = float(info["C_d_t"])
        alpha = float(info.get("alpha", 0.0))
        thermal_loads: Sequence[float] = info.get("thermal_loads", [])

        # R1 (Eq. 17): Baseline gap - reward when agent uses less grid energy
        R1 = C_g_t * (E_r_baseline_t - E_r_t)

        # R2a (Eq. 18): Battery degradation on discharge only
        discharged = max(0.0, self.prev_soc - soc_t)
        R2a = (1.0 - self.zeta) * C_d_t * discharged * (1.0 - alpha)

        # R2b (Eq. 18): Thermal grid cost
        # Paper assumes thermal breakdown exists: Σ(E^r + e)
        # Fallback: Use E^r as proxy when breakdown unavailable
        if thermal_loads:
            n_thermal = len(thermal_loads)
            sum_thermal = sum(thermal_loads)
            R2b = self.zeta * C_g_t * (n_thermal * E_r_t + sum_thermal)
        else:
            # Fallback: No thermal breakdown available
            # Using residual energy as practical approximation
            R2b = self.zeta * C_g_t * E_r_t
        
        # R2a and R2b are costs (positive), so subtract them
        return R1 - R2a - R2b