# ======================================
# hems/rewards/custom_rewards.py
# ======================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseRewardFunction

__all__ = [
    "CustomRewardV5",
    "SimpleReward",
    "BatteryHealthReward",
    "PVMaximizationReward",
]


def _merge_config(config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a dict-like config with kwargs (kwargs override)."""
    merged = dict(config or {})
    merged.update(kwargs or {})
    return merged


class CustomRewardV5(BaseRewardFunction):
    """Advanced custom reward function for HEMS optimization (environment-agnostic).

    Objectives (combined):
    1. Cost minimization (primary)
    2. PV self-consumption maximization
    3. Battery health protection
    4. Peak demand reduction
    5. Grid stability support
    """

    def __init__(self, *args: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        # CityLearn passe un 1er arg positionnel (env=None). On l’ignore.
        cfg = _merge_config(config, kwargs)
        super().__init__(cfg)

        # Cost-focused parameters
        self.alpha_import_hp = float(self.config.get("alpha_import_hp", 0.6))
        self.price_high_threshold = float(self.config.get("price_high_threshold", 0.70))

        # Peak penalty (moderate)
        self.alpha_peak = float(self.config.get("alpha_peak", 0.01))

        # PV self-consumption shaping (strong incentive)
        self.alpha_pv_base = float(self.config.get("alpha_pv_base", 0.10))
        self.alpha_pv_soc = float(self.config.get("alpha_pv_soc", 0.35))

        # Battery health (SoC management)
        self.alpha_soc = float(self.config.get("alpha_soc", 0.05))
        self.soc_lo = float(self.config.get("soc_lo", 0.30))
        self.soc_hi = float(self.config.get("soc_hi", 0.60))

        # Dynamic price tracking
        self.max_price_seen: float = 1e-6  # évite division par zéro

        # Init tracking components
        self.reset_tracking()

    def reset(self) -> None:
        self.max_price_seen = 1e-6
        self.reset_tracking()

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        # Update price normalization baseline
        for obs in observations:
            price = float(obs.get("electricity_pricing", 0.0))
            if price > self.max_price_seen:
                self.max_price_seen = price

        total_reward = 0.0
        step_components: Dict[str, float] = {
            "cost": 0.0,
            "import_penalty": 0.0,
            "peak_penalty": 0.0,
            "pv_base": 0.0,
            "pv_soc": 0.0,
            "soc_band": 0.0,
        }

        for obs in observations:
            building_reward, building_components = self._calculate_building_reward(obs)
            total_reward += building_reward
            for k, v in building_components.items():
                step_components[k] += float(v)

        for key, value in step_components.items():
            self.track(key, float(value))

        return [float(total_reward)]

    def _calculate_building_reward(self, obs: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        net = float(obs.get("net_electricity_consumption", 0.0))  # +import, -export
        price = float(obs.get("electricity_pricing", 0.0))
        pv = float(obs.get("solar_generation", 0.0))
        soc = float(np.clip(obs.get("electrical_storage_soc", 0.0), 0.0, 1.0))

        price_norm = price / self.max_price_seen if self.max_price_seen > 0.0 else 0.0

        import_kwh = max(0.0, net)
        export_kwh = max(0.0, -net)

        components: Dict[str, float] = {}

        components["cost"] = -net * price_norm

        if price_norm > self.price_high_threshold:
            components["import_penalty"] = -self.alpha_import_hp * import_kwh * (
                price_norm - self.price_high_threshold
            )
        else:
            components["import_penalty"] = 0.0

        components["peak_penalty"] = -self.alpha_peak * (import_kwh**2)

        if pv > 0.0:
            components["pv_base"] = -self.alpha_pv_base * export_kwh
            components["pv_soc"] = -self.alpha_pv_soc * export_kwh if soc < self.soc_hi else 0.0
        else:
            components["pv_base"] = 0.0
            components["pv_soc"] = 0.0

        soc_excess = max(0.0, soc - self.soc_hi)
        soc_deficit = max(0.0, self.soc_lo - soc)
        components["soc_band"] = -self.alpha_soc * (soc_excess**2 + soc_deficit**2)

        total_reward = float(sum(components.values()))
        return total_reward, components


class SimpleReward(BaseRewardFunction):
    """Baseline: cost minimization only."""

    def __init__(self, *args: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        cfg = _merge_config(config, kwargs)
        super().__init__(cfg)
        self.cost_weight = float(self.config.get("cost_weight", 1.0))
        self.reset_tracking()

    def reset(self) -> None:
        self.reset_tracking()

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        total_cost = 0.0
        for obs in observations:
            net = float(obs.get("net_electricity_consumption", 0.0))
            price = float(obs.get("electricity_pricing", 0.0))
            total_cost += net * price

        reward = -self.cost_weight * total_cost  # minimize cost
        self.track("cost", float(-reward))  # store the *cost* as a component (positive)
        return [float(reward)]


class BatteryHealthReward(BaseRewardFunction):
    """Emphasizes battery health and longevity (with cost)."""

    def __init__(self, *args: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        cfg = _merge_config(config, kwargs)
        super().__init__(cfg)

        self.cost_weight = float(self.config.get("cost_weight", 0.7))
        self.health_weight = float(self.config.get("health_weight", 0.3))
        self.optimal_soc_min = float(self.config.get("optimal_soc_min", 0.2))
        self.optimal_soc_max = float(self.config.get("optimal_soc_max", 0.8))

        self.previous_soc: Dict[int, float] = {}
        self.reset_tracking()

    def reset(self) -> None:
        self.previous_soc.clear()
        self.reset_tracking()

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        total_reward = 0.0
        step_cost = 0.0
        step_health = 0.0

        for i, obs in enumerate(observations):
            net = float(obs.get("net_electricity_consumption", 0.0))
            price = float(obs.get("electricity_pricing", 0.0))
            cost_penalty = -self.cost_weight * net * price
            step_cost += float(-cost_penalty)

            soc = float(obs.get("electrical_storage_soc", 0.5))

            if soc < self.optimal_soc_min:
                health_penalty = -self.health_weight * (self.optimal_soc_min - soc) ** 2
            elif soc > self.optimal_soc_max:
                health_penalty = -self.health_weight * (soc - self.optimal_soc_max) ** 2
            else:
                health_penalty = 0.0

            prev = self.previous_soc.get(i, soc)
            soc_change = abs(soc - prev)
            if soc_change > 0.1:
                health_penalty -= self.health_weight * 0.1 * soc_change

            self.previous_soc[i] = soc
            step_health += float(-health_penalty)

            building_reward = cost_penalty + health_penalty
            total_reward += building_reward

        self.track("cost", step_cost)
        self.track("health_cost", step_health)

        return [float(total_reward)]


class PVMaximizationReward(BaseRewardFunction):
    """Strongly emphasizes PV self-consumption (with cost)."""

    def __init__(self, *args: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        cfg = _merge_config(config, kwargs)
        super().__init__(cfg)

        self.cost_weight = float(self.config.get("cost_weight", 0.5))
        self.pv_weight = float(self.config.get("pv_weight", 0.5))
        self.export_penalty = float(self.config.get("export_penalty", 2.0))
        self.reset_tracking()

    def reset(self) -> None:
        self.reset_tracking()

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        total_reward = 0.0
        step_cost = 0.0
        step_pv_term = 0.0
        step_export_pen = 0.0

        for obs in observations:
            net = float(obs.get("net_electricity_consumption", 0.0))
            price = float(obs.get("electricity_pricing", 0.0))
            cost_component = -self.cost_weight * net * price
            step_cost += float(-cost_component)

            pv = float(obs.get("solar_generation", 0.0))
            export = max(0.0, -net)

            if pv > 0.0:
                self_consumption = max(0.0, min(pv, pv - export))
                self_consumption_ratio = self_consumption / pv if pv > 0.0 else 0.0

                pv_reward = self.pv_weight * self_consumption_ratio
                export_penalty = -self.export_penalty * export
            else:
                pv_reward = 0.0
                export_penalty = 0.0

            step_pv_term += float(pv_reward)
            step_export_pen += float(-export_penalty)

            building_reward = cost_component + pv_reward + export_penalty
            total_reward += building_reward

        self.track("cost", step_cost)
        self.track("pv_term", step_pv_term)
        self.track("export_penalty", step_export_pen)

        return [float(total_reward)]
