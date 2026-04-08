# =====================
# hems/rewards/base.py 
# =====================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
import numpy as np
import time

__all__ = ["BaseRewardFunction"]


class BaseRewardFunction(ABC):
    """Environment-agnostic base reward function.

    This class is completely independent of any specific environment (CityLearn, Gym, etc.).
    All reward functions inherit from this base class and implement pure business logic.
    
    The API expects standardized observations and returns reward values that can be
    adapted to any environment through appropriate adapter classes.

    Recommended Observation Format:
    -------------------------------
    Each observation dict should contain:
    - "net_electricity_consumption": float  # +import, -export (kWh)
    - "electricity_pricing": float          # current price (€/kWh)
    - "solar_generation": float             # PV generation (kWh) 
    - "electrical_storage_soc": float       # battery SoC [0, 1]
    
    Additional keys can be included as needed by specific reward functions.
    """

    # Standard observation keys - used for validation and documentation
    RECOMMENDED_KEYS: Set[str] = {
        "net_electricity_consumption",  # +import, -export
        "electricity_pricing",          # current price
        "solar_generation",             # PV generation
        "electrical_storage_soc",       # battery SoC [0, 1]
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize reward function.

        Args:
            config: Configuration parameters for the reward function
        """
        self.config: Dict[str, Any] = config or {}
        
        # Component tracking for analysis
        self.reward_components: Dict[str, List[float]] = {}
        
        # Validation settings
        self._validation_enabled = bool(self.config.get("enable_validation", True))
        self._strict_validation = bool(self.config.get("strict_validation", False))
        
        # Performance tracking
        self._calculation_count = 0
        self._last_calculation_time = 0.0
        self._total_calculation_time = 0.0
        
        print(f"[{self.__class__.__name__}] Initialized environment-agnostic reward function")

    # ===============================================================
    # Abstract Interface - Must be implemented by all reward classes
    # ===============================================================

    @abstractmethod
    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        """Compute reward from standardized observations.

        This is the core method that all reward functions must implement.
        It should contain only pure business logic with no environment-specific code.

        Args:
            observations: List of standardized observation dictionaries.
                         Each dict should follow the recommended format.

        Returns:
            List of reward values. For centralized control, typically [total_reward].
            For distributed control, one reward per building/agent.
            
        Raises:
            ValueError: If observations are invalid (when validation enabled) 
        """
        raise NotImplementedError("Subclasses must implement calculate()")

    def __call__(self, observations: List[Dict[str, float]]) -> List[float]:
        """Convenient callable interface - delegates to calculate()."""
        return self.calculate(observations)

    # ===============================================================
    # Validation System - Prevents bugs and ensures data integrity
    # ===============================================================

    def validate_observations(
        self,
        observations: Sequence[Mapping[str, float]],
        required_keys: Optional[Sequence[str]] = None,
        strict: bool = None,
    ) -> None:
        """Validate observation format and content.

        Args:
            observations: Observations to validate
            required_keys: Keys that must be present. If None, uses RECOMMENDED_KEYS
            strict: Override strict validation setting
            
        Raises:
            ValueError: If validation fails and strict mode is enabled
            
        Note:
            This helps prevent bugs by catching format mismatches early.
        """
        if not self._validation_enabled:
            return
            
        strict = strict if strict is not None else self._strict_validation
        check_keys = required_keys or list(self.RECOMMENDED_KEYS)
        
        if not observations:
            if strict:
                raise ValueError("Empty observations list")
            return

        for idx, obs in enumerate(observations):
            if not isinstance(obs, Mapping):
                if strict:
                    raise ValueError(f"Observation[{idx}] is not a dict-like object")
                continue
                
            missing_keys = [k for k in check_keys if k not in obs]
            if missing_keys:
                msg = f"Missing keys in observation[{idx}]: {missing_keys}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"[{self.__class__.__name__}] Warning: {msg}")
                    
            # Check for non-numeric values
            for key, value in obs.items():
                if not isinstance(value, (int, float, np.number)):
                    msg = f"Non-numeric value in observation[{idx}][{key}]: {type(value)}"
                    if strict:
                        raise ValueError(msg)

    def _safe_calculate_with_validation(
        self, observations: List[Dict[str, float]]
    ) -> List[float]:
        """Internal wrapper that adds validation and performance tracking."""
        start_time = time.perf_counter()
        
        try:
            # Validate inputs if enabled
            if self._validation_enabled:
                self.validate_observations(observations)
            
            # Call the actual implementation
            result = self.calculate(observations)
            
            # Validate outputs
            if not isinstance(result, list):
                raise ValueError(f"calculate() must return a list, got {type(result)}")
                
            if not all(isinstance(x, (int, float, np.number)) for x in result):
                raise ValueError("All reward values must be numeric")
                
            # Update performance tracking
            self._calculation_count += 1
            calculation_time = time.perf_counter() - start_time
            self._last_calculation_time = calculation_time
            self._total_calculation_time += calculation_time
            
            return result
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error in calculate(): {e}")
            raise

    # ===============================================================
    # Component Tracking System - For analysis and debugging
    # ===============================================================

    def track(self, component: str, value: float) -> None:
        """Track a reward component value for analysis.
        
        Args:
            component: Name of the reward component (e.g., 'cost', 'pv_reward')
            value: Component value to track
        """
        if component not in self.reward_components:
            self.reward_components[component] = []
        self.reward_components[component].append(float(value))

    def reset_tracking(self) -> None:
        """Reset all component tracking data."""
        for key in list(self.reward_components.keys()):
            self.reward_components[key] = []
        print(f"[{self.__class__.__name__}] Reset reward component tracking")

    def get_reward_summary(self) -> Dict[str, float]:
        """Get summary statistics of all tracked components.
        
        Returns:
            Dictionary with mean, std, and total for each component
        """
        summary: Dict[str, float] = {}
        
        for component, values in self.reward_components.items():
            if values:
                arr = np.asarray(values, dtype=float)
                summary[f"{component}_mean"] = float(np.mean(arr))
                summary[f"{component}_std"] = float(np.std(arr))
                summary[f"{component}_total"] = float(np.sum(arr))
                summary[f"{component}_count"] = len(values)
            else:
                summary[f"{component}_mean"] = 0.0
                summary[f"{component}_std"] = 0.0
                summary[f"{component}_total"] = 0.0
                summary[f"{component}_count"] = 0
                
        return summary

    def get_component_history(self, component: str) -> List[float]:
        """Get full history of a specific component.
        
        Args:
            component: Name of component to retrieve
            
        Returns:
            List of all tracked values for the component
        """
        return self.reward_components.get(component, []).copy()

    # ===============================================================
    # State Management - For episodic environments
    # ===============================================================

    def reset(self) -> None:
        """Reset reward function state at the start of a new episode.
        
        Override this method if your reward function maintains state
        between steps (e.g., price tracking, SoC history).
        """
        self.reset_tracking()
        print(f"[{self.__class__.__name__}] Reset reward function state")

    # ===============================================================
    # Metadata and Information
    # ===============================================================

    def get_info(self) -> Dict[str, Any]:
        """Get reward function metadata and performance information.
        
        Returns:
            Dictionary with reward function information
        """
        return {
            "name": self.__class__.__name__,
            "type": "reward_function",
            "config": self.config.copy(),
            "is_environment_agnostic": True,
            "validation_enabled": self._validation_enabled,
            "strict_validation": self._strict_validation,
            "calculation_count": self._calculation_count,
            "total_calculation_time": self._total_calculation_time,
            "average_calculation_time": (
                self._total_calculation_time / max(1, self._calculation_count)
            ),
            "tracked_components": list(self.reward_components.keys()),
            "recommended_observation_keys": list(self.RECOMMENDED_KEYS),
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get detailed performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "calculation_count": self._calculation_count,
            "total_time_seconds": self._total_calculation_time,
            "average_time_ms": (
                self._total_calculation_time * 1000 / max(1, self._calculation_count)
            ),
            "last_calculation_time_ms": self._last_calculation_time * 1000,
        }

    # ===============================================================
    # Utility Methods
    # ===============================================================

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get a configuration value with default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            key: Configuration key to update
            value: New value
        """
        self.config[key] = value
        print(f"[{self.__class__.__name__}] Updated config: {key} = {value}")

    def enable_validation(self, strict: bool = False) -> None:
        """Enable observation validation.
        
        Args:
            strict: Whether to raise exceptions on validation errors
        """
        self._validation_enabled = True
        self._strict_validation = strict
        print(f"[{self.__class__.__name__}] Enabled validation (strict={strict})")

    def disable_validation(self) -> None:
        """Disable observation validation for performance."""
        self._validation_enabled = False
        print(f"[{self.__class__.__name__}] Disabled validation")

    def __repr__(self) -> str:
        """String representation of the reward function."""
        return (
            f"{self.__class__.__name__}("
            f"calculations={self._calculation_count}, "
            f"components={len(self.reward_components)})"
        )