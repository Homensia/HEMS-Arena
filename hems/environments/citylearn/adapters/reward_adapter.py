# =========================================================================
# hems/environments/citylearn/adapters/reward_adapter.py - CityLearn Adapter
# =========================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import time

__all__ = ["CityLearnRewardAdapter"]

# ========================================================================
# CityLearn imports - ONLY HERE to isolate the dependency completely
# ========================================================================
try:
    # Try CityLearn >= 2.4 first
    from citylearn.reward_function import RewardFunction as CityLearnRewardFunction
except ImportError:
    try:
        # Fallback to CityLearn 2.3.x
        from citylearn.reward import RewardFunction as CityLearnRewardFunction
    except ImportError:
        # Minimal fallback if CityLearn not available
        print("[WARNING] CityLearn not available, using minimal fallback")
        
        class CityLearnRewardFunction:
            """Minimal fallback for when CityLearn is not installed."""
            def __init__(self, env_metadata: Optional[Dict[str, Any]] = None) -> None:
                self.env_metadata = env_metadata or {}


class CityLearnRewardAdapter(CityLearnRewardFunction):
    """Adapter to integrate environment-agnostic HEMS rewards with CityLearn.
    
    This class serves as the ONLY bridge between HEMS reward functions and CityLearn.
    It handles:
    1. Observation format conversion (CityLearn -> standardized)
    2. Reward delegation to pure HEMS reward functions
    3. Result format conversion (standardized -> CityLearn)
    
    Key Features:
    - Zero dependencies in core reward functions
    - Robust observation format conversion
    - Performance monitoring and validation
    - Error handling and logging
    """

    def __init__(
        self,
        hems_reward: "BaseRewardFunction",
        env_metadata: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_validation: bool = True,
    ) -> None:
        """Initialize CityLearn reward adapter.

        Args:
            hems_reward: Environment-agnostic HEMS reward function
            env_metadata: CityLearn environment metadata
            enable_logging: Whether to enable detailed logging
            enable_validation: Whether to validate observations
        """
        # Initialize parent CityLearn class
        safe_metadata = env_metadata or {"buildings": []}
        super().__init__(safe_metadata)

        # Store HEMS reward function (this is the core business logic)
        self.hems_reward = hems_reward
        self.env_metadata = safe_metadata
        
        # Adapter configuration
        self.enable_logging = enable_logging
        self.enable_validation = enable_validation
        
        # Performance tracking
        self._adapter_calls = 0
        self._total_conversion_time = 0.0
        self._last_observation_count = 0
        
        # Validation tracking
        self._validation_warnings = 0
        self._unknown_keys_seen: Set[str] = set()
        
        # Expected CityLearn observation keys (can be extended)
        self._known_citylearn_keys = {
            "net_electricity_consumption",
            "electricity_pricing", 
            "solar_generation",
            "electrical_storage_soc",
            "hour",  # Time information
            "month",  # Seasonal information
            "day_type",  # Weekend/weekday
            # Add more as needed
        }
        
        if self.enable_logging:
            print(f"[CityLearnRewardAdapter] Initialized adapter")
            print(f"  -> Wrapped reward: {self.hems_reward.__class__.__name__}")
            print(f"  -> Buildings: {len(safe_metadata.get('buildings', []))}")
            print(f"  -> Validation: {enable_validation}")

    # ====================================================================
    # CityLearn Interface Implementation
    # ====================================================================

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        """CityLearn reward interface - main entry point.

        Args:
            observations: List of CityLearn observation dictionaries

        Returns:
            List of reward values in CityLearn format
            
        Raises:
            ValueError: If observation conversion fails
            RuntimeError: If HEMS reward calculation fails
        """
        start_time = time.perf_counter()
        
        try:
            # Track adapter usage
            self._adapter_calls += 1
            self._last_observation_count = len(observations)
            
            # Convert CityLearn observations to standardized format
            standardized_obs = self._convert_observations(observations)
            
            # Delegate to HEMS reward function (pure business logic)
            reward_values = self.hems_reward.calculate(standardized_obs)
            
            # Track performance
            conversion_time = time.perf_counter() - start_time
            self._total_conversion_time += conversion_time
            
            if self.enable_logging and self._adapter_calls % 100 == 0:
                avg_time = self._total_conversion_time / self._adapter_calls
                print(f"[CityLearnRewardAdapter] Performance update: "
                      f"calls={self._adapter_calls}, avg_time={avg_time*1000:.2f}ms")
            
            return reward_values
            
        except Exception as e:
            print(f"[CityLearnRewardAdapter] ERROR in calculate(): {e}")
            print(f"  -> Observations count: {len(observations)}")
            print(f"  -> HEMS reward: {self.hems_reward.__class__.__name__}")
            raise RuntimeError(f"Reward calculation failed: {e}") from e

    def __call__(self, observations: List[Dict[str, float]]) -> List[float]:
        """Alternative callable interface for compatibility."""
        return self.calculate(observations)

    # ====================================================================
    # Observation Format Conversion - The Critical Translation Layer
    # ====================================================================

    def _convert_observations(
        self, citylearn_obs: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Convert CityLearn observations to standardized HEMS format.
        
        This is the critical translation layer that allows HEMS rewards to work
        with CityLearn without any direct dependencies.

        Args:
            citylearn_obs: List of CityLearn observation dictionaries

        Returns:
            List of standardized observation dictionaries

        Raises:
            ValueError: If conversion fails
        """
        if not citylearn_obs:
            raise ValueError("Empty observation list received from CityLearn")

        standardized: List[Dict[str, float]] = []

        for idx, obs in enumerate(citylearn_obs):
            if not isinstance(obs, dict):
                raise ValueError(f"Observation[{idx}] is not a dictionary: {type(obs)}")

            try:
                # Core required observations with robust defaults
                standard_obs = {
                    "net_electricity_consumption": self._safe_get_float(
                        obs, "net_electricity_consumption", 0.0
                    ),
                    "electricity_pricing": self._safe_get_float(
                        obs, "electricity_pricing", 0.0
                    ), 
                    "solar_generation": self._safe_get_float(
                        obs, "solar_generation", 0.0
                    ),
                    "electrical_storage_soc": self._safe_get_float(
                        obs, "electrical_storage_soc", 0.0
                    ),
                }

                # Optional additional fields (preserved if present)
                optional_fields = ["hour", "month", "day_type"]
                for field in optional_fields:
                    if field in obs:
                        standard_obs[field] = self._safe_get_float(obs, field, 0.0)

                # Validation checks
                if self.enable_validation:
                    self._validate_converted_observation(standard_obs, idx)

                # Track unknown keys for debugging
                if self.enable_logging:
                    unknown_keys = set(obs.keys()) - self._known_citylearn_keys
                    if unknown_keys:
                        new_unknown = unknown_keys - self._unknown_keys_seen
                        if new_unknown:
                            self._unknown_keys_seen.update(new_unknown)
                            print(f"[CityLearnRewardAdapter] New observation keys: {new_unknown}")

                standardized.append(standard_obs)

            except Exception as e:
                raise ValueError(f"Failed to convert observation[{idx}]: {e}") from e

        return standardized

    def _safe_get_float(
        self, obs: Dict[str, Any], key: str, default: float
    ) -> float:
        """Safely extract and convert a numeric value from observations.
        
        Args:
            obs: Observation dictionary
            key: Key to extract
            default: Default value if key missing or invalid
            
        Returns:
            Float value
        """
        try:
            value = obs.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            if self.enable_logging:
                print(f"[CityLearnRewardAdapter] Warning: Invalid {key} value: {obs.get(key)}, using {default}")
            return default

    def _validate_converted_observation(
        self, obs: Dict[str, float], idx: int
    ) -> None:
        """Validate a converted observation for sanity.
        
        Args:
            obs: Converted observation dictionary
            idx: Observation index (for error reporting)
        """
        # Basic range checks
        soc = obs.get("electrical_storage_soc", 0.0)
        if not (0.0 <= soc <= 1.0):
            self._validation_warnings += 1
            if self.enable_logging and self._validation_warnings <= 10:
                print(f"[CityLearnRewardAdapter] Warning: SoC out of range [0,1]: {soc} in obs[{idx}]")

        # Check for reasonable price values  
        price = obs.get("electricity_pricing", 0.0)
        if price < 0:
            self._validation_warnings += 1
            if self.enable_logging and self._validation_warnings <= 10:
                print(f"[CityLearnRewardAdapter] Warning: Negative price: {price} in obs[{idx}]")

    # ====================================================================
    # State Management and Delegation
    # ====================================================================

    def reset(self) -> None:
        """Reset adapter and delegate to HEMS reward function."""
        # Reset adapter state
        self._adapter_calls = 0
        self._total_conversion_time = 0.0
        self._validation_warnings = 0
        self._unknown_keys_seen.clear()
        
        # Delegate to HEMS reward function
        if hasattr(self.hems_reward, "reset") and callable(self.hems_reward.reset):
            self.hems_reward.reset()
        elif hasattr(self.hems_reward, "reset_tracking"):
            # Fallback for older interface
            self.hems_reward.reset_tracking()
            
        if self.enable_logging:
            print(f"[CityLearnRewardAdapter] Reset adapter and delegated to {self.hems_reward.__class__.__name__}")

    def reset_tracking(self) -> None:
        """Reset reward component tracking (delegate to HEMS reward)."""
        if hasattr(self.hems_reward, "reset_tracking"):
            self.hems_reward.reset_tracking()

    # ====================================================================
    # Information and Statistics Delegation
    # ====================================================================

    def get_reward_summary(self) -> Dict[str, float]:
        """Get reward component summary (delegate to HEMS reward)."""
        if hasattr(self.hems_reward, "get_reward_summary"):
            return self.hems_reward.get_reward_summary()
        return {}

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive adapter and reward information."""
        # Get info from HEMS reward
        hems_info = {}
        if hasattr(self.hems_reward, "get_info"):
            hems_info = self.hems_reward.get_info()

        # Combine with adapter info
        adapter_info = {
            "adapter_type": "CityLearn",
            "adapter_version": "2.0",
            "environment": "citylearn",
            "wrapped_reward": self.hems_reward.__class__.__name__,
            "adapter_calls": self._adapter_calls,
            "total_conversion_time": self._total_conversion_time,
            "average_conversion_time": (
                self._total_conversion_time / max(1, self._adapter_calls)
            ),
            "validation_warnings": self._validation_warnings,
            "unknown_keys_seen": list(self._unknown_keys_seen),
            "logging_enabled": self.enable_logging,
            "validation_enabled": self.enable_validation,
        }

        return {**hems_info, **adapter_info}

    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get detailed adapter performance statistics."""
        return {
            "total_calls": self._adapter_calls,
            "total_conversion_time_seconds": self._total_conversion_time,
            "average_conversion_time_ms": (
                self._total_conversion_time * 1000 / max(1, self._adapter_calls)
            ),
            "last_observation_count": self._last_observation_count,
            "validation_warnings": self._validation_warnings,
            "unknown_keys_count": len(self._unknown_keys_seen),
        }

    # ====================================================================
    # Configuration and Control
    # ====================================================================

    def enable_logging(self) -> None:
        """Enable detailed logging."""
        self.enable_logging = True
        print("[CityLearnRewardAdapter] Enabled logging")

    def disable_logging(self) -> None:
        """Disable logging for performance."""
        self.enable_logging = False

    def enable_validation(self) -> None:
        """Enable observation validation."""
        self.enable_validation = True
        print("[CityLearnRewardAdapter] Enabled validation")

    def disable_validation(self) -> None:
        """Disable validation for performance."""
        self.enable_validation = False
        print("[CityLearnRewardAdapter] Disabled validation")

    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"CityLearnRewardAdapter("
            f"reward={self.hems_reward.__class__.__name__}, "
            f"calls={self._adapter_calls}, "
            f"buildings={len(self.env_metadata.get('buildings', []))})"
        )