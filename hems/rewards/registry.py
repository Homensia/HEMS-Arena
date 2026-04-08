# ===============================
# hems/rewards/registry.py 
# ===============================

from __future__ import annotations
from typing import Any, Dict, Optional, Type, List

from .base import BaseRewardFunction
from .custom_rewards import CustomRewardV5, SimpleReward, BatteryHealthReward
from .Chen_Bu_p2p_variant import P2PReward
from .mp_ppo_reward import MPPPOReward
from .no_reward import NoReward

__all__ = [
    "REWARD_REGISTRY",
    "create_reward",
    "get_available_rewards",
    "list_rewards",
    "is_reward_available", 
    "register_reward",
    "validate_reward_function",
    "test_reward_function",
]

# ========================================================================
# Pure Environment-Agnostic Registry (NO environment mentions)
# ========================================================================

REWARD_REGISTRY: Dict[str, Type[BaseRewardFunction]] = {
    "custom_v5": CustomRewardV5,
    "custom": CustomRewardV5,  # Default alias
    "simple": SimpleReward,
    "battery_health": BatteryHealthReward,
    "Chen_Bu_p2p": P2PReward,
    "mp_ppo": MPPPOReward, 
    "none": NoReward,
}

def create_reward(reward_name: str, **config: Any) -> BaseRewardFunction:
    """Create a pure, environment-agnostic reward function.
    
    This function ONLY creates pure reward functions with zero dependencies.
    For environment integration, use the environment-specific factories.

    Args:
        reward_name: Name of the reward function (registry key)
        **config: Configuration parameters

    Returns:
        Pure HEMS reward function instance

    Examples:
        >>> reward = create_reward('custom_v5', alpha_import_hp=0.7)
        >>> results = reward.calculate(observations)  # Pure calculation
    """
    try:
        reward_class = REWARD_REGISTRY[reward_name]
    except KeyError as e:
        available = list(REWARD_REGISTRY.keys())
        raise ValueError(f"Unknown reward function: {reward_name}. Available: {available}") from e

    try:
        reward = reward_class(config=config)
        print(f"[rewards.registry] Created pure reward: {reward_name}")
        return reward
    except Exception as e:
        raise RuntimeError(f"Failed to create reward function {reward_name}: {e}") from e

def get_available_rewards() -> Dict[str, str]:
    """Get available reward functions with their descriptions.

    Iterates over the reward registry and extracts the first line of each
    reward class's docstring as its description.

    Returns:
        A dictionary mapping reward names to a string of the form
        ``ClassName: first line of docstring``.
    """
    descriptions: Dict[str, str] = {}
    for name, reward_class in REWARD_REGISTRY.items():
        doc = (reward_class.__doc__ or "No description").strip()
        first_line = doc.splitlines()[0] if doc else "No description"
        descriptions[name] = f"{reward_class.__name__}: {first_line}"
    return descriptions

def list_rewards() -> List[str]:
    """Get the list of all registered reward function names.

    Returns:
        A list of reward name strings currently in the registry.
    """
    return list(REWARD_REGISTRY.keys())

def is_reward_available(name: str) -> bool:
    """Check whether a reward function is registered.

    Args:
        name: The reward function name to look up.

    Returns:
        True if the name exists in the registry, False otherwise.
    """
    return name in REWARD_REGISTRY

def register_reward(name: str, reward_class: Type[BaseRewardFunction]) -> None:
    """Register a new reward function in the global registry.

    Args:
        name: The name to register the reward function under.
        reward_class: The reward class to register. Must be a subclass of
            ``BaseRewardFunction``.

    Raises:
        TypeError: If ``reward_class`` does not inherit from
            ``BaseRewardFunction``.
    """
    if not issubclass(reward_class, BaseRewardFunction):
        raise TypeError(f"Reward class must inherit from BaseRewardFunction")
    REWARD_REGISTRY[name] = reward_class
    print(f"[rewards.registry] Registered: {name} -> {reward_class.__name__}")

def validate_reward_function(reward_class: Type[BaseRewardFunction]) -> Dict[str, Any]:
    """Validate that a reward function class meets framework requirements.

    Checks that the class inherits from ``BaseRewardFunction``, implements the
    required ``calculate()`` method, and has a docstring.

    Args:
        reward_class: The reward function class to validate.

    Returns:
        A dictionary containing:
            - ``valid`` (bool): Whether all required checks passed.
            - ``errors`` (list): Critical issues that make the class unusable.
            - ``warnings`` (list): Non-critical issues (e.g., missing docstring).
            - ``info`` (dict): Metadata about the inspected class.
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {
            "name": reward_class.__name__,
            "has_calculate": hasattr(reward_class, "calculate"),
            "has_docstring": bool(reward_class.__doc__),
            "inherits_from_base": issubclass(reward_class, BaseRewardFunction),
        }
    }
    
    if not hasattr(reward_class, "calculate"):
        results["valid"] = False
        results["errors"].append("Missing required calculate() method")
        
    if not issubclass(reward_class, BaseRewardFunction):
        results["valid"] = False
        results["errors"].append("Must inherit from BaseRewardFunction")
        
    if not reward_class.__doc__:
        results["warnings"].append("Missing docstring")
        
    return results

def test_reward_function(reward_name: str, **config: Any) -> Dict[str, Any]:
    """Test a reward function by instantiating it and running sample data.

    Creates the reward function via ``create_reward`` and invokes its
    ``calculate`` method with a minimal set of sample observations.

    Args:
        reward_name: Name of the reward function to test (must be in the
            registry).
        **config: Additional configuration parameters forwarded to
            ``create_reward``.

    Returns:
        A dictionary containing:
            - ``success`` (bool): Whether the test completed without errors.
            - ``reward_values``: The computed reward values (on success).
            - ``reward_type``: Class name of the reward (on success).
            - ``error``: Error message string (on failure).
            - ``error_type``: Exception class name (on failure).
    """
    sample_observations = [
        {
            "net_electricity_consumption": 1.5,
            "electricity_pricing": 0.20,
            "solar_generation": 0.8,
            "electrical_storage_soc": 0.4,
        }
    ]
    
    try:
        reward = create_reward(reward_name, **config)
        results = reward.calculate(sample_observations)
        
        return {
            "success": True,
            "reward_values": results,
            "reward_type": reward.__class__.__name__,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }