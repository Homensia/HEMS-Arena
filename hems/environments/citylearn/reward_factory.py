# ===============================================================================
# hems/environments/reward_factory.py - Environment-Specific Reward Integration
# ===============================================================================

"""
Environment-specific reward factory that handles adaptation.

This module is separate from the core rewards registry and handles
the integration of pure rewards with specific environments.

This is where ALL environment-specific reward integration happens,
keeping the core rewards module completely environment-agnostic.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

# Import pure reward system (zero environment dependencies)
try:
    from hems.rewards import create_reward, BaseRewardFunction
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from rewards import create_reward, BaseRewardFunction

__all__ = [
    "create_reward_for_environment",
    "get_supported_environments", 
    "create_citylearn_reward",  # Legacy
    "detect_environment_type",
    "create_reward_for_env_instance",
]

if TYPE_CHECKING:
    try:
        from citylearn.reward_function import RewardFunction as CityLearnRewardFunction
    except ImportError:
        CityLearnRewardFunction = Any


def create_reward_for_environment(
    reward_name: str,
    environment_type: str = "citylearn",
    env_metadata: Optional[Dict[str, Any]] = None,
    **config: Any,
) -> Union[BaseRewardFunction, Any]:
    """Create a reward function adapted for a specific environment.
    
    This function creates pure rewards first, then adapts them as needed.
    This is the main entry point for environment-specific reward integration.
    
    Args:
        reward_name: Name of the reward function
        environment_type: Target environment ('citylearn', 'gym', 'generic')
        env_metadata: Environment metadata for adapter configuration
        **config: Reward function configuration
        
    Returns:
        Pure reward function or environment-adapted reward function
        
    Examples:
        >>> # Pure reward (no adaptation)
        >>> reward = create_reward_for_environment('custom_v5', 'generic')
        >>> 
        >>> # CityLearn-adapted reward
        >>> reward = create_reward_for_environment('custom_v5', 'citylearn', metadata)
    """
    # Step 1: Always create pure reward first (zero dependencies)
    pure_reward = create_reward(reward_name, **config)
    
    # Step 2: Apply environment-specific adaptation
    env_type = environment_type.lower()
    
    if env_type in ("generic", "pure", "clean"):
        # No adaptation needed - return pure reward
        print(f"[environment.reward_factory] Created pure reward: {reward_name}")
        return pure_reward
    
    elif env_type == "citylearn":
        # CityLearn adaptation
        try:
            # Import adapter from the correct location
            from hems.environments.citylearn.adapters.reward_adapter import CityLearnRewardAdapter
        except ImportError:
            try:
                # Fallback: try old location for backward compatibility
                from hems.environments.citylearn.adapters import CityLearnRewardAdapter
            except ImportError as e:
                print(f"[environment.reward_factory] CityLearn adapter not found. Falling back to pure reward.")
                print(f"[environment.reward_factory] Error: {e}")
                print(f"[environment.reward_factory] Install CityLearn or use 'generic' environment type.")
                return pure_reward
        
        try:
            adapted_reward = CityLearnRewardAdapter(pure_reward, env_metadata)
            print(f"[environment.reward_factory] Created CityLearn-adapted reward: {reward_name}")
            return adapted_reward
        except Exception as e:
            print(f"[environment.reward_factory] Failed to create CityLearn adapter: {e}")
            print(f"[environment.reward_factory] Falling back to pure reward")
            return pure_reward
    
    elif env_type == "gym":
        # Future: Gym adapter
        print(f"[environment.reward_factory] Gym adapter not implemented, returning pure reward")
        return pure_reward
    
    else:
        # Unknown environment - return pure reward with warning
        print(f"[environment.reward_factory] Unknown environment '{environment_type}', returning pure reward")
        return pure_reward


def get_supported_environments() -> Dict[str, Dict[str, Any]]:
    """Get information about supported environments."""
    environments = {
        "generic": {
            "description": "Pure reward functions with no adaptation",
            "adapter_required": False,
            "dependencies": [],
            "status": "available"
        },
        "citylearn": {
            "description": "CityLearn environment with automatic observation conversion", 
            "adapter_required": True,
            "dependencies": ["citylearn"],
            "status": "available" if _check_citylearn_available() else "missing_dependencies"
        },
        "gym": {
            "description": "OpenAI Gym/Gymnasium environments",
            "adapter_required": True, 
            "dependencies": ["gymnasium"],
            "status": "not_implemented"
        }
    }
    
    return environments


def _check_citylearn_available() -> bool:
    """Check if CityLearn is available."""
    try:
        import citylearn
        return True
    except ImportError:
        return False


# ========================================================================
# Legacy Compatibility Functions
# ========================================================================

def create_citylearn_reward(
    reward_name: str,
    env_metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Legacy interface for creating CityLearn-compatible rewards.
    
    Args:
        reward_name: Name of reward function
        env_metadata: CityLearn environment metadata
        **kwargs: Reward configuration
        
    Returns:
        CityLearn-adapted reward function
    """
    return create_reward_for_environment(
        reward_name=reward_name,
        environment_type="citylearn", 
        env_metadata=env_metadata,
        **kwargs,
    )


# ========================================================================
# Environment Detection Utilities
# ========================================================================

def detect_environment_type(env) -> str:
    """Automatically detect environment type from environment instance."""
    if env is None:
        return "generic"
        
    env_class_name = env.__class__.__name__.lower()
    env_module = getattr(env.__class__, '__module__', '').lower()
    
    # CityLearn detection
    if 'citylearn' in env_class_name or 'citylearn' in env_module:
        return 'citylearn'
    
    # Gym detection
    if any(x in env_class_name for x in ['gym', 'gymnasium']):
        return 'gym'
    
    # Test environments
    if any(x in env_class_name for x in ['dummy', 'test', 'mock']):
        return 'generic'
    
    # Default
    return 'generic'


def create_reward_for_env_instance(
    reward_name: str,
    env,
    **config: Any
) -> Union[BaseRewardFunction, Any]:
    """Create reward adapted for a specific environment instance.
    
    Args:
        reward_name: Name of reward function
        env: Environment instance
        **config: Reward configuration
        
    Returns:
        Appropriately adapted reward function
    """
    env_type = detect_environment_type(env)
    env_metadata = _extract_env_metadata(env)
    
    return create_reward_for_environment(
        reward_name=reward_name,
        environment_type=env_type,
        env_metadata=env_metadata,
        **config
    )


def _extract_env_metadata(env) -> Dict[str, Any]:
    """Extract metadata from environment instance."""
    metadata = {}
    
    # Common attributes
    common_attrs = ['metadata', 'buildings', 'action_space', 'observation_space']
    for attr in common_attrs:
        if hasattr(env, attr):
            try:
                metadata[attr] = getattr(env, attr)
            except Exception:
                pass
    
    # Ensure buildings key exists
    if 'buildings' not in metadata:
        metadata['buildings'] = []
    
    return metadata


# ========================================================================
# Module Initialization
# ========================================================================

def _initialize_environment_factory():
    """Initialize the environment reward factory."""
    print(f"[environment.reward_factory] Environment-specific reward factory initialized")
    
    # Check available environments
    environments = get_supported_environments()
    available_envs = [name for name, info in environments.items() if info['status'] == 'available']
    
    print(f"[environment.reward_factory] Supported environments: {available_envs}")
    
    # Check CityLearn specifically
    if _check_citylearn_available():
        print(f"[environment.reward_factory] ✅ CityLearn available - automatic adaptation enabled")
    else:
        print(f"[environment.reward_factory] ⚠️  CityLearn not available - will use pure rewards")


# Initialize on import
_initialize_environment_factory()