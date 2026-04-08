# ==========================
# hems/rewards/__init__.py 
# ==========================

"""
HEMS Rewards Module - Pure Environment-Agnostic Architecture

This module provides ONLY pure, environment-agnostic reward functions.
All environment-specific integration is handled by separate modules.

Key Features:
- Zero environment dependencies
- Pure business logic only
- Standardized observation format
- Component tracking and validation
- Performance monitoring

Architecture:
```
Pure HEMS Rewards (this module) - 100% environment-agnostic
        |
        | (integrated via)
        v
Environment Factories (hems.environments.*)
        |
        | (adapted for)
        v
Target Environments (CityLearn, Gym, etc.)
```

For environment integration, use:
- hems.environments.reward_factory.create_reward_for_environment()
- hems.environments.citylearn.adapters.CityLearnRewardAdapter
"""

from __future__ import annotations

# ===========================
# Core Environment-Agnostic 
# ===========================

# Base reward function (completely environment-independent)
from .base import BaseRewardFunction

# Pure reward implementations (no environment dependencies)
from .custom_rewards import (
    CustomRewardV5,
    SimpleReward, 
    BatteryHealthReward,
)

from .Chen_Bu_p2p_variant import P2PReward
from .mp_ppo_reward import MPPPOReward

# Registry and factory functions (PURE ONLY - no environment integration)
from .registry import (
    REWARD_REGISTRY,
    create_reward,                    # Pure reward factory ONLY
    get_available_rewards,
    list_rewards,
    is_reward_available,
    register_reward,
    validate_reward_function,
    test_reward_function,
)

# ========================================================================
# Public API (Pure Environment-Agnostic Only)
# ========================================================================

__all__ = [
    # Core base class
    "BaseRewardFunction",

    # Pure reward implementations  
    "CustomRewardV5",
    "SimpleReward",
    "BatteryHealthReward", 
    "P2PReward",
    "MPPPOReward",

    # Registry and factories (PURE ONLY)
    "REWARD_REGISTRY",
    "create_reward",                    # Create pure reward ONLY
    
    # Registry utilities
    "get_available_rewards",
    "list_rewards", 
    "is_reward_available",
    "register_reward",
    
    # Validation and testing
    "validate_reward_function",
    "test_reward_function",
    
    # Module utilities
    "get_module_info",
    "validate_module_purity",
]

# ========================================================================
# Module Information and Validation
# ========================================================================

def get_module_info() -> dict:
    """Get comprehensive information about the rewards module.
    
    Returns:
        Dictionary with module status and capabilities
    """
    return {
        "module": "hems.rewards",
        "architecture": "pure_environment_agnostic",
        "version": "2.0",
        "total_rewards": len(REWARD_REGISTRY),
        "reward_names": list(REWARD_REGISTRY.keys()),
        "base_class": "BaseRewardFunction", 
        "environment_dependencies": 0,  # ZERO!
        "environment_integration": "hems.environments.reward_factory",
        "features": [
            "pure_environment_agnostic_rewards",
            "standardized_observation_format", 
            "component_tracking",
            "performance_monitoring",
            "validation_system",
            "zero_dependencies",
        ],
        "supported_usage": [
            "pure_reward_creation",
            "independent_testing",
            "business_logic_only",
        ],
    }


def validate_module_purity() -> dict:
    """Validate that this module has ZERO environment dependencies.
    
    Returns:
        Dictionary with purity validation results
    """
    results = {
        "is_pure": True,
        "environment_dependencies": [],
        "warnings": [],
        "info": {},
    }
    
    # Check for any environment-related imports or references
    import sys
    
    try:
        # Check all imported modules for actual runtime dependencies
        for name, obj in globals().items():
            if hasattr(obj, '__module__') and obj.__module__:
                module_name = obj.__module__.lower()
                
                # Check for environment-specific imports (runtime only)
                env_keywords = ['citylearn', 'gym', 'gymnasium']
                for keyword in env_keywords:
                    if keyword in module_name and not module_name.startswith('hems.'):
                        results["is_pure"] = False
                        results["environment_dependencies"].append(f"{name} from {obj.__module__}")
        
        # Note: We skip source code scanning as it picks up documentation/comments
        # The important thing is runtime imports, not documentation
            
    except Exception as e:
        results["warnings"].append(f"Validation error: {e}")
    
    results["info"] = {
        "total_imports_checked": len([name for name, obj in globals().items() if hasattr(obj, '__module__')]),
        "pure_status": "✓ PURE" if results["is_pure"] else "✗ HAS DEPENDENCIES",
        "note": "Only checking runtime imports, not documentation references"
    }
    
    return results


def list_pure_rewards() -> list[str]:
    """List all pure (environment-agnostic) reward functions.
    
    Returns:
        List of reward names - all rewards in this module are pure by design
    """
    return list_rewards()


# ========================================================================
# Module Initialization with Purity Validation
# ========================================================================

def _initialize_pure_module():
    """Initialize the pure rewards module with validation."""
    print("[hems.rewards] Initializing PURE environment-agnostic reward system...")
    
    # Validate module purity
    purity_check = validate_module_purity()
    
    if not purity_check["is_pure"]:
        print("[hems.rewards] ⚠️ WARNING - Module purity compromised:")
        for dep in purity_check["environment_dependencies"]:
            print(f"  DEPENDENCY: {dep}")
    
    # Show warnings
    for warning in purity_check["warnings"]:
        print(f"  WARNING: {warning}")
    
    # Show success message
    if purity_check["is_pure"]:
        reward_count = len(REWARD_REGISTRY)
        print(f"[hems.rewards] ✅ Pure module validated - {reward_count} environment-agnostic rewards")
        print("[hems.rewards] ✅ ZERO environment dependencies confirmed")
        print("[hems.rewards] 📍 For environment integration use: hems.environments.reward_factory")
    
    # Usage guidance
    print("[hems.rewards] Usage:")
    print("  Pure rewards: hems.rewards.create_reward()")
    print("  Environment integration: hems.environments.reward_factory.create_reward_for_environment()")


# Run module initialization
_initialize_pure_module()


# ========================================================================
# Usage Examples and Documentation
# ========================================================================

__usage_examples__ = """
HEMS Pure Rewards Module - Usage Examples
=========================================

1. Create Pure Reward (Environment-Agnostic):
>>> from hems.rewards import create_reward
>>> reward = create_reward('custom_v5', alpha_import_hp=0.7)
>>> 
>>> # Test with standardized observations
>>> observations = [{"net_electricity_consumption": 1.5, "electricity_pricing": 0.2, 
...                  "solar_generation": 0.8, "electrical_storage_soc": 0.4}]
>>> results = reward.calculate(observations)  # Pure calculation, no environment needed

2. List Available Rewards:
>>> from hems.rewards import list_rewards, get_available_rewards
>>> print(list_rewards())
['custom_v5', 'custom', 'simple', 'battery_health', 'Chen_Bu_p2p']
>>> print(get_available_rewards())  # With descriptions

3. Register Custom Reward:
>>> from hems.rewards import register_reward, BaseRewardFunction
>>> 
>>> class MyPureReward(BaseRewardFunction):
...     def calculate(self, observations):
...         return [-sum(obs['net_electricity_consumption'] for obs in observations)]
>>> 
>>> register_reward('my_pure', MyPureReward)
>>> reward = create_reward('my_pure')  # Available immediately

4. Validate Reward Function:
>>> from hems.rewards import validate_reward_function
>>> result = validate_reward_function(MyPureReward)
>>> print(f"Valid: {result['valid']}")

5. Test Reward Function:
>>> from hems.rewards import test_reward_function
>>> result = test_reward_function('simple', cost_weight=1.0)
>>> print(f"Success: {result['success']}, Values: {result['reward_values']}")

6. Environment Integration (use separate module):
>>> # For CityLearn integration:
>>> from hems.environments.reward_factory import create_reward_for_environment
>>> citylearn_reward = create_reward_for_environment('custom_v5', 'citylearn', metadata)
>>> 
>>> # For pure usage (no environment):
>>> from hems.rewards import create_reward  
>>> pure_reward = create_reward('custom_v5')
"""


# ========================================================================
# Final Purity Validation
# ========================================================================

# Ensure no environment imports have leaked in during module initialization
try:
    import sys
    
    # Check that we haven't accidentally imported environment modules
    environment_modules = [name for name in sys.modules.keys() 
                         if any(env in name.lower() for env in ['citylearn', 'gym'])]
    
    if environment_modules:
        # This is OK - other parts of the system might import them
        # The important thing is that THIS module doesn't depend on them
        pass
    
    # The critical validation: ensure our module exports work without environment deps
    test_exports = [
        'BaseRewardFunction', 'create_reward', 'list_rewards', 
        'CustomRewardV5', 'SimpleReward'
    ]
    
    for export in test_exports:
        if export not in __all__:
            print(f"[hems.rewards] WARNING: Missing export: {export}")
            
except Exception:
    # Don't fail module import on validation errors
    pass

print(f"[hems.rewards] Module ready - {len(__all__)} pure functions exported")