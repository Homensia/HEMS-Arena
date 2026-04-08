# =========================================================================
# hems/environments/citylearn/adapters/__init__.py - CityLearn Adapters
# =========================================================================

"""
CityLearn Adapters Module

This module contains all CityLearn-specific adapters that bridge the gap
between pure environment-agnostic HEMS components and CityLearn's API.

This is the ONLY place in the HEMS system where CityLearn dependencies exist.
All core HEMS modules (rewards, algorithms, strategies) remain completely
environment-agnostic.

Key Components:
- reward_adapter.py: Adapts pure HEMS reward functions to CityLearn interface
- (future) observation_adapter.py: Standardizes CityLearn observations  
- (future) action_adapter.py: Converts actions between formats

Design Principles:
- Complete isolation of CityLearn dependencies
- Zero leakage of CityLearn imports into core HEMS modules
- Robust error handling and validation
- Performance monitoring and debugging support
- Clear separation between pure business logic and environment integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional

# ========================================================================
# Import Adapters with Comprehensive Error Handling
# ========================================================================

try:
    from .reward_adapter import CityLearnRewardAdapter
    REWARD_ADAPTER_AVAILABLE = True
    REWARD_ADAPTER_ERROR = None
except ImportError as e:
    print(f"[CityLearn Adapters] WARNING: CityLearnRewardAdapter not available: {e}")
    REWARD_ADAPTER_AVAILABLE = False
    REWARD_ADAPTER_ERROR = str(e)
    CityLearnRewardAdapter = None

# Type hints only imports (no runtime dependencies)
if TYPE_CHECKING:
    try:
        from citylearn.reward_function import RewardFunction as CityLearnRewardFunction
    except ImportError:
        try:
            from citylearn.reward import RewardFunction as CityLearnRewardFunction
        except ImportError:
            CityLearnRewardFunction = None

# ========================================================================
# Public API
# ========================================================================

__all__ = [
    "CityLearnRewardAdapter",
    "REWARD_ADAPTER_AVAILABLE",
    "get_adapter_info",
    "validate_citylearn_availability", 
    "create_reward_adapter",
    "get_citylearn_info",
]

# ========================================================================
# Adapter Information and Validation
# ========================================================================

def get_adapter_info() -> Dict[str, Any]:
    """Get comprehensive information about available CityLearn adapters.
    
    Returns:
        Dictionary with adapter availability, version info, and status
    """
    info = {
        "adapters_module": "hems.environments.citylearn.adapters",
        "purpose": "Bridge pure HEMS components to CityLearn environment",
        "adapters": {},
        "citylearn_status": {},
        "system_status": {},
    }
    
    # Check CityLearn availability and version
    citylearn_info = get_citylearn_info()
    info["citylearn_status"] = citylearn_info
    
    # Add reward adapter details
    if REWARD_ADAPTER_AVAILABLE:
        info["adapters"]["reward"] = {
            "class": "CityLearnRewardAdapter",
            "description": "Adapts pure HEMS reward functions to CityLearn interface",
            "status": "available",
            "location": "hems.environments.citylearn.adapters.reward_adapter",
            "features": [
                "observation_format_conversion",
                "performance_monitoring", 
                "validation_system",
                "error_handling",
                "component_tracking_delegation"
            ]
        }
    else:
        info["adapters"]["reward"] = {
            "class": "CityLearnRewardAdapter", 
            "description": "Adapts pure HEMS reward functions to CityLearn interface",
            "status": "unavailable",
            "error": REWARD_ADAPTER_ERROR,
            "location": "hems.environments.citylearn.adapters.reward_adapter"
        }
    
    # System status
    info["system_status"] = {
        "total_adapters": len([k for k, v in info["adapters"].items() if v["status"] == "available"]),
        "available_adapters": [k for k, v in info["adapters"].items() if v["status"] == "available"],
        "unavailable_adapters": [k for k, v in info["adapters"].items() if v["status"] == "unavailable"],
        "core_isolation": "complete" if REWARD_ADAPTER_AVAILABLE else "not_applicable",
    }
    
    return info


def get_citylearn_info() -> Dict[str, Any]:
    """Get detailed information about CityLearn installation and compatibility.
    
    Returns:
        Dictionary with CityLearn status and version information
    """
    info = {
        "installed": False,
        "version": None,
        "compatible": False,
        "reward_function_available": False,
        "import_path": None,
        "error": None,
    }
    
    try:
        import citylearn
        info["installed"] = True
        info["version"] = getattr(citylearn, '__version__', 'unknown')
        
        # Check for reward function compatibility
        reward_function_info = _check_reward_function_compatibility()
        info.update(reward_function_info)
        
        # Overall compatibility
        info["compatible"] = info["reward_function_available"]
        
    except ImportError as e:
        info["error"] = f"CityLearn not installed: {e}"
    except Exception as e:
        info["error"] = f"CityLearn check failed: {e}"
    
    return info


def _check_reward_function_compatibility() -> Dict[str, Any]:
    """Check CityLearn reward function compatibility."""
    result = {
        "reward_function_available": False,
        "import_path": None,
    }
    
    # Try CityLearn >= 2.4 first
    try:
        from citylearn.reward_function import RewardFunction
        result["reward_function_available"] = True
        result["import_path"] = "citylearn.reward_function.RewardFunction"
        return result
    except ImportError:
        pass
    
    # Try CityLearn 2.3.x
    try:
        from citylearn.reward import RewardFunction
        result["reward_function_available"] = True
        result["import_path"] = "citylearn.reward.RewardFunction"
        return result
    except ImportError:
        pass
    
    return result


def validate_citylearn_availability() -> tuple[bool, str]:
    """Validate CityLearn installation and compatibility.
    
    Returns:
        Tuple of (is_available, detailed_status_message)
    """
    citylearn_info = get_citylearn_info()
    
    if not citylearn_info["installed"]:
        return False, f"CityLearn not installed: {citylearn_info.get('error', 'Unknown error')}"
    
    if not citylearn_info["compatible"]:
        return False, f"CityLearn installed but incompatible (version: {citylearn_info['version']})"
    
    version = citylearn_info["version"]
    import_path = citylearn_info["import_path"]
    return True, f"CityLearn {version} available and compatible (using {import_path})"


# ========================================================================
# Adapter Factory Functions
# ========================================================================

def create_reward_adapter(
    hems_reward, 
    env_metadata: Optional[Dict[str, Any]] = None,
    enable_logging: bool = True,
    enable_validation: bool = True,
    **kwargs
):
    """Convenience function to create a CityLearn reward adapter.
    
    Args:
        hems_reward: Pure HEMS reward function instance (must have calculate() method)
        env_metadata: CityLearn environment metadata
        enable_logging: Whether to enable detailed logging
        enable_validation: Whether to enable observation validation
        **kwargs: Additional adapter configuration
        
    Returns:
        CityLearnRewardAdapter instance
        
    Raises:
        ImportError: If CityLearn or adapter is not available
        ValueError: If hems_reward is not valid
        
    Example:
        >>> from hems.rewards import create_reward
        >>> from hems.environments.citylearn.adapters import create_reward_adapter
        >>> 
        >>> pure_reward = create_reward('custom_v5', alpha_cost=0.6)
        >>> adapted_reward = create_reward_adapter(pure_reward, env_metadata)
    """
    # Check adapter availability
    if not REWARD_ADAPTER_AVAILABLE:
        available, status = validate_citylearn_availability()
        if not available:
            raise ImportError(f"CityLearnRewardAdapter is not available: {status}")
        else:
            raise ImportError(f"CityLearnRewardAdapter import failed: {REWARD_ADAPTER_ERROR}")
    
    if CityLearnRewardAdapter is None:
        raise ImportError("CityLearnRewardAdapter class is None")
    
    # Validate HEMS reward
    if not hasattr(hems_reward, 'calculate'):
        raise ValueError(
            f"hems_reward must have a calculate() method. "
            f"Got {type(hems_reward).__name__} with methods: {dir(hems_reward)}"
        )
    
    if not callable(hems_reward.calculate):
        raise ValueError(f"hems_reward.calculate must be callable")
    
    # Validate reward is from pure HEMS system
    if hasattr(hems_reward, '__module__'):
        module_name = hems_reward.__module__
        if 'citylearn' in module_name.lower():
            print(f"[CityLearn Adapters] WARNING: Reward appears to have CityLearn dependency: {module_name}")
    
    # Create adapter with configuration
    try:
        adapter = CityLearnRewardAdapter(
            hems_reward, 
            env_metadata, 
            enable_logging=enable_logging,
            enable_validation=enable_validation,
            **kwargs
        )
        
        print(f"[CityLearn Adapters] ✅ Created adapter for {hems_reward.__class__.__name__}")
        return adapter
        
    except Exception as e:
        raise RuntimeError(f"Failed to create CityLearnRewardAdapter: {e}") from e


# ========================================================================
# Integration Validation
# ========================================================================

def validate_adapter_integration(hems_reward, sample_observations: Optional[list] = None) -> Dict[str, Any]:
    """Validate that a HEMS reward can be properly adapted for CityLearn.
    
    Args:
        hems_reward: Pure HEMS reward function to validate
        sample_observations: Optional sample CityLearn observations for testing
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    
    # Default sample observations if none provided
    if sample_observations is None:
        sample_observations = [
            {
                "net_electricity_consumption": 1.5,
                "electricity_pricing": 0.20,
                "solar_generation": 0.8,
                "electrical_storage_soc": 0.4,
                "hour": 14.0,
                "month": 6.0,
            }
        ]
    
    try:
        # Test pure reward function
        pure_results = hems_reward.calculate(sample_observations)
        results["info"]["pure_reward_works"] = True
        results["info"]["pure_results"] = pure_results
        
        # Test adapter creation
        if REWARD_ADAPTER_AVAILABLE:
            adapter = create_reward_adapter(hems_reward, enable_logging=False)
            adapted_results = adapter.calculate(sample_observations)
            
            results["info"]["adapter_works"] = True
            results["info"]["adapted_results"] = adapted_results
            
            # Compare results
            if pure_results == adapted_results:
                results["info"]["results_consistent"] = True
            else:
                results["warnings"].append("Pure and adapted results differ")
                results["info"]["results_consistent"] = False
                
        else:
            results["warnings"].append("Adapter not available for testing")
            results["info"]["adapter_works"] = False
        
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Validation failed: {e}")
        results["info"]["error_details"] = str(e)
    
    return results


# ========================================================================
# Module Initialization and Status
# ========================================================================

def _initialize_adapters_module():
    """Initialize the CityLearn adapters module."""
    print("[CityLearn Adapters] Initializing environment-specific adapter system...")
    
    # Check CityLearn availability
    citylearn_available, citylearn_status = validate_citylearn_availability()
    
    if citylearn_available:
        print(f"[CityLearn Adapters] ✅ {citylearn_status}")
    else:
        print(f"[CityLearn Adapters] ⚠️  {citylearn_status}")
    
    # Check adapter availability
    if REWARD_ADAPTER_AVAILABLE:
        print("[CityLearn Adapters] ✅ CityLearnRewardAdapter ready")
        print("[CityLearn Adapters] ✅ Pure HEMS rewards can be adapted to CityLearn")
    else:
        print(f"[CityLearn Adapters] ✗ CityLearnRewardAdapter unavailable: {REWARD_ADAPTER_ERROR}")
    
    print("[CityLearn Adapters] 🔒 CityLearn dependencies isolated to this module only")


# Run module initialization
_initialize_adapters_module()


# ========================================================================
# Usage Examples
# ========================================================================

__usage_examples__ = """
CityLearn Adapters - Usage Examples
==================================

1. Basic Adapter Creation:
>>> from hems.rewards import create_reward
>>> from hems.environments.citylearn.adapters import create_reward_adapter
>>> 
>>> pure_reward = create_reward('custom_v5', alpha_cost=0.6)
>>> adapted_reward = create_reward_adapter(pure_reward, env_metadata)
>>> 
>>> # Use with CityLearn
>>> citylearn_results = adapted_reward.calculate(citylearn_observations)

2. Direct Adapter Import:
>>> from hems.rewards import create_reward
>>> from hems.environments.citylearn.adapters import CityLearnRewardAdapter
>>> 
>>> pure_reward = create_reward('simple')
>>> adapter = CityLearnRewardAdapter(pure_reward, env_metadata)

3. Check Adapter Availability:
>>> from hems.environments.citylearn.adapters import get_adapter_info
>>> info = get_adapter_info()
>>> print(f"Reward adapter available: {info['adapters']['reward']['status']}")

4. Validate Integration:
>>> from hems.environments.citylearn.adapters import validate_adapter_integration
>>> from hems.rewards import create_reward
>>> 
>>> reward = create_reward('custom_v5')
>>> results = validate_adapter_integration(reward)
>>> print(f"Integration valid: {results['valid']}")

5. Check CityLearn Status:
>>> from hems.environments.citylearn.adapters import validate_citylearn_availability
>>> available, status = validate_citylearn_availability()
>>> print(f"CityLearn: {status}")
"""

# Make usage examples available
__examples__ = __usage_examples__

print(f"[CityLearn Adapters] Module ready - {len(__all__)} functions exported")
print(f"[CityLearn Adapters] Adapter status: {'✅ Available' if REWARD_ADAPTER_AVAILABLE else '❌ Unavailable'}")