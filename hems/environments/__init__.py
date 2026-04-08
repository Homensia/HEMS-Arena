"""
HEMS Environments Module
Provides environment abstraction and factory for different environment types.
"""

from .base import BaseHEMSEnvironment, BaseEnvironmentManager, EnvironmentRegistry
from .factory import EnvironmentFactory, HEMSEnvironmentManager, HEMSEnvironment

# Import environment implementations to register them
try:
    from .citylearn.citylearn_wrapper import CityLearnHEMSEnvironment, CityLearnEnvironmentManager
    from .citylearn.synthetic_env import SyntheticCityLearnEnv
    CITYLEARN_AVAILABLE = True
except ImportError:
    CITYLEARN_AVAILABLE = False
    SyntheticCityLearnEnv = None

try:
    from .dummy.dummy_env import DummyHEMSEnvironment, DummyEnvironmentManager
    DUMMY_AVAILABLE = True
except ImportError:
    DUMMY_AVAILABLE = False

__all__ = [
    # Base classes
    'BaseHEMSEnvironment',
    'BaseEnvironmentManager', 
    'EnvironmentRegistry',
    
    # Factory
    'EnvironmentFactory',
    'HEMSEnvironmentManager',
    'HEMSEnvironment',  # Backward compatibility
    
    # Environment implementations
    'CityLearnHEMSEnvironment' if CITYLEARN_AVAILABLE else None,
    'SyntheticCityLearnEnv' if CITYLEARN_AVAILABLE else None,
    'DummyHEMSEnvironment' if DUMMY_AVAILABLE else None,
]

# Remove None values
__all__ = [item for item in __all__ if item is not None]


def list_available_environments():
    """Utility function to list available environments."""
    return EnvironmentFactory.list_available_environments()


def get_environment_info():
    """Get detailed information about all environments."""
    info = {
        'available_environments': list_available_environments(),
        'total_count': len(EnvironmentRegistry.get_available_environments()),
        'capabilities': {
            'citylearn': CITYLEARN_AVAILABLE,
            'dummy': DUMMY_AVAILABLE
        }
    }
    return info