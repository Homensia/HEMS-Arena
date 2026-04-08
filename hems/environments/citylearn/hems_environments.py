#=====================================================
#hems/environment/citylearn/hems_environments.py
#=====================================================

"""
Compatibility wrapper for existing HEMS codebase.
Provides HEMSEnvironment interface using new CityLearnEnvironmentManager.
"""

from .citylearn_wrapper import CityLearnEnvironmentManager

HEMSEnvironment = CityLearnEnvironmentManager

__all__ = ['HEMSEnvironment']