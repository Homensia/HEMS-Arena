"""
HEMS Agents Module
Contains the main agent composition and factory for creating HEMS agents.
"""

from .agent import HEMSAgent
from .factory import (
    create_agent,
    create_agent_from_legacy_name,
    
)

# Import agent from the proper configuration location
from hems.core.config import SimulationConfig

def get_agent_config_template():
    """Get agent config template from configuration system."""
    # Use the existing config system instead of duplicating
    return SimulationConfig().to_dict()

__all__ = [
    'HEMSAgent',
    'create_agent',
    'create_agent_from_legacy_name',
    'get_agent_config_template'
]