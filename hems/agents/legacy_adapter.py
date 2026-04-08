"""
Legacy compatibility adapter for backward compatibility with existing runner code.
This allows the new modular architecture to work with existing code unchanged.
"""

from typing import Dict, Any
from .factory import create_agent_from_legacy_name


def create_agent(agent_type: str, env, config):
    """
    Legacy interface for creating agents (backward compatibility).
    
    This function maintains the exact same interface as the original 
    hems_agents.create_agent function to ensure existing code works unchanged.
    
    Args:
        agent_type: Type of agent ('baseline', 'rbc', 'dqn', 'mp_ppo', 'tql', 'sac')
        env: CityLearn environment
        config: Agent configuration (SimulationConfig or Dict)
        
    Returns:
        HEMS agent instance compatible with existing code
    """
    
    # Handle both SimulationConfig and Dict configurations
    if hasattr(config, 'to_dict'):
        # It's a SimulationConfig object
        config_dict = config.to_dict()
    elif hasattr(config, 'copy'):
        # It's a dict-like object
        config_dict = config.copy()
    else:
        # It's already a dict or SimulationConfig without copy method
        try:
            config_dict = config.copy() if hasattr(config, 'copy') else dict(config)
        except (TypeError, AttributeError):
            # SimulationConfig without copy method - convert manually
            config_dict = {
                'random_seed': getattr(config, 'random_seed', 42),
                'use_gpu': getattr(config, 'use_gpu', False),
                'device': getattr(config, 'device', 'cpu'),
                'train_episodes': getattr(config, 'train_episodes', 100),
            }
            
            # Add agent-specific configs
            agent_config_map = {
                'dqn': getattr(config, 'dqn_config', {}),
                'rbc': getattr(config, 'rbc_config', {}),
                'tql': getattr(config, 'tql_config', {}),
                'sac': getattr(config, 'sac_config', {}),
                'Memdqn': getattr(config, 'dqn_config', {}),
                'mp_ppo': getattr(config, 'mp_ppo_config', {}),
            }
            
            if agent_type in agent_config_map:
                config_dict.update(agent_config_map[agent_type])

    if hasattr(config, 'get_agent_config'):
        agent_specific = config.get_agent_config(agent_type)
        config_dict.update(agent_specific)
        print(f"[DEBUG] Merged agent config for {agent_type}: {list(agent_specific.keys())}")
    
    # Transform config to match new structure
    new_config = {
        'algorithm_config': config_dict,
        'reward_config': config_dict.get('reward_config', {}),
        'strategy_config': config_dict.get('strategy_config', {}),
        'random_seed': config_dict.get('random_seed', 42),
        'use_gpu': config_dict.get('use_gpu', False),
        'device': config_dict.get('device', 'gpu')
    }
    
    # Create agent using new architecture
    try:
        agent = create_agent_from_legacy_name(agent_type, env, new_config)
        
        # Ensure training mode is set for trainable agents
        if agent_type in ['dqn', 'Chen_Bu_p2p', 'mp_ppo','tql', 'sac']:
            agent.set_training_mode(True)
        
        return agent
    except Exception as e:
        print(f"Failed to create agent {agent_type}: {e}")
        raise

# Legacy registry for backward compatibility
AGENT_REGISTRY = {
    'baseline': 'baseline',
    'rbc': 'rbc', 
    'dqn': 'dqn',
    'tql': 'tql',
    'sac': 'sac',
    'mp_ppo': 'mp_ppo',
    'Chen_Bu_p2p': 'Chen_Bu_p2p',
    'mpc_forecast': 'mpc_forecast',
    'ambitious_engineers': 'ambitious_engineers',
}