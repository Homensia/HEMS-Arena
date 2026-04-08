# ========================
# hems/agents/factory.py
# ========================
"""Agent factory for composing HEMS agents from modular components.

This module provides the primary entry point for creating ``HEMSAgent``
instances by composing an Algorithm, a Reward function, and a Strategy.
It handles environment detection, metadata extraction, reward adapter
selection, and component validation.

Typical usage::

    agent = create_agent_from_legacy_name('dqn', env, config)

Or with explicit component names::

    agent = create_agent('dqn', 'custom_v5', 'single_agent', env, config)

Legacy agent names (e.g., ``'dqn'``, ``'mp_ppo'``, ``'Chen_Bu_p2p'``) are
mapped to their respective (algorithm, reward, strategy) tuples via
``create_agent_from_legacy_name``.
"""

from typing import Dict, Any, Optional, Union
from .agent import HEMSAgent

# Import registries
from hems.algorithms.registry import create_algorithm
from hems.environments.citylearn.reward_factory import create_reward_for_environment  # New location  # Updated system
from hems.strategies.registry import create_strategy


def create_agent(
    algorithm_name: str, 
    reward_name: str, 
    strategy_name: str,
    env, 
    config: Dict[str, Any],
    agent_name: Optional[str] = None
) -> HEMSAgent:
    """
    Create HEMS agent from component specifications.
    
    This uses the new environment-agnostic reward system with proper adapters
    located in the environments module.
    
    Args:
        algorithm_name: Name of algorithm to use
        reward_name: Name of reward function to use  
        strategy_name: Name of strategy to use
        env: Environment instance (e.g., CityLearn)
        config: Configuration dictionary with sections for each component
        agent_name: Optional custom agent name
        
    Returns:
        Composed HEMS agent
        
    Example:
        config = {
            'algorithm_config': {'learning_rate': 0.001, 'batch_size': 128},
            'reward_config': {'alpha_cost': 0.6, 'alpha_peak': 0.01},
            'strategy_config': {'centralized_control': True}
        }
        
        agent = create_agent('dqn', 'custom_v5', 'mp_ppo', 'Chen_Bu_P2P' 'single_agent', env, config)
    """
    
    # Extract component configurations
    algorithm_config = config.get('algorithm_config', {})
    reward_config = config.get('reward_config', {})
    strategy_config = config.get('strategy_config', {})
    
    # Add common parameters to algorithm config
    common_params = ['random_seed', 'use_gpu', 'device']
    for param in common_params:
        if param in config:
            algorithm_config[param] = config[param]

    if hasattr(config, 'get_agent_config'):
        # Config is a SimulationConfig object - get agent-specific settings
        agent_specific_config = config.get_agent_config(algorithm_name)
        algorithm_config.update(agent_specific_config)
        print(f"[AgentFactory] Merged agent-specific config for {algorithm_name}")
        if 'pretrained_model_path' in agent_specific_config:
            print(f"[AgentFactory] ✓ Found pretrained_model_path: {agent_specific_config['pretrained_model_path']}")
    
    try:
        print(f"[AgentFactory] Creating agent components...")
        
        # Create algorithm
        print(f"[AgentFactory] Creating algorithm: {algorithm_name}")
        algorithm = create_algorithm(algorithm_name, env, algorithm_config)
        
        # Create reward function using NEW environment-agnostic system
        print(f"[AgentFactory] Creating reward function: {reward_name}")
        
        # Determine environment type
        env_type = _detect_environment_type(env)
        print(f"[AgentFactory] Detected environment type: {env_type}")
        
        # Extract environment metadata
        env_metadata = _extract_environment_metadata(env)
        
        # Create reward with proper adapter (adapter is in environments module now)
        reward_function = create_reward_for_environment(
            reward_name=reward_name,
            environment_type=env_type,
            env_metadata=env_metadata,
            **reward_config
        )
        
        # Validate reward function creation
        _validate_reward_function(reward_function, reward_name, env_type)
        
        # Create strategy
        print(f"[AgentFactory] Creating strategy: {strategy_name}")
        strategy = create_strategy(
            strategy_name, env, algorithm, reward_function, strategy_config
        )
        
        # Create composed agent
        agent = HEMSAgent(algorithm, reward_function, strategy, agent_name)
        
        # Success logging
        print(f"[AgentFactory] ✓ Successfully created agent: {agent.name}")
        print(f"[AgentFactory]   Algorithm: {algorithm.__class__.__name__}")
        print(f"[AgentFactory]   Reward: {reward_function.__class__.__name__}")
        print(f"[AgentFactory]   Strategy: {strategy.__class__.__name__}")
        
        # Show environment agnostic status
        is_adapted = hasattr(reward_function, 'hems_reward')
        if is_adapted:
            pure_reward_name = reward_function.hems_reward.__class__.__name__
            print(f"[AgentFactory]   Environment-agnostic: ✓ (Pure: {pure_reward_name})")
        else:
            print(f"[AgentFactory]   Environment-agnostic: ✓ (Pure reward)")
        
        return agent
        
    except Exception as e:
        print(f"[AgentFactory] ✗ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        raise


def _detect_environment_type(env) -> str:
    """Detect the type of environment being used.
    
    Args:
        env: Environment instance
        
    Returns:
        Environment type string
    """
    env_class_name = env.__class__.__name__.lower()
    env_module = getattr(env.__class__, '__module__', '').lower()
    
    # Check for CityLearn
    if 'citylearn' in env_class_name or 'citylearn' in env_module:
        return 'citylearn'
    
    # Check for other known environments
    if 'gym' in env_class_name or 'gymnasium' in env_class_name:
        return 'gym'
    
    if 'dummy' in env_class_name or 'test' in env_class_name:
        return 'dummy'
    
    # Default fallback
    print(f"[AgentFactory] Unknown environment type, defaulting to 'generic': {env_class_name}")
    return 'generic'


def _extract_environment_metadata(env) -> Dict[str, Any]:
    """Extract metadata from environment for adapter configuration.
    
    Args:
        env: Environment instance
        
    Returns:
        Metadata dictionary
    """
    metadata = {}
    
    # Try to get standard metadata
    if hasattr(env, 'metadata'):
        metadata.update(env.metadata)
    
    # Try to get buildings information (common for CityLearn)
    if hasattr(env, 'buildings'):
        metadata['buildings'] = getattr(env, 'buildings', [])
    
    # Try to get other common attributes
    common_attrs = ['action_space', 'observation_space', 'num_buildings']
    for attr in common_attrs:
        if hasattr(env, attr):
            try:
                metadata[attr] = getattr(env, attr)
            except Exception:
                # Skip if attribute access fails
                pass
    
    # Ensure we always have a buildings entry
    if 'buildings' not in metadata:
        metadata['buildings'] = []
    
    return metadata


def _validate_reward_function(
    reward_function: Any, 
    reward_name: str, 
    env_type: str
) -> None:
    """Validate that reward function was created correctly.
    
    Args:
        reward_function: Created reward function
        reward_name: Name of the reward
        env_type: Environment type
        
    Raises:
        ValueError: If validation fails
    """
    # Check basic interface
    if not hasattr(reward_function, 'calculate'):
        raise ValueError(
            f"Reward function {reward_name} missing calculate() method"
        )
    
    # Check if it's properly adapted for the environment
    if env_type == 'citylearn':
        # For CityLearn, should be adapted unless it's 'generic'
        is_adapted = hasattr(reward_function, 'hems_reward')
        adapter_name = reward_function.__class__.__name__
        
        if is_adapted and 'adapter' in adapter_name.lower():
            print(f"[AgentFactory] ✓ Reward properly adapted for CityLearn")
        elif not is_adapted:
            print(f"[AgentFactory] ⚠ Using pure reward with CityLearn - "
                  f"ensure observation format compatibility")
    
    # Try a basic functionality test
    try:
        if hasattr(reward_function, 'get_info'):
            info = reward_function.get_info()
            print(f"[AgentFactory] Reward function info: {info.get('name', 'unknown')}")
    except Exception as e:
        print(f"[AgentFactory] Warning: Could not get reward function info: {e}")


def create_agent_from_legacy_name(
    legacy_agent_name: str, 
    env, 
    config: Dict[str, Any]
) -> HEMSAgent:
    """
    Create agent using legacy naming convention for backward compatibility.
    
    This allows the new modular architecture to work with existing code unchanged.
    
    Args:
        legacy_agent_name: Legacy agent type ('baseline', 'dqn', 'mp_ppo', 'Chen_BU_P2P', 'sac', etc.)
        env: Environment instance
        config: Configuration dictionary
        
    Returns:
        HEMS agent instance
    """
    # Legacy name mappings
   
    legacy_mappings = {
        'baseline': ('baseline', 'simple', 'single_agent'),
        'rbc': ('rbc', 'custom_v5', 'single_agent'),
        'dqn': ('dqn', 'custom_v5', 'single_agent'),
        'tql': ('tql', 'custom_v5', 'single_agent'),
        'sac': ('sac', 'custom_v5', 'single_agent'),
        'Chen_Bu_p2p': ('Memdqn', 'Chen_Bu_p2p', 'single_agent'),
        'mp_ppo': ('mp_ppo', 'mp_ppo', 'single_agent'),
        'mpc_forecast':('mpc_forecast', 'none', 'single_agent'),
        'ambitious_engineers': ('ambitious_engineers', 'none', 'single_agent'),
    }

    if legacy_agent_name not in legacy_mappings:
        raise ValueError(f"Unknown legacy agent: {legacy_agent_name}")

    algorithm_name, reward_name, strategy_name = legacy_mappings[legacy_agent_name]

    return create_agent(
        algorithm_name=algorithm_name,
        reward_name=reward_name,
        strategy_name=strategy_name,
        env=env,
        config=config,
        agent_name=legacy_agent_name
    )


def validate_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent configuration before creation.
    
    Args:
        config: Agent configuration dictionary
        
    Returns:
        Validation results with any issues found
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "normalized_config": {}
    }
    
    # Check required sections
    required_sections = ['algorithm_config', 'reward_config', 'strategy_config']
    for section in required_sections:
        if section not in config:
            results["warnings"].append(f"Missing config section: {section}")
            config[section] = {}
    
    # Validate algorithm config
    algo_config = config.get('algorithm_config', {})
    if 'learning_rate' in algo_config:
        lr = algo_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            results["errors"].append("learning_rate must be positive number")
    
    # Validate reward config
    reward_config = config.get('reward_config', {})
    for key, value in reward_config.items():
        if 'alpha' in key and not isinstance(value, (int, float)):
            results["errors"].append(f"Reward parameter {key} must be numeric")
    
    # Set validation status
    results["valid"] = len(results["errors"]) == 0
    results["normalized_config"] = config.copy()
    
    return results


def get_factory_info() -> Dict[str, Any]:
    """Get information about the agent factory.
    
    Returns:
        Dictionary with factory capabilities and status
    """
    return {
        "factory_version": "2.0",
        "architecture": "modular_composition",
        "supported_components": {
            "algorithms": "via hems.algorithms.registry",
            "rewards": "via hems.rewards.registry (environment-agnostic)",
            "strategies": "via hems.strategies.registry"
        },
        "adapter_support": {
            "citylearn": "hems.environments.citylearn.adapters.reward_adapter",
            "generic": "pure_rewards_no_adaptation"
        },
        "legacy_support": True,
        "validation": True,
        "features": [
            "environment_detection",
            "metadata_extraction", 
            "reward_validation",
            "performance_logging",
            "error_handling"
        ]
    }


# ========================================================================
# Testing and Development Support
# ========================================================================

def test_agent_creation(
    algorithm_name: str = 'dqn',
    reward_name: str = 'simple',
    strategy_name: str = 'single_agent',
    env_type: str = 'dummy'
) -> Dict[str, Any]:
    """Test agent creation with dummy components.
    
    Args:
        algorithm_name: Algorithm to test
        reward_name: Reward to test
        strategy_name: Strategy to test
        env_type: Environment type to simulate
        
    Returns:
        Test results
    """
    try:
        # Create dummy environment
        class DummyEnv:
            def __init__(self):
                self.buildings = ['Building1']
                self.metadata = {'buildings': self.buildings}
            
        dummy_env = DummyEnv()
        
        # Basic test config
        test_config = {
            'algorithm_config': {'learning_rate': 0.001},
            'reward_config': {'cost_weight': 1.0},
            'strategy_config': {'centralized_control': True},
            'random_seed': 42
        }
        
        # Test agent creation
        agent = create_agent(
            algorithm_name, reward_name, strategy_name, 
            dummy_env, test_config, f"test_{algorithm_name}"
        )
        
        return {
            "success": True,
            "agent_name": agent.name,
            "components": {
                "algorithm": agent.algorithm.__class__.__name__,
                "reward": agent.reward_function.__class__.__name__,
                "strategy": agent.strategy.__class__.__name__
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Run basic tests when module is executed directly
    print("Testing agent factory...")
    
    result = test_agent_creation()
    if result["success"]:
        print(f"✓ Test passed: {result['agent_name']}")
        for component, name in result["components"].items():
            print(f"  {component}: {name}")
    else:
        print(f"✗ Test failed: {result['error']}")