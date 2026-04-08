#================================================
#hems/environment/citylearn/citylearn_wrapper.py
#================================================


"""
CityLearn environment wrapper for HEMS framework.
Provides clean, isolated interface to CityLearn simulation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import (
    NormalizedObservationWrapper, 
    StableBaselines3Wrapper, 
    TabularQLearningWrapper
)


class CityLearnWrapper:
    """
    Clean wrapper for CityLearn environment.
    Handles building selection, period selection, and agent-specific configuration.
    Completely reward-agnostic - reward functions are passed as parameters.
    """
    
    def __init__(self, config):
        """
        Initialize CityLearn wrapper.
        
        Args:
            config: SimulationConfig object with dataset_name, building selection, etc.
        """
        self.config = config
        self.dataset_name = config.dataset_name
        np.random.seed(config.random_seed)
        
    def _get_schema(self):
        """
        Get schema with proper API compatibility for different CityLearn versions.
        
        Returns:
            Schema dictionary
        """
        from citylearn.data import DataSet
        
        try:
            dataset = DataSet()
            schema = dataset.get_schema(self.dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load schema for {self.dataset_name}: {e}")
        
        if not isinstance(schema, dict):
            raise RuntimeError(f"Schema is not a dictionary, got {type(schema)}")
        
        return schema
        
    def select_buildings(self, exclude_buildings: Optional[List[str]] = None) -> List[str]:
        """
        Select buildings for simulation based on configuration.
        
        Args:
            exclude_buildings: Buildings to exclude from selection
            
        Returns:
            List of selected building names
        """
        if self.config.building_id:
            return [self.config.building_id]
        
        schema = self._get_schema()
        
        if 'buildings' not in schema:
            raise KeyError(f"Schema missing 'buildings' key. Available keys: {list(schema.keys())}")
        
        available_buildings = list(schema['buildings'].keys())
        
        exclude_buildings = exclude_buildings or []
        exclude_buildings.extend(['Building_12', 'Building_15'])
        
        for building in exclude_buildings:
            if building in available_buildings:
                available_buildings.remove(building)
        
        if not available_buildings:
            raise ValueError("No buildings available after exclusion")
        
        if self.config.building_count == 1:
            selected = ['Building_1'] if 'Building_1' in available_buildings else [available_buildings[0]]
        else:
            count = min(self.config.building_count, len(available_buildings))
            selected = np.random.choice(available_buildings, size=count, replace=False).tolist()
            
            try:
                building_ids = [int(b.split('_')[-1]) for b in selected]
                building_ids = sorted(building_ids)
                selected = [f'Building_{i}' for i in building_ids]
            except (ValueError, IndexError):
                selected = sorted(selected)
        
        return selected
    
    def select_simulation_period(self) -> Tuple[int, int]:
        """
        Select simulation period based on configuration.
        
        Returns:
            Tuple of (start_time_step, end_time_step)
        """
        schema = self._get_schema()
        
        total_time_steps = 8760
        
        root_directory = schema.get('root_directory', '')
        if root_directory and 'buildings' in schema:
            try:
                first_building = list(schema['buildings'].keys())[0]
                building_data_path = Path(root_directory) / schema['buildings'][first_building]['energy_simulation']
                
                if building_data_path.exists():
                    import pandas as pd
                    building_data = pd.read_csv(building_data_path)
                    total_time_steps = len(building_data)
            except (KeyError, IndexError, FileNotFoundError):
                pass
        
        simulation_length = 24 * self.config.simulation_days
        max_start = max(0, total_time_steps - simulation_length)
        
        start_options = np.arange(0, max_start + 1, 24)
        
        if len(start_options) == 0:
            start_time_step = 0
        else:
            start_idx = self.config.random_seed % len(start_options)
            start_time_step = int(start_options[start_idx])
        
        end_time_step = min(start_time_step + simulation_length - 1, total_time_steps - 1)
        
        return start_time_step, end_time_step
    
    def _initialize_battery(self, storage, capacity: float):
        """
        Force battery initialization with proper parameters.
        Clear any pre-loaded data to enable simulation mode.
        
        Args:
            storage: ElectricalStorage object
            capacity: Battery capacity in kWh
        """
        power_value = capacity * 0.5  # C-rate of 0.5 (2-hour charge time)
        soc_init_value = 0.5  # 50% initial SoC
        
        # Set power limits
        storage.nominal_power = power_value
        if hasattr(storage, 'power_rating'):
            storage.power_rating = power_value
        
        # Set initial SoC  
        storage.soc_init = soc_init_value
        if hasattr(storage, 'initial_soc'):
            storage.initial_soc = soc_init_value
        if hasattr(storage, '_soc_init'):
            storage._soc_init = soc_init_value
        
        # Set efficiency
        if not hasattr(storage, 'efficiency') or storage.efficiency is None:
            storage.efficiency = 0.9
        
        # CRITICAL: Reset the SoC array
        # CityLearn loads SoC from CSV with all zeros
        # We need to keep the array structure but set the first value correctly
        if hasattr(storage, 'soc') and hasattr(storage.soc, '__len__'):
            # Don't resize the array, just set all values to soc_init
            # This preserves CityLearn's array structure while initializing properly
            if isinstance(storage.soc, np.ndarray):
                storage.soc.fill(soc_init_value)
            elif isinstance(storage.soc, list):
                storage.soc = [soc_init_value] * len(storage.soc)
            
            # Ensure at least one value is set
            if len(storage.soc) > 0:
                storage.soc[0] = soc_init_value
    
    def create_environment(
        self, 
        buildings: List[str], 
        start_time: int, 
        end_time: int,
        reward_function=None,
        use_custom_battery=False,
        central_agent: bool = True
    ):
        """
        Create CityLearn environment instance.
        
        Args:
            buildings: List of building names
            start_time: Simulation start timestep
            end_time: Simulation end timestep
            reward_function: Optional reward function (CityLearn-compatible)
            use_custom_battery: If True, use custom battery simulator instead of
                               CityLearn's built-in battery (useful for datasets
                               with pre-recorded/frozen battery data)
            
        Returns:
            Configured CityLearnEnv instance
        """
        schema = self._get_schema()
        
        if 'buildings' not in schema:
            raise KeyError("Schema missing 'buildings' key")
        
        schema_dict = schema.copy()
        schema_dict['buildings'] = {b: schema['buildings'][b] for b in buildings if b in schema['buildings']}
        
        if not schema_dict['buildings']:
            raise ValueError(f"None of the requested buildings {buildings} found in schema")
        
        try:
            env = CityLearnEnv(
                schema=schema_dict,
                entral_agent=central_agent,
                simulation_start_time_step=start_time,
                simulation_end_time_step=end_time
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create CityLearn environment: {e}")
        
        # CRITICAL FIX: Initialize battery after environment creation
        for building in env.buildings:
            if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                storage = building.electrical_storage
                capacity = getattr(storage, 'capacity', 6.4)
                self._initialize_battery(storage, capacity)
        
        # Wrap the environment to fix battery on each reset
        env = BatteryResetWrapper(env)
        
        # Optionally use custom battery simulator
        if use_custom_battery:
            from hems.environments.citylearn.custom_battery_wrapper import CustomBatteryWrapper
            env = CustomBatteryWrapper(env, initial_soc=0.5, efficiency=0.9)
        
        if reward_function is not None:
            env.reward_function = reward_function
        
        return env


class BatteryResetWrapper:
    """
    Wrapper that ensures battery is properly initialized after each reset.
    Fixes issues with CityLearn loading pre-recorded battery data from CSV files.
    """
    
    def __init__(self, env):
        self.env = env
        self._fix_battery_on_reset = True
    
    def reset(self, **kwargs):
        """Reset environment and reinitialize battery."""
        result = self.env.reset(**kwargs)
        
        # Fix battery SoC after reset (CityLearn might reload from CSV)
        if self._fix_battery_on_reset:
            for building in self.env.buildings:
                if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                    storage = building.electrical_storage
                    
                    # Get initial SoC value
                    soc_init = getattr(storage, 'soc_init', 0.5)
                    
                    # Reset SoC array: fill entire array with initial value
                    # (CityLearn pre-loads with zeros from CSV)
                    if hasattr(storage, 'soc'):
                        if isinstance(storage.soc, np.ndarray):
                            storage.soc.fill(soc_init)
                        elif isinstance(storage.soc, list):
                            storage.soc = [soc_init] * len(storage.soc)
                        
                        # Ensure first element is set (this is the "current" SoC)
                        if len(storage.soc) > 0:
                            storage.soc[0] = soc_init
        
        return result
    
    def step(self, actions):
        """Forward step to wrapped environment."""
        return self.env.step(actions)
    
    def __getattr__(self, name):
        """Forward all other attributes to wrapped environment."""
        return getattr(self.env, name)
    
    def create_environment_for_agent(
        self, 
        agent_type: str, 
        buildings: List[str], 
        start_time: int, 
        end_time: int,
        reward_function=None
    ):
        """
        Create environment with agent-specific wrappers.
        
        Args:
            agent_type: Type of agent ('dqn', 'sac', 'tql', etc.)
            buildings: List of building names
            start_time: Simulation start timestep
            end_time: Simulation end timestep
            reward_function: Optional reward function
            
        Returns:
            Environment with appropriate wrappers applied
        """
        env = self.create_environment(buildings, start_time, end_time, reward_function)
        
        if agent_type == 'tql':
            tql_obs_bins = self.config.tql_config.get('observation_bins', {
                'hour': 24,
                'electricity_pricing': 10, 
                'net_electricity_consumption': 20,
                'electrical_storage_soc': 10,
                'solar_generation': 10
            })
            
            tql_action_bins = self.config.tql_config.get('action_bins', {
                'electrical_storage': 12
            })
            
            observation_bin_sizes = [tql_obs_bins] * len(buildings)
            action_bin_sizes = [tql_action_bins] * len(buildings)
            
            try:
                env = TabularQLearningWrapper(
                    env,
                    observation_bin_sizes=observation_bin_sizes,
                    action_bin_sizes=action_bin_sizes
                )
            except Exception as e:
                raise RuntimeError(f"Failed to apply TQL wrapper: {e}")
        
        return env
    
    def validate_environment(self, env) -> Dict[str, bool]:
        """
        Validate environment setup and configuration.
        
        Args:
            env: Environment instance
            
        Returns:
            Dictionary with validation results
        """
        validations = {
            'has_buildings': hasattr(env, 'buildings') and len(env.buildings) > 0,
            'has_observation_space': hasattr(env, 'observation_space'),
            'has_action_space': hasattr(env, 'action_space'),
            'has_time_steps': hasattr(env, 'time_steps'),
            'battery_configured': True,
            'battery_power_set': True,
            'battery_soc_initialized': True
        }
        
        if hasattr(env, 'buildings'):
            for building in env.buildings:
                if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                    storage = building.electrical_storage
                    
                    # Check capacity
                    capacity = getattr(storage, 'capacity', None)
                    if capacity is None or capacity <= 0:
                        validations['battery_configured'] = False
                    
                    # Check nominal power (CRITICAL for charge/discharge)
                    nominal_power = getattr(storage, 'nominal_power', None)
                    if nominal_power is None or nominal_power <= 0:
                        validations['battery_power_set'] = False
                    
                    # Check initial SoC (check both soc_init and current soc)
                    soc_init = getattr(storage, 'soc_init', None)
                    current_soc = getattr(storage, 'soc', [None])[-1] if hasattr(storage, 'soc') and len(storage.soc) > 0 else None
                    
                    if soc_init is None and (current_soc is None or current_soc < 0.01):
                        validations['battery_soc_initialized'] = False
        
        return validations
    
    def validate_battery_dynamics(self, env, num_steps: int = 10) -> Dict[str, Any]:
        """
        Validate battery dynamics by testing charge/discharge actions.
        
        Args:
            env: Environment instance
            num_steps: Number of test steps
            
        Returns:
            Dictionary with battery validation results
        """
        if not hasattr(env, 'buildings') or len(env.buildings) == 0:
            return {'valid': False, 'error': 'No buildings in environment'}
        
        building = env.buildings[0]
        if not hasattr(building, 'electrical_storage') or building.electrical_storage is None:
            return {'valid': False, 'error': 'No battery in building'}
        
        storage = building.electrical_storage
        
        # Reset environment
        env.reset()
        
        # Track SoC changes
        soc_history = []
        action_history = []
        
        for i in range(num_steps):
            # Alternate charge/discharge
            action_val = 0.5 if i % 2 == 0 else -0.5
            action = [[action_val]]
            
            env.step(action)
            
            if hasattr(storage, 'soc') and len(storage.soc) > 0:
                soc_history.append(float(storage.soc[-1]))
                action_history.append(action_val)
        
        if not soc_history:
            return {'valid': False, 'error': 'No SoC data collected'}
        
        soc_array = np.array(soc_history)
        
        results = {
            'valid': True,
            'soc_range': [float(soc_array.min()), float(soc_array.max())],
            'soc_mean': float(soc_array.mean()),
            'soc_utilization': float(soc_array.max() - soc_array.min()),
            'soc_changes': len(set(soc_array.round(4))) > 1,
            'bounds_respected': float(soc_array.min()) >= 0 and float(soc_array.max()) <= 1
        }
        
        # Check if battery is actually responding to actions
        if results['soc_utilization'] < 0.01:
            results['valid'] = False
            results['error'] = 'Battery SoC not changing (actions may be ignored)'
        
        if not results['bounds_respected']:
            results['valid'] = False
            results['error'] = f"SoC out of bounds: [{results['soc_range'][0]}, {results['soc_range'][1]}]"
        
        return results


class CityLearnEnvironmentManager:
    """
    Manager for CityLearn environment lifecycle.
    Integrates with HEMS framework patterns.
    """
    
    def __init__(self, config):
        """
        Initialize environment manager.
        
        Args:
            config: SimulationConfig object
        """
        self.config = config
        self.wrapper = CityLearnWrapper(config)
        self._training_env = None
        self._eval_env = None
        
    def get_training_environment(self, reward_function=None):
        """
        Get or create training environment.
        
        Args:
            reward_function: Optional reward function for training
            
        Returns:
            Training environment instance
        """
        if self._training_env is None:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            
            self._training_env = self.wrapper.create_environment(
                buildings, start_time, end_time, reward_function
            )
        
        return self._training_env
    
    def get_evaluation_environment(self, reward_function=None):
        """
        Get or create evaluation environment.
        
        Args:
            reward_function: Optional reward function for evaluation
            
        Returns:
            Evaluation environment instance
        """
        if self._eval_env is None:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            
            self._eval_env = self.wrapper.create_environment(
                buildings, start_time, end_time, reward_function
            )
        
        return self._eval_env
    
    def create_environment_for_agent(
        self, 
        agent_type: str,
        reward_function=None
    ):
        """
        Create environment configured for specific agent.
        
        Args:
            agent_type: Type of agent
            reward_function: Reward function for this agent
            
        Returns:
            Agent-specific environment
        """
        buildings = self.wrapper.select_buildings()
        start_time, end_time = self.wrapper.select_simulation_period()
        
        return self.wrapper.create_environment_for_agent(
            agent_type, buildings, start_time, end_time, reward_function
        )
    
    def reset_environments(self):
        """Reset cached environments."""
        self._training_env = None
        self._eval_env = None


class CityLearnEnvironmentManagerAdapter:
    """
    Adapter to make CityLearnEnvironmentManager compatible with BaseEnvironmentManager.
    This bridges the new clean wrapper with the existing framework structure.
    """
    
    def __init__(self, config):
        self.config = config
        self.manager = CityLearnEnvironmentManager(config)
    
    def is_available(self) -> bool:
        """Check if CityLearn is available."""
        try:
            import citylearn
            return True
        except ImportError:
            return False
    
    def get_supported_datasets(self) -> List[str]:
        """Get supported CityLearn datasets."""
        return [
            'citylearn_challenge_2022_phase_all',
            'citylearn_challenge_2022_phase_1', 
            'citylearn_challenge_2022_phase_2',
            'citylearn_challenge_2022_phase_3'
        ]
    
    def create_environment_instance(self):
        """Create environment manager instance."""
        return self.manager


def register_citylearn_environment():
    """Register CityLearn environment with the framework registry."""
    try:
        from hems.environments.base import EnvironmentRegistry
        EnvironmentRegistry.register('citylearn', CityLearnEnvironmentManagerAdapter)
    except ImportError:
        pass