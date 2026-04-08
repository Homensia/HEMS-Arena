"""
Dummy environment implementation for testing and development.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ..base import BaseHEMSEnvironment, BaseEnvironmentManager, EnvironmentRegistry


class DummyHEMSEnvironment(BaseHEMSEnvironment):
    """
    Simple dummy environment for testing algorithms without CityLearn dependency.
    
    This environment simulates basic energy management scenarios with:
    - Synthetic load profiles
    - Simple PV generation patterns
    - Basic battery dynamics
    - Configurable pricing schemes
    """
    
    def __init__(self, config):
        """Initialize dummy environment."""
        super().__init__(config)
        self.dataset_name = "dummy_dataset"
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dummy dataset information."""
        return {
            'dataset_type': 'dummy',
            'dataset_name': self.dataset_name,
            'num_buildings': self.config.building_count,
            'total_timesteps': 24 * self.config.simulation_days,
            'time_resolution_minutes': 60,
            'description': 'Synthetic dummy environment for testing'
        }
    
    def select_buildings(self, exclude_buildings: List[str] = None) -> List[str]:
        """Select buildings for dummy environment."""
        building_count = self.config.building_count
        buildings = [f"DummyBuilding_{i+1}" for i in range(building_count)]
        
        print(f"Selected dummy buildings: {buildings}")
        return buildings
    
    def select_simulation_period(self) -> Tuple[int, int]:
        """Select simulation period for dummy environment."""
        start_time = 0
        end_time = 24 * self.config.simulation_days - 1
        
        print(f"Dummy simulation period: {self.config.simulation_days} days "
              f"({start_time} to {end_time}, {end_time - start_time + 1} steps)")
        
        return start_time, end_time
    
    def create_environment(self, buildings: List[str], start_time: int, end_time: int):
        """Create dummy environment."""
        env = DummyEnergyEnvironment(
            buildings=buildings,
            start_time=start_time,
            end_time=end_time,
            config=self.config
        )
        
        print(f"Dummy environment created:")
        print(f"  - Buildings: {len(buildings)}")
        print(f"  - Time steps: {end_time - start_time + 1}")
        
        return env
    
    def inject_tariff(self, env) -> np.ndarray:
        """Inject electricity tariff into dummy environment."""
        T = env.time_steps
        
        if self.config.tariff_type == 'hp_hc':
            # Simple HP/HC pattern
            hours = np.arange(T) % 24
            prices = np.where(
                np.isin(hours, self.config.hc_hours),
                self.config.price_hc,
                self.config.price_hp
            ).astype(np.float32)
        elif self.config.tariff_type == 'tempo':
            # Simplified Tempo-like pattern
            prices = np.random.choice([0.15, 0.25, 0.45], size=T, p=[0.7, 0.25, 0.05])
        else:
            # Flat rate
            prices = np.full(T, 0.15, dtype=np.float32)
        
        # Inject into environment
        env.electricity_pricing = prices
        
        print(f"Injected {self.config.tariff_type} tariff into dummy environment")
        return prices
    
    def create_environment_for_agent(self, agent_type: str, buildings: List[str], 
                                   start_time: int, end_time: int):
        """Create dummy environment configured for specific agent type."""
        env = self.create_environment(buildings, start_time, end_time)
        
        # Inject tariff
        self.inject_tariff(env)
        
        # Apply agent-specific configurations if needed
        if agent_type == 'tql':
            # For TQL, we might want to discretize observations
            env.discrete_mode = True
        elif agent_type == 'sac':
            # For SAC, ensure continuous action space
            env.continuous_mode = True
        
        return env


class DummyEnergyEnvironment:
    """
    Simple energy management environment for testing.
    
    Mimics the CityLearn interface but with synthetic data.
    """
    
    def __init__(self, buildings: List[str], start_time: int, end_time: int, config):
        """Initialize dummy energy environment."""
        self.building_names = buildings
        self.start_time = start_time
        self.end_time = end_time
        self.time_steps = end_time - start_time + 1
        self.config = config
        
        # Environment state
        self.current_step = 0
        self.terminated = False
        self.central_agent = True
        
        # Initialize buildings
        self.buildings = []
        for name in buildings:
            building = self._create_dummy_building(name)
            self.buildings.append(building)
        
        # Environment data
        self.electricity_pricing = np.full(self.time_steps, 0.15, dtype=np.float32)
        self.net_electricity_consumption = [0.0] * self.time_steps
        
        # Generate synthetic patterns
        self._generate_synthetic_data()
        
        # Mock attributes for compatibility
        self.discrete_mode = False
        self.continuous_mode = True
    
    def _create_dummy_building(self, name: str):
        """Create a mock building object."""
        building = type('DummyBuilding', (), {})()
        building.name = name
        building.active_actions = ['electrical_storage']
        
        # Mock electrical storage
        storage = type('DummyStorage', (), {})()
        storage.capacity = 10.0  # 10 kWh
        storage.nominal_power = 5.0  # 5 kW
        storage.efficiency = 0.95
        storage.soc = [0.5] * self.time_steps  # Initialize SoC
        building.electrical_storage = storage
        
        # Mock PV
        pv = type('DummyPV', (), {})()
        pv.nominal_power = 5.0  # 5 kW
        building.pv = pv
        
        # Initialize data arrays
        building.net_electricity_consumption = [0.0] * self.time_steps
        building.solar_generation = [0.0] * self.time_steps
        
        return building
    
    def _generate_synthetic_data(self):
        """Generate synthetic energy data patterns."""
        hours = np.arange(self.time_steps) % 24
        
        for building in self.buildings:
            # Generate daily load pattern (higher during day, lower at night)
            base_load = 2.0 + np.sin((hours - 6) * np.pi / 12) * 1.5
            base_load = np.maximum(base_load, 0.5)  # Minimum load
            
            # Add some randomness
            np.random.seed(42 + hash(building.name) % 1000)
            noise = np.random.normal(0, 0.3, self.time_steps)
            consumption = base_load + noise
            
            # Generate PV pattern (peak at noon, zero at night)
            pv_pattern = np.maximum(0, 4.0 * np.sin((hours - 6) * np.pi / 12))
            pv_pattern = np.where((hours >= 6) & (hours <= 18), pv_pattern, 0)
            
            # Store data
            building.net_electricity_consumption = consumption.tolist()
            building.solar_generation = pv_pattern.tolist()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.terminated = False
        
        # Reset building states
        for building in self.buildings:
            building.electrical_storage.soc = [0.5] * self.time_steps
        
        self.net_electricity_consumption = [0.0] * self.time_steps
        
        # Return initial observations
        observations = self._get_observations()
        info = {}
        
        return observations, info
    
    def step(self, actions):
        """Take one step in the environment."""
        if self.terminated:
            raise RuntimeError("Environment is terminated. Call reset() first.")
        
        # Process actions and update state
        rewards = self._process_actions(actions)
        
        # Update time step
        self.current_step += 1
        self.terminated = self.current_step >= self.time_steps
        
        # Get new observations
        observations = self._get_observations()
        
        # Calculate total reward
        if isinstance(rewards, list):
            total_reward = sum(rewards)
        else:
            total_reward = float(rewards)
        
        info = {}
        truncated = False
        
        return observations, total_reward, self.terminated, truncated, info
    
    def _get_observations(self) -> List[List[float]]:
        """Get current observations for all buildings."""
        if self.terminated or self.current_step >= self.time_steps:
            time_idx = min(self.current_step - 1, self.time_steps - 1)
        else:
            time_idx = self.current_step
        
        observations = []
        
        for i, building in enumerate(self.buildings):
            obs = []
            
            # Hour of day
            hour = time_idx % 24
            obs.append(float(hour))
            
            # Electricity pricing
            price = self.electricity_pricing[time_idx] if time_idx < len(self.electricity_pricing) else 0.15
            obs.append(float(price))
            
            # Net electricity consumption
            consumption = building.net_electricity_consumption[time_idx] if time_idx < len(building.net_electricity_consumption) else 2.0
            obs.append(float(consumption))
            
            # Battery SoC
            soc = building.electrical_storage.soc[time_idx] if time_idx < len(building.electrical_storage.soc) else 0.5
            obs.append(float(soc))
            
            # Solar generation
            pv_gen = building.solar_generation[time_idx] if time_idx < len(building.solar_generation) else 0.0
            obs.append(float(pv_gen))
            
            observations.append(obs)
        
        return observations
    
    def _process_actions(self, actions) -> float:
        """Process actions and update environment state."""
        # Actions should be in format [[action1, action2, ...]]
        if len(actions) > 0 and isinstance(actions[0], list):
            building_actions = actions[0]
        else:
            building_actions = actions
        
        total_reward = 0.0
        time_idx = self.current_step
        
        for i, (building, action) in enumerate(zip(self.buildings, building_actions)):
            # Get current state
            base_consumption = building.net_electricity_consumption[time_idx] if time_idx < len(building.net_electricity_consumption) else 2.0
            pv_generation = building.solar_generation[time_idx] if time_idx < len(building.solar_generation) else 0.0
            
            # Apply battery action
            current_soc = building.electrical_storage.soc[time_idx] if time_idx < len(building.electrical_storage.soc) else 0.5
            battery_power = float(action) * building.electrical_storage.nominal_power
            
            # Simple battery model
            soc_change = -(battery_power / building.electrical_storage.capacity)
            new_soc = np.clip(current_soc + soc_change, 0.0, 1.0)
            
            # Update SoC for next timestep
            if time_idx + 1 < len(building.electrical_storage.soc):
                building.electrical_storage.soc[time_idx + 1] = new_soc
            
            # Calculate net consumption (positive = import, negative = export)
            net_consumption = base_consumption + battery_power - pv_generation
            
            # Store for tracking
            if time_idx < len(building.net_electricity_consumption):
                building.net_electricity_consumption[time_idx] = net_consumption
        
        # Simple reward calculation (negative cost)
        district_consumption = sum(
            building.net_electricity_consumption[time_idx] 
            for building in self.buildings 
            if time_idx < len(building.net_electricity_consumption)
        )
        
        if time_idx < len(self.net_electricity_consumption):
            self.net_electricity_consumption[time_idx] = district_consumption
        
        # Simple cost-based reward
        price = self.electricity_pricing[time_idx] if time_idx < len(self.electricity_pricing) else 0.15
        cost = district_consumption * price
        reward = -cost  # Negative cost as reward
        
        return reward
    
    def evaluate(self):
        """Evaluate environment performance (compatibility method)."""
        # Create basic evaluation DataFrame
        import pandas as pd
        
        data = []
        
        for building in self.buildings:
            # Building-level metrics
            total_consumption = sum(building.net_electricity_consumption)
            peak_consumption = max(building.net_electricity_consumption) if building.net_electricity_consumption else 0
            
            data.append({
                'level': 'building',
                'name': building.name,
                'cost_function': 'electricity_cost',
                'value': total_consumption * 0.15
            })
            
            data.append({
                'level': 'building',
                'name': building.name,
                'cost_function': 'peak_demand',
                'value': peak_consumption
            })
        
        # District-level metrics
        total_district_consumption = sum(self.net_electricity_consumption)
        peak_district_consumption = max(self.net_electricity_consumption) if self.net_electricity_consumption else 0
        
        data.append({
            'level': 'district',
            'name': 'district',
            'cost_function': 'electricity_cost',
            'value': total_district_consumption * 0.15
        })
        
        data.append({
            'level': 'district',
            'name': 'district',
            'cost_function': 'peak_demand',
            'value': peak_district_consumption
        })
        
        return pd.DataFrame(data)
    
    @property
    def observation_space(self):
        """Mock observation space for compatibility."""
        try:
            from gymnasium.spaces import Box
            # Create Box spaces for each building
            spaces = []
            for _ in self.building_names:
                obs_dim = 5  # hour, pricing, consumption, soc, solar
                spaces.append(Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,)))
            return spaces
        except ImportError:
            # Fallback if gymnasium not available
            return None
    
    @property
    def action_space(self):
        """Mock action space for compatibility."""
        try:
            from gymnasium.spaces import Box
            # Single continuous action per building (battery control)
            return Box(low=-1.0, high=1.0, shape=(len(self.building_names),))
        except ImportError:
            # Fallback if gymnasium not available
            return None
    
    @property
    def unwrapped(self):
        """Return self for unwrapped access."""
        return self


class DummyEnvironmentManager(BaseEnvironmentManager):
    """Manager for dummy environments."""
    
    def is_available(self) -> bool:
        """Dummy environment is always available."""
        return True
    
    def get_supported_datasets(self) -> List[str]:
        """Get supported dummy datasets."""
        return ['dummy_dataset', 'test_scenario_1', 'test_scenario_2']
    
    def create_environment_instance(self) -> BaseHEMSEnvironment:
        """Create dummy environment instance."""
        return DummyHEMSEnvironment(self.config)


# Register dummy environment
EnvironmentRegistry.register('dummy', DummyEnvironmentManager)