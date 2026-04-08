#=====================================================
#hems/environment/citylearn/custom_battery_wrapper.py
#=====================================================


"""
Custom battery simulator for CityLearn.
This wrapper completely overrides CityLearn's battery dynamics
to work around datasets that have pre-recorded (frozen) battery data.
"""
import numpy as np


class CustomBatteryWrapper:
    """
    Wrapper that implements custom battery dynamics.
    
    Use this when CityLearn's battery is frozen/pre-recorded and doesn't
    respond to actions (common in challenge datasets).
    """
    
    def __init__(self, env, initial_soc=0.5, efficiency=0.9):
        """
        Args:
            env: CityLearn environment
            initial_soc: Initial state of charge (0-1)
            efficiency: Round-trip efficiency (0-1)
        """
        self.env = env
        self.initial_soc = initial_soc
        self.efficiency = efficiency
        
        # Initialize custom battery state for each building
        self.battery_states = []
        for building in env.buildings:
            if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                storage = building.electrical_storage
                capacity = getattr(storage, 'capacity', 6.4)
                nominal_power = getattr(storage, 'nominal_power', capacity * 0.5)
                
                self.battery_states.append({
                    'soc': initial_soc,
                    'capacity': capacity,
                    'nominal_power': nominal_power,
                    'efficiency': efficiency,
                    'soc_history': [initial_soc]
                })
            else:
                self.battery_states.append(None)
    
    def reset(self, **kwargs):
        """Reset environment and battery states."""
        result = self.env.reset(**kwargs)
        
        # Reset custom battery states
        for i, battery_state in enumerate(self.battery_states):
            if battery_state is not None:
                battery_state['soc'] = self.initial_soc
                battery_state['soc_history'] = [self.initial_soc]
        
        # Update CityLearn's battery SoC to match our custom state
        for i, building in enumerate(self.env.buildings):
            if self.battery_states[i] is not None:
                storage = building.electrical_storage
                soc = self.battery_states[i]['soc']
                
                # Update the SoC in CityLearn's storage object
                if hasattr(storage, 'soc'):
                    if isinstance(storage.soc, np.ndarray):
                        storage.soc[0] = soc  # Set current SoC
                    elif isinstance(storage.soc, list) and len(storage.soc) > 0:
                        storage.soc[0] = soc
        
        return result
    
    def step(self, actions):
        """
        Take a step in the environment and update custom battery dynamics.
        
        Args:
            actions: List of actions for each building
            
        Returns:
            observations, rewards, done, truncated, info
        """
        # Update custom battery states based on actions
        for i, (action, battery_state) in enumerate(zip(actions, self.battery_states)):
            if battery_state is not None and len(action) > 0:
                self._update_battery(battery_state, action[0])
        
        # CRITICAL: Update CityLearn's battery SoC at CURRENT time_step BEFORE calling env.step
        # This ensures the next observation includes the updated SoC
        for i, building in enumerate(self.env.buildings):
            if self.battery_states[i] is not None:
                storage = building.electrical_storage
                soc = self.battery_states[i]['soc']
                time_step = building.time_step  # Current time_step BEFORE step
                
                # Update SoC at current position
                if hasattr(storage, 'soc'):
                    if isinstance(storage.soc, np.ndarray) and time_step < len(storage.soc):
                        storage.soc[time_step] = soc
                    elif isinstance(storage.soc, list) and time_step < len(storage.soc):
                        storage.soc[time_step] = soc
        
        # Take step in wrapped environment (this advances time_step)
        result = self.env.step(actions)
        
        # No need for post-update since we updated at the right time_step before
        
        return result
    
    def _update_battery(self, battery_state, action):
        """
        Update battery state based on action.
        
        Args:
            battery_state: Dict with battery parameters and state
            action: Scalar action (-1 to 1, where positive=charge, negative=discharge)
        """
        soc = battery_state['soc']
        capacity = battery_state['capacity']
        nominal_power = battery_state['nominal_power']
        efficiency = battery_state['efficiency']
        
        # Convert action to energy (kWh per hour timestep)
        # action is normalized (-1 to 1), nominal_power is in kW
        # Assuming 1-hour timesteps
        energy_requested = action * nominal_power  # kWh
        
        if energy_requested > 0:
            # Charging: apply efficiency loss
            energy_stored = energy_requested * efficiency
            soc_change = energy_stored / capacity
        else:
            # Discharging: apply efficiency loss
            energy_delivered = abs(energy_requested)
            energy_removed = energy_delivered / efficiency
            soc_change = -energy_removed / capacity
        
        # Apply change with bounds [0, 1]
        new_soc = np.clip(soc + soc_change, 0.0, 1.0)
        
        # Update state
        battery_state['soc'] = float(new_soc)
        battery_state['soc_history'].append(float(new_soc))
    
    def get_battery_soc(self, building_idx=0):
        """Get current SoC for a building."""
        if building_idx < len(self.battery_states) and self.battery_states[building_idx] is not None:
            return self.battery_states[building_idx]['soc']
        return 0.0
    
    def get_battery_history(self, building_idx=0):
        """Get SoC history for a building."""
        if building_idx < len(self.battery_states) and self.battery_states[building_idx] is not None:
            return self.battery_states[building_idx]['soc_history']
        return []
    
    def __getattr__(self, name):
        """Forward all other attributes to wrapped environment."""
        return getattr(self.env, name)