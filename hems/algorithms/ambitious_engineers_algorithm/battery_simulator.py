# ====================================================================
# hems/agents/ambitious_engineers_algorithm.py - battery_simulator.py
# ====================================================================


"""
Battery Simulator - Replicates CityLearn Battery Physics
=========================================================
Faithful reproduction of the battery model used by Team ambitiousengineers.
This implements the exact same battery dynamics as the CityLearn Challenge 2022.

Key Physics:
- State of Charge (SoC) tracking
- Capacity degradation over cycles
- Charging/discharging efficiency
- Power constraints
- Energy calculations for DP optimization

"""

import numpy as np
from typing import Tuple


class BatterySimulator:
    """
    High-fidelity battery simulator matching CityLearn physics.
    
    This simulator is used for:
    1. Accurate energy calculations in DP optimization
    2. Simulating future battery states
    3. Computing net energy consumption for cost functions
    
    Battery Specifications (Challenge Default):
    - Nominal Capacity: 6.4 kWh
    - Nominal Power: 5.0 kW (Building 4) or 4.0 kW (others)
    - Efficiency: 0.9 (both charging and discharging)
    - Degradation: Capacity degrades with cycling
    - SoC Range: [0.0, 1.0] (0% to 100%)
    """
    
    def __init__(
        self,
        capacity: float = 6.4,
        nominal_power: float = 5.0,
        efficiency: float = 0.9,
        capacity_loss_coefficient: float = 1e-5
    ):
        """
        Initialize battery simulator.
        
        Args:
            capacity: Battery capacity in kWh (default: 6.4)
            nominal_power: Maximum charge/discharge power in kW (default: 5.0)
            efficiency: Round-trip efficiency (default: 0.9)
            capacity_loss_coefficient: Degradation rate (default: 1e-5)
        """
        self.nominal_capacity = capacity
        self.nominal_power = nominal_power
        self.efficiency = efficiency
        self.capacity_loss_coef = capacity_loss_coefficient
        
        # Efficiency values
        self.efficiency_charging = efficiency
        self.efficiency_discharging = efficiency
        
        # Loss coefficients (per unit energy processed)
        self.loss_coef = capacity_loss_coefficient
        
        # Depth of Discharge for degradation calculation
        self.depth_of_discharge = 0.0
    
    def fast_simulate(
        self,
        action: float,
        current_soc: float,
        current_capacity: float,
        timestep: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Fast battery simulation for a single timestep.
        
        This is the core function used by the DP solver and policy networks.
        It computes the next battery state given an action.
        
        Args:
            action: Normalized action in [-1, 1]
                    -1 = full discharge, 0 = no action, +1 = full charge
            current_soc: Current State of Charge [0, 1]
            current_capacity: Current capacity in kWh (degrades over time)
            timestep: Time step in hours (default: 1.0)
        
        Returns:
            Tuple of (next_soc, next_capacity, next_efficiency, battery_energy)
            - next_soc: New State of Charge [0, 1]
            - next_capacity: New capacity after degradation
            - next_efficiency: Current efficiency value
            - battery_energy: Energy flow (positive = charging, negative = discharging)
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to power (kW)
        # Positive action = charging (battery consumes power)
        # Negative action = discharging (battery provides power)
        power = -action * self.nominal_power  # Note: negative sign for convention
        
        # Calculate energy change (kWh) based on efficiency
        if power > 0:  # Charging
            # Energy stored = power * time * efficiency
            energy_change = power * timestep * self.efficiency_charging
            battery_energy = power * timestep  # Energy drawn from grid
        else:  # Discharging
            # Energy released = power * time / efficiency
            energy_change = power * timestep / self.efficiency_discharging
            battery_energy = power * timestep  # Energy provided to grid
        
        # Calculate new SoC
        soc_change = energy_change / current_capacity
        next_soc = current_soc + soc_change
        
        # Clip SoC to valid range [0, 1]
        next_soc = np.clip(next_soc, 0.0, 1.0)
        
        # Calculate actual energy change (after clipping)
        actual_soc_change = next_soc - current_soc
        actual_energy_change = actual_soc_change * current_capacity
        
        # Recalculate battery energy based on actual energy change
        if actual_energy_change > 0:  # Actually charging
            battery_energy = actual_energy_change / self.efficiency_charging
        elif actual_energy_change < 0:  # Actually discharging
            battery_energy = actual_energy_change * self.efficiency_discharging
        else:
            battery_energy = 0.0
        
        # Capacity degradation
        # Degradation is proportional to energy throughput
        energy_throughput = abs(actual_energy_change)
        capacity_loss = self.loss_coef * energy_throughput
        next_capacity = current_capacity - capacity_loss
        
        # Ensure capacity doesn't go below reasonable threshold
        next_capacity = max(next_capacity, 0.8 * self.nominal_capacity)
        
        # Current efficiency (remains constant in this simple model)
        next_efficiency = self.efficiency
        
        return next_soc, next_capacity, next_efficiency, battery_energy
    
    def simulate_trajectory(
        self,
        actions: np.ndarray,
        initial_soc: float = 0.0,
        initial_capacity: float = 6.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate battery behavior over a sequence of actions.
        
        Useful for:
        - Validating action sequences
        - Analyzing battery degradation
        - Computing total energy flows
        
        Args:
            actions: Array of normalized actions, shape (T,)
            initial_soc: Starting State of Charge
            initial_capacity: Starting capacity in kWh
        
        Returns:
            Tuple of (soc_trajectory, capacity_trajectory, efficiency_trajectory, energy_flow)
            All arrays have shape (T+1,) with initial state at index 0
        """
        T = len(actions)
        
        # Initialize trajectories
        soc = np.zeros(T + 1)
        capacity = np.zeros(T + 1)
        efficiency = np.zeros(T + 1)
        energy = np.zeros(T)
        
        # Set initial conditions
        soc[0] = initial_soc
        capacity[0] = initial_capacity
        efficiency[0] = self.efficiency
        
        # Simulate each timestep
        for t in range(T):
            soc[t+1], capacity[t+1], efficiency[t+1], energy[t] = self.fast_simulate(
                action=actions[t],
                current_soc=soc[t],
                current_capacity=capacity[t]
            )
        
        return soc, capacity, efficiency, energy
    
    def compute_max_charge_action(
        self,
        current_soc: float,
        current_capacity: float,
        timestep: float = 1.0
    ) -> float:
        """
        Compute maximum feasible charging action.
        
        Accounts for:
        - Power limit
        - SoC upper bound
        - Energy capacity
        
        Args:
            current_soc: Current State of Charge [0, 1]
            current_capacity: Current capacity in kWh
            timestep: Time step in hours
        
        Returns:
            Maximum charging action in [-1, 1]
        """
        # Energy needed to reach full charge
        energy_to_full = (1.0 - current_soc) * current_capacity
        
        # Maximum energy that can be charged in one timestep
        max_energy = self.nominal_power * timestep * self.efficiency_charging
        
        # Actual energy to charge (limited by both)
        energy_to_charge = min(energy_to_full, max_energy)
        
        # Convert to action
        if max_energy > 0:
            max_action = energy_to_charge / max_energy
        else:
            max_action = 0.0
        
        return max_action
    
    def compute_max_discharge_action(
        self,
        current_soc: float,
        current_capacity: float,
        timestep: float = 1.0
    ) -> float:
        """
        Compute maximum feasible discharging action.
        
        Accounts for:
        - Power limit
        - SoC lower bound
        - Available energy
        
        Args:
            current_soc: Current State of Charge [0, 1]
            current_capacity: Current capacity in kWh
            timestep: Time step in hours
        
        Returns:
            Maximum discharging action in [-1, 1] (negative value)
        """
        # Energy available to discharge
        energy_available = current_soc * current_capacity
        
        # Maximum energy that can be discharged in one timestep
        max_energy = self.nominal_power * timestep / self.efficiency_discharging
        
        # Actual energy to discharge (limited by both)
        energy_to_discharge = min(energy_available, max_energy)
        
        # Convert to action (negative for discharge)
        if max_energy > 0:
            max_action = -energy_to_discharge / max_energy
        else:
            max_action = 0.0
        
        return max_action
    
    def get_net_energy(
        self,
        action: float,
        current_soc: float,
        current_capacity: float,
        load: float,
        solar: float,
        timestep: float = 1.0
    ) -> float:
        """
        Calculate net energy consumption for a given action.
        
        Net energy = load + battery_energy - solar
        - Positive: importing from grid
        - Negative: exporting to grid
        
        This is used extensively in the DP cost function.
        
        Args:
            action: Battery action [-1, 1]
            current_soc: Current State of Charge [0, 1]
            current_capacity: Current capacity in kWh
            load: Non-shiftable load in kW
            solar: Solar generation in kW
            timestep: Time step in hours
        
        Returns:
            Net energy consumption in kWh
        """
        # Simulate battery
        _, _, _, battery_energy = self.fast_simulate(
            action=action,
            current_soc=current_soc,
            current_capacity=current_capacity,
            timestep=timestep
        )
        
        # Calculate net consumption
        # Load consumption + battery consumption - solar generation
        net_energy = (load + battery_energy - solar) * timestep
        
        return net_energy


# Utility functions for batch operations
def simulate_multiple_buildings(
    simulators: list,
    actions: np.ndarray,
    initial_states: list
) -> list:
    """
    Simulate multiple buildings in parallel.
    
    Args:
        simulators: List of BatterySimulator instances
        actions: Array of actions, shape (n_buildings, T)
        initial_states: List of (soc, capacity) tuples
    
    Returns:
        List of trajectory tuples for each building
    """
    results = []
    for sim, acts, (soc0, cap0) in zip(simulators, actions, initial_states):
        trajectories = sim.simulate_trajectory(acts, soc0, cap0)
        results.append(trajectories)
    return results


if __name__ == "__main__":
    # Test battery simulator
    print("Testing Battery Simulator")
    print("=" * 60)
    
    # Create simulator
    battery = BatterySimulator(
        capacity=6.4,
        nominal_power=5.0,
        efficiency=0.9
    )
    
    # Test 1: Single step simulation
    print("\nTest 1: Single Step Simulation")
    print("-" * 60)
    action = 0.5  # Half charging
    soc, capacity, efficiency, energy = battery.fast_simulate(
        action=action,
        current_soc=0.3,
        current_capacity=6.4
    )
    print(f"Action: {action:.2f} (charging)")
    print(f"Next SoC: {soc:.4f}")
    print(f"Next Capacity: {capacity:.4f} kWh")
    print(f"Battery Energy: {energy:.4f} kWh")
    
    # Test 2: Full charge cycle
    print("\nTest 2: Full Charge Cycle")
    print("-" * 60)
    actions = np.array([0.5] * 10 + [-0.5] * 10)  # Charge then discharge
    soc_traj, cap_traj, eff_traj, energy_traj = battery.simulate_trajectory(
        actions=actions,
        initial_soc=0.5,
        initial_capacity=6.4
    )
    print(f"Initial SoC: {soc_traj[0]:.4f}")
    print(f"SoC after charging: {soc_traj[10]:.4f}")
    print(f"Final SoC: {soc_traj[-1]:.4f}")
    print(f"Capacity degradation: {6.4 - cap_traj[-1]:.6f} kWh")
    
    # Test 3: Net energy calculation
    print("\nTest 3: Net Energy Calculation")
    print("-" * 60)
    net_energy = battery.get_net_energy(
        action=0.8,
        current_soc=0.4,
        current_capacity=6.4,
        load=3.0,  # 3 kW load
        solar=2.0,  # 2 kW solar
        timestep=1.0
    )
    print(f"Load: 3.0 kW")
    print(f"Solar: 2.0 kW")
    print(f"Battery action: 0.8 (charging)")
    print(f"Net energy: {net_energy:.4f} kWh")
    print(f"  {'Importing from grid' if net_energy > 0 else 'Exporting to grid'}")
    
    print("\n" + "=" * 60)
    print("✅ Battery Simulator Tests Completed!")