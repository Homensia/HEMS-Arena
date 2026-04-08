# ======================================================================
# hems/agents/ambitious_engineers_algorithm.py - dynamic_programming.py
# ======================================================================

"""
Dynamic Programming Solver for Single-Agent Optimization
=========================================================
Faithful reproduction of Team ambitiousengineers' Phase 1 DP approach.

Key Innovation: Proxy Cost Function
------------------------------------
Since grid costs are non-causal (depend on aggregate behavior), we optimize
using a proxy cost function that penalizes large net energy usage.

Proxy Cost Formula:
    cost = (power_price[t+1] / base_power) * clip(net_energy, 0, None)
         + (emission_price[t+1] / base_emission) * clip(net_energy, 0, None)
         + w1 * |net_energy| / base_proxy1
         + w2 * |net_energy²| / base_proxy2
         + w3 * |net_energy³| / base_proxy3

This penalizes:
- Energy imports (via pricing)
- Absolute energy usage (L1 penalty)
- Squared energy usage (L2 penalty)
- Cubic energy usage (L3 penalty for extreme peaks)

The DP solver uses backward induction to find optimal actions for each state.

"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from tqdm import tqdm
import multiprocessing
from functools import partial

from .battery_simulator import BatterySimulator


class DynamicProgrammingSolver:
    """
    Single-agent dynamic programming solver with proxy cost optimization.
    
    This solver computes optimal battery control actions for a single building
    by solving a finite-horizon DP problem with discretized state space.
    
    State Space:
    - State: Battery State of Charge (SoC) in [0, 1]
    - Discretized into n_states grid points (default: 101)
    
    Action Space:
    - Action: Normalized battery action in [-1, 1]
    - Optimized continuously for each state
    
    Time Horizon:
    - Typically 8760 hours (1 year) for annual optimization
    
    Algorithm:
    - Backward induction from T to 0
    - For each time step and state, find optimal action
    - Uses scipy.optimize for continuous action optimization
    """
    
    def __init__(
        self,
        building_data: pd.DataFrame,
        price_data: pd.DataFrame,
        emission_data: pd.DataFrame,
        building_id: int = 1,
        proxy_weights: Optional[Dict[str, float]] = None,
        n_states: int = 101,
        battery_capacity: float = 6.4,
        battery_power: float = 4.0,  # 5.0 for building 4
        verbose: bool = True
    ):
        """
        Initialize DP solver.
        
        Args:
            building_data: DataFrame with columns ['non_shiftable_load', 'solar_generation']
            price_data: DataFrame with electricity pricing
            emission_data: DataFrame with carbon intensity
            building_id: Building identifier (1-5)
            proxy_weights: Dictionary with proxy cost weights
            n_states: Number of discrete SoC states
            battery_capacity: Battery capacity in kWh
            battery_power: Battery nominal power in kW
            verbose: Whether to show progress
        """
        self.building_data = building_data
        self.price_data = price_data
        self.emission_data = emission_data
        self.building_id = building_id
        self.n_states = n_states
        self.verbose = verbose
        
        # Battery parameters
        self.battery_capacity = battery_capacity
        self.battery_power = battery_power
        
        # Initialize battery simulator
        self.simulator = BatterySimulator(
            capacity=battery_capacity,
            nominal_power=battery_power
        )
        
        # Extract time series data
        self.load = building_data['non_shiftable_load'].values
        self.solar = building_data['solar_generation'].values
        self.prices = price_data.values.flatten() if isinstance(price_data, pd.DataFrame) else price_data
        self.emissions = emission_data.values.flatten() if isinstance(emission_data, pd.DataFrame) else emission_data
        
        self.n_horizon = len(self.load)
        
        # Proxy cost weights (from their tuned hyperparameters)
        if proxy_weights is None:
            proxy_weights = {
                'l1_penalty': 0.1300366652733203,
                'l2_penalty': 0.009508553021774526,
                'proxy_weight1': 0.011557095624191585,
                'proxy_weight2': 0.08482562812998984,
                'proxy_weight3': 2.61560899997805
            }
        
        self.l1_penalty = proxy_weights['l1_penalty']
        self.l2_penalty = proxy_weights['l2_penalty']
        self.proxy_weight1 = proxy_weights['proxy_weight1']
        self.proxy_weight2 = proxy_weights['proxy_weight2']
        self.proxy_weight3 = proxy_weights['proxy_weight3']
        
        # Compute baseline costs (with no battery control)
        self._compute_baseline_costs()
        
        # State grid (SoC values from 0 to battery_capacity)
        self.state_grid = np.linspace(0, battery_capacity, n_states)
        
        # DP tables
        self.J = None  # Cost-to-go function: J[t, i] = optimal cost from state i at time t
        self.mu = None  # Policy function: mu[t, i] = optimal action from state i at time t
        
    def _compute_baseline_costs(self):
        """
        Compute baseline costs with zero battery action.
        
        These are used to normalize the proxy cost function terms.
        """
        if self.verbose:
            print("Computing baseline costs (no battery control)...")
        
        # Simulate battery with zero actions
        actions_baseline = np.zeros(self.n_horizon)
        soc_hist, cap_hist, eff_hist, battery_energy = self.simulator.simulate_trajectory(
            actions=actions_baseline,
            initial_soc=0.0,
            initial_capacity=self.battery_capacity
        )
        
        # Compute net energy without storage
        net_energy_baseline = (
            battery_energy + self.load[:self.n_horizon] - self.solar[:self.n_horizon]
        )
        
        # Baseline power cost
        self.base_power_cost = np.sum(
            self.prices[:self.n_horizon] * np.clip(net_energy_baseline, 0, None)
        )
        
        # Baseline emission cost
        self.base_emission_cost = np.sum(
            self.emissions[:self.n_horizon] * np.clip(net_energy_baseline, 0, None)
        )
        
        # Baseline proxy costs
        self.base_proxy_cost1 = np.mean(np.abs(net_energy_baseline))
        self.base_proxy_cost2 = np.mean(np.abs(net_energy_baseline ** 2))
        self.base_proxy_cost3 = np.mean(np.abs(net_energy_baseline ** 3))
        
        if self.verbose:
            print(f"  Base power cost: {self.base_power_cost:.2f}")
            print(f"  Base emission cost: {self.base_emission_cost:.2f}")
            print(f"  Base proxy costs: {self.base_proxy_cost1:.4f}, "
                  f"{self.base_proxy_cost2:.4f}, {self.base_proxy_cost3:.4f}")
    
    def _evaluate_action(
        self,
        action: float,
        current_soc: float,
        time_step: int,
        J_next: Callable
    ) -> float:
        """
        Evaluate cost of taking an action from a given state.
        
        This is the core Bellman equation evaluation:
        Q(s, a, t) = immediate_cost(s, a, t) + J(s', t+1)
        
        Args:
            action: Normalized action in [-1, 1]
            current_soc: Current battery SoC
            time_step: Current time step
            J_next: Interpolated cost-to-go function for next time step
        
        Returns:
            Total cost (immediate + future)
        """
        # Simulate battery action
        next_soc, next_capacity, _, battery_energy = self.simulator.fast_simulate(
            action=action,
            current_soc=current_soc,
            current_capacity=self.battery_capacity  # Assume constant for DP
        )
        
        # Calculate net energy consumption
        net_external_load = self.load[time_step + 1] - self.solar[time_step + 1]
        net_energy = battery_energy + net_external_load
        
        # Proxy cost function
        cost = 0.0
        
        # Power cost component (only for positive net energy - imports)
        cost += (self.prices[time_step + 1] / self.base_power_cost) * np.clip(net_energy, 0, None)
        
        # Emission cost component (only for positive net energy)
        cost += (self.emissions[time_step + 1] / self.base_emission_cost) * np.clip(net_energy, 0, None)
        
        # L1 penalty on absolute net energy
        cost += self.proxy_weight1 * np.abs(net_energy) / self.base_proxy_cost1
        
        # L2 penalty on squared net energy
        cost += self.proxy_weight2 * np.abs(net_energy ** 2) / self.base_proxy_cost2
        
        # L3 penalty on cubed net energy (penalizes extreme peaks)
        cost += self.proxy_weight3 * np.abs(net_energy ** 3) / self.base_proxy_cost3
        
        # Action penalty (L1 and L2 on action to encourage smoothness)
        penalty = self.l1_penalty * abs(action) + self.l2_penalty * action ** 2
        cost += penalty
        
        # Add future cost
        cost += J_next(next_soc)
        
        return cost
    
    def _optimize_state(
        self,
        state_index: int,
        time_step: int,
        J_next: Callable
    ) -> Tuple[float, float]:
        """
        Find optimal action for a given state and time.
        
        Uses scipy's minimize_scalar for continuous optimization over actions.
        
        Args:
            state_index: Index in state grid
            time_step: Current time step
            J_next: Interpolated cost-to-go function
        
        Returns:
            Tuple of (optimal_cost, optimal_action)
        """
        current_soc = self.state_grid[state_index]
        
        # Objective function
        def objective(action):
            return self._evaluate_action(action, current_soc, time_step, J_next)
        
        # Optimize action in [-1, 1]
        result = minimize_scalar(
            objective,
            bounds=(-1.0, 1.0),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        return result.fun, result.x
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the DP problem using backward induction.
        
        Returns:
            Tuple of (J, mu):
            - J: Cost-to-go function, shape (n_horizon+1, n_states)
            - mu: Policy function (optimal actions), shape (n_horizon, n_states)
        """
        if self.verbose:
            print(f"\nSolving DP for Building {self.building_id}")
            print(f"  Horizon: {self.n_horizon} time steps")
            print(f"  States: {self.n_states} SoC grid points")
            print("  Starting backward induction...\n")
        
        # Initialize DP tables
        self.J = np.zeros((self.n_horizon + 1, self.n_states))
        self.mu = np.zeros((self.n_horizon, self.n_states))
        
        # Terminal cost is zero
        self.J[-1, :] = 0.0
        
        # Backward induction
        iterator = range(self.n_horizon - 1)[::-1]
        if self.verbose:
            iterator = tqdm(iterator, desc="DP Optimization")
        
        for time_step in iterator:
            # Create interpolated cost-to-go function for next time step
            J_next = interp1d(
                self.state_grid,
                self.J[time_step + 1],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Optimize each state
            for state_idx in range(self.n_states):
                optimal_cost, optimal_action = self._optimize_state(
                    state_idx, time_step, J_next
                )
                
                self.J[time_step, state_idx] = optimal_cost
                self.mu[time_step, state_idx] = optimal_action
        
        if self.verbose:
            print("\n✅ DP optimization completed!")
        
        return self.J, self.mu
    
    def get_action_trajectory(
        self,
        initial_soc: float = 0.0
    ) -> np.ndarray:
        """
        Extract optimal action trajectory from policy.
        
        Simulates forward from initial state following the optimal policy.
        
        Args:
            initial_soc: Initial battery State of Charge
        
        Returns:
            Array of optimal actions, shape (n_horizon,)
        """
        if self.mu is None:
            raise ValueError("Must call solve() before extracting trajectory")
        
        actions = np.zeros(self.n_horizon)
        current_soc = initial_soc
        
        # Create interpolator for policy at each time step
        for t in range(self.n_horizon):
            # Interpolate policy at current SoC
            policy_interp = interp1d(
                self.state_grid,
                self.mu[t, :],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Get optimal action
            action = policy_interp(current_soc)
            actions[t] = np.clip(action, -1.0, 1.0)
            
            # Simulate to get next state
            next_soc, _, _, _ = self.simulator.fast_simulate(
                action=actions[t],
                current_soc=current_soc,
                current_capacity=self.battery_capacity
            )
            current_soc = next_soc
        
        return actions
    
    def evaluate_trajectory(
        self,
        actions: np.ndarray,
        initial_soc: float = 0.0
    ) -> Dict[str, float]:
        """
        Evaluate a trajectory of actions.
        
        Computes various cost metrics for analysis.
        
        Args:
            actions: Array of actions, shape (n_horizon,)
            initial_soc: Initial State of Charge
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Simulate battery
        soc_traj, cap_traj, eff_traj, battery_energy = self.simulator.simulate_trajectory(
            actions=actions,
            initial_soc=initial_soc,
            initial_capacity=self.battery_capacity
        )
        
        # Calculate net energy
        net_energy = battery_energy + self.load[:len(actions)] - self.solar[:len(actions)]
        
        # Calculate costs
        power_cost = np.sum(self.prices[:len(actions)] * np.clip(net_energy, 0, None))
        emission_cost = np.sum(self.emissions[:len(actions)] * np.clip(net_energy, 0, None))
        
        # Peak demand
        peak_demand = np.max(np.clip(net_energy, 0, None))
        
        # Energy imported/exported
        energy_imported = np.sum(np.clip(net_energy, 0, None))
        energy_exported = np.sum(np.clip(-net_energy, 0, None))
        
        # Battery utilization
        battery_cycles = np.sum(np.abs(np.diff(soc_traj))) / 2  # Full cycle = 2.0
        
        return {
            'power_cost': power_cost,
            'emission_cost': emission_cost,
            'total_cost': power_cost + emission_cost,
            'peak_demand': peak_demand,
            'energy_imported': energy_imported,
            'energy_exported': energy_exported,
            'battery_cycles': battery_cycles,
            'final_soc': soc_traj[-1],
            'final_capacity': cap_traj[-1],
            'capacity_degradation': self.battery_capacity - cap_traj[-1]
        }


def solve_multiple_buildings(
    buildings_data: Dict[int, pd.DataFrame],
    price_data: pd.DataFrame,
    emission_data: pd.DataFrame,
    proxy_weights: Optional[Dict[str, float]] = None,
    n_states: int = 101,
    n_jobs: int = 1,
    verbose: bool = True
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve DP for multiple buildings in parallel.
    
    Args:
        buildings_data: Dict mapping building_id to DataFrame with load/solar data
        price_data: Electricity pricing data
        emission_data: Carbon intensity data
        proxy_weights: Proxy cost weights
        n_states: Number of SoC grid points
        n_jobs: Number of parallel jobs
        verbose: Show progress
    
    Returns:
        Dict mapping building_id to (J, mu) tuples
    """
    def solve_single_building(building_id):
        # Building-specific battery power (5.0 kW for building 4, 4.0 kW otherwise)
        battery_power = 5.0 if building_id == 4 else 4.0
        
        solver = DynamicProgrammingSolver(
            building_data=buildings_data[building_id],
            price_data=price_data,
            emission_data=emission_data,
            building_id=building_id,
            proxy_weights=proxy_weights,
            n_states=n_states,
            battery_power=battery_power,
            verbose=verbose and n_jobs == 1
        )
        
        J, mu = solver.solve()
        actions = solver.get_action_trajectory()
        
        return building_id, (J, mu, actions)
    
    # Solve in parallel if requested
    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            results = pool.map(solve_single_building, buildings_data.keys())
    else:
        results = [solve_single_building(bid) for bid in buildings_data.keys()]
    
    # Convert to dictionary
    return {bid: (J, mu, actions) for bid, (J, mu, actions) in results}


if __name__ == "__main__":
    # Test DP solver
    print("Testing Dynamic Programming Solver")
    print("=" * 80)
    
    # Create synthetic test data
    n_hours = 24 * 7  # 1 week
    
    # Building data
    building_data = pd.DataFrame({
        'non_shiftable_load': 2.0 + np.sin(np.arange(n_hours) * 2 * np.pi / 24),  # Daily pattern
        'solar_generation': np.maximum(0, 3.0 * np.sin(np.arange(n_hours) * 2 * np.pi / 24 - np.pi/2))  # Daytime
    })
    
    # Price data (higher during day)
    prices = 0.15 + 0.10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
    price_data = pd.DataFrame({'price': prices})
    
    # Emission data
    emissions = 0.5 + 0.2 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
    emission_data = pd.DataFrame({'emission': emissions})
    
    # Test 1: Solve DP
    print("\nTest 1: Solving DP Problem")
    print("-" * 80)
    solver = DynamicProgrammingSolver(
        building_data=building_data,
        price_data=price_data,
        emission_data=emission_data,
        building_id=1,
        n_states=51,  # Fewer states for faster testing
        verbose=True
    )
    
    J, mu = solver.solve()
    
    print(f"\nCost-to-go function shape: {J.shape}")
    print(f"Policy function shape: {mu.shape}")
    print(f"Initial state value: J[0, 25] = {J[0, 25]:.4f}")
    
    # Test 2: Extract and evaluate trajectory
    print("\nTest 2: Extracting Optimal Trajectory")
    print("-" * 80)
    actions = solver.get_action_trajectory(initial_soc=0.0)
    print(f"Action trajectory shape: {actions.shape}")
    print(f"Action statistics:")
    print(f"  Mean: {np.mean(actions):.4f}")
    print(f"  Std: {np.std(actions):.4f}")
    print(f"  Min: {np.min(actions):.4f}")
    print(f"  Max: {np.max(actions):.4f}")
    
    # Test 3: Evaluate trajectory
    print("\nTest 3: Evaluating Trajectory")
    print("-" * 80)
    metrics = solver.evaluate_trajectory(actions)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 4: Compare with baseline (no control)
    print("\nTest 4: Comparing with Baseline")
    print("-" * 80)
    baseline_actions = np.zeros(n_hours)
    baseline_metrics = solver.evaluate_trajectory(baseline_actions)
    
    print("Baseline (no control):")
    print(f"  Total cost: {baseline_metrics['total_cost']:.2f}")
    print(f"  Peak demand: {baseline_metrics['peak_demand']:.2f} kW")
    
    print("\nOptimal DP policy:")
    print(f"  Total cost: {metrics['total_cost']:.2f}")
    print(f"  Peak demand: {metrics['peak_demand']:.2f} kW")
    
    cost_savings = (baseline_metrics['total_cost'] - metrics['total_cost']) / baseline_metrics['total_cost'] * 100
    peak_reduction = (baseline_metrics['peak_demand'] - metrics['peak_demand']) / baseline_metrics['peak_demand'] * 100
    
    print(f"\nImprovement:")
    print(f"  Cost savings: {cost_savings:.1f}%")
    print(f"  Peak reduction: {peak_reduction:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ DP Solver Tests Completed!")
    print("\nNext: This DP solver will be used as the foundation for Phase 1 policy.")
    print("The Phase 1 neural network will refine these DP actions using CMA-ES.")