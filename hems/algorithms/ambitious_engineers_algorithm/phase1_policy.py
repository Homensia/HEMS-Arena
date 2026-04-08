# ================================================================
# hems/agents/ambitious_engineers_algorithm.py - phase1_policy.py
# ================================================================




"""
Phase 1 Policy Network - Single-Agent Refinement with CMA-ES
=============================================================
Faithful reproduction of Team ambitiousengineers' Phase 1 policy refinement.

Architecture:
    DP Action + Observation + Future Costs + Future Energy
           ↓
    Neural Network (5 hidden neurons)
           ↓
    Action Refinement (added to DP action)

Key Components:
1. Base network (model0): Takes DP action + current obs + future leads
2. Two refinement networks (model1a, model1b): Fine-tune the action
3. Time-dependent bias: Different behavior for different hours of day
4. Trained with CMA-ES optimization (3000 iterations × 5 seeds)

The policy adjusts DP actions based on:
- Current battery state (SoC)
- Future pricing signals (5 steps ahead)
- Future emission signals (5 steps ahead)
- Future energy forecasts (local + group)

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .battery_simulator import BatterySimulator


class Phase1PolicyNetwork:
    """
    Phase 1 single-agent policy network.
    
    This network refines DP actions using learned weights optimized with CMA-ES.
    
    Network Structure:
    - Input: 18 features (DP action, state, future costs, forecasts)
    - Hidden: 5 neurons with ReLU
    - Output: 1 action refinement
    
    Total Parameters: 184
    - bias: 5
    - time_bias: 24 (one per hour)
    - weights00: 5×18 = 90
    - weights01: 1×5 = 5
    - weights10a: 5×5 = 25
    - weights11a: 1×5 = 5
    - weights10b: 5×5 = 25
    - weights11b: 1×5 = 5
    Total: 5+24+90+5+25+5+25+5 = 184 parameters
    """
    
    def __init__(self, params: np.ndarray, building_id: int = 1):
        """
        Initialize policy network with learned parameters.
        
        Args:
            params: Flat array of 184 parameters (from CMA-ES)
            building_id: Building identifier (1-5)
        """
        assert len(params) == 184, f"Expected 184 parameters, got {len(params)}"
        
        self.building_id = building_id
        self.n_leads = 5  # Future steps to consider
        
        # Parse parameters
        # Model 0: Main network (18 features, 5 hidden, 1 output)
        self.bias = params[:5]
        self.time_bias = params[5:29]  # 24 hours
        self.weights00 = params[29:119].reshape(5, 18)
        self.weights01 = params[119:124].reshape(1, 5)
        
        # Model 1a: First refinement network (5 features, 5 hidden, 1 output)
        self.weights10a = params[124:149].reshape(5, 5)
        self.weights11a = params[149:154].reshape(1, 5)
        
        # Model 1b: Second refinement network (5 features, 5 hidden, 1 output)
        self.weights10b = params[154:179].reshape(5, 5)
        self.weights11b = params[179:184].reshape(1, 5)
        
        # Normalization parameters (computed from training data)
        # These are the mean and std of features from Phase 1 data
        self.mu0 = np.array([
            1.06639756e+00,  # DP action (normalized)
            6.99354999e-01,  # Battery SoC
            0.00000000e+00,  # Placeholder
            *[0.1565307] * self.n_leads,   # Emission leads (5)
            *[0.2731312] * self.n_leads,   # Price leads (5)
            *[0.3679884] * self.n_leads,   # Local forecast (5)
        ])
        
        self.sig0 = np.array([
            8.89049950e-01,  # DP action std
            1.01712690e+00,  # SoC std
            1.00000000e+00,  # Placeholder
            *[0.035367] * self.n_leads,    # Emission std (5)
            *[0.117795] * self.n_leads,    # Price std (5)
            *[1.041145] * self.n_leads,    # Local forecast std (5)
        ])
        
        # Refinement network normalization
        self.mu1a = np.array([0.0, 0.0, 0.0, 0.1565307, 0.273131])
        self.sig1a = np.array([1.0, 1.0, 1.0, 0.035367, 0.117795])
        
        self.mu1b = np.array([0.0, 1.445, 0.2890, 0.1565307, 0.273131])
        self.sig1b = np.array([1.0, 8.325, 1.6650, 0.035367, 0.117795])
        
        # Battery simulator for energy calculations
        nominal_power = 5.0 if building_id == 4 else 4.0
        self.simulator = BatterySimulator(
            capacity=6.4,
            nominal_power=nominal_power
        )
        
        # State tracking
        self.reset()
    
    def reset(self):
        """Reset policy state for new episode."""
        self.soc_history = [0.0]
        self.capacity_history = [6.4]
        self.net_energy_history = []
        self.action_history = []
        self.timestep = 0
    
    def compute_action_stage0(
        self,
        dp_action: float,
        observation: np.ndarray,
        emission_leads: np.ndarray,
        price_leads: np.ndarray,
        local_forecast: np.ndarray
    ) -> float:
        """
        Stage 0: Base action from main network.
        
        Args:
            dp_action: Action from DP solver
            observation: Current observation [hour, soc, ...]
            emission_leads: Future emissions (5 steps)
            price_leads: Future prices (5 steps)
            local_forecast: Future net energy forecast (5 steps)
        
        Returns:
            Base action
        """
        # Extract features
        hour = int(observation[2])  # Hour of day
        soc = self.soc_history[-1]
        
        # Construct feature vector (18 features)
        features = np.array([
            dp_action,
            soc,
            0.0,  # Placeholder
            *emission_leads,  # 5 features
            *price_leads,     # 5 features
            *local_forecast   # 5 features
        ])
        
        # Normalize features
        features = (features - self.mu0) / self.sig0
        
        # Apply time-dependent bias
        bias = self.bias * self.time_bias[hour - 1]
        
        # Forward pass: hidden = ReLU(bias + W × features)
        hidden = (bias + np.dot(self.weights00, features)).clip(0)
        
        # Output: action = W × hidden
        action = self.weights01.dot(hidden)[0]
        
        return action
    
    def compute_action_stage1a(
        self,
        action: float,
        next_agent_action: float,
        next_group_action: float,
        emission_lead1: float,
        price_lead1: float
    ) -> float:
        """
        Stage 1a: First refinement based on neighboring actions.
        
        Args:
            action: Current action
            next_agent_action: Next building's action
            next_group_action: Group average action
            emission_lead1: Next emission value
            price_lead1: Next price value
        
        Returns:
            Refined action
        """
        # Construct features (5 features)
        features = np.array([
            0.0,  # Placeholder
            0.0,  # Placeholder
            0.0,  # Placeholder
            emission_lead1,
            price_lead1
        ])
        
        # Normalize
        features = (features - self.mu1a) / self.sig1a
        
        # Forward pass
        hidden = np.dot(self.weights10a, features).clip(0)
        adjustment = np.tanh(self.weights11a.dot(hidden)[0])
        
        # Apply adjustment (signed)
        action += np.sign(action) * adjustment
        
        return action
    
    def compute_action_stage1b(
        self,
        action: float,
        next_agent_net_energy: float,
        next_group_net_energy: float,
        emission_lead1: float,
        price_lead1: float
    ) -> float:
        """
        Stage 1b: Second refinement based on energy forecasts.
        
        Args:
            action: Current action
            next_agent_net_energy: Predicted agent net energy
            next_group_net_energy: Predicted group net energy
            emission_lead1: Next emission value
            price_lead1: Next price value
        
        Returns:
            Final refined action
        """
        # Construct features (5 features)
        features = np.array([
            0.0,  # Placeholder
            next_agent_net_energy,
            next_group_net_energy,
            emission_lead1,
            price_lead1
        ])
        
        # Normalize
        features = (features - self.mu1b) / self.sig1b
        
        # Forward pass
        hidden = np.dot(self.weights10b, features).clip(0)
        adjustment = np.tanh(self.weights11b.dot(hidden)[0])
        
        # Apply adjustment (signed)
        action += np.sign(action) * adjustment
        
        return action
    
    def compute_action(
        self,
        dp_action: float,
        observation: np.ndarray,
        emission_data: np.ndarray,
        price_data: np.ndarray,
        local_forecast: np.ndarray,
        load: float,
        solar: float
    ) -> float:
        """
        Full action computation pipeline.
        
        Args:
            dp_action: DP baseline action
            observation: Current observation
            emission_data: Full emission time series
            price_data: Full price time series
            local_forecast: Local energy forecast
            load: Current load
            solar: Current solar generation
        
        Returns:
            Final action in [-1, 1]
        """
        # Get future leads
        t = self.timestep
        emission_leads = emission_data[t+1:t+1+self.n_leads]
        price_leads = price_data[t+1:t+1+self.n_leads]
        forecast_leads = local_forecast[t:t+self.n_leads]
        
        # Pad if needed
        if len(emission_leads) < self.n_leads:
            emission_leads = np.pad(emission_leads, (0, self.n_leads - len(emission_leads)))
        if len(price_leads) < self.n_leads:
            price_leads = np.pad(price_leads, (0, self.n_leads - len(price_leads)))
        if len(forecast_leads) < self.n_leads:
            forecast_leads = np.pad(forecast_leads, (0, self.n_leads - len(forecast_leads)))
        
        # Stage 0: Base action
        action = self.compute_action_stage0(
            dp_action=dp_action,
            observation=observation,
            emission_leads=emission_leads,
            price_leads=price_leads,
            local_forecast=forecast_leads
        )
        
        # Stage 1a: Refinement (simplified for single-agent)
        action = self.compute_action_stage1a(
            action=action,
            next_agent_action=0.0,
            next_group_action=0.0,
            emission_lead1=emission_leads[0],
            price_lead1=price_leads[0]
        )
        
        # Stage 1b: Final refinement
        # Simulate action to get net energy
        _, _, _, battery_energy = self.simulator.fast_simulate(
            action=np.clip(action, -1, 1),
            current_soc=self.soc_history[-1],
            current_capacity=self.capacity_history[-1]
        )
        
        predicted_net_energy = battery_energy + load - solar
        
        action = self.compute_action_stage1b(
            action=action,
            next_agent_net_energy=predicted_net_energy,
            next_group_net_energy=predicted_net_energy,
            emission_lead1=emission_leads[0],
            price_lead1=price_leads[0]
        )
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Update state
        next_soc, next_capacity, _, battery_energy = self.simulator.fast_simulate(
            action=action,
            current_soc=self.soc_history[-1],
            current_capacity=self.capacity_history[-1]
        )
        
        net_energy = battery_energy + load - solar
        
        self.soc_history.append(next_soc)
        self.capacity_history.append(next_capacity)
        self.net_energy_history.append(net_energy)
        self.action_history.append(action)
        self.timestep += 1
        
        return action
    
    def get_trajectory(
        self,
        dp_actions: np.ndarray,
        observations: np.ndarray,
        emission_data: np.ndarray,
        price_data: np.ndarray,
        local_forecasts: np.ndarray,
        loads: np.ndarray,
        solar: np.ndarray
    ) -> np.ndarray:
        """
        Generate full action trajectory.
        
        Args:
            dp_actions: DP baseline actions
            observations: Observation sequence
            emission_data: Emission time series
            price_data: Price time series
            local_forecasts: Energy forecasts
            loads: Load time series
            solar: Solar generation time series
        
        Returns:
            Array of refined actions
        """
        self.reset()
        
        actions = []
        for t in range(len(dp_actions)):
            action = self.compute_action(
                dp_action=dp_actions[t],
                observation=observations[t],
                emission_data=emission_data,
                price_data=price_data,
                local_forecast=local_forecasts,
                load=loads[t],
                solar=solar[t]
            )
            actions.append(action)
        
        return np.array(actions)


def initialize_random_policy() -> np.ndarray:
    """
    Initialize random policy parameters for CMA-ES.
    
    Returns:
        Random parameter vector of size 184
    """
    return np.zeros(184)  # Start from zero (neutral initialization)


def add_noise_to_params(params: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to parameters.
    
    Used for creating population in CMA-ES.
    
    Args:
        params: Base parameters
        noise_level: Std of noise
    
    Returns:
        Noisy parameters
    """
    return params + np.random.randn(len(params)) * noise_level


if __name__ == "__main__":
    print("Testing Phase 1 Policy Network")
    print("=" * 80)
    
    # Test 1: Initialize with random parameters
    print("\nTest 1: Parameter Initialization")
    print("-" * 80)
    params = initialize_random_policy()
    print(f"Parameter vector size: {len(params)}")
    
    policy = Phase1PolicyNetwork(params, building_id=1)
    print(f"✅ Policy network initialized")
    print(f"   Building ID: {policy.building_id}")
    print(f"   Number of future leads: {policy.n_leads}")
    
    # Test 2: Single action computation
    print("\nTest 2: Single Action Computation")
    print("-" * 80)
    
    dp_action = 0.3
    observation = np.array([0, 0, 12, 0.5])  # hour=12, soc=0.5
    emission_leads = np.array([0.15] * 5)
    price_leads = np.array([0.20] * 5)
    local_forecast = np.array([1.5] * 5)
    
    action0 = policy.compute_action_stage0(
        dp_action=dp_action,
        observation=observation,
        emission_leads=emission_leads,
        price_leads=price_leads,
        local_forecast=local_forecast
    )
    print(f"Stage 0 action: {action0:.4f}")
    
    action1a = policy.compute_action_stage1a(
        action=action0,
        next_agent_action=0.0,
        next_group_action=0.0,
        emission_lead1=emission_leads[0],
        price_lead1=price_leads[0]
    )
    print(f"Stage 1a action: {action1a:.4f}")
    
    action1b = policy.compute_action_stage1b(
        action=action1a,
        next_agent_net_energy=1.0,
        next_group_net_energy=1.0,
        emission_lead1=emission_leads[0],
        price_lead1=price_leads[0]
    )
    print(f"Stage 1b action (final): {action1b:.4f}")
    
    # Test 3: Full trajectory
    print("\nTest 3: Full Trajectory Generation")
    print("-" * 80)
    
    n_steps = 24 * 7  # 1 week
    
    dp_actions = np.random.randn(n_steps) * 0.3
    observations = np.zeros((n_steps, 4))
    observations[:, 2] = np.tile(np.arange(24), 7)  # Hour of day
    emission_data = np.random.rand(n_steps + 10) * 0.2 + 0.1
    price_data = np.random.rand(n_steps + 10) * 0.15 + 0.10
    local_forecasts = np.random.randn(n_steps + 10) * 1.0 + 1.5
    loads = np.random.rand(n_steps) * 2.0 + 1.0
    solar = np.random.rand(n_steps) * 3.0
    
    policy.reset()
    actions = policy.get_trajectory(
        dp_actions=dp_actions,
        observations=observations,
        emission_data=emission_data,
        price_data=price_data,
        local_forecasts=local_forecasts,
        loads=loads,
        solar=solar
    )
    
    print(f"Generated {len(actions)} actions")
    print(f"Action statistics:")
    print(f"  Mean: {np.mean(actions):.4f}")
    print(f"  Std: {np.std(actions):.4f}")
    print(f"  Min: {np.min(actions):.4f}")
    print(f"  Max: {np.max(actions):.4f}")
    print(f"  All in [-1, 1]: {np.all((actions >= -1) & (actions <= 1))}")
    
    # Test 4: State tracking
    print("\nTest 4: State Tracking")
    print("-" * 80)
    print(f"SoC history length: {len(policy.soc_history)}")
    print(f"Initial SoC: {policy.soc_history[0]:.4f}")
    print(f"Final SoC: {policy.soc_history[-1]:.4f}")
    print(f"Capacity degradation: {6.4 - policy.capacity_history[-1]:.6f} kWh")
    
    print("\n" + "=" * 80)
    print("✅ Phase 1 Policy Network Tests Completed!")
    print("\nNext: CMA-ES optimizer will optimize these 184 parameters")
    print("      to minimize the total cost over the evaluation period.")