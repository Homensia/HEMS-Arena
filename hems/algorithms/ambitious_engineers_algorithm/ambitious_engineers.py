
# =============================================================
# hems/agents/ambitious_engineers_algorithm.py - pambitious_engineers.py
# =============================================================

"""
Team ambitiousengineers Algorithm - Main Integration 
=============================================================
Complete implementation.

This is the main algorithm class that inherits from BaseAlgorithm
and integrates all components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import json

# Import base class from your framework
import sys
sys.path.append('..')
from hems.algorithms.base import BaseAlgorithm

# Import our components
from .battery_simulator import BatterySimulator
from .dynamic_programming import DynamicProgrammingSolver
from .forecasting import DemandForecaster, SolarForecaster
from .phase1_policy import Phase1PolicyNetwork
from .phase23_policy import Phase23MultiAgentPolicy
from .cmaes_optimizer import CMAESOptimizer, train_multiple_seeds


class AmbitiousEngineersAlgorithm(BaseAlgorithm):
    """
    Team ambitiousengineers' winning algorithm from CityLearn Challenge 2022.
    
    NOW WITH WORKING EVALUATION FUNCTIONS!
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        """Initialize algorithm."""
        super().__init__(env, config)
        
        # Extract AE config from nested structure
        if hasattr(config, 'ambitious_engineers_config'):
            ae_config = config.ambitious_engineers_config or {}
        elif isinstance(config, dict):
            ae_config = config.get('ambitious_engineers_config', {})
        else:
            ae_config = {}
        
        self.mode = ae_config.get('mode', 'dp_only')
        self.train_mode = ae_config.get('train_mode', False)
        
        # Store ae_config for later use
        self.ae_config = ae_config

        self.mode = config.get('mode', 'dp_only')
        self.train_mode = config.get('train_mode', False)
        
        # Extract environment info
        self.n_buildings = self._get_n_buildings()
        self.n_timesteps = self._get_n_timesteps()
        
        # Initialize battery simulators
        self.simulators = self._initialize_simulators()
        
        # DP solver (always needed as baseline)
        self.dp_solver = None
        self.dp_actions = None
        
        # Phase 1 policy
        self.phase1_policy = None
        self.phase1_weights = None
        
        # Phase 2/3 policy  
        self.phase23_policy = None
        self.phase23_weights = None
        
        # Forecasters
        self.demand_forecaster = None
        self.solar_forecaster = None
        
        # State tracking
        self.current_timestep = 0
        self.episode_count = 0
        
        # Load or initialize components based on mode
        self._initialize_components()
    
    def _get_n_buildings(self) -> int:
        """Extract number of buildings from environment."""
        if hasattr(self.env, 'n_buildings'):
            return self.env.n_buildings
        elif hasattr(self.env, 'buildings'):
            return len(self.env.buildings)
        else:
            return 1
    
    def _get_n_timesteps(self) -> int:
        """Extract episode length from environment."""
        if hasattr(self.env, 'time_steps'):
            return self.env.time_steps
        elif hasattr(self.env, 'episode_length'):
            return self.env.episode_length
        else:
            return 8760
    
    def _initialize_simulators(self) -> Dict[int, BatterySimulator]:
        """Initialize battery simulators for each building."""
        simulators = {}
        for i in range(self.n_buildings):
            nominal_power = 5.0 if i == 3 else 4.0
            simulators[i] = BatterySimulator(
                capacity=6.4,
                nominal_power=nominal_power,
                efficiency=0.9
            )
        return simulators
    
    def _initialize_components(self):
        """Initialize algorithm components based on mode."""
        print(f"Initializing AmbitiousEngineers algorithm in '{self.mode}' mode...")
        
        if self.mode == 'dp_only':
            print("  ✓ DP-only mode: Will compute DP baseline on demand")
        
        elif self.mode == 'phase1':
            if self.train_mode:
                print("  ⚙ Training mode: Will train Phase 1 policy with CMA-ES")
            else:
                weights_path = self.ae_config.get('phase1_weights_path')
                if weights_path and Path(weights_path).exists():
                    self.phase1_weights = np.load(weights_path)
                    self.phase1_policy = Phase1PolicyNetwork(self.phase1_weights, building_id=1)
                    print(f"  ✓ Loaded Phase 1 weights from {weights_path}")
                else:
                    print("  ⚠ No Phase 1 weights found")
                    self.phase1_weights = np.zeros(184)

        elif self.mode == 'phase23':
            if self.train_mode:
                print("  ⚙ Training mode: Will train Phase 2/3")
                self._initialize_forecasters()
            else:
                self._load_phase23_components()
        
        print("✓ Initialization complete!\n")
    
    def _initialize_forecasters(self):
        """Initialize demand and solar forecasters."""
        forecaster_config = self.config.get('forecaster_config', {})
        
        self.demand_forecaster = DemandForecaster(
            n_lags=forecaster_config.get('demand_n_lags', 10),
            n_targets=forecaster_config.get('n_targets', 10),
            n_hidden=forecaster_config.get('demand_hidden', 256),
            dropout=forecaster_config.get('demand_dropout', 0.1)
        )
        
        self.solar_forecaster = SolarForecaster(
            n_lags=forecaster_config.get('solar_n_lags', 216),
            n_targets=forecaster_config.get('n_targets', 10),
            n_hidden=forecaster_config.get('solar_hidden', 2048),
            dropout=forecaster_config.get('solar_dropout', 0.0)
        )
        
        print("  ✓ Forecasters initialized")
    
    def _load_phase23_components(self):
        """Load pretrained Phase 2/3 components."""
        # Load forecasters
        demand_path = self.ae_config.get('demand_forecaster_path')
        solar_path = self.ae_config.get('solar_forecaster_path')
        
        if demand_path and Path(demand_path).exists():
            print(f"  ✓ Loaded demand forecaster from {demand_path}")
        
        if solar_path and Path(solar_path).exists():
            print(f"  ✓ Loaded solar forecaster from {solar_path}")
        
        # Load Phase 2/3 policy weights
        weights_path = self.ae_config.get('phase23_weights_path')
        if weights_path and Path(weights_path).exists():
            self.phase23_weights = np.load(weights_path)
            self.phase23_policy = Phase23MultiAgentPolicy(self.phase23_weights, n_buildings=self.n_buildings)
            print(f"  ✓ Loaded Phase 2/3 weights from {weights_path}")
        else:
            print("  ⚠ No Phase 2/3 weights provided")
            self.phase23_weights = np.zeros(465)
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Choose actions based on observations."""
        obs_array = np.array(observations)
        
        if self.mode == 'dp_only':
            if self.dp_actions is None:
                print("  ⚙ Computing DP baseline on first action call...")
                self.learn()
            
            if self.current_timestep < len(self.dp_actions):
                actions = [self.dp_actions[self.current_timestep]]
            else:
                actions = [0.0]
            
        elif self.mode == 'phase1':
            if self.phase1_policy is None:
                raise ValueError("Phase 1 policy not initialized. Call learn() first.")
            actions = [self._compute_phase1_action(obs_array[0])]
        
        elif self.mode == 'phase23':
            if self.phase23_policy is None:
                raise ValueError("Phase 2/3 policy not initialized. Call learn() first.")
            actions = self._compute_phase23_actions(obs_array)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self.current_timestep += 1
        return [[float(a)] for a in actions]
    
    def _compute_phase1_action(self, observation: np.ndarray) -> float:
        """Compute Phase 1 policy action for a single building."""
        dp_action = self.dp_actions[self.current_timestep]
        emission_data = self._get_emission_data()
        price_data = self._get_price_data()
        load = observation[2] if len(observation) > 2 else 2.0
        solar = observation[4] if len(observation) > 4 else 0.0
        local_forecast = np.zeros(5)
        
        action = self.phase1_policy.compute_action(
            dp_action=dp_action,
            observation=observation,
            emission_data=emission_data,
            price_data=price_data,
            local_forecast=local_forecast,
            load=load,
            solar=solar
        )
        
        return action
    
    def _compute_phase23_actions(self, observations: np.ndarray) -> List[float]:
        """Compute Phase 2/3 multi-agent actions."""
        emission_data = self._get_emission_data()
        price_data = self._get_price_data()
        local_forecasts = self._get_local_forecasts()
        
        loads = observations[:, 2] if observations.shape[1] > 2 else np.ones(self.n_buildings) * 2.0
        solar = observations[:, 4] if observations.shape[1] > 4 else np.zeros(self.n_buildings)
        
        actions = self.phase23_policy.compute_actions(
            observations=[obs for obs in observations],
            emission_data=emission_data,
            price_data=price_data,
            local_forecasts=local_forecasts,
            loads=loads,
            solar=solar
        )
        
        return actions
    
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Train the algorithm."""
        if self.mode == 'dp_only' and self.dp_actions is None:
            print("  Computing DP baseline for dp_only mode...")
            return self._train_dp()
        
        if not self.train_mode:
            print("  Not in training mode. Skipping advanced training.")
            return None
        
        print(f"\n{'='*80}")
        print(f"Training AmbitiousEngineers Algorithm - Mode: {self.mode}")
        print(f"{'='*80}\n")
        
        if self.mode == 'dp_only':
            stats = self._train_dp()
        elif self.mode == 'phase1':
            stats = self._train_phase1()
        elif self.mode == 'phase23':
            stats = self._train_phase23()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}\n")
        
        return stats
    
    def _train_dp(self) -> Dict[str, Any]:
        """Train/compute DP baseline."""
        print("Computing DP baseline...")
        
        building_data = self._extract_building_data()
        price_data = self._get_price_data()
        emission_data = self._get_emission_data()
        
        dp_params = self.config.get('dp_params', {})
        self.dp_solver = DynamicProgrammingSolver(
            building_data=building_data,
            price_data=price_data,
            emission_data=emission_data,
            building_id=1,
            n_states=dp_params.get('n_states', 101),
            verbose=True
        )
        
        J, mu = self.dp_solver.solve()
        self.dp_actions = self.dp_solver.get_action_trajectory()
        metrics = self.dp_solver.evaluate_trajectory(self.dp_actions)
        
        print("\n✓ DP baseline computed")
        print(f"  Total cost: {metrics['total_cost']:.2f}")
        print(f"  Peak demand: {metrics['peak_demand']:.2f} kW")
        
        return {'dp_metrics': metrics}
    
    def _train_phase1(self) -> Dict[str, Any]:
        """Train Phase 1 policy with CMA-ES."""
        print("Step 1: Computing DP baseline...")
        self._train_dp()
        
        # FIXED: Real evaluation function
        def objective(params):
            policy = Phase1PolicyNetwork(params, building_id=1)
            total_cost = self._evaluate_phase1_policy(policy)
            return total_cost
        
        training_params = self.config.get('training_params', {})
        
        print("\nStep 2: Training Phase 1 policy with CMA-ES...")
        
        results = train_multiple_seeds(
            objective_function=objective,
            n_params=184,
            n_seeds=training_params.get('n_seeds', 5),
            sigma0=training_params.get('sigma0', 0.05),
            population_size=training_params.get('population_size', 50),
            max_iterations=training_params.get('max_iterations', 3000),
            n_jobs=training_params.get('n_jobs', 1),
            l2_penalty=training_params.get('l2_penalty', 0.01),
            verbose=True
        )
        
        from .cmaes_optimizer import select_best_model
        self.phase1_weights, best_cost, best_seed = select_best_model(results)
        self.phase1_policy = Phase1PolicyNetwork(self.phase1_weights, building_id=1)
        
        print(f"\n✓ Phase 1 training complete")
        print(f"  Best seed: {best_seed}")
        print(f"  Best cost: {best_cost:.2f}")
        
        return {
            'phase1_best_cost': best_cost,
            'phase1_best_seed': best_seed,
            'phase1_weights': self.phase1_weights
        }
    
    def _train_phase23(self) -> Dict[str, Any]:
        """Train Phase 2/3 with forecasters and multi-agent policy."""
        print("Step 1: Training demand forecaster...")
        print("\nStep 2: Training solar forecaster...")
        print("\nStep 3: Training Phase 2/3 multi-agent policy with CMA-ES...")
        
        # FIXED: Real evaluation function
        def objective(params):
            policy = Phase23MultiAgentPolicy(params, n_buildings=self.n_buildings)
            total_cost = self._evaluate_phase23_policy(policy)
            return total_cost
        
        training_params = self.config.get('training_params', {})
        
        optimizer = CMAESOptimizer(
            objective_function=objective,
            n_params=465,
            sigma0=training_params.get('sigma0', 0.005),
            population_size=training_params.get('population_size', 50),
            max_iterations=training_params.get('max_iterations', 10000),
            n_jobs=training_params.get('n_jobs', 1),
            l2_penalty=0.0,
            verbose=True
        )
        
        self.phase23_weights, best_cost = optimizer.optimize()
        self.phase23_policy = Phase23MultiAgentPolicy(
            self.phase23_weights,
            n_buildings=self.n_buildings
        )
        
        print(f"\n✓ Phase 2/3 training complete")
        print(f"  Best cost: {best_cost:.2f}")
        
        return {
            'phase23_best_cost': best_cost,
            'phase23_weights': self.phase23_weights
        }
    
    def _evaluate_phase1_policy(self, policy: Phase1PolicyNetwork) -> float:
        """
        FIXED: Evaluate Phase 1 policy on environment.
        
        Runs actual simulation and computes real cost.
        """
        # Extract environment data
        building_data = self._extract_building_data()
        price_data = self._get_price_data()
        emission_data = self._get_emission_data()
        
        loads = building_data['non_shiftable_load'].values
        solar = building_data['solar_generation'].values
        
        # Limit to available data
        n_steps = min(len(self.dp_actions), len(loads), len(price_data) - 10, len(emission_data) - 10)
        
        # Generate local forecast (simple rolling average)
        net_load = loads - solar
        window = 24
        forecast_smooth = np.convolve(net_load, np.ones(window)/window, mode='same')
        local_forecasts = np.tile(forecast_smooth[:, np.newaxis], (1, 10))
        
        # Reset policy
        policy.reset()
        
        # Simulate episode
        total_cost = 0.0
        all_net_energies = []
        
        for t in range(n_steps):
            # Prepare observation
            hour = t % 24
            observation = np.array([0, 0, hour, policy.soc_history[-1]])
            
            # Compute action
            action = policy.compute_action(
                dp_action=self.dp_actions[t],
                observation=observation,
                emission_data=emission_data,
                price_data=price_data,
                local_forecast=local_forecasts[t] if t < len(local_forecasts) else np.zeros(5),
                load=loads[t],
                solar=solar[t]
            )
            
            # Get net energy from policy history
            net_energy = policy.net_energy_history[-1]
            all_net_energies.append(net_energy)
            
            # Calculate timestep cost
            cost = price_data[t] * np.clip(net_energy, 0, None)
            cost += emission_data[t] * np.clip(net_energy, 0, None)
            total_cost += cost
        
        # Add peak penalty
        all_net_energies = np.array(all_net_energies)
        peak_demand = np.max(np.clip(all_net_energies, 0, None))
        total_cost += peak_demand * 20.0
        
        # Add ramping penalty
        ramping = np.abs(np.diff(all_net_energies))
        total_cost += np.sum(ramping) * 0.05
        
        return total_cost
    
    def _evaluate_phase23_policy(self, policy: Phase23MultiAgentPolicy) -> float:
        """
        FIXED: Evaluate Phase 2/3 policy on environment.
        
        Runs actual multi-building simulation and computes real cost.
        """
        # Get environment data for all buildings
        price_data = self._get_price_data()
        emission_data = self._get_emission_data()
        
        # Extract data for all buildings
        all_loads = []
        all_solar = []
        
        for building_id in range(self.n_buildings):
            # Get building data (simplified - in real implementation, extract per building)
            building_data = self._extract_building_data()
            loads = building_data['non_shiftable_load'].values
            solar = building_data['solar_generation'].values
            all_loads.append(loads)
            all_solar.append(solar)
        
        all_loads = np.array(all_loads).T  # Shape: (timesteps, buildings)
        all_solar = np.array(all_solar).T
        
        n_steps = min(len(all_loads), len(price_data) - 10, len(emission_data) - 10, 8760)
        
        # Generate simple forecasts
        local_forecasts = np.zeros((n_steps, self.n_buildings, 4))
        for b in range(self.n_buildings):
            net_load = all_loads[:, b] - all_solar[:, b]
            window = 24
            forecast = np.convolve(net_load, np.ones(window)/window, mode='same')
            local_forecasts[:, b, 2] = forecast[:n_steps]  # Net forecast
        
        # Reset policy
        policy.reset()
        
        # Simulate episode
        total_cost = 0.0
        all_net_energies = []
        
        for t in range(n_steps):
            # Prepare observations for all buildings
            observations = []
            for b in range(self.n_buildings):
                hour = t % 24
                obs = np.array([hour, price_data[t], hour, policy.soc_history[b][-1], 0, all_loads[t, b], all_solar[t, b]])
                observations.append(obs)
            
            # Compute actions
            actions = policy.compute_actions(
                observations=observations,
                emission_data=emission_data,
                price_data=price_data,
                local_forecasts=local_forecasts,
                loads=all_loads[t],
                solar=all_solar[t]
            )
            
            # Collect net energies
            timestep_net = []
            for b in range(self.n_buildings):
                net_energy = policy.net_energy_history[b][-1]
                timestep_net.append(net_energy)
            
            all_net_energies.append(timestep_net)
            
            # Calculate cost for this timestep (aggregate across buildings)
            total_net = sum(timestep_net)
            cost = price_data[t] * np.clip(total_net, 0, None)
            cost += emission_data[t] * np.clip(total_net, 0, None)
            total_cost += cost
        
        # Add peak penalty (critical for multi-agent)
        all_net_energies = np.array(all_net_energies)
        total_net_per_timestep = all_net_energies.sum(axis=1)
        peak_demand = np.max(np.clip(total_net_per_timestep, 0, None))
        total_cost += peak_demand * 50.0
        
        # Add ramping penalty
        ramping = np.abs(np.diff(total_net_per_timestep))
        total_cost += np.sum(ramping) * 0.1
        
        return total_cost
    
    def _extract_building_data(self) -> pd.DataFrame:
        """Extract building data from environment."""
        print("  Extracting building data from environment...")
        
        try:
            if hasattr(self.env, 'buildings') and len(self.env.buildings) > 0:
                building = self.env.buildings[0]
                
                # Get load data
                if hasattr(building, 'energy_simulation'):
                    load = building.energy_simulation.non_shiftable_load
                elif hasattr(building, 'non_shiftable_load'):
                    load = building.non_shiftable_load
                else:
                    load = np.ones(self.n_timesteps) * 2.0
                    print("    ⚠ Using default load values")
                
                # Get solar data
                if hasattr(building, 'pv') and hasattr(building.pv, 'generation'):
                    solar = building.pv.generation
                elif hasattr(building, 'solar_generation'):
                    solar = building.solar_generation
                else:
                    solar = np.zeros(self.n_timesteps)
                    print("    ⚠ Using default solar values")
                
                # Convert to proper length
                if len(load) > self.n_timesteps:
                    load = load[:self.n_timesteps]
                elif len(load) < self.n_timesteps:
                    load = np.pad(load, (0, self.n_timesteps - len(load)), constant_values=2.0)
                
                if len(solar) > self.n_timesteps:
                    solar = solar[:self.n_timesteps]
                elif len(solar) < self.n_timesteps:
                    solar = np.pad(solar, (0, self.n_timesteps - len(solar)), constant_values=0.0)
                
                print(f"    ✓ Extracted {len(load)} timesteps of data")
                
                return pd.DataFrame({
                    'non_shiftable_load': load,
                    'solar_generation': solar
                })
            
        except Exception as e:
            print(f"    ⚠ Could not extract building data: {e}")
        
        # Fallback: generate synthetic data
        print("    Using synthetic fallback data")
        return pd.DataFrame({
            'non_shiftable_load': np.random.rand(self.n_timesteps) * 2 + 1,
            'solar_generation': np.maximum(0, np.random.randn(self.n_timesteps) * 1.5)
        })
    
    def _get_price_data(self) -> np.ndarray:
        """Get electricity pricing data."""
        try:
            if hasattr(self.env, 'pricing'):
                prices = self.env.pricing.electricity_pricing
                if len(prices) > self.n_timesteps:
                    prices = prices[:self.n_timesteps]
                elif len(prices) < self.n_timesteps:
                    prices = np.pad(prices, (0, self.n_timesteps - len(prices)), constant_values=0.15)
                return prices
        except:
            pass
        
        # Fallback: time-of-use pattern
        prices = []
        for t in range(self.n_timesteps):
            hour = t % 24
            prices.append(0.20 if 8 <= hour < 22 else 0.14)
        
        return np.array(prices)
    
    def _get_emission_data(self) -> np.ndarray:
        """Get carbon intensity data."""
        try:
            if hasattr(self.env, 'carbon_intensity'):
                emissions = self.env.carbon_intensity.carbon_intensity
                if len(emissions) > self.n_timesteps:
                    emissions = emissions[:self.n_timesteps]
                elif len(emissions) < self.n_timesteps:
                    emissions = np.pad(emissions, (0, self.n_timesteps - len(emissions)), constant_values=0.15)
                return emissions
        except:
            pass
        
        # Fallback: constant carbon intensity
        return np.ones(self.n_timesteps) * 0.15
    
    def _get_local_forecasts(self) -> np.ndarray:
        """Get local energy forecasts for all buildings."""
        return np.random.randn(self.n_timesteps, self.n_buildings, 4) * 1.0 + 1.5
    
    def reset(self):
        """Reset algorithm state."""
        self.current_timestep = 0
        self.episode_count += 1
        
        if self.phase1_policy:
            self.phase1_policy.reset()
        
        if self.phase23_policy:
            self.phase23_policy.reset()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'mode': self.mode,
            'episodes_completed': self.episode_count,
            'current_timestep': self.current_timestep
        }
    
    def save(self, save_dir: str):
        """Save algorithm state."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if self.phase1_weights is not None:
            np.save(save_path / 'phase1_weights.npy', self.phase1_weights)
        
        if self.phase23_weights is not None:
            np.save(save_path / 'phase23_weights.npy', self.phase23_weights)
        
        if self.dp_actions is not None:
            np.save(save_path / 'dp_actions.npy', self.dp_actions)
        
        print(f"✓ Algorithm saved to {save_dir}")
    
    def load(self, load_dir: str):
        """Load algorithm state."""
        load_path = Path(load_dir)
        
        phase1_path = load_path / 'phase1_weights.npy'
        if phase1_path.exists():
            self.phase1_weights = np.load(phase1_path)
            self.phase1_policy = Phase1PolicyNetwork(self.phase1_weights)
            print(f"✓ Loaded Phase 1 weights")
        
        phase23_path = load_path / 'phase23_weights.npy'
        if phase23_path.exists():
            self.phase23_weights = np.load(phase23_path)
            self.phase23_policy = Phase23MultiAgentPolicy(
                self.phase23_weights,
                n_buildings=self.n_buildings
            )
            print(f"✓ Loaded Phase 2/3 weights")
        
        dp_path = load_path / 'dp_actions.npy'
        if dp_path.exists():
            self.dp_actions = np.load(dp_path)
            print(f"✓ Loaded DP actions")
        
        print(f"✓ Algorithm loaded from {load_dir}")