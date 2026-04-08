# ======================================================================
# hems/agents/ambitious_engineers_algorithm.py - policy_optimization.py
# ======================================================================


"""
policy_optimization.py - Phase 1 Single-Agent Policy Training
================================================================
Expected runtime: ~2.5 days with 50 parallel workers

This script trains the Phase 1 single-agent policy refinement:
1. Loads DP baseline actions from Step 4
2. Trains Phase 1 policy network (184 parameters) with CMA-ES
3. Uses 5 random seeds for robustness
4. 3000 iterations per seed, population size 50
5. Saves 30 checkpoint models (5 seeds × 6 checkpoints)
6. Refines DP actions using neural network with future forecasts

The Phase 1 policy takes DP actions as input and refines them using:
- Current battery state (SoC)
- Future pricing signals (5 steps ahead)
- Future emission signals (5 steps ahead)
- Future energy forecasts (local)

Output:
- data/models/phase1_policy_seed{0-4}_iter{2500,2600,2700,2800,2900,3000}.npy
- data/models/phase1_best_policy.npy (best overall)
- data/models/phase1_training_stats.json (training statistics)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hems.algorithms.ambitious_engineers_algorithm.phase1_policy import Phase1PolicyNetwork
from hems.algorithms.ambitious_engineers_algorithm.battery_simulator import BatterySimulator
from hems.algorithms.ambitious_engineers_algorithm.data_loader import CityLearnDataLoader
from hems.algorithms.ambitious_engineers_algorithm.cmaes_optimizer import train_multiple_seeds, select_best_model


# Global evaluator for pickling support in multiprocessing
_global_evaluator = None


def _evaluate_policy_wrapper(params):
    """
    Wrapper function for parallel evaluation.
    
    Must be at module level (not nested) for multiprocessing pickling.
    """
    global _global_evaluator
    if _global_evaluator is None:
        raise RuntimeError("Global evaluator not initialized")
    return _global_evaluator.evaluate_policy(params)


class Phase1Evaluator:
    """
    Evaluator for Phase 1 single-agent policy.
    
    This evaluates the Phase 1 policy by:
    1. Loading DP baseline actions
    2. Using policy to refine actions
    3. Simulating battery and computing total cost
    """
    
    def __init__(
        self,
        data_path: str = "datasets/citylearn_datasets/citylearn_challenge_2022_phase_1",
        dp_actions_path: str = "data/external/single_agent_dp.npy",
        building_id: int = 1
    ):
        print("Initializing Phase 1 evaluator...")
        
        self.building_id = building_id
        
        # Load data
        loader = CityLearnDataLoader(data_path)
        self.observations = loader.create_observations_dataframe()
        self.carbon = loader.load_carbon_intensity()
        self.pricing = loader.load_pricing()
        
        # Filter for specific building
        building_obs = self.observations[
            self.observations['building_num'] == building_id
        ].copy()
        
        self.loads = building_obs['non_shiftable_load'].values
        self.solar = building_obs['solar_generation'].values
        self.hours = building_obs['hour'].values
        
        n_steps = len(self.loads)
        
        # Pad carbon and pricing to match
        if len(self.carbon) < n_steps + 100:
            self.carbon = np.pad(self.carbon, (0, n_steps + 100 - len(self.carbon)))
        if len(self.pricing) < n_steps + 100:
            self.pricing = np.pad(self.pricing, (0, n_steps + 100 - len(self.pricing)))
        
        # Load DP baseline actions
        print(f"  Loading DP baseline actions from {dp_actions_path}")
        self.dp_actions = np.load(dp_actions_path)
        
        # Truncate to match available data
        min_len = min(len(self.dp_actions), len(self.loads), len(self.carbon) - 10, len(self.pricing) - 10)
        self.dp_actions = self.dp_actions[:min_len]
        self.loads = self.loads[:min_len]
        self.solar = self.solar[:min_len]
        self.hours = self.hours[:min_len]
        
        # Initialize battery simulator
        nominal_power = 5.0 if building_id == 4 else 4.0
        self.simulator = BatterySimulator(
            capacity=6.4,
            nominal_power=nominal_power
        )
        
        # Create simple local forecast (rolling average)
        print("  Generating local energy forecasts...")
        net_load = self.loads - self.solar
        window = 24  # 1 day
        forecast_smooth = np.convolve(net_load, np.ones(window)/window, mode='same')
        self.local_forecasts = np.tile(forecast_smooth[:, np.newaxis], (1, 10))  # Replicate for 10 steps
        
        print(f"  ✓ Evaluator initialized for Building {building_id}")
        print(f"    Timesteps: {len(self.dp_actions)}")
        print(f"    DP actions loaded: {len(self.dp_actions)}")
    
    def evaluate_policy(self, params: np.ndarray) -> float:
        """
        Evaluate Phase 1 policy parameters.
        
        Args:
            params: Policy parameters (184 dimensions)
        
        Returns:
            Total cost for the episode
        """
        # Create policy with these parameters
        policy = Phase1PolicyNetwork(params, building_id=self.building_id)
        
        # Simulate episode
        total_cost = 0.0
        all_net_energies = []
        
        policy.reset()
        
        for t in range(len(self.dp_actions)):
            # Prepare observation
            observation = np.array([
                0,  # placeholder
                0,  # placeholder
                self.hours[t],  # hour
                policy.soc_history[-1]  # current SoC
            ])
            
            # Compute action
            action = policy.compute_action(
                dp_action=self.dp_actions[t],
                observation=observation,
                emission_data=self.carbon,
                price_data=self.pricing,
                local_forecast=self.local_forecasts[t] if t < len(self.local_forecasts) else np.zeros(5),
                load=self.loads[t],
                solar=self.solar[t]
            )
            
            # Get net energy from policy's history (already computed)
            net_energy = policy.net_energy_history[-1]
            all_net_energies.append(net_energy)
            
            # Calculate cost for this timestep
            # Power cost (only for imports)
            cost = self.pricing[t] * np.clip(net_energy, 0, None)
            
            # Emission cost (only for imports)
            cost += self.carbon[t] * np.clip(net_energy, 0, None)
            
            total_cost += cost
        
        # Add peak demand penalty (critical for grid cost)
        all_net_energies = np.array(all_net_energies)
        peak_demand = np.max(np.clip(all_net_energies, 0, None))
        peak_penalty = peak_demand * 20.0  # Peak penalty weight
        total_cost += peak_penalty
        
        # Add ramping penalty (smooth transitions)
        ramping = np.abs(np.diff(all_net_energies))
        ramping_penalty = np.sum(ramping) * 0.05
        total_cost += ramping_penalty
        
        # Add L2 penalty on actions (from policy history)
        if len(policy.action_history) > 0:
            action_penalty = 0.01 * np.sum(np.array(policy.action_history) ** 2)
            total_cost += action_penalty
        
        return total_cost
    
    def evaluate_policy_parallel(self, params_list: List[np.ndarray]) -> List[float]:
        """
        Evaluate multiple policies (used by CMA-ES).
        
        Args:
            params_list: List of parameter vectors
        
        Returns:
            List of costs
        """
        return [self.evaluate_policy(params) for params in params_list]


# Global evaluator for pickling support
_global_evaluator = None

def _evaluate_policy_wrapper(params):
    """Wrapper function for parallel evaluation (must be at module level for pickling)."""
    global _global_evaluator
    if _global_evaluator is None:
        raise RuntimeError("Global evaluator not initialized")
    return _global_evaluator.evaluate_policy(params)


def train_phase1_policy(
    n_seeds: int = 5,
    n_iterations: int = 3000,
    population_size: int = 50,
    sigma0: float = 0.05,
    l2_penalty: float = 0.01,
    n_jobs: int = 50,
    building_id: int = 1
) -> Dict:
    """
    Train Phase 1 policy with multiple random seeds.
    
    Args:
        n_seeds: Number of random seeds (default: 5)
        n_iterations: CMA-ES iterations per seed (default: 3000)
        population_size: CMA-ES population size (default: 50)
        sigma0: Initial step size (default: 0.05)
        l2_penalty: L2 regularization strength (default: 0.01)
        n_jobs: Number of parallel workers (default: 50)
        building_id: Building to train on (default: 1)
    
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*80}")
    print(f"  Training Phase 1 Single-Agent Policy")
    print(f"{'='*80}\n")
    
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Initialize evaluator and set as global for pickling
    global _global_evaluator
    _global_evaluator = Phase1Evaluator(building_id=building_id)
    
    # Use module-level wrapper function (pickable)
    objective_function = _evaluate_policy_wrapper
    
    print(f"Training Configuration:")
    print(f"  Building ID: {building_id}")
    print(f"  Parameters: 184")
    print(f"  Random seeds: {n_seeds}")
    print(f"  Iterations per seed: {n_iterations}")
    print(f"  Population size: {population_size}")
    print(f"  Initial sigma: {sigma0}")
    print(f"  L2 penalty: {l2_penalty}")
    print(f"  Parallel workers: {n_jobs}")
    print(f"  Checkpoint iterations: 2500, 2600, 2700, 2800, 2900, 3000\n")
    
    # Train with multiple seeds
    results = train_multiple_seeds(
        objective_function=objective_function,
        n_params=184,
        n_seeds=n_seeds,
        sigma0=sigma0,
        population_size=population_size,
        max_iterations=n_iterations,
        n_jobs=n_jobs,
        l2_penalty=l2_penalty,
        checkpoint_dir=str(output_dir),
        save_all_models=True,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Select best model
    print(f"\nSelecting best model from {n_seeds} seeds...")
    best_params, best_cost, best_seed = select_best_model(
        results,
        save_path=str(output_dir / "phase1_best_policy.npy")
    )
    
    print(f"\nBest Model:")
    print(f"  Seed: {best_seed}")
    print(f"  Cost: {best_cost:.2f}")
    print(f"  Saved to: phase1_best_policy.npy")
    
    # Compile results
    training_stats = {
        'building_id': building_id,
        'n_seeds': n_seeds,
        'n_iterations': n_iterations,
        'population_size': population_size,
        'sigma0': sigma0,
        'l2_penalty': l2_penalty,
        'training_time_seconds': training_time,
        'best_seed': int(best_seed),
        'best_cost': float(best_cost),
        'seed_results': [
            {
                'seed': r['seed'],
                'best_cost': float(r['best_cost']),
                'final_params': r['best_params'].tolist()
            }
            for r in results
        ]
    }
    
    # Clean up global evaluator
    _global_evaluator = None
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"  Phase 1 Training Complete")
    print(f"{'='*80}")
    print(f"Training time: {training_time/3600:.1f} hours ({training_time/86400:.1f} days)")
    print(f"Best cost: {best_cost:.2f} (Seed {best_seed})")
    print(f"Total models saved: {n_seeds * 6} (checkpoints + final)")
    print(f"{'='*80}\n")
    
    return training_stats


def main():
    print("="*80)
    print("  STEP 5: Phase 1 Single-Agent Policy Training")
    print("="*80)
    print("\nThis script trains the Phase 1 policy refinement network")
    print("Expected runtime: ~2.5 days with 50 parallel workers\n")
    print("Configuration:")
    print("  - 5 random seeds")
    print("  - 3000 iterations per seed")
    print("  - Population size: 50")
    print("  - Initial sigma: 0.05")
    print("  - L2 penalty: 0.01")
    print("  - Saves 6 checkpoints per seed (30 total models)\n")
    
    response = input("This will take 2.5 days. Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return 1
    
    start_time = time.time()
    
    try:
        # Train Phase 1 policy
        results = train_phase1_policy(
            n_seeds=5,
            n_iterations=3000,
            population_size=50,
            sigma0=0.05,
            l2_penalty=0.01,
            n_jobs=50,
            building_id=1
        )
        
        # Save training statistics
        stats_path = Path("data/models") / "phase1_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved training statistics: {stats_path}")
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("  PHASE 1 TRAINING COMPLETE")
        print("="*80)
        print(f"\nTotal training time: {total_time/3600:.1f} hours ({total_time/86400:.1f} days)")
        print(f"Best cost: {results['best_cost']:.2f}")
        print(f"Best seed: {results['best_seed']}")
        print(f"\nGenerated files:")
        print(f"  phase1_best_policy.npy - Best model overall")
        print(f"  phase1_policy_seed{{0-4}}_iter{{...}}.npy - Checkpoint models (30 total)")
        print(f"  phase1_training_stats.json - Training statistics")
        
        print("\n" + "="*80)
        print("  NEXT STEP: Run 5_create_training_multi_agents.py")
        print("  (Phase 2/3 Multi-Agent Policy Training - 1.5 days)")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())