# ===============================================================================
# hems/agents/ambitious_engineers_algorithm.py - create_training_single_agent.py
# ===============================================================================


"""
4_create_training_single_agents.py - Compute DP Baseline Actions
==================================================================
Expected runtime: ~15 minutes

This script computes the Dynamic Programming baseline for Building 1:
1. Loads building data, pricing, and carbon intensity
2. Solves the single-agent DP problem using proxy cost function
3. Extracts optimal action trajectory
4. Evaluates performance metrics
5. Saves DP actions for use in Phase 1 policy training

The DP solver uses a proxy cost function since grid costs are non-causal.

Output:
- data/external/single_agent_dp.npy (DP actions for Building 1)
- data/external/dp_baseline_metrics.json (performance metrics)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hems.algorithms.ambitious_engineers_algorithm.data_loader import CityLearnDataLoader
from hems.algorithms.ambitious_engineers_algorithm.dynamic_programming import DynamicProgrammingSolver


def main():
    print("="*80)
    print("  STEP 4: Compute DP Baseline for Single Agent")
    print("="*80)
    print("\nThis script computes the Dynamic Programming baseline for Building 1")
    print("Expected runtime: ~15 minutes\n")
    
    start_time = time.time()
    
    # Create output directory
    output_dir = Path("data/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        print("Loading CityLearn Challenge data...")
        loader = CityLearnDataLoader(
            data_path="datasets/citylearn_datasets/citylearn_challenge_2022_phase_1"
        )
        
        # Prepare data for DP (Building 1)
        building_id = 1
        print(f"\nPreparing data for Building {building_id}...")
        building_data, price_data, emission_data = loader.prepare_for_dp(building_id)
        
        print(f"  Building data: {len(building_data)} timesteps")
        print(f"  Price data: {len(price_data)} values")
        print(f"  Emission data: {len(emission_data)} values")
        
        # Display data statistics
        print(f"\nBuilding {building_id} Statistics:")
        print(f"  Average Load: {building_data['non_shiftable_load'].mean():.2f} kW")
        print(f"  Peak Load: {building_data['non_shiftable_load'].max():.2f} kW")
        print(f"  Average Solar: {building_data['solar_generation'].mean():.2f} kW")
        print(f"  Peak Solar: {building_data['solar_generation'].max():.2f} kW")
        print(f"  Average Price: {price_data.mean():.4f} $/kWh")
        print(f"  Average Emission: {emission_data.mean():.4f} kgCO2/kWh")
        
        # Initialize DP solver with tuned hyperparameters
        print("\nInitializing DP solver...")
        print("  Using tuned proxy cost weights from Team ambitiousengineers")
        
        proxy_weights = {
            'l1_penalty': 0.1300366652733203,
            'l2_penalty': 0.009508553021774526,
            'proxy_weight1': 0.011557095624191585,
            'proxy_weight2': 0.08482562812998984,
            'proxy_weight3': 2.61560899997805
        }
        
        # Building 1 has 4kW battery (Building 4 has 5kW)
        battery_power = 4.0 if building_id != 4 else 5.0
        
        solver = DynamicProgrammingSolver(
            building_data=building_data,
            price_data=price_data,
            emission_data=emission_data,
            building_id=building_id,
            proxy_weights=proxy_weights,
            n_states=101,  # 101 state discretization
            battery_capacity=6.4,
            battery_power=battery_power,
            verbose=True
        )
        
        print(f"\nDP Solver Configuration:")
        print(f"  State discretization: {solver.n_states} states")
        print(f"  Time horizon: {solver.n_horizon} hours")
        print(f"  Battery capacity: {solver.battery_capacity} kWh")
        print(f"  Battery power: {solver.battery_power} kW")
        print(f"  Proxy weights: {proxy_weights}")
        
        # Solve DP problem
        print("\n" + "="*80)
        print("  Solving DP Problem")
        print("="*80)
        print("\nThis will take approximately 15 minutes...")
        print("Progress will be displayed during backward induction.\n")
        
        dp_start = time.time()
        J, mu = solver.solve()
        dp_time = time.time() - dp_start
        
        print(f"\nDP solution completed in {dp_time/60:.1f} minutes")
        
        # Extract optimal action trajectory
        print("\nExtracting optimal action trajectory...")
        actions = solver.get_action_trajectory(initial_soc=0.0)
        
        print(f"  Action trajectory shape: {actions.shape}")
        print(f"  Action statistics:")
        print(f"    Mean: {actions.mean():.4f}")
        print(f"    Std: {actions.std():.4f}")
        print(f"    Min: {actions.min():.4f}")
        print(f"    Max: {actions.max():.4f}")
        
        # Evaluate trajectory
        print("\nEvaluating DP trajectory...")
        metrics = solver.evaluate_trajectory(actions, initial_soc=0.0)
        
        print(f"\nDP Baseline Performance:")
        print(f"  Total Cost: ${metrics['total_cost']:.2f}")
        print(f"  Power Cost: ${metrics['power_cost']:.2f}")
        print(f"  Emission Cost: ${metrics['emission_cost']:.2f}")
        print(f"  Peak Demand: {metrics['peak_demand']:.2f} kW")
        print(f"  Energy Imported: {metrics['energy_imported']:.2f} kWh")
        print(f"  Energy Exported: {metrics['energy_exported']:.2f} kWh")
        print(f"  Battery Cycles: {metrics['battery_cycles']:.2f}")
        print(f"  Final SoC: {metrics['final_soc']:.4f}")
        print(f"  Capacity Degradation: {metrics['capacity_degradation']:.6f} kWh")
        
        # Compare with baseline (no control)
        print("\nComparing with baseline (no battery control)...")
        baseline_actions = np.zeros(len(actions))
        baseline_metrics = solver.evaluate_trajectory(baseline_actions, initial_soc=0.0)
        
        print(f"\nBaseline (No Control) Performance:")
        print(f"  Total Cost: ${baseline_metrics['total_cost']:.2f}")
        print(f"  Peak Demand: {baseline_metrics['peak_demand']:.2f} kW")
        
        # Calculate improvements
        cost_improvement = (baseline_metrics['total_cost'] - metrics['total_cost']) / baseline_metrics['total_cost'] * 100
        peak_reduction = (baseline_metrics['peak_demand'] - metrics['peak_demand']) / baseline_metrics['peak_demand'] * 100
        
        print(f"\nDP Improvements over Baseline:")
        print(f"  Cost Savings: {cost_improvement:.2f}%")
        print(f"  Peak Reduction: {peak_reduction:.2f}%")
        print(f"  Absolute Cost Savings: ${baseline_metrics['total_cost'] - metrics['total_cost']:.2f}")
        
        # Save DP actions
        print("\nSaving DP actions...")
        actions_path = output_dir / "single_agent_dp.npy"
        np.save(actions_path, actions)
        print(f"  Saved: {actions_path}")
        
        # Save metrics
        metrics_data = {
            'building_id': building_id,
            'dp_metrics': {k: float(v) for k, v in metrics.items()},
            'baseline_metrics': {k: float(v) for k, v in baseline_metrics.items()},
            'improvements': {
                'cost_savings_percent': float(cost_improvement),
                'peak_reduction_percent': float(peak_reduction),
                'absolute_cost_savings': float(baseline_metrics['total_cost'] - metrics['total_cost'])
            },
            'proxy_weights': proxy_weights,
            'solver_config': {
                'n_states': solver.n_states,
                'n_horizon': solver.n_horizon,
                'battery_capacity': float(solver.battery_capacity),
                'battery_power': float(solver.battery_power)
            },
            'computation_time_seconds': float(dp_time)
        }
        
        metrics_path = output_dir / "dp_baseline_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"  Saved: {metrics_path}")
        
        # Save cost-to-go function for analysis
        print("\nSaving DP solution (cost-to-go and policy)...")
        np.save(output_dir / "dp_cost_to_go.npy", J)
        np.save(output_dir / "dp_policy.npy", mu)
        print(f"  Saved: {output_dir / 'dp_cost_to_go.npy'}")
        print(f"  Saved: {output_dir / 'dp_policy.npy'}")
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("  DP BASELINE COMPUTATION COMPLETE")
        print("="*80)
        print(f"\nTotal time: {total_time/60:.1f} minutes")
        print(f"  DP solving: {dp_time/60:.1f} minutes")
        print(f"  Data prep & evaluation: {(total_time-dp_time)/60:.1f} minutes")
        
        print(f"\nGenerated files in {output_dir}:")
        print(f"  single_agent_dp.npy - DP action trajectory")
        print(f"  dp_baseline_metrics.json - Performance metrics")
        print(f"  dp_cost_to_go.npy - Cost-to-go function J")
        print(f"  dp_policy.npy - Policy function μ")
        
        print("\n" + "="*80)
        print("  NEXT STEP: Run 5_policy_optimization.py")
        print("  (This will take 2.5 days with 50 parallel workers)")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ DP computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())