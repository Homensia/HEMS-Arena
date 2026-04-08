# ============================================================
# hems/agents/ambitious_engineers_algorithm.py - intialize.py
# ============================================================


"""
1_initialize.py - Data Preparation for Team ambitiousengineers
===============================================================
Expected runtime: ~5 seconds

This script prepares all data needed for training:
1. Loads CityLearn Challenge 2022 Phase 1 data
2. Creates observations DataFrame
3. Extracts carbon intensity and pricing
4. Saves processed data for training scripts

Output:
- data/external/observations.csv
- data/external/carbon_intensity.npy
- data/external/pricing.npy
- data/external/building_stats.json
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hems.algorithms.ambitious_engineers_algorithm.data_loader import CityLearnDataLoader


def main():
    print("="*80)
    print("  STEP 1: Initialize Data")
    print("="*80)
    print("\nThis script prepares data from CityLearn Challenge 2022 Phase 1")
    print("Expected runtime: ~5 seconds\n")
    
    start_time = time.time()
    
    # Create output directories
    output_dir = Path("data/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data loader
        print("Loading CityLearn Challenge 2022 Phase 1 data...")
        loader = CityLearnDataLoader(
            data_path="datasets/citylearn_datasets/citylearn_challenge_2022_phase_1"
        )
        
        # Load all buildings
        print("\nLoading building data...")
        buildings = loader.load_all_buildings()
        
        for building_id, df in buildings.items():
            print(f"  Building {building_id}: {len(df)} timesteps")
        
        # Create observations DataFrame
        print("\nCreating observations DataFrame...")
        observations = loader.create_observations_dataframe()
        observations_path = output_dir / "observations.csv"
        observations.to_csv(observations_path, index=False)
        print(f"  ✓ Saved: {observations_path}")
        print(f"    Shape: {observations.shape}")
        print(f"    Buildings: {observations['building_num'].nunique()}")
        
        # Load and save carbon intensity
        print("\nLoading carbon intensity...")
        carbon = loader.load_carbon_intensity()
        carbon_path = output_dir / "carbon_intensity.npy"
        np.save(carbon_path, carbon)
        print(f"  ✓ Saved: {carbon_path}")
        print(f"    Length: {len(carbon)} timesteps")
        print(f"    Range: [{carbon.min():.4f}, {carbon.max():.4f}]")
        
        # Load and save pricing
        print("\nLoading electricity pricing...")
        pricing = loader.load_pricing()
        pricing_path = output_dir / "pricing.npy"
        np.save(pricing_path, pricing)
        print(f"  ✓ Saved: {pricing_path}")
        print(f"    Length: {len(pricing)} timesteps")
        print(f"    Range: [{pricing.min():.4f}, {pricing.max():.4f}]")
        
        # Save building statistics
        print("\nComputing building statistics...")
        stats = loader.get_building_stats()
        
        import json
        stats_path = output_dir / "building_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved: {stats_path}")
        
        # Print statistics
        print("\nBuilding Statistics:")
        print("-"*80)
        for building, building_stats in stats.items():
            print(f"\n{building.replace('_', ' ').title()}:")
            print(f"  Timesteps: {building_stats['timesteps']}")
            print(f"  Average Load: {building_stats['avg_load_kw']:.2f} kW")
            print(f"  Peak Load: {building_stats['max_load_kw']:.2f} kW")
            print(f"  Average Solar: {building_stats['avg_solar_kw']:.2f} kW")
            print(f"  Peak Solar: {building_stats['max_solar_kw']:.2f} kW")
            print(f"  Total Energy: {building_stats['total_energy_kwh']:.2f} kWh")
        
        # Create summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("  INITIALIZATION COMPLETE")
        print("="*80)
        print(f"\nElapsed time: {elapsed_time:.2f} seconds")
        print(f"\nGenerated files in {output_dir}:")
        print("  ✓ observations.csv - All building observations")
        print("  ✓ carbon_intensity.npy - Carbon intensity data")
        print("  ✓ pricing.npy - Electricity pricing data")
        print("  ✓ building_stats.json - Building statistics")
        
        print("\n" + "="*80)
        print("  NEXT STEP: Run 2_create_demand_model.py")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())