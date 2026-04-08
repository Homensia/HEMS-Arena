# ================================================================
# hems/agents/ambitious_engineers_algorithm.py - phase1_policy.py
# ================================================================

"""
CityLearn Challenge 2022 Data Loader
=====================================
Loads and preprocesses data from the CityLearn Challenge 2022 Phase 1 dataset.

Directory structure expected:
datasets/citylearn_datasets/citylearn_challenge_2022_phase_1/
├── Building_1.csv
├── Building_2.csv
├── Building_3.csv
├── Building_4.csv
├── Building_5.csv
├── carbon_intensity.csv
├── pricing.csv
└── weather.csv

This loader extracts:
- Building observations (load, solar, etc.)
- Carbon intensity
- Electricity pricing
- Weather data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class CityLearnDataLoader:
    """
    Loader for CityLearn Challenge 2022 Phase 1 data.
    
    This matches the exact format used by Team ambitiousengineers.
    """
    
    def __init__(self, data_path: str = "datasets/citylearn_datasets/citylearn_challenge_2022_phase_1"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CityLearn Challenge 2022 Phase 1 data directory
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_path}\n"
                f"Please ensure CityLearn Challenge 2022 Phase 1 data is in the correct location."
            )
        
        # Check for required files
        self.required_files = [
            'Building_1.csv', 'Building_2.csv', 'Building_3.csv',
            'Building_4.csv', 'Building_5.csv',
            'carbon_intensity.csv', 'pricing.csv', 'weather.csv'
        ]
        
        missing_files = [f for f in self.required_files if not (self.data_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        print(f"✓ Data directory validated: {self.data_path}")
    
    def load_building_data(self, building_id: int) -> pd.DataFrame:
        """
        Load data for a specific building.
        
        Args:
            building_id: Building ID (1-5)
        
        Returns:
            DataFrame with building data
        """
        if building_id < 1 or building_id > 5:
            raise ValueError(f"Building ID must be 1-5, got {building_id}")
        
        filepath = self.data_path / f"Building_{building_id}.csv"
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Ensure required columns exist
        required_cols = ['non_shiftable_load', 'solar_generation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try alternative names
            if 'equipment_electric_power' in df.columns:
                df['non_shiftable_load'] = df['equipment_electric_power']
            if 'solar_generation' not in df.columns and 'pv_generation' in df.columns:
                df['solar_generation'] = df['pv_generation']
        
        return df
    
    def load_all_buildings(self) -> Dict[int, pd.DataFrame]:
        """
        Load data for all 5 buildings.
        
        Returns:
            Dictionary mapping building_id to DataFrame
        """
        buildings = {}
        for building_id in range(1, 6):
            buildings[building_id] = self.load_building_data(building_id)
        
        print(f"✓ Loaded data for {len(buildings)} buildings")
        return buildings
    
    def load_carbon_intensity(self) -> np.ndarray:
        """
        Load carbon intensity data.
        
        Returns:
            1D array of carbon intensity values
        """
        filepath = self.data_path / "carbon_intensity.csv"
        df = pd.read_csv(filepath)
        
        # Get the carbon intensity column
        if 'carbon_intensity' in df.columns:
            carbon = df['carbon_intensity'].values
        elif len(df.columns) == 1:
            carbon = df.iloc[:, 0].values
        else:
            # Try to find the right column
            carbon = df.iloc[:, 0].values
        
        print(f"✓ Loaded carbon intensity: {len(carbon)} timesteps")
        return carbon
    
    def load_pricing(self) -> np.ndarray:
        """
        Load electricity pricing data.
        
        Returns:
            1D array of price values
        """
        filepath = self.data_path / "pricing.csv"
        df = pd.read_csv(filepath)
        
        # Get the pricing column
        if 'electricity_pricing' in df.columns:
            pricing = df['electricity_pricing'].values
        elif 'price' in df.columns:
            pricing = df['price'].values
        elif len(df.columns) == 1:
            pricing = df.iloc[:, 0].values
        else:
            pricing = df.iloc[:, 0].values
        
        print(f"✓ Loaded pricing data: {len(pricing)} timesteps")
        return pricing
    
    def load_weather(self) -> pd.DataFrame:
        """
        Load weather data.
        
        Returns:
            DataFrame with weather variables
        """
        filepath = self.data_path / "weather.csv"
        df = pd.read_csv(filepath)
        
        print(f"✓ Loaded weather data: {len(df)} timesteps, {len(df.columns)} variables")
        return df
    
    def create_observations_dataframe(self) -> pd.DataFrame:
        """
        Create consolidated observations DataFrame for all buildings.
        
        This matches the format used in their code:
        - building_num: Building ID (1-5)
        - timestep: Time index
        - non_shiftable_load: Load in kW
        - solar_generation: Solar in kW
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0-6)
        
        Returns:
            DataFrame with all building observations
        """
        all_data = []
        
        buildings = self.load_all_buildings()
        
        for building_id, df in buildings.items():
            df = df.copy()
            df['building_num'] = building_id
            df['timestep'] = np.arange(len(df))
            
            # Add time features if not present
            if 'hour' not in df.columns:
                df['hour'] = df['timestep'] % 24
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = (df['timestep'] // 24) % 7
            
            # Select relevant columns
            cols = ['building_num', 'timestep', 'non_shiftable_load', 
                   'solar_generation', 'hour', 'day_of_week']
            
            all_data.append(df[cols])
        
        observations = pd.concat(all_data, ignore_index=True)
        
        print(f"✓ Created observations DataFrame: {len(observations)} rows")
        return observations
    
    def get_building_stats(self) -> Dict:
        """
        Get statistics about the buildings.
        
        Returns:
            Dictionary with building statistics
        """
        buildings = self.load_all_buildings()
        
        stats = {}
        for building_id, df in buildings.items():
            stats[f'building_{building_id}'] = {
                'timesteps': len(df),
                'avg_load_kw': df['non_shiftable_load'].mean(),
                'max_load_kw': df['non_shiftable_load'].max(),
                'avg_solar_kw': df['solar_generation'].mean(),
                'max_solar_kw': df['solar_generation'].max(),
                'total_energy_kwh': df['non_shiftable_load'].sum(),
            }
        
        return stats
    
    def prepare_for_dp(self, building_id: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare data for Dynamic Programming solver.
        
        Args:
            building_id: Building ID (1-5)
        
        Returns:
            Tuple of (building_data, price_data, emission_data)
        """
        building_df = self.load_building_data(building_id)
        prices = self.load_pricing()
        emissions = self.load_carbon_intensity()
        
        # Ensure arrays are same length
        n_steps = min(len(building_df), len(prices), len(emissions))
        
        building_data = building_df[['non_shiftable_load', 'solar_generation']].iloc[:n_steps]
        price_data = prices[:n_steps]
        emission_data = emissions[:n_steps]
        
        return building_data, price_data, emission_data
    
    def save_processed_data(self, output_dir: str = "data/processed"):
        """
        Save processed data for later use.
        
        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save observations
        observations = self.create_observations_dataframe()
        observations.to_csv(output_path / "observations.csv", index=False)
        
        # Save carbon and pricing
        carbon = self.load_carbon_intensity()
        pricing = self.load_pricing()
        
        np.save(output_path / "carbon_intensity.npy", carbon)
        np.save(output_path / "pricing.npy", pricing)
        
        # Save statistics
        stats = self.get_building_stats()
        with open(output_path / "building_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved processed data to: {output_path}")
        
        return output_path


def test_data_loader():
    """Test the data loader."""
    print("="*80)
    print("Testing CityLearn Data Loader")
    print("="*80)
    
    try:
        # Initialize loader
        loader = CityLearnDataLoader()
        
        # Test loading individual building
        print("\nTest 1: Load Building 1")
        print("-"*80)
        building1 = loader.load_building_data(1)
        print(f"Shape: {building1.shape}")
        print(f"Columns: {building1.columns.tolist()}")
        print(f"First few rows:\n{building1.head()}")
        
        # Test loading all buildings
        print("\nTest 2: Load All Buildings")
        print("-"*80)
        buildings = loader.load_all_buildings()
        for bid, df in buildings.items():
            print(f"Building {bid}: {len(df)} timesteps")
        
        # Test carbon and pricing
        print("\nTest 3: Load Carbon Intensity and Pricing")
        print("-"*80)
        carbon = loader.load_carbon_intensity()
        pricing = loader.load_pricing()
        print(f"Carbon intensity: {len(carbon)} values, range [{carbon.min():.4f}, {carbon.max():.4f}]")
        print(f"Pricing: {len(pricing)} values, range [{pricing.min():.4f}, {pricing.max():.4f}]")
        
        # Test observations DataFrame
        print("\nTest 4: Create Observations DataFrame")
        print("-"*80)
        obs = loader.create_observations_dataframe()
        print(f"Shape: {obs.shape}")
        print(f"Buildings: {obs['building_num'].unique()}")
        print(f"Sample:\n{obs.head(10)}")
        
        # Test stats
        print("\nTest 5: Building Statistics")
        print("-"*80)
        stats = loader.get_building_stats()
        for building, building_stats in stats.items():
            print(f"\n{building}:")
            for key, value in building_stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Test DP preparation
        print("\nTest 6: Prepare Data for DP")
        print("-"*80)
        building_data, price_data, emission_data = loader.prepare_for_dp(1)
        print(f"Building data shape: {building_data.shape}")
        print(f"Price data length: {len(price_data)}")
        print(f"Emission data length: {len(emission_data)}")
        
        # Test save
        print("\nTest 7: Save Processed Data")
        print("-"*80)
        output_path = loader.save_processed_data("data/processed")
        print(f"Data saved to: {output_path}")
        
        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loader()