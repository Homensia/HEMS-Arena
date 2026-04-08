"""
HEMS Exploratory Data Analysis Module
Comprehensive data analysis for CityLearn datasets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path
from hems.utils.dataset import DataSet

from citylearn.citylearn import CityLearnEnv


class HEMSDataAnalyzer:
    """Comprehensive data analyzer for HEMS simulation environment."""
    
    def __init__(self, config):
        """
        Initialize data analyzer.
        
        Args:
            config: SimulationConfig object
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.output_dir = Path(config.output_dir) / 'eda'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"HEMS Data Analyzer initialized for dataset: {self.dataset_name}")
        print(f"Results will be saved to: {self.output_dir}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive data analysis.
        
        Returns:
            Dictionary with analysis results
        """
        print("Starting comprehensive data analysis...")
        
        results = {}
        
        # 1. Dataset overview
        print("1. Analyzing dataset overview...")
        results['overview'] = self.analyze_dataset_overview()
        
        # 2. Building characteristics
        print("2. Analyzing building characteristics...")
        results['buildings'] = self.analyze_building_characteristics()
        
        # 3. Time series analysis
        print("3. Analyzing time series patterns...")
        results['timeseries'] = self.analyze_timeseries_patterns()
        
        # 4. Energy patterns
        print("4. Analyzing energy consumption patterns...")
        results['energy'] = self.analyze_energy_patterns()
        
        # 5. Weather and solar analysis
        print("5. Analyzing weather and solar patterns...")
        results['weather'] = self.analyze_weather_patterns()
        
        # 6. Correlations
        print("6. Analyzing correlations...")
        results['correlations'] = self.analyze_correlations()
        
        # 7. Seasonal analysis
        print("7. Analyzing seasonal patterns...")
        results['seasonal'] = self.analyze_seasonal_patterns()
        
        # 8. Generate summary report
        print("8. Generating summary report...")
        self.generate_summary_report(results)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
        return results
    
    def analyze_dataset_overview(self) -> Dict[str, Any]:
        """Analyze basic dataset characteristics."""
        schema = str(DataSet().get_schema())
        
        overview = {
            'dataset_name': self.dataset_name,
            'total_buildings': len(schema['buildings']),
            'building_names': list(schema['buildings'].keys()),
            'root_directory': schema['root_directory']
        }
        
        # Sample data to get time information
        sample_building = list(schema['buildings'].keys())[0]
        carbon_file = schema['buildings'][sample_building]['carbon_intensity']
        carbon_path = os.path.join(schema['root_directory'], carbon_file)
        
        if os.path.exists(carbon_path):
            sample_data = pd.read_csv(carbon_path)
            overview['total_timesteps'] = len(sample_data)
            overview['total_days'] = len(sample_data) // 24
            overview['total_hours'] = len(sample_data)
        
        # Create overview plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Building count
        axes[0, 0].bar(['Buildings'], [overview['total_buildings']], color='skyblue')
        axes[0, 0].set_title('Total Buildings in Dataset')
        axes[0, 0].set_ylabel('Count')
        
        # Time information
        time_info = ['Days', 'Hours', 'Timesteps']
        time_values = [overview.get('total_days', 0), 
                      overview.get('total_hours', 0), 
                      overview.get('total_timesteps', 0)]
        axes[0, 1].bar(time_info, time_values, color='lightgreen')
        axes[0, 1].set_title('Dataset Time Coverage')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Building names (if reasonable number)
        if len(overview['building_names']) <= 15:
            axes[1, 0].barh(range(len(overview['building_names'])), 
                           [1] * len(overview['building_names']))
            axes[1, 0].set_yticks(range(len(overview['building_names'])))
            axes[1, 0].set_yticklabels(overview['building_names'])
            axes[1, 0].set_title('Available Buildings')
            axes[1, 0].set_xlabel('Available')
        else:
            axes[1, 0].text(0.5, 0.5, f'{len(overview["building_names"])} buildings\n(too many to display)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Available Buildings')
        
        # Dataset info text
        info_text = f"Dataset: {self.dataset_name}\n"
        info_text += f"Buildings: {overview['total_buildings']}\n"
        info_text += f"Time Coverage: {overview.get('total_days', 0)} days\n"
        info_text += f"Timesteps: {overview.get('total_timesteps', 0)}"
        
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Dataset Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return overview
    
    def analyze_building_characteristics(self) -> Dict[str, Any]:
        """Analyze individual building characteristics."""
        # Create a temporary environment to extract building info
        env = CityLearnEnv(
            self.dataset_name,
            central_agent=True,
            buildings=None,  # All buildings
            simulation_start_time_step=0,
            simulation_end_time_step=168  # One week
        )
        
        building_info = []
        for building in env.buildings:
            info = {
                'name': building.name,
                'battery_capacity': building.electrical_storage.capacity,
                'battery_power': building.electrical_storage.nominal_power,
                'battery_efficiency': building.electrical_storage.efficiency,
                'pv_capacity': building.pv.nominal_power,
                'has_battery': building.electrical_storage.capacity > 0,
                'has_pv': building.pv.nominal_power > 0
            }
            building_info.append(info)
        
        df_buildings = pd.DataFrame(building_info)
        
        # Create characteristics plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Battery capacity distribution
        axes[0, 0].hist(df_buildings['battery_capacity'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Battery Capacity Distribution')
        axes[0, 0].set_xlabel('Capacity (kWh)')
        axes[0, 0].set_ylabel('Count')
        
        # PV capacity distribution
        axes[0, 1].hist(df_buildings['pv_capacity'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('PV Capacity Distribution')
        axes[0, 1].set_xlabel('Capacity (kW)')
        axes[0, 1].set_ylabel('Count')
        
        # Battery vs PV scatter
        axes[0, 2].scatter(df_buildings['battery_capacity'], df_buildings['pv_capacity'], alpha=0.6)
        axes[0, 2].set_title('Battery vs PV Capacity')
        axes[0, 2].set_xlabel('Battery Capacity (kWh)')
        axes[0, 2].set_ylabel('PV Capacity (kW)')
        
        # Battery power distribution
        axes[1, 0].hist(df_buildings['battery_power'], bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Battery Power Distribution')
        axes[1, 0].set_xlabel('Power (kW)')
        axes[1, 0].set_ylabel('Count')
        
        # Efficiency distribution
        axes[1, 1].hist(df_buildings['battery_efficiency'], bins=20, alpha=0.7, color='red')
        axes[1, 1].set_title('Battery Efficiency Distribution')
        axes[1, 1].set_xlabel('Efficiency')
        axes[1, 1].set_ylabel('Count')
        
        # System availability
        system_counts = {
            'Battery Only': sum(df_buildings['has_battery'] & ~df_buildings['has_pv']),
            'PV Only': sum(~df_buildings['has_battery'] & df_buildings['has_pv']),
            'Both': sum(df_buildings['has_battery'] & df_buildings['has_pv']),
            'Neither': sum(~df_buildings['has_battery'] & ~df_buildings['has_pv'])
        }
        
        axes[1, 2].pie(system_counts.values(), labels=system_counts.keys(), autopct='%1.1f%%')
        axes[1, 2].set_title('System Configuration')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'building_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save building data
        df_buildings.to_csv(self.output_dir / 'building_characteristics.csv', index=False)
        
        return {
            'building_count': len(df_buildings),
            'avg_battery_capacity': df_buildings['battery_capacity'].mean(),
            'avg_pv_capacity': df_buildings['pv_capacity'].mean(),
            'buildings_with_battery': df_buildings['has_battery'].sum(),
            'buildings_with_pv': df_buildings['has_pv'].sum(),
            'building_data': df_buildings.to_dict('records')
        }
    
    def analyze_timeseries_patterns(self) -> Dict[str, Any]:
        """Analyze time series patterns in the data."""
        # Select a representative building for time series analysis
        schema = str(DataSet().get_schema())
        sample_building = 'Building_1' if 'Building_1' in schema['buildings'] else list(schema['buildings'].keys())[0]
        
        # Load various time series data
        building_data = schema['buildings'][sample_building]
        root_dir = schema['root_directory']
        
        timeseries_data = {}
        
        # Load available data files
        for data_type, filename in building_data.items():
            if filename and isinstance(filename, str):
                filepath = os.path.join(root_dir, filename)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        if len(df) > 0:
                            timeseries_data[data_type] = df.iloc[:, 0].values  # First column
                    except Exception as e:
                        print(f"Could not load {data_type}: {e}")
        
        if not timeseries_data:
            return {'error': 'No time series data could be loaded'}
        
        # Create time series plots
        n_series = len(timeseries_data)
        cols = min(3, n_series)
        rows = (n_series + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if n_series == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        analysis_results = {}
        
        for i, (data_type, data) in enumerate(timeseries_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot first week of data
            week_data = data[:168] if len(data) >= 168 else data
            hours = range(len(week_data))
            
            ax.plot(hours, week_data)
            ax.set_title(f'{data_type.replace("_", " ").title()}\n(First Week)')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Calculate statistics
            analysis_results[data_type] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'length': len(data)
            }
        
        # Hide unused subplots
        for i in range(len(timeseries_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timeseries_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'sample_building': sample_building,
            'available_series': list(timeseries_data.keys()),
            'statistics': analysis_results
        }
    
    def analyze_energy_patterns(self) -> Dict[str, Any]:
        """Analyze energy consumption and generation patterns."""
        # Create environment for a few days
        env = CityLearnEnv(
            self.dataset_name,
            central_agent=True,
            buildings=['Building_1'] if 'Building_1' in DataSet().get_schema(self.dataset_name)['buildings'] else None,
            simulation_start_time_step=0,
            simulation_end_time_step=168  # One week
        )
        
        # Run simulation to collect data
        obs, _ = env.reset()
        energy_data = {
            'net_consumption': [],
            'solar_generation': [],
            'battery_soc': [],
            'hour': []
        }
        
        while not env.terminated:
            # Extract energy data
            if len(obs[0]) >= 5:
                energy_data['hour'].append(obs[0][0])
                energy_data['net_consumption'].append(obs[0][2])
                energy_data['battery_soc'].append(obs[0][3])
                energy_data['solar_generation'].append(obs[0][4])
            
            # Take no action (baseline)
            actions = [[0.0]]
            obs, _, _, _, _ = env.step(actions)
        
        # Convert to DataFrame
        df_energy = pd.DataFrame(energy_data)
        
        # Create energy pattern plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Daily consumption pattern
        if len(df_energy) >= 24:
            daily_consumption = df_energy.groupby(df_energy.index % 24)['net_consumption'].mean()
            axes[0, 0].plot(range(24), daily_consumption, marker='o')
            axes[0, 0].set_title('Average Daily Consumption Pattern')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Net Consumption (kWh)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Daily solar pattern
        if len(df_energy) >= 24:
            daily_solar = df_energy.groupby(df_energy.index % 24)['solar_generation'].mean()
            axes[0, 1].plot(range(24), daily_solar, marker='o', color='orange')
            axes[0, 1].set_title('Average Daily Solar Generation Pattern')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Solar Generation (kWh)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Net consumption vs solar generation
        axes[1, 0].scatter(df_energy['solar_generation'], df_energy['net_consumption'], alpha=0.6)
        axes[1, 0].set_title('Net Consumption vs Solar Generation')
        axes[1, 0].set_xlabel('Solar Generation (kWh)')
        axes[1, 0].set_ylabel('Net Consumption (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Battery SoC over time
        axes[1, 1].plot(df_energy['battery_soc'])
        axes[1, 1].set_title('Battery State of Charge Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('SoC')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save energy data
        df_energy.to_csv(self.output_dir / 'energy_patterns.csv', index=False)
        
        return {
            'peak_consumption_hour': df_energy.groupby(df_energy.index % 24)['net_consumption'].mean().idxmax(),
            'peak_solar_hour': df_energy.groupby(df_energy.index % 24)['solar_generation'].mean().idxmax(),
            'avg_consumption': df_energy['net_consumption'].mean(),
            'avg_solar': df_energy['solar_generation'].mean(),
            'consumption_solar_correlation': df_energy['net_consumption'].corr(df_energy['solar_generation'])
        }
    
    def analyze_weather_patterns(self) -> Dict[str, Any]:
        """Analyze weather and solar irradiance patterns."""
        schema = str(DataSet().get_schema())
        sample_building = list(schema['buildings'].keys())[0]
        building_data = schema['buildings'][sample_building]
        root_dir = schema['root_directory']
        
        weather_data = {}
        weather_files = ['solar_irradiance', 'outdoor_dry_bulb_temperature']
        
        for weather_type in weather_files:
            if weather_type in building_data:
                filepath = os.path.join(root_dir, building_data[weather_type])
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        weather_data[weather_type] = df.iloc[:, 0].values
                    except Exception as e:
                        print(f"Could not load {weather_type}: {e}")
        
        if not weather_data:
            return {'error': 'No weather data available'}
        
        # Create weather plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'solar_irradiance' in weather_data:
            irradiance = weather_data['solar_irradiance'][:168]  # First week
            axes[0, 0].plot(irradiance)
            axes[0, 0].set_title('Solar Irradiance (First Week)')
            axes[0, 0].set_xlabel('Hour')
            axes[0, 0].set_ylabel('Irradiance (W/m²)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Daily solar pattern
            if len(irradiance) >= 24:
                daily_irradiance = pd.Series(irradiance).groupby(pd.Series(irradiance).index % 24).mean()
                axes[0, 1].plot(range(24), daily_irradiance, marker='o', color='orange')
                axes[0, 1].set_title('Average Daily Solar Irradiance')
                axes[0, 1].set_xlabel('Hour of Day')
                axes[0, 1].set_ylabel('Irradiance (W/m²)')
                axes[0, 1].grid(True, alpha=0.3)
        
        if 'outdoor_dry_bulb_temperature' in weather_data:
            temperature = weather_data['outdoor_dry_bulb_temperature'][:168]
            axes[1, 0].plot(temperature, color='red')
            axes[1, 0].set_title('Outdoor Temperature (First Week)')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Daily temperature pattern
            if len(temperature) >= 24:
                daily_temp = pd.Series(temperature).groupby(pd.Series(temperature).index % 24).mean()
                axes[1, 1].plot(range(24), daily_temp, marker='o', color='red')
                axes[1, 1].set_title('Average Daily Temperature')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Temperature (°C)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weather_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        weather_stats = {}
        for weather_type, data in weather_data.items():
            weather_stats[weather_type] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data))
            }
        
        return {
            'available_weather_data': list(weather_data.keys()),
            'statistics': weather_stats
        }
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different variables."""
        # Create environment and collect data
        env = CityLearnEnv(
            self.dataset_name,
            central_agent=True,
            buildings=['Building_1'] if 'Building_1' in DataSet().get_schema(self.dataset_name)['buildings'] else None,
            simulation_start_time_step=0,
            simulation_end_time_step=720  # One month
        )
        
        # Collect correlation data
        obs, _ = env.reset()
        correlation_data = []
        
        while not env.terminated:
            if len(obs[0]) >= 5:
                correlation_data.append({
                    'hour': obs[0][0],
                    'pricing': obs[0][1] if len(obs[0]) > 1 else 0,
                    'net_consumption': obs[0][2],
                    'battery_soc': obs[0][3],
                    'solar_generation': obs[0][4]
                })
            
            actions = [[0.0]]
            obs, _, _, _, _ = env.step(actions)
        
        df_corr = pd.DataFrame(correlation_data)
        
        # Calculate correlation matrix
        correlation_matrix = df_corr.corr()
        
        # Create correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0])
        axes[0].set_title('Variable Correlations')
        
        # Pairwise scatter plots for key variables
        if len(df_corr) > 10:
            sample_indices = np.random.choice(len(df_corr), size=min(1000, len(df_corr)), replace=False)
            df_sample = df_corr.iloc[sample_indices]
            
            axes[1].scatter(df_sample['solar_generation'], df_sample['net_consumption'], 
                          alpha=0.6, label='Solar vs Consumption')
            axes[1].set_xlabel('Solar Generation')
            axes[1].set_ylabel('Net Consumption')
            axes[1].set_title('Solar Generation vs Net Consumption')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation data
        correlation_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        df_corr.to_csv(self.output_dir / 'correlation_data.csv', index=False)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix)
        }
    
    def analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """Analyze seasonal patterns in energy data."""
        try:
            # Create environment for full year if possible
            env = CityLearnEnv(
                self.dataset_name,
                central_agent=True,
                buildings=['Building_1'] if 'Building_1' in DataSet().get_schema(self.dataset_name)['buildings'] else None,
                simulation_start_time_step=0,
                simulation_end_time_step=min(8760, 2000)  # Full year or 2000 steps
            )
            
            # Collect seasonal data
            obs, _ = env.reset()
            seasonal_data = []
            
            while not env.terminated:
                if len(obs[0]) >= 5:
                    seasonal_data.append({
                        'timestep': len(seasonal_data),
                        'hour': obs[0][0],
                        'net_consumption': obs[0][2],
                        'solar_generation': obs[0][4]
                    })
                
                actions = [[0.0]]
                obs, _, _, _, _ = env.step(actions)
            
            df_seasonal = pd.DataFrame(seasonal_data)
            
            if len(df_seasonal) < 168:  # Less than a week
                return {'error': 'Insufficient data for seasonal analysis'}
            
            # Add time features
            df_seasonal['day'] = df_seasonal['timestep'] // 24
            df_seasonal['week'] = df_seasonal['day'] // 7
            df_seasonal['month'] = df_seasonal['day'] // 30
            
            # Create seasonal plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Weekly patterns
            if df_seasonal['week'].max() > 1:
                weekly_consumption = df_seasonal.groupby('week')['net_consumption'].mean()
                weekly_solar = df_seasonal.groupby('week')['solar_generation'].mean()
                
                axes[0, 0].plot(weekly_consumption.index, weekly_consumption, label='Consumption', marker='o')
                axes[0, 0].plot(weekly_solar.index, weekly_solar, label='Solar', marker='s')
                axes[0, 0].set_title('Weekly Patterns')
                axes[0, 0].set_xlabel('Week')
                axes[0, 0].set_ylabel('Energy (kWh)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Monthly patterns (if enough data)
            if df_seasonal['month'].max() > 1:
                monthly_consumption = df_seasonal.groupby('month')['net_consumption'].mean()
                monthly_solar = df_seasonal.groupby('month')['solar_generation'].mean()
                
                axes[0, 1].plot(monthly_consumption.index, monthly_consumption, label='Consumption', marker='o')
                axes[0, 1].plot(monthly_solar.index, monthly_solar, label='Solar', marker='s')
                axes[0, 1].set_title('Monthly Patterns')
                axes[0, 1].set_xlabel('Month')
                axes[0, 1].set_ylabel('Energy (kWh)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Daily patterns by week
            daily_by_week = df_seasonal.groupby(['week', df_seasonal['timestep'] % 24])['net_consumption'].mean().unstack(level=0)
            if not daily_by_week.empty:
                daily_by_week.plot(ax=axes[1, 0], alpha=0.7)
                axes[1, 0].set_title('Daily Consumption Patterns by Week')
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Net Consumption (kWh)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend().set_visible(False)
            
            # Solar patterns by week
            solar_by_week = df_seasonal.groupby(['week', df_seasonal['timestep'] % 24])['solar_generation'].mean().unstack(level=0)
            if not solar_by_week.empty:
                solar_by_week.plot(ax=axes[1, 1], alpha=0.7)
                axes[1, 1].set_title('Daily Solar Patterns by Week')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Solar Generation (kWh)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend().set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'seasonal_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'data_length_days': df_seasonal['day'].max() + 1,
                'weekly_consumption_trend': self._calculate_trend(df_seasonal.groupby('week')['net_consumption'].mean()) if df_seasonal['week'].max() > 1 else None,
                'weekly_solar_trend': self._calculate_trend(df_seasonal.groupby('week')['solar_generation'].mean()) if df_seasonal['week'].max() > 1 else None
            }
            
        except Exception as e:
            return {'error': f'Seasonal analysis failed: {str(e)}'}
    
    def _find_strong_correlations(self, corr_matrix, threshold=0.7):
        """Find strong correlations in correlation matrix."""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        return strong_corr
    
    def _calculate_trend(self, series):
        """Calculate trend direction of a time series."""
        if len(series) < 2:
            return 'insufficient_data'
        
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        
        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HEMS DATA ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset overview
            if 'overview' in results:
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 20 + "\n")
                overview = results['overview']
                f.write(f"Dataset: {overview['dataset_name']}\n")
                f.write(f"Total buildings: {overview['total_buildings']}\n")
                f.write(f"Time coverage: {overview.get('total_days', 'N/A')} days\n")
                f.write(f"Total timesteps: {overview.get('total_timesteps', 'N/A')}\n\n")
            
            # Building characteristics
            if 'buildings' in results:
                f.write("BUILDING CHARACTERISTICS\n")
                f.write("-" * 25 + "\n")
                buildings = results['buildings']
                f.write(f"Buildings analyzed: {buildings['building_count']}\n")
                f.write(f"Average battery capacity: {buildings['avg_battery_capacity']:.2f} kWh\n")
                f.write(f"Average PV capacity: {buildings['avg_pv_capacity']:.2f} kW\n")
                f.write(f"Buildings with battery: {buildings['buildings_with_battery']}\n")
                f.write(f"Buildings with PV: {buildings['buildings_with_pv']}\n\n")
            
            # Energy patterns
            if 'energy' in results:
                f.write("ENERGY PATTERNS\n")
                f.write("-" * 15 + "\n")
                energy = results['energy']
                f.write(f"Peak consumption hour: {energy.get('peak_consumption_hour', 'N/A')}\n")
                f.write(f"Peak solar hour: {energy.get('peak_solar_hour', 'N/A')}\n")
                f.write(f"Average consumption: {energy.get('avg_consumption', 0):.4f} kWh\n")
                f.write(f"Average solar generation: {energy.get('avg_solar', 0):.4f} kWh\n")
                f.write(f"Consumption-solar correlation: {energy.get('consumption_solar_correlation', 0):.3f}\n\n")
            
            # Correlations
            if 'correlations' in results and 'strong_correlations' in results['correlations']:
                f.write("STRONG CORRELATIONS\n")
                f.write("-" * 20 + "\n")
                for corr in results['correlations']['strong_correlations']:
                    f.write(f"{corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}\n")
                f.write("\n")
            
            # Analysis files
            f.write("GENERATED FILES\n")
            f.write("-" * 15 + "\n")
            f.write("- dataset_overview.png: Basic dataset information\n")
            f.write("- building_characteristics.png: Building system analysis\n")
            f.write("- timeseries_patterns.png: Time series data patterns\n")
            f.write("- energy_patterns.png: Energy consumption and generation\n")
            f.write("- weather_patterns.png: Weather and environmental data\n")
            f.write("- correlations.png: Variable correlations\n")
            f.write("- seasonal_patterns.png: Seasonal trends\n")
            f.write("- *.csv files: Raw data for further analysis\n")
        
        print(f"Summary report saved to: {report_path}")