#================================
#hems/core/validation_testing.py
#================================
"""
Validation and Testing Phase 
Runs evaluation episodes and collects detailed data for KPI calculation.

Key Features:
- Loads best model for evaluation
- Runs episodes in evaluation mode (deterministic)
- Collects detailed episode data (consumption, PV, cost, SoC, actions)
- Uses different seeds for val/test (generalization testing)
- Computes KPIs using MetricsCalculator
- Compares agent vs baseline
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    from .adapters import ObservationAdapter, ActionAdapter
except ImportError:
    from adapters import ObservationAdapter, ActionAdapter

logger = logging.getLogger(__name__)


class ValidationTester:
    """
    Runs validation or testing phase with proper data collection.
    """
    
    def __init__(
        self,
        agent,
        env_manager,
        config,
        phase: str,  # "validation" or "testing"
        output_dir: Path,
        logger_instance: logging.Logger
    ):
        """
        Initialize validation/testing runner.
        
        Args:
            agent: Agent to evaluate
            env_manager: Environment manager
            config: Benchmark configuration
            phase: "validation" or "testing"
            output_dir: Output directory
            logger_instance: Logger
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.phase = phase
        self.output_dir = Path(output_dir)
        self.logger = logger_instance
        self.training_results = None  # Will be set if training data available
        
        # Get phase config
        if phase == "validation":
            self.phase_config = config.validation
            self.seed_offset = 1000  # seed + 1000
        else:  # testing
            self.phase_config = config.testing
            self.seed_offset = 2000  # seed + 2000
        
        self.episodes = self.phase_config.episodes
        self.building_ids = self.phase_config.buildings.ids if self.phase_config.buildings else []
        
        # Baseline agent for comparison
        self.baseline_agent = None
        
        # Get central_agent setting from environment (config.environment is a DICT)
        central_agent = config.environment.get('central_agent', True) if isinstance(config.environment, dict) else getattr(config.environment, 'central_agent', True)
        
        # Observation/action adapters
        self.obs_adapter = ObservationAdapter()
        self.action_adapter = ActionAdapter(central_agent=central_agent)
    
    def set_training_results(self, training_results: Optional[Dict[str, Any]]):
        """Set training results for KPI calculation."""
        self.training_results = training_results
    
    def run(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run validation or testing phase.
        
        Testing is ALWAYS sequential: each building is tested individually in its own
        1-building environment, regardless of training mode. Results are aggregated.
        
        Args:
            model_path: Path to model to load (if None, uses current agent state)
            
        Returns:
            Phase results with aggregated KPIs across all buildings
        """
        phase_name = self.phase.upper()
        self.logger.info(f"[{phase_name}] Starting {self.phase} phase")
        self.logger.info(f"[{phase_name}] Buildings: {self.building_ids}")
        self.logger.info(f"[{phase_name}] Episodes per building: {self.episodes}")
        self.logger.info(f"[{phase_name}] Testing mode: Sequential (one building at a time)")
        
        # Set seed for this phase (different from training)
        phase_seed = self.config.seed + self.seed_offset
        self._set_seeds(phase_seed)
        self.logger.info(f"[{phase_name}] Seed: {phase_seed}")
        
        # Load model if provided
        if model_path and model_path.exists():
            try:
                self._load_model(model_path)
                self.logger.info(f"[{phase_name}] Loaded model from {model_path}")
            except Exception as e:
                self.logger.warning(f"[{phase_name}] Could not load model: {e}")
        
        # Test each building individually
        all_agent_data = []
        all_baseline_data = []
        
        start_time, end_time = self.env_manager.wrapper.select_simulation_period()
        
        # Progress bar for buildings
        try:
            from tqdm import tqdm
            building_iterator = tqdm(
                self.building_ids,
                desc=f"[{phase_name}] Testing Buildings",
                unit="building",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        except ImportError:
            building_iterator = self.building_ids
            self.logger.info(f"[{phase_name}] Progress bar not available (install tqdm)")
        
        for building_id in building_iterator:
            self.logger.info(f"[{phase_name}] Testing on building: {building_id}")
            
            # Create 1-building environment for this building
            env = self.env_manager.wrapper.create_environment(
                buildings=[building_id],  # Single building
                start_time=start_time,
                end_time=end_time,
                reward_function=None
            )
            
            # Initialize baseline agent for THIS specific 1-building environment
            baseline_agent = self._create_baseline_for_building(env, building_id)
            
            # Run episodes on this building
            try:
                building_agent_data = self._run_episodes(env, self.agent, "agent")
                building_agent_data['building_id'] = building_id
                all_agent_data.append(building_agent_data)
            except Exception as e:
                self.logger.error(f"[{phase_name}] Agent episodes failed for {building_id}: {e}")
                all_agent_data.append({
                    'building_id': building_id,
                    'agent_type': 'agent',
                    'episodes': 0,
                    'rewards': [],
                    'avg_reward': 0.0,
                    'std_reward': 0.0,
                    'episode_data': [],
                    'aggregated': {}
                })
            
            try:
                building_baseline_data = self._run_episodes(env, baseline_agent, "baseline")
                building_baseline_data['building_id'] = building_id
                all_baseline_data.append(building_baseline_data)
            except Exception as e:
                self.logger.error(f"[{phase_name}] Baseline episodes failed for {building_id}: {e}")
                all_baseline_data.append({
                    'building_id': building_id,
                    'agent_type': 'baseline',
                    'episodes': 0,
                    'rewards': [],
                    'avg_reward': 0.0,
                    'std_reward': 0.0,
                    'episode_data': [],
                    'aggregated': {}
                })
        
        # Aggregate results across all buildings
        agent_data = self._aggregate_building_data(all_agent_data, 'agent')
        baseline_data = self._aggregate_building_data(all_baseline_data, 'baseline')
        
        # Store per-building data for CSV export
        self._last_per_building_agent = all_agent_data
        self._last_per_building_baseline = all_baseline_data
        
        # Calculate KPIs
        import sys
        from pathlib import Path
        
        eval_dir = Path(__file__).parent / 'benchmark_evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        try:
            from metrics_calculator import MetricsCalculator
        finally:
            if str(eval_dir) in sys.path:
                sys.path.remove(str(eval_dir))
        
        calculator = MetricsCalculator(self.logger)
        
        # Calculate KPIs with training data if available
        agent_kpis = calculator.calculate_kpis(agent_data, training_data=self.training_results)
        baseline_kpis = calculator.calculate_kpis(baseline_data)
        
        # Calculate battery value captured
        battery_value = calculator.calculate_battery_value(agent_kpis, baseline_kpis)
        agent_kpis['battery_value_captured'] = battery_value
        
        # Compute savings
        savings = calculator.compute_savings(agent_kpis, baseline_kpis)
        
        results = {
            'phase': self.phase,
            'buildings': self.building_ids,
            'episodes': self.episodes,
            'seed': phase_seed,
            'agent_data': agent_data,
            'baseline_data': baseline_data,
            'agent_kpis': agent_kpis,
            'baseline_kpis': baseline_kpis,
            'savings': savings,
            'status': 'completed',
            'per_building_agent': all_agent_data,  # Keep individual building results
            'per_building_baseline': all_baseline_data
        }
        
        self.logger.info(f"[{phase_name}] Phase complete")
        self._log_summary(results)
        
        # Print per-building performance comparison
        self._print_per_building_summary(all_agent_data, all_baseline_data)
        
        # Export detailed CSV files for analysis
        self._export_detailed_csv(agent_data, baseline_data)
        
        # Export per-building comparison CSV
        self._export_per_building_csv(all_agent_data, all_baseline_data)
        
        # Export comprehensive results for deep analysis (NEW!)
        # This creates a parallel 'results/' folder with ALL episodes and analysis data
        self._export_comprehensive_results(
            agent_data=agent_data,
            baseline_data=baseline_data,
            per_building_agent=all_agent_data,
            per_building_baseline=all_baseline_data
        )
        
        return results
    
    def _run_episodes(self, env, agent, agent_type: str) -> Dict[str, Any]:
        """
        Run episodes and collect detailed data.
        
        Args:
            env: Environment
            agent: Agent to evaluate
            agent_type: "agent" or "baseline"
            
        Returns:
            Episode data dictionary
        """
        self.logger.info(f"  Running {self.episodes} episodes for {agent_type}")
        
        # Disable training mode
        if hasattr(agent, 'is_training'):
            agent.is_training = False
        
        # Storage for episode data
        all_rewards = []
        all_episode_data = []
        
        for episode in range(1, self.episodes + 1):
            episode_data = self._run_single_episode(env, agent, episode)
            all_rewards.append(episode_data['total_reward'])
            all_episode_data.append(episode_data)
            
            if episode % 10 == 0:
                avg_reward = np.mean(all_rewards)
                self.logger.info(f"    Episode {episode}/{self.episodes} | Avg Reward: {avg_reward:.2f}")
        
        # Aggregate data
        aggregated = self._aggregate_episode_data(all_episode_data)
        
        return {
            'agent_type': agent_type,
            'episodes': self.episodes,
            'rewards': all_rewards,
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'episode_data': all_episode_data,
            'aggregated': aggregated
        }
    
    def _run_single_episode(self, env, agent, episode: int) -> Dict[str, Any]:
        """
        Run a single episode and collect detailed data.
        
        Returns:
            Episode data with consumption, PV, cost, SoC, actions
        """
        citylearn_obs = env.reset()
        obs = self.obs_adapter.adapt(citylearn_obs)
        done = False
        total_reward = 0
        
        # Data collection
        timestep_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'net_consumption': [],       # Grid import/export (agent-influenced)
            'non_shiftable_load': [],    # Appliances, lighting, plugs
            'cooling_demand': [],        # Cooling load (if electric)
            'heating_demand': [],        # Heating load (if electric)
            'dhw_demand': [],            # Domestic hot water (if electric)
            'building_load': [],         # Sum of above (total building electrical load)
            'pv_generation': [],
            'electricity_price': [],
            'battery_soc': [],
            'import_grid': [],
            'export_grid': [],
        }
        
        step = 0
        while not done:
            try:
                action = agent.act(obs, deterministic=True)
            except Exception as e:
                if step == 0:
                    error_msg = str(e)
                    if "cannot be multiplied" in error_msg or "mat1 and mat2 shapes" in error_msg:
                        self.logger.warning(f"  [WARN] Dimension mismatch detected!")
                        self.logger.warning(f"  Model was trained with different number of buildings than validation.")
                        self.logger.warning(f"  Parallel-trained models cannot be validated/tested on different building counts.")
                        self.logger.warning(f"  Using zero actions for this evaluation.")
                    else:
                        self.logger.warning(f"  [WARN] Action error: {e}")
                action = [[0.0]] if isinstance(obs, list) else [0.0]
            
            citylearn_action = self.action_adapter.adapt(action)
            
            result = env.step(citylearn_action)
            if len(result) == 5:
                next_citylearn_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_citylearn_obs, reward, done, info = result
            
            next_obs = self.obs_adapter.adapt(next_citylearn_obs)
            
            # Collect data
            timestep_data['observations'].append(obs)
            timestep_data['actions'].append(action)
            
            # Handle reward - convert to scalar
            # CityLearn returns rewards in various formats: scalar, list, nested list
            if isinstance(reward, (list, tuple, np.ndarray)):
                # Flatten completely to handle nested structures
                reward_flat = np.array(reward).flatten()
                step_reward = float(np.sum(reward_flat))
            else:
                step_reward = float(reward)
            
            timestep_data['rewards'].append(step_reward)
            total_reward += step_reward
            
            # Extract data from environment directly
            # CityLearn observations are arrays, not dicts, so we get data from env.buildings
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                building = env.buildings[0]
                
                # Get net consumption from building (grid import/export - agent-influenced)
                if hasattr(building, 'net_electricity_consumption'):
                    net_consumption = float(building.net_electricity_consumption[building.time_step])
                else:
                    net_consumption = 0.0
                
                # Capture individual building load components
                non_shiftable = 0.0
                dhw = 0.0
                cooling = 0.0
                heating = 0.0
                
                # Non-shiftable load (appliances, lighting, plugs)
                if hasattr(building, 'non_shiftable_load') and building.non_shiftable_load is not None:
                    try:
                        non_shiftable = float(building.non_shiftable_load[building.time_step])
                    except (IndexError, TypeError):
                        pass
                
                # DHW demand (if electric)
                if hasattr(building, 'dhw_demand') and building.dhw_demand is not None:
                    try:
                        if hasattr(building, 'dhw_heating_device') and building.dhw_heating_device is not None:
                            dhw = float(building.dhw_demand[building.time_step])
                    except (IndexError, TypeError, AttributeError):
                        pass
                
                # Cooling demand (if electric)
                if hasattr(building, 'cooling_demand') and building.cooling_demand is not None:
                    try:
                        if hasattr(building, 'cooling_device') and building.cooling_device is not None:
                            cooling = float(building.cooling_demand[building.time_step])
                    except (IndexError, TypeError):
                        pass
                
                # Heating demand (if electric)
                if hasattr(building, 'heating_demand') and building.heating_demand is not None:
                    try:
                        if hasattr(building, 'heating_device') and building.heating_device is not None:
                            heating = float(building.heating_demand[building.time_step])
                    except (IndexError, TypeError):
                        pass
                
                # Calculate total building load
                building_load = non_shiftable + dhw + cooling + heating
                
                # Get PV generation
                if hasattr(building, 'solar_generation') and building.solar_generation is not None:
                    pv_gen = float(building.solar_generation[building.time_step])
                else:
                    pv_gen = 0.0
                
                # Get electricity price from pricing object
                if hasattr(building, 'pricing') and building.pricing is not None:
                    price = float(building.pricing.electricity_pricing[building.time_step])
                else:
                    price = 0.22  # Fallback only if no pricing data exists
                
                # Get battery SoC
                if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                    storage = building.electrical_storage
                    if hasattr(storage, 'soc') and len(storage.soc) > building.time_step:
                        soc = float(storage.soc[building.time_step])
                    else:
                        soc = 0.0
                else:
                    soc = 0.0
                
                timestep_data['net_consumption'].append(net_consumption)
                timestep_data['non_shiftable_load'].append(non_shiftable)
                timestep_data['cooling_demand'].append(cooling)
                timestep_data['heating_demand'].append(heating)
                timestep_data['dhw_demand'].append(dhw)
                timestep_data['building_load'].append(building_load)
                timestep_data['pv_generation'].append(pv_gen)
                timestep_data['electricity_price'].append(price)
                timestep_data['battery_soc'].append(soc)
                
                import_val, export_val = self._split_import_export(net_consumption)
                timestep_data['import_grid'].append(import_val)
                timestep_data['export_grid'].append(export_val)
            else:
                # If no building data available, use zeros (shouldn't happen)
                timestep_data['net_consumption'].append(0.0)
                timestep_data['non_shiftable_load'].append(0.0)
                timestep_data['cooling_demand'].append(0.0)
                timestep_data['heating_demand'].append(0.0)
                timestep_data['dhw_demand'].append(0.0)
                timestep_data['building_load'].append(0.0)
                timestep_data['pv_generation'].append(0.0)
                timestep_data['electricity_price'].append(0.0)
                timestep_data['battery_soc'].append(0.0)
                timestep_data['import_grid'].append(0.0)
                timestep_data['export_grid'].append(0.0)
            
            obs = next_obs
            citylearn_obs = next_citylearn_obs
            step += 1
        
        # Convert lists to numpy arrays
        for key in timestep_data:
            if key not in ['observations', 'actions']:
                timestep_data[key] = np.array(timestep_data[key])
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': step,
            'timestep_data': timestep_data
        }
    
    def _split_import_export(self, net_consumption: float) -> Tuple[float, float]:
        """
        Split net consumption into import and export.
        
        Convention: net > 0 = import, net < 0 = export
        
        Args:
            net_consumption: Net consumption (+ = import, - = export)
            
        Returns:
            (import_value, export_value) both as positive numbers
        """
        if net_consumption > 0:
            return net_consumption, 0.0
        else:
            return 0.0, abs(net_consumption)
    
    def _aggregate_episode_data(self, episode_data_list: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate data across all episodes.
        
        Returns:
            Aggregated statistics
        """
        # Collect all timestep data across episodes
        all_net_consumption = []
        all_pv_generation = []
        all_prices = []
        all_battery_soc = []
        all_import_grid = []
        all_export_grid = []
        
        for ep_data in episode_data_list:
            td = ep_data['timestep_data']
            all_net_consumption.extend(td['net_consumption'].tolist())
            all_pv_generation.extend(td['pv_generation'].tolist())
            all_prices.extend(td['electricity_price'].tolist())
            all_battery_soc.extend(td['battery_soc'].tolist())
            all_import_grid.extend(td['import_grid'].tolist())
            all_export_grid.extend(td['export_grid'].tolist())
        
        # Convert to arrays
        all_net_consumption = np.array(all_net_consumption)
        all_pv_generation = np.array(all_pv_generation)
        all_prices = np.array(all_prices)
        all_battery_soc = np.array(all_battery_soc)
        all_import_grid = np.array(all_import_grid)
        all_export_grid = np.array(all_export_grid)
        
        return {
            'total_steps': len(all_net_consumption),
            'net_consumption': all_net_consumption,
            'pv_generation': all_pv_generation,
            'electricity_price': all_prices,
            'battery_soc': all_battery_soc,
            'import_grid': all_import_grid,
            'export_grid': all_export_grid,
        }
    
    def _aggregate_building_data(self, building_data_list: List[Dict[str, Any]], 
                                  agent_type: str) -> Dict[str, Any]:
        """
        Aggregate data from individual building tests into single dataset.
        
        Args:
            building_data_list: List of data dicts from individual buildings
            agent_type: "agent" or "baseline"
            
        Returns:
            Aggregated data dict compatible with existing KPI calculations
        """
        if not building_data_list:
            return {
                'agent_type': agent_type,
                'episodes': 0,
                'rewards': [],
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'episode_data': [],
                'aggregated': {}
            }
        
        # Aggregate rewards across all buildings
        all_rewards = []
        all_episode_data = []
        
        for building_data in building_data_list:
            all_rewards.extend(building_data.get('rewards', []))
            all_episode_data.extend(building_data.get('episode_data', []))
        
        # Compute statistics
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        std_reward = np.std(all_rewards) if all_rewards else 0.0
        
        # Aggregate timestep data across all buildings and episodes
        aggregated = {}
        if all_episode_data:
            all_net_consumption = []
            all_pv_generation = []
            all_prices = []
            all_battery_soc = []
            all_import_grid = []
            all_export_grid = []
            
            for episode in all_episode_data:
                td = episode.get('timestep_data', {})
                if td:
                    all_net_consumption.extend(td.get('net_consumption', []))
                    all_pv_generation.extend(td.get('pv_generation', []))
                    all_prices.extend(td.get('electricity_price', []))
                    all_battery_soc.extend(td.get('battery_soc', []))
                    all_import_grid.extend(td.get('import_grid', []))
                    all_export_grid.extend(td.get('export_grid', []))
            
            if all_net_consumption:
                aggregated = {
                    'total_steps': len(all_net_consumption),
                    'net_consumption': np.array(all_net_consumption),
                    'pv_generation': np.array(all_pv_generation),
                    'electricity_price': np.array(all_prices),
                    'battery_soc': np.array(all_battery_soc),
                    'import_grid': np.array(all_import_grid),
                    'export_grid': np.array(all_export_grid),
                }
        
        return {
            'agent_type': agent_type,
            'episodes': len(all_rewards),
            'rewards': all_rewards,
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'episode_data': all_episode_data,
            'aggregated': aggregated
        }
    
    def _create_baseline_for_building(self, env, building_id: str):
        """
        Create baseline agent for a specific building's environment.
        
        Each building gets its own baseline initialized with that building's environment.
        This ensures the baseline agent expects the same observation/action dimensions
        as the 1-building test environment.
        
        Args:
            env: The 1-building environment for this specific building
            building_id: ID of the building being tested
            
        Returns:
            Baseline agent configured for this building
        """
        try:
            from hems.agents.legacy_adapter import create_agent
            from hems.core.yaml_loader import YAMLConfigLoader
            
            yaml_loader = YAMLConfigLoader()
            sim_config = yaml_loader.to_simulation_config(self.config)
            
            baseline_agent = create_agent(
                agent_type='baseline',
                env=env,  # Use the 1-building environment
                config=sim_config
            )
            self.logger.info(f"  [OK] Baseline agent initialized for {building_id}")
            return baseline_agent
            
        except Exception as e:
            self.logger.warning(f"  [WARN] Could not initialize baseline for {building_id}: {e}")
            # Fallback to dummy baseline
            return DummyBaseline()
    
    def _load_model(self, model_path: Path):
        """Load model from file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if model_data.get('agent_state') and hasattr(self.agent.algorithm, 'load_state'):
            self.agent.algorithm.load_state(model_data['agent_state'])
    
    def _set_seeds(self, seed: int):
        """Set random seeds."""
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def _log_summary(self, results: Dict[str, Any]):
        """Log summary of results."""
        agent_kpis = results['agent_kpis']
        baseline_kpis = results['baseline_kpis']
        savings = results['savings']
        
        self.logger.info(f"\n  === {self.phase.upper()} SUMMARY ===")
        self.logger.info(f"  Agent Reward: {results['agent_data']['avg_reward']:.2f}")
        self.logger.info(f"  Baseline Reward: {results['baseline_data']['avg_reward']:.2f}")
        
        if 'total_cost' in agent_kpis:
            self.logger.info(f"  Agent Cost: â‚¬{agent_kpis['total_cost']:.2f}")
            self.logger.info(f"  Baseline Cost: â‚¬{baseline_kpis['total_cost']:.2f}")
        
        if 'cost_savings_percent' in savings:
            self.logger.info(f"  Cost Savings: {savings['cost_savings_percent']:.1f}%")
    
    def _export_detailed_csv(self, agent_data: Dict[str, Any], baseline_data: Dict[str, Any]):
        """
        Export detailed timestep-by-timestep CSV files for analysis.
        
        Creates one CSV per building containing:
        - timestep, day, hour, building_id
        - agent_action, baseline_action
        - soc, price, load, pv_generation, grid_import, grid_export
        - agent_reward, baseline_reward, agent_cost, baseline_cost
        
        Args:
            agent_data: Aggregated agent data (with per_building_agent in results)
            baseline_data: Aggregated baseline data (with per_building_baseline in results)
        """
        import csv
        from datetime import datetime
        
        csv_dir = self.output_dir / 'csv_exports' / self.phase
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Get agent name safely
        agent_name = getattr(self.agent, 'name', 'agent')
        
        # Get per-building data from parent results if available
        # (results are passed differently, need to get from self if set externally)
        per_building_agent = getattr(self, '_last_per_building_agent', None)
        per_building_baseline = getattr(self, '_last_per_building_baseline', None)
        
        if not per_building_agent or not per_building_baseline:
            self.logger.warning("  [WARN] No per-building data available for CSV export")
            return
        
        # Create mapping of building_id to data
        agent_by_building = {bd['building_id']: bd for bd in per_building_agent}
        baseline_by_building = {bd['building_id']: bd for bd in per_building_baseline}
        
        # Process each building
        for building_id in self.building_ids:
            if building_id not in agent_by_building or building_id not in baseline_by_building:
                self.logger.warning(f"  [WARN] Missing data for CSV export: {building_id}")
                continue
            
            agent_building_data = agent_by_building[building_id]
            baseline_building_data = baseline_by_building[building_id]
            
            csv_file = csv_dir / f"{agent_name}_{building_id}_{self.phase}.csv"
            
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'timestep', 'day', 'hour', 'building_id',
                        'agent_action', 'baseline_action',
                        'soc', 'price_eur_kwh', 
                        'load_kwh', 'pv_generation_kwh', 
                        'grid_import_kwh', 'grid_export_kwh',
                        'agent_reward', 'baseline_reward',
                        'agent_cost_eur', 'baseline_cost_eur'
                    ])
                    
                    # Get episode data for this building
                    agent_episodes = agent_building_data.get('episode_data', [])
                    baseline_episodes = baseline_building_data.get('episode_data', [])
                    
                    if not agent_episodes or not baseline_episodes:
                        self.logger.warning(f"  [WARN] No episode data for CSV export: {building_id}")
                        continue
                    
                    # Use first episode for detailed export
                    agent_ep = agent_episodes[0]['timestep_data']
                    baseline_ep = baseline_episodes[0]['timestep_data']
                    
                    num_steps = len(agent_ep.get('rewards', []))
                    
                    for step in range(num_steps):
                        # Calculate day and hour (assuming 1 timestep = 1 hour)
                        day = step // 24 + 1
                        hour = step % 24
                        
                        # Extract actions (handle different formats)
                        agent_action = agent_ep['actions'][step]
                        if isinstance(agent_action, list):
                            agent_action = agent_action[0][0] if isinstance(agent_action[0], list) else agent_action[0]
                        
                        baseline_action = baseline_ep['actions'][step]
                        if isinstance(baseline_action, list):
                            baseline_action = baseline_action[0][0] if isinstance(baseline_action[0], list) else baseline_action[0]
                        
                        # Get data
                        soc = agent_ep['battery_soc'][step]
                        price = agent_ep['electricity_price'][step]
                        load = agent_ep['net_consumption'][step]  # Actually net_consumption
                        pv_gen = agent_ep['pv_generation'][step]
                        grid_import = agent_ep['import_grid'][step]
                        grid_export = agent_ep['export_grid'][step]
                        agent_reward = agent_ep['rewards'][step]
                        baseline_reward = baseline_ep['rewards'][step]
                        
                        # Calculate costs separately for agent and baseline
                        # Agent uses its own import/export
                        agent_import = agent_ep['import_grid'][step]
                        agent_export = agent_ep['export_grid'][step]
                        agent_cost = (agent_import * price) - (agent_export * price * 0.5)
                        
                        # Baseline uses its own import/export
                        baseline_import = baseline_ep['import_grid'][step]
                        baseline_export = baseline_ep['export_grid'][step]
                        baseline_cost = (baseline_import * price) - (baseline_export * price * 0.5)
                        
                        # Write row
                        writer.writerow([
                            step, day, hour, building_id,
                            f"{agent_action:.6f}", f"{baseline_action:.6f}",
                            f"{soc:.4f}", f"{price:.4f}",
                            f"{load:.4f}", f"{pv_gen:.4f}",
                            f"{grid_import:.4f}", f"{grid_export:.4f}",
                            f"{agent_reward:.4f}", f"{baseline_reward:.4f}",
                            f"{agent_cost:.4f}", f"{baseline_cost:.4f}"
                        ])
                
                self.logger.info(f"  [CSV] Exported detailed data to {csv_file.name}")
                
            except Exception as e:
                self.logger.warning(f"  [WARN] Failed to export CSV for {building_id}: {e}")
    
    def _export_comprehensive_results(
        self, 
        agent_data: Dict[str, Any], 
        baseline_data: Dict[str, Any],
        per_building_agent: List[Dict[str, Any]],
        per_building_baseline: List[Dict[str, Any]]
    ):
        """
        Export comprehensive results for deep analysis.
        
        Creates a 'results/' folder parallel to existing exports with:
        - Full pickle data (all episodes)
        - Episode-level summaries
        - Timestep-level data
        - Behavioral analysis
        - Building comparisons
        
        This does NOT modify existing plots/, csv_exports/, or benchmark_results.json
        
        Args:
            agent_data: Aggregated agent data
            baseline_data: Aggregated baseline data
            per_building_agent: Per-building agent data
            per_building_baseline: Per-building baseline data
        """
        import json
        import pickle
        import pandas as pd
        from datetime import datetime
        
        # Create results directory parallel to existing structure
        results_dir = self.output_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("[RESULTS] Exporting comprehensive analysis data...")
        
        # =====================================================================
        # 1. FULL DATA (Pickle for reloading)
        # =====================================================================
        full_data_dir = results_dir / 'full_data'
        full_data_dir.mkdir(exist_ok=True)
        
        # Save complete results as pickle
        full_results = {
            'agent_data': agent_data,
            'baseline_data': baseline_data,
            'per_building_agent': per_building_agent,
            'per_building_baseline': per_building_baseline,
            'metadata': {
                'phase': self.phase,
                'num_episodes': len(agent_data.get('episode_data', [])),
                'num_buildings': len(per_building_agent),
                'building_ids': [b['building_id'] for b in per_building_agent],
                'export_timestamp': datetime.now().isoformat()
            }
        }
        
        pkl_file = full_data_dir / 'benchmark_results.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(full_results, f)
        self.logger.info(f"  [SAVED] Full data: {pkl_file.name}")
        
        # Save metadata as JSON
        metadata_file = full_data_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(full_results['metadata'], f, indent=2)
        
        # =====================================================================
        # 2. EPISODE ANALYSIS
        # =====================================================================
        episode_dir = results_dir / 'episode_analysis'
        episode_dir.mkdir(exist_ok=True)
        
        episode_summaries = []
        for building_data in per_building_agent:
            building_id = building_data['building_id']
            episodes = building_data.get('episode_data', [])
            
            for ep_num, episode in enumerate(episodes):
                td = episode.get('timestep_data', {})
                
                summary = {
                    'building_id': building_id,
                    'episode_num': ep_num,
                    'total_reward': sum(td.get('rewards', [])),
                    'avg_reward': np.mean(td.get('rewards', [0])),
                    'total_cost_eur': self._calculate_episode_cost_from_timesteps(td),
                    'total_import_kwh': sum(td.get('import_grid', [])),
                    'total_export_kwh': sum(td.get('export_grid', [])),
                    'total_pv_generation_kwh': sum(np.maximum(0, -np.array(td.get('pv_generation', [])))),
                    'avg_soc': np.mean(td.get('battery_soc', [0])),
                    'min_soc': np.min(td.get('battery_soc', [0])),
                    'max_soc': np.max(td.get('battery_soc', [1])),
                    'num_timesteps': len(td.get('rewards', []))
                }
                episode_summaries.append(summary)
        
        # Save episode summaries
        df_episodes = pd.DataFrame(episode_summaries)
        episodes_file = episode_dir / 'episode_summaries.csv'
        df_episodes.to_csv(episodes_file, index=False)
        self.logger.info(f"  [SAVED] Episode summaries: {len(episode_summaries)} episodes")
        
        # Save best/worst/median episodes info
        df_episodes_sorted = df_episodes.sort_values('total_cost_eur')
        best_worst = {
            'best_episode': {
                'building_id': df_episodes_sorted.iloc[0]['building_id'],
                'episode_num': int(df_episodes_sorted.iloc[0]['episode_num']),
                'cost': float(df_episodes_sorted.iloc[0]['total_cost_eur'])
            },
            'worst_episode': {
                'building_id': df_episodes_sorted.iloc[-1]['building_id'],
                'episode_num': int(df_episodes_sorted.iloc[-1]['episode_num']),
                'cost': float(df_episodes_sorted.iloc[-1]['total_cost_eur'])
            },
            'median_episode': {
                'building_id': df_episodes_sorted.iloc[len(df_episodes_sorted)//2]['building_id'],
                'episode_num': int(df_episodes_sorted.iloc[len(df_episodes_sorted)//2]['episode_num']),
                'cost': float(df_episodes_sorted.iloc[len(df_episodes_sorted)//2]['total_cost_eur'])
            }
        }
        with open(episode_dir / 'best_worst_median_episodes.json', 'w') as f:
            json.dump(best_worst, f, indent=2)
        
        # =====================================================================
        # 3. TIMESTEP ANALYSIS (ALL timesteps from ALL episodes)
        # =====================================================================
        timestep_dir = results_dir / 'timestep_analysis'
        timestep_dir.mkdir(exist_ok=True)
        
        # Aggregate ALL timesteps from ALL episodes
        all_agent_timesteps = []
        all_baseline_timesteps = []
        
        for agent_bldg, baseline_bldg in zip(per_building_agent, per_building_baseline):
            building_id = agent_bldg['building_id']
            
            for ep_num, (agent_ep, baseline_ep) in enumerate(
                zip(agent_bldg.get('episode_data', []), baseline_bldg.get('episode_data', []))
            ):
                agent_td = agent_ep.get('timestep_data', {})
                baseline_td = baseline_ep.get('timestep_data', {})
                
                num_steps = len(agent_td.get('rewards', []))
                
                for step in range(num_steps):
                    # Agent timestep
                    agent_row = {
                        'building_id': building_id,
                        'episode_num': ep_num,
                        'timestep': step,
                        'day': step // 24 + 1,
                        'hour': step % 24,
                        'action': self._extract_scalar(agent_td.get('actions', [])[step]) if step < len(agent_td.get('actions', [])) else 0,
                        'reward': agent_td.get('rewards', [])[step] if step < len(agent_td.get('rewards', [])) else 0,
                        'soc': agent_td.get('battery_soc', [])[step] if step < len(agent_td.get('battery_soc', [])) else 0,
                        'price_eur_kwh': agent_td.get('electricity_price', [])[step] if step < len(agent_td.get('electricity_price', [])) else 0,
                        'pv_generation_kwh': -agent_td.get('pv_generation', [])[step] if step < len(agent_td.get('pv_generation', [])) else 0,
                        'grid_import_kwh': agent_td.get('import_grid', [])[step] if step < len(agent_td.get('import_grid', [])) else 0,
                        'grid_export_kwh': agent_td.get('export_grid', [])[step] if step < len(agent_td.get('export_grid', [])) else 0,
                        'net_consumption_kwh': agent_td.get('net_consumption', [])[step] if step < len(agent_td.get('net_consumption', [])) else 0,
                        'non_shiftable_load_kwh': agent_td.get('non_shiftable_load', [])[step] if step < len(agent_td.get('non_shiftable_load', [])) else 0,
                        'cooling_demand_kwh': agent_td.get('cooling_demand', [])[step] if step < len(agent_td.get('cooling_demand', [])) else 0,
                        'heating_demand_kwh': agent_td.get('heating_demand', [])[step] if step < len(agent_td.get('heating_demand', [])) else 0,
                        'dhw_demand_kwh': agent_td.get('dhw_demand', [])[step] if step < len(agent_td.get('dhw_demand', [])) else 0,
                        'building_load_kwh': agent_td.get('building_load', [])[step] if step < len(agent_td.get('building_load', [])) else 0
                    }
                    all_agent_timesteps.append(agent_row)
                    
                    # Baseline timestep
                    baseline_row = {
                        'building_id': building_id,
                        'episode_num': ep_num,
                        'timestep': step,
                        'day': step // 24 + 1,
                        'hour': step % 24,
                        'action': self._extract_scalar(baseline_td.get('actions', [])[step]) if step < len(baseline_td.get('actions', [])) else 0,
                        'reward': baseline_td.get('rewards', [])[step] if step < len(baseline_td.get('rewards', [])) else 0,
                        'soc': baseline_td.get('battery_soc', [])[step] if step < len(baseline_td.get('battery_soc', [])) else 0,
                        'price_eur_kwh': baseline_td.get('electricity_price', [])[step] if step < len(baseline_td.get('electricity_price', [])) else 0,
                        'pv_generation_kwh': -baseline_td.get('pv_generation', [])[step] if step < len(baseline_td.get('pv_generation', [])) else 0,
                        'grid_import_kwh': baseline_td.get('import_grid', [])[step] if step < len(baseline_td.get('import_grid', [])) else 0,
                        'grid_export_kwh': baseline_td.get('export_grid', [])[step] if step < len(baseline_td.get('export_grid', [])) else 0,
                        'net_consumption_kwh': baseline_td.get('net_consumption', [])[step] if step < len(baseline_td.get('net_consumption', [])) else 0,
                        'non_shiftable_load_kwh': baseline_td.get('non_shiftable_load', [])[step] if step < len(baseline_td.get('non_shiftable_load', [])) else 0,
                        'cooling_demand_kwh': baseline_td.get('cooling_demand', [])[step] if step < len(baseline_td.get('cooling_demand', [])) else 0,
                        'heating_demand_kwh': baseline_td.get('heating_demand', [])[step] if step < len(baseline_td.get('heating_demand', [])) else 0,
                        'dhw_demand_kwh': baseline_td.get('dhw_demand', [])[step] if step < len(baseline_td.get('dhw_demand', [])) else 0,
                        'building_load_kwh': baseline_td.get('building_load', [])[step] if step < len(baseline_td.get('building_load', [])) else 0
                    }
                    all_baseline_timesteps.append(baseline_row)
        
        # Save all timesteps (WARNING: These will be large files!)
        df_agent_timesteps = pd.DataFrame(all_agent_timesteps)
        df_baseline_timesteps = pd.DataFrame(all_baseline_timesteps)
        
        agent_timesteps_file = timestep_dir / 'all_timesteps_agent.csv'
        baseline_timesteps_file = timestep_dir / 'all_timesteps_baseline.csv'
        
        df_agent_timesteps.to_csv(agent_timesteps_file, index=False)
        df_baseline_timesteps.to_csv(baseline_timesteps_file, index=False)
        
        self.logger.info(f"  [SAVED] All timesteps: {len(all_agent_timesteps)} rows (agent + baseline)")
        
        # =====================================================================
        # 4. BEHAVIORAL ANALYSIS
        # =====================================================================
        behavioral_dir = results_dir / 'behavioral_analysis'
        behavioral_dir.mkdir(exist_ok=True)
        
        # Analyze action patterns
        action_patterns = self._analyze_action_patterns(df_agent_timesteps, df_baseline_timesteps)
        action_file = behavioral_dir / 'action_patterns.csv'
        pd.DataFrame(action_patterns).to_csv(action_file, index=False)
        self.logger.info(f"  [SAVED] Action patterns analysis")
        
        # Analyze price-action coherence
        coherence = self._analyze_price_action_coherence(df_agent_timesteps)
        coherence_file = behavioral_dir / 'price_action_coherence.csv'
        pd.DataFrame(coherence).to_csv(coherence_file, index=False)
        
        # Analyze PV-action coherence
        pv_coherence = self._analyze_pv_action_coherence(df_agent_timesteps)
        pv_file = behavioral_dir / 'pv_action_coherence.csv'
        pd.DataFrame(pv_coherence).to_csv(pv_file, index=False)
        
        # =====================================================================
        # 5. BUILDING ANALYSIS
        # =====================================================================
        building_dir = results_dir / 'building_analysis'
        building_dir.mkdir(exist_ok=True)
        
        # Per-building summaries
        building_summaries = []
        for building_id in set(df_episodes['building_id']):
            building_episodes = df_episodes[df_episodes['building_id'] == building_id]
            
            summary = {
                'building_id': building_id,
                'num_episodes': len(building_episodes),
                'avg_cost_eur': building_episodes['total_cost_eur'].mean(),
                'std_cost_eur': building_episodes['total_cost_eur'].std(),
                'min_cost_eur': building_episodes['total_cost_eur'].min(),
                'max_cost_eur': building_episodes['total_cost_eur'].max(),
                'avg_reward': building_episodes['total_reward'].mean(),
                'avg_import_kwh': building_episodes['total_import_kwh'].mean(),
                'avg_export_kwh': building_episodes['total_export_kwh'].mean(),
                'avg_pv_generation_kwh': building_episodes['total_pv_generation_kwh'].mean()
            }
            building_summaries.append(summary)
        
        df_buildings = pd.DataFrame(building_summaries)
        df_buildings = df_buildings.sort_values('avg_cost_eur')
        df_buildings['rank'] = range(1, len(df_buildings) + 1)
        
        buildings_file = building_dir / 'per_building_summaries.csv'
        df_buildings.to_csv(buildings_file, index=False)
        self.logger.info(f"  [SAVED] Per-building summaries: {len(building_summaries)} buildings")
        
        # =====================================================================
        # 6. DETAILED EPISODES (ALL episodes, organized by building)
        # =====================================================================
        detailed_dir = results_dir / 'detailed_episodes'
        detailed_dir.mkdir(exist_ok=True)
        
        for building_data in per_building_agent:
            building_id = building_data['building_id']
            building_folder = detailed_dir / building_id
            building_folder.mkdir(exist_ok=True)
            
            episodes = building_data.get('episode_data', [])
            for ep_num, episode in enumerate(episodes):
                td = episode.get('timestep_data', {})
                
                # Create episode CSV
                episode_rows = []
                num_steps = len(td.get('rewards', []))
                
                for step in range(num_steps):
                    row = {
                        'timestep': step,
                        'day': step // 24 + 1,
                        'hour': step % 24,
                        'action': self._extract_scalar(td.get('actions', [])[step]) if step < len(td.get('actions', [])) else 0,
                        'reward': td.get('rewards', [])[step] if step < len(td.get('rewards', [])) else 0,
                        'soc': td.get('battery_soc', [])[step] if step < len(td.get('battery_soc', [])) else 0,
                        'price_eur_kwh': td.get('electricity_price', [])[step] if step < len(td.get('electricity_price', [])) else 0,
                        'pv_generation_kwh': max(0, -td.get('pv_generation', [])[step]) if step < len(td.get('pv_generation', [])) else 0,
                        'grid_import_kwh': td.get('import_grid', [])[step] if step < len(td.get('import_grid', [])) else 0,
                        'grid_export_kwh': td.get('export_grid', [])[step] if step < len(td.get('export_grid', [])) else 0
                    }
                    episode_rows.append(row)
                
                df_ep = pd.DataFrame(episode_rows)
                ep_file = building_folder / f'episode_{ep_num:02d}.csv'
                df_ep.to_csv(ep_file, index=False)
        
        self.logger.info(f"  [SAVED] Detailed episodes: {len(per_building_agent)} buildings × 20 episodes")
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        self.logger.info("[RESULTS] Comprehensive export complete!")
        self.logger.info(f"  Location: {results_dir}")
        self.logger.info(f"  Episodes: {len(episode_summaries)}")
        self.logger.info(f"  Timesteps: {len(all_agent_timesteps)}")
        self.logger.info(f"  Buildings: {len(building_summaries)}")
    
    def _extract_scalar(self, value):
        """Extract scalar from nested lists/arrays."""
        if isinstance(value, (list, np.ndarray)):
            while isinstance(value, (list, np.ndarray)) and len(value) > 0:
                value = value[0]
        return float(value) if value is not None else 0.0
    
    def _calculate_episode_cost_from_timesteps(self, timestep_data: Dict[str, Any]) -> float:
        """Calculate cost from timestep data."""
        import_grid = np.array(timestep_data.get('import_grid', []))
        export_grid = np.array(timestep_data.get('export_grid', []))
        prices = np.array(timestep_data.get('electricity_price', []))
        
        if len(import_grid) == 0 or len(prices) == 0:
            return 0.0
        
        import_cost = np.sum(import_grid * prices)
        export_revenue = np.sum(export_grid * prices) * 0.5  # Assuming 50% export price
        
        return import_cost - export_revenue
    
    def _analyze_action_patterns(self, df_agent, df_baseline):
        """
        Analyze when agent charges/discharges with complete price categorization.
        
        Price categories:
        - LOW: Bottom 25% of prices (< 25th percentile)
        - MEDIUM: Middle 50% of prices (25th-75th percentile)
        - HIGH: Top 25% of prices (> 75th percentile)
        
        Thresholds are calculated PER BUILDING to account for different price distributions.
        """
        patterns = []
        
        for building_id in df_agent['building_id'].unique():
            df_bldg = df_agent[df_agent['building_id'] == building_id].copy()
            
            # Calculate thresholds PER BUILDING (not globally)
            price_low = df_bldg['price_eur_kwh'].quantile(0.25)
            price_high = df_bldg['price_eur_kwh'].quantile(0.75)
            pv_low = df_bldg['pv_generation_kwh'].quantile(0.25)
            pv_high = df_bldg['pv_generation_kwh'].quantile(0.75)
            
            # Categorize ALL timesteps by price
            df_bldg['price_category'] = 'medium'  # Default
            df_bldg.loc[df_bldg['price_eur_kwh'] < price_low, 'price_category'] = 'low'
            df_bldg.loc[df_bldg['price_eur_kwh'] > price_high, 'price_category'] = 'high'
            
            # Categorize ALL timesteps by PV
            df_bldg['pv_category'] = 'medium'  # Default
            df_bldg.loc[df_bldg['pv_generation_kwh'] < pv_low, 'pv_category'] = 'low'
            df_bldg.loc[df_bldg['pv_generation_kwh'] > pv_high, 'pv_category'] = 'high'
            
            # Identify action types
            charge_mask = df_bldg['action'] > 0.1
            discharge_mask = df_bldg['action'] < -0.1
            hold_mask = (df_bldg['action'] >= -0.1) & (df_bldg['action'] <= 0.1)
            
            num_charge = int(charge_mask.sum())
            num_discharge = int(discharge_mask.sum())
            num_hold = int(hold_mask.sum())
            
            # Charge actions by price category
            charge_at_low_price = int((charge_mask & (df_bldg['price_category'] == 'low')).sum())
            charge_at_medium_price = int((charge_mask & (df_bldg['price_category'] == 'medium')).sum())
            charge_at_high_price = int((charge_mask & (df_bldg['price_category'] == 'high')).sum())
            
            # Charge actions by PV category
            charge_at_low_pv = int((charge_mask & (df_bldg['pv_category'] == 'low')).sum())
            charge_at_medium_pv = int((charge_mask & (df_bldg['pv_category'] == 'medium')).sum())
            charge_at_high_pv = int((charge_mask & (df_bldg['pv_category'] == 'high')).sum())
            
            # Discharge actions by price category
            discharge_at_low_price = int((discharge_mask & (df_bldg['price_category'] == 'low')).sum())
            discharge_at_medium_price = int((discharge_mask & (df_bldg['price_category'] == 'medium')).sum())
            discharge_at_high_price = int((discharge_mask & (df_bldg['price_category'] == 'high')).sum())
            
            pattern = {
                'building_id': building_id,
                'total_timesteps': len(df_bldg),
                
                # Action counts
                'num_charge_actions': num_charge,
                'num_discharge_actions': num_discharge,
                'num_hold_actions': num_hold,
                
                # Charge by price (counts)
                'charge_at_low_price': charge_at_low_price,
                'charge_at_medium_price': charge_at_medium_price,
                'charge_at_high_price': charge_at_high_price,
                
                # Charge by price (percentages)
                'pct_charge_at_low_price': round((charge_at_low_price / num_charge * 100), 1) if num_charge > 0 else 0,
                'pct_charge_at_medium_price': round((charge_at_medium_price / num_charge * 100), 1) if num_charge > 0 else 0,
                'pct_charge_at_high_price': round((charge_at_high_price / num_charge * 100), 1) if num_charge > 0 else 0,
                
                # Charge by PV (counts)
                'charge_at_low_pv': charge_at_low_pv,
                'charge_at_medium_pv': charge_at_medium_pv,
                'charge_at_high_pv': charge_at_high_pv,
                
                # Charge by PV (percentages)
                'pct_charge_at_low_pv': round((charge_at_low_pv / num_charge * 100), 1) if num_charge > 0 else 0,
                'pct_charge_at_medium_pv': round((charge_at_medium_pv / num_charge * 100), 1) if num_charge > 0 else 0,
                'pct_charge_at_high_pv': round((charge_at_high_pv / num_charge * 100), 1) if num_charge > 0 else 0,
                
                # Discharge by price (counts)
                'discharge_at_low_price': discharge_at_low_price,
                'discharge_at_medium_price': discharge_at_medium_price,
                'discharge_at_high_price': discharge_at_high_price,
                
                # Discharge by price (percentages)
                'pct_discharge_at_low_price': round((discharge_at_low_price / num_discharge * 100), 1) if num_discharge > 0 else 0,
                'pct_discharge_at_medium_price': round((discharge_at_medium_price / num_discharge * 100), 1) if num_discharge > 0 else 0,
                'pct_discharge_at_high_price': round((discharge_at_high_price / num_discharge * 100), 1) if num_discharge > 0 else 0,
                
                # Thresholds used (for reference)
                'price_low_threshold_eur': round(float(price_low), 4),
                'price_high_threshold_eur': round(float(price_high), 4),
                'pv_low_threshold_kwh': round(float(pv_low), 2),
                'pv_high_threshold_kwh': round(float(pv_high), 2)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_price_action_coherence(self, df_agent):
        """Analyze coherence between price signals and actions."""
        coherence = []
        
        price_low = df_agent['price_eur_kwh'].quantile(0.25)
        price_high = df_agent['price_eur_kwh'].quantile(0.75)
        
        for building_id in df_agent['building_id'].unique():
            df_bldg = df_agent[df_agent['building_id'] == building_id]
            
            # Count "good" decisions
            good_decisions = 0
            total_decisions = 0
            
            for _, row in df_bldg.iterrows():
                if abs(row['action']) > 0.1:  # Significant action
                    total_decisions += 1
                    
                    if row['action'] > 0.1 and row['price_eur_kwh'] < price_low:
                        # Charging at low price - GOOD
                        good_decisions += 1
                    elif row['action'] < -0.1 and row['price_eur_kwh'] > price_high:
                        # Discharging at high price - GOOD
                        good_decisions += 1
            
            coherence_score = (good_decisions / total_decisions * 100) if total_decisions > 0 else 0
            
            coherence.append({
                'building_id': building_id,
                'total_decisions': total_decisions,
                'good_decisions': good_decisions,
                'bad_decisions': total_decisions - good_decisions,
                'coherence_score_pct': coherence_score
            })
        
        return coherence
    
    def _analyze_pv_action_coherence(self, df_agent):
        """Analyze coherence between PV generation and actions."""
        coherence = []
        
        pv_high = df_agent['pv_generation_kwh'].quantile(0.75)
        
        for building_id in df_agent['building_id'].unique():
            df_bldg = df_agent[df_agent['building_id'] == building_id]
            
            # Count decisions that align with PV
            good_pv_decisions = 0
            total_charge_decisions = 0
            
            for _, row in df_bldg.iterrows():
                if row['action'] > 0.1:  # Charging
                    total_charge_decisions += 1
                    if row['pv_generation_kwh'] > pv_high:
                        # Charging when PV is high - GOOD
                        good_pv_decisions += 1
            
            pv_coherence_score = (good_pv_decisions / total_charge_decisions * 100) if total_charge_decisions > 0 else 0
            
            coherence.append({
                'building_id': building_id,
                'total_charge_decisions': total_charge_decisions,
                'charge_with_high_pv': good_pv_decisions,
                'charge_with_low_pv': total_charge_decisions - good_pv_decisions,
                'pv_coherence_score_pct': pv_coherence_score
            })
        
        return coherence
    
    def _calculate_cost_from_episodes(self, building_data: Dict[str, Any]) -> float:
        """
        Calculate total cost from episode timestep data.
        
        Args:
            building_data: Building data with episode_data list
            
        Returns:
            Total cost in euros
        """
        total_cost = 0.0
        
        episode_data_list = building_data.get('episode_data', [])
        if not episode_data_list:
            return 0.0
        
        for episode in episode_data_list:
            timestep_data = episode.get('timestep_data', {})
            
            # Get arrays
            import_grid = np.array(timestep_data.get('import_grid', []))
            export_grid = np.array(timestep_data.get('export_grid', []))
            prices = np.array(timestep_data.get('electricity_price', []))
            
            # Calculate cost: import_cost - export_revenue
            if len(import_grid) > 0 and len(prices) > 0:
                import_cost = np.sum(import_grid * prices)
                export_revenue = np.sum(export_grid * prices)
                episode_cost = import_cost - export_revenue
                total_cost += episode_cost
        
        # Average across episodes
        if len(episode_data_list) > 0:
            total_cost = total_cost / len(episode_data_list)
        
        return total_cost
    
    def _calculate_building_metrics(self, building_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate import, export, and PV self-consumption metrics from episode data.
        
        Args:
            building_data: Building data with episode_data list
            
        Returns:
            Dict with total_import, total_export, pv_self_consumption
        """
        total_import = 0.0
        total_export = 0.0
        total_pv_self_consumption = 0.0
        
        episode_data_list = building_data.get('episode_data', [])
        if not episode_data_list:
            return {'total_import': 0.0, 'total_export': 0.0, 'pv_self_consumption': 0.0}
        
        for episode in episode_data_list:
            timestep_data = episode.get('timestep_data', {})
            
            # Get arrays
            import_grid = np.array(timestep_data.get('import_grid', []))
            export_grid = np.array(timestep_data.get('export_grid', []))
            net_consumption = np.array(timestep_data.get('net_consumption', []))
            pv_gen_raw = np.array(timestep_data.get('pv_generation', []))
            
            # PV is negative in CityLearn, rectify it
            pv_gen = np.maximum(0, -pv_gen_raw)
            
            # Calculate consumption (as per user's correction: |net|)
            consumption = np.abs(net_consumption)
            
            # PV self-consumption = min(PV generated, consumption)
            # This represents how much PV is used directly by the building
            pv_self_consumed = np.minimum(pv_gen, consumption)
            
            # Sum for this episode
            total_import += np.sum(import_grid)
            total_export += np.sum(export_grid)
            total_pv_self_consumption += np.sum(pv_self_consumed)
        
        # Average across episodes
        num_episodes = len(episode_data_list)
        if num_episodes > 0:
            total_import /= num_episodes
            total_export /= num_episodes
            total_pv_self_consumption /= num_episodes
        
        return {
            'total_import': total_import,
            'total_export': total_export,
            'pv_self_consumption': total_pv_self_consumption
        }
    
    def _print_per_building_summary(
        self, 
        per_building_agent: List[Dict[str, Any]], 
        per_building_baseline: List[Dict[str, Any]]
    ):
        """
        Print formatted per-building performance comparison.
        
        Args:
            per_building_agent: List of agent results (each with 'building_id')
            per_building_baseline: List of baseline results (each with 'building_id')
        """
        self.logger.info("\n  === PER-BUILDING PERFORMANCE ===")
        
        # Calculate metrics for each building
        building_comparisons = []
        
        # Match agent and baseline data by building_id
        for agent_data in per_building_agent:
            building_id = agent_data.get('building_id', 'Unknown')
            
            # Find matching baseline data
            baseline_data = None
            for b_data in per_building_baseline:
                if b_data.get('building_id') == building_id:
                    baseline_data = b_data
                    break
            
            if not baseline_data:
                self.logger.warning(f"  No baseline data found for {building_id}")
                continue
            
            agent_reward = agent_data.get('avg_reward', 0)
            baseline_reward = baseline_data.get('avg_reward', 0)
            
            # Calculate cost from timestep data (since aggregated doesn't have KPIs)
            agent_cost = self._calculate_cost_from_episodes(agent_data)
            baseline_cost = self._calculate_cost_from_episodes(baseline_data)
            
            # Calculate savings
            if baseline_cost > 0:
                cost_savings = ((baseline_cost - agent_cost) / baseline_cost) * 100
            else:
                cost_savings = 0
            
            if baseline_reward != 0:
                reward_improvement = ((agent_reward - baseline_reward) / abs(baseline_reward)) * 100
            else:
                reward_improvement = 0
            
            building_comparisons.append({
                'building_id': building_id,
                'agent_reward': agent_reward,
                'baseline_reward': baseline_reward,
                'reward_improvement': reward_improvement,
                'agent_cost': agent_cost,
                'baseline_cost': baseline_cost,
                'cost_savings': cost_savings
            })
            
            # Format status indicator
            if cost_savings > 0:
                status = "✓"
            else:
                status = "✗"
            
            # Log this building
            self.logger.info(
                f"  {building_id}: "
                f"Agent={agent_reward:.2f}, Baseline={baseline_reward:.2f}, "
                f"Cost: €{agent_cost:.2f} vs €{baseline_cost:.2f}, "
                f"Savings={cost_savings:+.1f}% {status}"
            )
        
        # Find best and worst performers
        if building_comparisons:
            best_building = max(building_comparisons, key=lambda x: x['cost_savings'])
            worst_building = min(building_comparisons, key=lambda x: x['cost_savings'])
            
            self.logger.info(f"\n  Best Building: {best_building['building_id']} ({best_building['cost_savings']:+.1f}% savings)")
            self.logger.info(f"  Worst Building: {worst_building['building_id']} ({worst_building['cost_savings']:+.1f}% savings)")
            
            # Calculate averages
            avg_savings = np.mean([b['cost_savings'] for b in building_comparisons])
            avg_reward_improvement = np.mean([b['reward_improvement'] for b in building_comparisons])
            
            self.logger.info(f"\n  Average Cost Savings: {avg_savings:+.1f}%")
            self.logger.info(f"  Average Reward Improvement: {avg_reward_improvement:+.1f}%")
    
    def _export_per_building_csv(
        self,
        per_building_agent: List[Dict[str, Any]],
        per_building_baseline: List[Dict[str, Any]]
    ):
        """
        Export per-building comparison to CSV file.
        
        Args:
            per_building_agent: List of agent results (each with 'building_id')
            per_building_baseline: List of baseline results (each with 'building_id')
        """
        import csv
        
        phase_name = self.phase.upper()
        csv_file = self.output_dir / f"per_building_comparison_{self.phase}.csv"
        
        try:
            # First, collect all data
            building_rows = []
            
            # Match agent and baseline data by building_id
            for agent_data in per_building_agent:
                building_id = agent_data.get('building_id', 'Unknown')
                
                # Find matching baseline data
                baseline_data = None
                for b_data in per_building_baseline:
                    if b_data.get('building_id') == building_id:
                        baseline_data = b_data
                        break
                
                if not baseline_data:
                    continue
                
                agent_reward = agent_data.get('avg_reward', 0)
                baseline_reward = baseline_data.get('avg_reward', 0)
                
                # Calculate costs from timestep data
                agent_cost = self._calculate_cost_from_episodes(agent_data)
                baseline_cost = self._calculate_cost_from_episodes(baseline_data)
                
                # Calculate additional metrics from episode data
                agent_metrics = self._calculate_building_metrics(agent_data)
                baseline_metrics = self._calculate_building_metrics(baseline_data)
                
                # Calculate improvements
                if baseline_cost > 0:
                    cost_savings = ((baseline_cost - agent_cost) / baseline_cost) * 100
                else:
                    cost_savings = 0
                
                if baseline_reward != 0:
                    reward_improvement = ((agent_reward - baseline_reward) / abs(baseline_reward)) * 100
                else:
                    reward_improvement = 0
                
                # Get episode counts
                agent_episodes = agent_data.get('episodes', 0)
                baseline_episodes = baseline_data.get('episodes', 0)
                
                building_rows.append({
                    'building_id': building_id,
                    'agent_reward': agent_reward,
                    'baseline_reward': baseline_reward,
                    'reward_improvement': reward_improvement,
                    'agent_cost': agent_cost,
                    'baseline_cost': baseline_cost,
                    'cost_savings': cost_savings,
                    'agent_import': agent_metrics['total_import'],
                    'baseline_import': baseline_metrics['total_import'],
                    'agent_export': agent_metrics['total_export'],
                    'baseline_export': baseline_metrics['total_export'],
                    'agent_pv_self': agent_metrics['pv_self_consumption'],
                    'baseline_pv_self': baseline_metrics['pv_self_consumption'],
                    'agent_episodes': agent_episodes,
                    'baseline_episodes': baseline_episodes
                })
            
            # Sort by cost savings (descending - best first)
            building_rows.sort(key=lambda x: x['cost_savings'], reverse=True)
            
            # Write to CSV
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'rank',
                    'building_id',
                    'agent_reward',
                    'baseline_reward',
                    'reward_improvement_%',
                    'agent_cost_eur',
                    'baseline_cost_eur',
                    'cost_savings_%',
                    'agent_import_kwh',
                    'baseline_import_kwh',
                    'agent_export_kwh',
                    'baseline_export_kwh',
                    'agent_pv_self_consumption_kwh',
                    'baseline_pv_self_consumption_kwh',
                    'agent_episodes',
                    'baseline_episodes'
                ])
                
                # Write sorted data with rank
                for rank, row in enumerate(building_rows, 1):
                    writer.writerow([
                        rank,
                        row['building_id'],
                        f"{row['agent_reward']:.4f}",
                        f"{row['baseline_reward']:.4f}",
                        f"{row['reward_improvement']:.2f}",
                        f"{row['agent_cost']:.2f}",
                        f"{row['baseline_cost']:.2f}",
                        f"{row['cost_savings']:.2f}",
                        f"{row['agent_import']:.2f}",
                        f"{row['baseline_import']:.2f}",
                        f"{row['agent_export']:.2f}",
                        f"{row['baseline_export']:.2f}",
                        f"{row['agent_pv_self']:.2f}",
                        f"{row['baseline_pv_self']:.2f}",
                        row['agent_episodes'],
                        row['baseline_episodes']
                    ])
            
            self.logger.info(f"  [CSV] Exported per-building comparison to {csv_file.name}")
            
        except Exception as e:
            self.logger.warning(f"  [WARN] Failed to export per-building CSV: {e}")



class DummyBaseline:
    """Baseline that does nothing (no-op actions)."""
    
    def __init__(self):
        self.is_training = False
        self.name = "baseline"
    
    def act(self, obs, deterministic=True):
        """Return no-op actions matching observation structure."""
        if isinstance(obs, list):
            return [[0.0] for _ in obs]
        return [[0.0]]