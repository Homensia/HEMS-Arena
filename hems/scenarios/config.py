"""
hems/scenarios/config.py
Enhanced scenario configuration system with full SimulationConfig compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from hems.core.config import SimulationConfig


@dataclass
class ScenarioConfig:
    """
    Complete scenario configuration that extends SimulationConfig.
    
    This class captures all possible CLI parameters and converts them
    to SimulationConfig for execution.
    """
    
    # ============================================================================
    # SCENARIO METADATA
    # ============================================================================
    name: str
    description: str
    category: str  # 'benchmark', 'research', 'testing', 'development'
    scientific_purpose: str
    
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    dataset_type: str = 'original'  # 'original', 'synthetic', 'dummy'
    dataset_name: str = 'citylearn_challenge_2022_phase_all'
    synthetic_dataset_name: str = 'basic_data_1'
    datasets_root: str = 'datasets'
    
    # ============================================================================
    # ENVIRONMENT CONFIGURATION  
    # ============================================================================
    environment_type: str = 'citylearn'  # 'citylearn', 'dummy'
    
    # ============================================================================
    # SIMULATION PARAMETERS
    # ============================================================================
    building_count: int = 1
    simulation_days: int = 30
    building_id: Optional[str] = None
    
    # ============================================================================
    # AGENT CONFIGURATION
    # ============================================================================
    agents_to_evaluate: List[str] = field(default_factory=lambda: ['baseline', 'rbc', 'dqn'])
    train_episodes: int = 100
    
    # ============================================================================
    # HARDWARE & PERFORMANCE
    # ============================================================================
    use_gpu: bool = False
    random_seed: int = 42
    
    # ============================================================================
    # TARIFF & ENVIRONMENT SETTINGS
    # ============================================================================
    tariff_type: str = 'hp_hc'
    price_hp: float = 0.22
    price_hc: float = 0.14
    hc_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    
    # ============================================================================
    # OBSERVATION SPACE
    # ============================================================================
    active_observations: List[str] = field(default_factory=lambda: [
        'hour', 'electricity_pricing', 'net_electricity_consumption',
        'electrical_storage_soc', 'solar_generation'
    ])
    
    # ============================================================================
    # ANALYSIS & OUTPUT
    # ============================================================================
    perform_eda: bool = False
    save_plots: bool = True
    results_root: str = 'results'
    
    # ============================================================================
    # EXPERIMENT MANAGEMENT
    # ============================================================================
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    
    # ============================================================================
    # SCIENTIFIC PARAMETERS
    # ============================================================================
    expected_outcome: Optional[str] = None
    hypothesis: Optional[str] = None
    metrics_focus: List[str] = field(default_factory=lambda: ['cost', 'peak_demand'])
    
    # ============================================================================
    # AGENT-SPECIFIC CONFIGURATIONS
    # ============================================================================
    dqn_config: Dict[str, Any] = field(default_factory=lambda: {
        'action_bins': 31,
        'buffer_size': 300_000,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.08,
        'epsilon_decay_steps': 150_000,
        'target_update_freq': 1000,
        'train_start_steps': 1000,
        'hidden_layers': [256, 256]
    })
    
    rbc_config: Dict[str, Any] = field(default_factory=lambda: {
        'charge_level': 0.7,
        'discharge_level': 0.7
    })
    
    tql_config: Dict[str, Any] = field(default_factory=lambda: {
        'observation_bins': {'hour': 24},
        'action_bins': {'electrical_storage': 12},
        'epsilon': 1.0,
        'minimum_epsilon': 0.01,
        'epsilon_decay': 0.0001,
        'learning_rate': 0.005,
        'discount_factor': 0.99
    })
    
    sac_config: Dict[str, Any] = field(default_factory=lambda: {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'buffer_size': 1_000_000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1
    })
    
    # ============================================================================
    # REWARD CONFIGURATION
    # ============================================================================
    reward_config: Dict[str, Any] = field(default_factory=lambda: {
        'alpha_import_hp': 0.6,
        'price_high_threshold': 0.70,
        'alpha_peak': 0.01,
        'alpha_pv_base': 0.10,
        'alpha_pv_soc': 0.35,
        'alpha_soc': 0.05,
        'soc_lo': 0.30,
        'soc_hi': 0.60
    })
    
    def to_simulation_config(self) -> SimulationConfig:
        """
        Convert ScenarioConfig to SimulationConfig for execution.
        
        This is the bridge between scenarios and the core system.
        """
        # Determine output directory
        output_dir = Path(self.results_root) / (self.experiment_name or self.name)
        
        return SimulationConfig(
            # Basic simulation parameters
            building_count=self.building_count,
            simulation_days=self.simulation_days,
            building_id=self.building_id,
            dataset_name=self._get_active_dataset_name(),
            
            # Environment configuration
            environment_type=self.environment_type,
            
            # Agent configuration
            agents_to_evaluate=self.agents_to_evaluate,
            train_episodes=self.train_episodes,
            
            # Hardware settings
            use_gpu=self.use_gpu,
            random_seed=self.random_seed,
            
            # Tariff settings
            tariff_type=self.tariff_type,
            price_hp=self.price_hp,
            price_hc=self.price_hc,
            hc_hours=self.hc_hours,
            
            # Observation space
            active_observations=self.active_observations,
            
            # Output settings
            output_dir=str(output_dir),
            save_plots=self.save_plots,
            perform_eda=self.perform_eda,
            
            # Agent-specific parameters
            dqn_config=self.dqn_config,
            rbc_config=self.rbc_config,
            tql_config=self.tql_config,
            sac_config=self.sac_config,
            
            # Reward configuration
            reward_config=self.reward_config,
        )
    
    def _get_active_dataset_name(self) -> str:
        """Get the active dataset name based on dataset type."""
        if self.dataset_type == 'synthetic':
            return self.synthetic_dataset_name
        elif self.dataset_type == 'dummy':
            return 'dummy_dataset'
        else:
            return self.dataset_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != [] and v != ''
        }
    
    def to_cli_args(self) -> List[str]:
        """
        Convert scenario to CLI arguments for the original main.py.
        
        This enables scenarios to generate equivalent CLI commands.
        """
        args = []
        
        # Environment selection
        if self.environment_type != 'citylearn':
            args.extend(['--environment', self.environment_type])
        
        # Basic simulation parameters
        args.extend(['--buildings', str(self.building_count)])
        args.extend(['--days', str(self.simulation_days)])
        
        if self.building_id:
            args.extend(['--building-id', self.building_id])
        
        # Agents
        args.extend(['--agents'] + self.agents_to_evaluate)
        
        if self.train_episodes != 100:
            args.extend(['--train-episodes', str(self.train_episodes)])
        
        # Hardware
        if self.use_gpu:
            args.append('--gpu')
        
        if self.random_seed != 42:
            args.extend(['--seed', str(self.random_seed)])
        
        # Tariff
        if self.tariff_type != 'hp_hc':
            args.extend(['--tariff', self.tariff_type])
        
        # Analysis
        if self.perform_eda:
            args.append('--eda')
        
        if not self.save_plots:
            args.append('--no-save-plots')
        
        # Output
        if self.results_root != 'results':
            args.extend(['--output-dir', self.results_root])
        
        return args
    
    def get_cli_command(self) -> str:
        """Get complete CLI command for this scenario."""
        args = self.to_cli_args()
        return f"python main.py {' '.join(args)}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        """Create ScenarioConfig from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_simulation_config(cls, sim_config: SimulationConfig, 
                             name: str, description: str, 
                             category: str = 'custom',
                             scientific_purpose: str = 'Custom scenario') -> 'ScenarioConfig':
        """Create ScenarioConfig from existing SimulationConfig."""
        return cls(
            name=name,
            description=description,
            category=category,
            scientific_purpose=scientific_purpose,
            
            # Copy all relevant parameters
            building_count=sim_config.building_count,
            simulation_days=sim_config.simulation_days,
            building_id=sim_config.building_id,
            dataset_name=sim_config.dataset_name,
            
            environment_type=getattr(sim_config, 'environment_type', 'citylearn'),
            
            agents_to_evaluate=sim_config.agents_to_evaluate,
            train_episodes=sim_config.train_episodes,
            
            use_gpu=sim_config.use_gpu,
            random_seed=sim_config.random_seed,
            
            tariff_type=sim_config.tariff_type,
            price_hp=sim_config.price_hp,
            price_hc=sim_config.price_hc,
            hc_hours=sim_config.hc_hours,
            
            active_observations=sim_config.active_observations,
            
            output_dir=sim_config.output_dir,
            save_plots=sim_config.save_plots,
            perform_eda=sim_config.perform_eda,
            
            dqn_config=sim_config.dqn_config,
            rbc_config=sim_config.rbc_config,
            tql_config=sim_config.tql_config,
            sac_config=sim_config.sac_config,
            
            reward_config=sim_config.reward_config,
        )


def save_scenario_config(scenario_config: ScenarioConfig, output_path: str):
    """Save scenario configuration to JSON file."""
    config_dict = scenario_config.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Scenario config saved to: {output_path}")


def load_scenario_config(config_path: str) -> ScenarioConfig:
    """Load scenario configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return ScenarioConfig.from_dict(config_dict)