#====================
#hems/core/config.py
#====================

"""
Configuration management for HEMS simulation environment.
Centralizes all simulation parameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import copy
from dataclasses import asdict


@dataclass
class SimulationConfig:
    """Central configuration for HEMS simulation."""

    # Environment and dataset configuration
    environment_type: str = 'citylearn'
    dataset_name: str = 'citylearn_challenge_2022_phase_all'
    
    # Basic simulation parameters
    building_count: int = 1
    simulation_days: int = 30
    building_id: Optional[str] = None
    
    # Agent configuration
    agents_to_evaluate: List[str] = field(default_factory=lambda: ['baseline', 'rbc', 'dqn','mp_ppo','mpc_forecast','ambitious_engineers', 'Chen_Bu_p2p'])
    train_episodes: int = 100
    
    # Hardware settings
    use_gpu: bool = True
    device: str = field(init=False)
    
    # Dataset tariff selection
    tariff_type: str = 'default'
    
    # Observation space
    active_observations: List[str] = field(default_factory=lambda: [
        'hour',
        'electricity_pricing', 
        'net_electricity_consumption',
        'electrical_storage_soc',
        'solar_generation'
    ])
    
    # Reproducibility
    random_seed: int = 42
    
    # Output settings
    output_dir: str = 'results'
    save_plots: bool = True
    perform_eda: bool = False
    
    # Agent-specific parameters
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
        'hidden_layers': [256, 256],
        'history_length': 5
    })
    
    rbc_config: Dict[str, Any] = field(default_factory=lambda: {
        'charge_level': 0.7,
        'discharge_level': 0.7
    })
    
    # Current config can stay as-is, but if you want to optimize:
    tql_config: Dict[str, Any] = field(default_factory=lambda: {
    # Simple parameters like in CityLearn tutorial
    'epsilon': 0.1,              # Exploration rate
    'learning_rate': 0.1,        # Learning rate
    'discount_factor': 0.99,     # Discount factor
    'random_seed': 42            # Random seed

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

    # mp_ppo algorithm configurations

    mp_ppo_config: Dict[str, Any] = field(default_factory=lambda: {
    'horizon': 24,
    'ctx': 48,
    'in_dim': 1,
    'd_model': 32,
    'nhead': 2,
    'layers': 4,
    'ff': 32,
    'lr_policy': 3e-4,
    'lr_pred': 3e-4,
    'train_iters': 10,
    'pred_updates_per_iter': 10,
    'buffer_steps': 8000,
    'minibatch': 2048,
    'gamma': 0.99,
    'lam': 0.95,
    'clip_ratio': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'pretrained_model_path': 'models/mp_ppo_predictor_best.pt'
    })


    # AmbitiousEngineers configuration
    ambitious_engineers_config: Dict[str, Any] = field(default_factory=lambda: {
            'mode': 'dp_only',  # 'dp_only', 'phase1', or 'phase23'
            'train_mode': False,  # False for inference, True for training
            'phase1_weights_path': None,
            'phase23_weights_path': None,
            'demand_forecaster_path': None,
            'solar_forecaster_path': None,
            'dp_params': {
                'n_states': 101,
                'battery_capacity': 6.4,
                'battery_power': 4.0
            },
            'training_params': {
                'n_seeds': 5,
                'n_iterations': 3000,
                'population_size': 50,
                'sigma0': 0.05,
                'l2_penalty': 0.01,
                'n_jobs': 50
            },
            'forecaster_config': {
                'demand_n_lags': 10,
                'solar_n_lags': 216,
                'n_targets': 10,
                'demand_hidden': 256,
                'solar_hidden': 2048,
                'demand_dropout': 0.1,
                'solar_dropout': 0.0
            }
        })

    # Custom reward parameters
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

    def copy(self) -> 'SimulationConfig':
        """
        Create a deep copy of the configuration.
        
        Returns:
            A new SimulationConfig instance with the same values
        """
        # Get all current values as dict
        config_dict = asdict(self)
        
        # Create new instance with same values
        return SimulationConfig(**config_dict)
    
    def get_active_dataset_name(self) -> str:
        """Get the dataset name for compatibility."""
        return self.dataset_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            'building_count': self.building_count,
            'simulation_days': self.simulation_days,
            'building_id': self.building_id,
            'agents_to_evaluate': self.agents_to_evaluate,
            'train_episodes': self.train_episodes,
            'device': self.device,
            'tariff_type': self.tariff_type,
            'random_seed': self.random_seed,
            'output_dir': self.output_dir,
            'environment_type': getattr(self, 'environment_type', 'citylearn'),
            'dataset_name': getattr(self, 'dataset_name', 'citylearn_challenge_2022_phase_all')
        }        
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set device based on GPU availability and user preference
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Set all random seeds for reproducibility
        self.set_seeds()
        
        # Validate configuration
        self.validate()
    
    def set_seeds(self):
        """Set all random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Additional reproducibility settings for PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def validate(self):
        """Validate configuration parameters."""
        assert 1 <= self.building_count <= 15, f"Building count must be 1-15, got {self.building_count}"
        assert 1 <= self.simulation_days <= 365, f"Simulation days must be 1-365, got {self.simulation_days}"
        assert self.train_episodes > 0, f"Train episodes must be positive, got {self.train_episodes}"
        
        # Validate agents
        valid_agents = ['baseline', 'rbc', 'tql', 'sac', 'dqn', 'mp_ppo', 'ambitious_engineers', 'mpc_forecast', 'Chen_Bu_p2p', 'custom']
        for agent in self.agents_to_evaluate:
            assert agent in valid_agents, f"Unknown agent: {agent}. Valid agents: {valid_agents}"
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for specific agent."""
        config_map = {
            'dqn': self.dqn_config,
            'rbc': self.rbc_config,
            'tql': self.tql_config,
            'sac': self.sac_config,
            'mp_ppo': self.mp_ppo_config
        }
        return config_map.get(agent_name, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            'building_count': self.building_count,
            'simulation_days': self.simulation_days,
            'building_id': self.building_id,
            'agents_to_evaluate': self.agents_to_evaluate,
            'train_episodes': self.train_episodes,
            'device': self.device,
            'tariff_type': self.tariff_type,
            'random_seed': self.random_seed,
            'output_dir': self.output_dir
        }


# Predefined configurations for common scenarios
QUICK_TEST_CONFIG = SimulationConfig(
    building_count=1,
    simulation_days=7,
    agents_to_evaluate=['baseline', 'rbc'],
    train_episodes=10,
    perform_eda=False
)

FULL_EVALUATION_CONFIG = SimulationConfig(
    building_count=2,
    simulation_days=365,
    agents_to_evaluate=['baseline', 'rbc', 'tql', 'sac', 'dqn', 'mp_ppo', 'ambitious_engineers',  'mpc_forecast','Chen_Bu_p2p'],
    train_episodes=500,
    perform_eda=True
)

SINGLE_BUILDING_CONFIG = SimulationConfig(
    building_count=1,
    simulation_days=30,
    agents_to_evaluate=['baseline', 'rbc', 'dqn', 'mp_ppo',  'mpc_forecast', 'ambitious_engineers', 'Chen_Bu_p2p'],
    train_episodes=100,
    perform_eda=True
)

def create_config_from_args(args) -> SimulationConfig:
    """Create configuration from command line arguments."""
    config = SimulationConfig(
        building_count=args.buildings,
        simulation_days=args.days,
        building_id=args.building_id,
        agents_to_evaluate=args.agents,
        train_episodes=args.train_episodes,
        use_gpu=args.gpu,
        tariff_type=args.tariff,
        random_seed=args.seed,
        results_root=args.output_dir,
        save_plots=args.save_plots,
        perform_eda=args.eda,
        experiment_name=getattr(args, 'experiment_name', None),
        experiment_description=getattr(args, 'experiment_description', None),
        environment_type=getattr(args, 'environment', 'citylearn')  # ADD THIS LINE
    )
    
    # ... rest of the function
    return config