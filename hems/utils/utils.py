"""
HEMS Utilities Module
Contains utility classes and functions for the HEMS simulation environment.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path


class ObservationNormalizer:
    """Running normalization for observations using Welford's algorithm."""
    
    def __init__(self, dim: int, epsilon: float = 1e-8):
        """
        Initialize observation normalizer.
        
        Args:
            dim: Dimension of observations
            epsilon: Small constant for numerical stability
        """
        self.dim = dim
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
    
    def update(self, observation: np.ndarray):
        """
        Update running statistics with new observation.
        
        Args:
            observation: New observation vector
        """
        observation = observation.astype(np.float64)
        self.count += 1
        
        delta = observation - self.mean
        self.mean += delta / self.count
        delta2 = observation - self.mean
        self.M2 += delta * delta2
        
        if self.count > 1:
            self.var = self.M2 / (self.count - 1)
    
    def normalize(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation using current statistics.
        
        Args:
            observation: Observation to normalize
            
        Returns:
            Normalized observation
        """
        if self.count == 0:
            return observation.astype(np.float32)
        
        std = np.sqrt(np.maximum(self.var, self.epsilon))
        normalized = (observation - self.mean) / std
        return normalized.astype(np.float32)
    
    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current mean and standard deviation."""
        std = np.sqrt(np.maximum(self.var, self.epsilon))
        return self.mean.copy(), std


class ReplayBuffer:
    """Experience replay buffer for RL agents."""
    
    def __init__(self, capacity: int, obs_dim: int, device: str = 'cpu'):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            obs_dim: Dimension of observations
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        
        # Pre-allocate memory
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, obs: np.ndarray, action: int, reward: float, 
             next_obs: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.observations[self.position] = torch.from_numpy(obs)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = torch.from_numpy(next_obs)
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = torch.randint(0, self.size, (batch_size,))
        
        return (
            self.observations[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_observations[indices].to(self.device),
            self.dones[indices].to(self.device)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


def setup_logger(name: str, log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_path / f'{name}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directory(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """
    Create output directory for experiment results.
    
    Args:
        base_dir: Base directory path
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    if experiment_name:
        output_dir = Path(base_dir) / experiment_name
    else:
        # Use timestamp if no experiment name provided
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(base_dir) / f'experiment_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(config, output_dir: Path):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        output_dir: Output directory
    """
    import json
    
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(config_path: Path):
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        return json.load(f)


class ProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, total_steps: int, update_freq: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            update_freq: How often to update display
        """
        self.total_steps = total_steps
        self.update_freq = update_freq
        self.current_step = 0
        self.metrics = {}
    
    def update(self, metrics: dict = None):
        """
        Update progress.
        
        Args:
            metrics: Optional metrics dictionary
        """
        self.current_step += 1
        
        if metrics:
            self.metrics.update(metrics)
        
        if self.current_step % self.update_freq == 0:
            self._display_progress()
    
    def _display_progress(self):
        """Display current progress."""
        progress = self.current_step / self.total_steps * 100
        
        metrics_str = ""
        if self.metrics:
            metrics_str = " | " + " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in self.metrics.items()
            ])
        
        print(f"Progress: {progress:.1f}% ({self.current_step}/{self.total_steps}){metrics_str}")
    
    def is_complete(self) -> bool:
        """Check if tracking is complete."""
        return self.current_step >= self.total_steps


def calculate_cost_savings(baseline_cost: float, agent_cost: float) -> dict:
    """
    Calculate cost savings metrics.
    
    Args:
        baseline_cost: Cost with baseline agent
        agent_cost: Cost with trained agent
        
    Returns:
        Dictionary with savings metrics
    """
    absolute_savings = baseline_cost - agent_cost
    relative_savings = (absolute_savings / baseline_cost * 100) if baseline_cost != 0 else 0
    
    return {
        'baseline_cost': baseline_cost,
        'agent_cost': agent_cost,
        'absolute_savings': absolute_savings,
        'relative_savings_percent': relative_savings,
        'cost_reduction_ratio': agent_cost / baseline_cost if baseline_cost != 0 else 1.0
    }


def format_number(value: float, precision: int = 4) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1000:
        return f"{value:.0f}"
    elif abs(value) >= 1:
        return f"{value:.{min(precision, 3)}f}"
    else:
        return f"{value:.{precision}f}"


def validate_environment_compatibility(env) -> bool:
    """
    Validate that environment is compatible with HEMS framework.
    
    Args:
        env: CityLearn environment
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check if central agent
        if not env.central_agent:
            print("Warning: Environment should use central_agent=True for best compatibility")
            return False
        
        # Check required attributes
        required_attrs = ['buildings', 'time_steps', 'observation_space', 'action_space']
        for attr in required_attrs:
            if not hasattr(env, attr):
                print(f"Error: Environment missing required attribute: {attr}")
                return False
        
        # Check buildings have required components
        for building in env.buildings:
            if not hasattr(building, 'electrical_storage'):
                print(f"Error: Building {building.name} has no electrical storage")
                return False
            
            if not hasattr(building, 'pv'):
                print(f"Warning: Building {building.name} has no PV system")
        
        return True
        
    except Exception as e:
        print(f"Error validating environment: {e}")
        return False


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for minimization or maximization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop