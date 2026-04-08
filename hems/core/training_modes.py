#==============================
#hems/core/training_modes.py
#==============================

"""
Training Modes for HEMS Benchmark 

Implements three training modes:
1. Sequential: Train on buildings one-by-one with optional transfer learning
2. Parallel: Train on all buildings simultaneously
3. Both: Run sequential first, then parallel

Key Features:
- Proper seed management (different seeds per phase)
- Model saving (best, final, average for sequential)
- Early stopping support
- Episode tracking and metrics collection
"""

import numpy as np
import pickle
import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Import torch for memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .adapters import ObservationAdapter, ActionAdapter
except ImportError:
    from adapters import ObservationAdapter, ActionAdapter

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def set_random_seeds(seed: int, context: str = ""):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        context: Context description for logging
    """
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    if context:
        logger.info(f"[SEED] Set random seed to {seed} for {context}")


def setup_gpu_optimization():
    """
    Setup GPU optimizations for the training pipeline.
    
    Returns:
        torch.device: Device to use (cuda or cpu)
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            device = torch.device('cuda')
            logger.info(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.info("[GPU] CUDA not available, using CPU")
            return torch.device('cpu')
    except ImportError:
        logger.info("[GPU] PyTorch not available, using CPU")
        return None


def save_model(agent, filepath: Path, metadata: Optional[Dict] = None):
    """
    Save agent model with metadata.
    
    Args:
        agent: Agent instance to save
        filepath: Path to save file
        metadata: Optional metadata dictionary
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'agent_state': agent.algorithm.get_state() if hasattr(agent.algorithm, 'get_state') else None,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"[SAVE] Model saved to {filepath}")


def load_model(agent, filepath: Path) -> Dict[str, Any]:
    """
    Load agent model from file.
    
    Args:
        agent: Agent instance to load into
        filepath: Path to model file
        
    Returns:
        Model metadata
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    if model_data.get('agent_state') and hasattr(agent.algorithm, 'load_state'):
        agent.algorithm.load_state(model_data['agent_state'])
    
    logger.info(f"[LOAD] Model loaded from {filepath}")
    return model_data.get('metadata', {})


# ============================================================================
# Sequential Training Mode
# ============================================================================

class SequentialTrainer:
    """
    Sequential training mode: train on buildings one-by-one.
    
    Features:
    - Transfer learning: weights from Building_i to Building_{i+1}
    - Per-building model saving (best and final)
    - Average model creation from all buildings
    """
    
    def __init__(
        self,
        agent,
        env_manager,
        config,
        output_dir: Path,
        logger_instance: logging.Logger
    ):
        """
        Initialize sequential trainer.
        
        Args:
            agent: Agent instance to train
            env_manager: Environment manager
            config: Benchmark configuration
            output_dir: Output directory for models
            logger_instance: Logger instance
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger_instance
        
        # Training settings
        self.episodes = config.training.episodes
        self.building_ids = config.training.buildings.ids
        self.save_frequency = config.training.save_frequency
        self.save_best = config.training.save_best
        self.save_final = config.training.save_final
        self.transfer_learning = config.advanced.get('sequential', {}).get('transfer_learning', False)
        
        # Early stopping
        self.early_stopping_config = config.training.early_stopping
        
        # Results tracking
        self.results = {
            'buildings': {},
            'summary': {}
        }
        
        # Get central_agent setting from environment
        central_agent = getattr(config.environment, 'central_agent', True)
        
        # Observation/action adapters
        self.obs_adapter = ObservationAdapter()
        self.action_adapter = ActionAdapter(central_agent=central_agent)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute sequential training across all buildings.
        
        Returns:
            Training results dictionary
        """
        self.logger.info("[SEQUENTIAL] Starting sequential training mode")
        self.logger.info(f"[SEQUENTIAL] Buildings: {self.building_ids}")
        self.logger.info(f"[SEQUENTIAL] Episodes per building: {self.episodes}")
        self.logger.info(f"[SEQUENTIAL] Transfer learning: {self.transfer_learning}")
        
        # Track total training time
        training_start_time = time.time()
        
        all_building_results = []
        
        # Progress bar for buildings
        try:
            from tqdm import tqdm
            building_iterator = tqdm(
                enumerate(self.building_ids),
                total=len(self.building_ids),
                desc="[SEQUENTIAL] Training Buildings",
                unit="building",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        except ImportError:
            building_iterator = enumerate(self.building_ids)
            self.logger.info("[SEQUENTIAL] Progress bar not available (install tqdm)")
        
        for idx, building_id in building_iterator:
            self.logger.info(f"\n[SEQUENTIAL] Building {idx+1}/{len(self.building_ids)}: {building_id}")
            
            # Set seed for this building (deterministic but different per building)
            building_seed = self.config.seed + idx * 100
            set_random_seeds(building_seed, f"Building {building_id}")
            
            # Train on this building
            building_results = self._train_on_building(building_id, idx)
            all_building_results.append(building_results)
            
            self.results['buildings'][building_id] = building_results
            
            # =====================================================================
            # MEMORY MANAGEMENT: Clear replay buffer between buildings
            # This prevents OOM while preserving transfer learning (network weights)
            # =====================================================================
            if idx < len(self.building_ids) - 1:  # Not the last building
                self.logger.info(f"[MEMORY] Clearing replay buffer after {building_id}")
                
                # Clear replay buffer (experiences from previous building)
                # Network weights are preserved for transfer learning!
                if hasattr(self.agent, 'algorithm') and hasattr(self.agent.algorithm, 'replay_buffer'):
                    try:
                        # Reset buffer position and size (keeps structure, clears data)
                        self.agent.algorithm.replay_buffer.position = 0
                        self.agent.algorithm.replay_buffer.size = 0
                        self.logger.info(f"[MEMORY] Replay buffer cleared (network weights preserved)")
                    except Exception as e:
                        self.logger.warning(f"[MEMORY] Could not clear replay buffer: {e}")
                
                # Clear PyTorch GPU cache if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info(f"[MEMORY] PyTorch GPU cache cleared")
                
                # Force Python garbage collection
                gc.collect()
                self.logger.info(f"[MEMORY] Garbage collection completed")
            
            # Transfer learning: keep weights for next building
            if self.transfer_learning and idx < len(self.building_ids) - 1:
                self.logger.info(f"[TRANSFER] Keeping weights for next building")
        
        # Create average model if multiple buildings
        if len(self.building_ids) > 1:
            self._create_average_model()
        
        # Compute summary statistics
        self.results['summary'] = self._compute_summary(all_building_results)
        
        # Add total training time
        training_duration = time.time() - training_start_time
        self.results['training_time'] = training_duration
        
        self.logger.info("[SEQUENTIAL] Sequential training complete")
        return self.results
    
    def _create_environment(self, building_ids: List[str]):
        """
        Create environment for given buildings.
        
        Args:
            building_ids: List of building IDs
            
        Returns:
            Environment instance
        """
        # Get simulation period
        start_time, end_time = self.env_manager.wrapper.select_simulation_period()
        
        # Create environment
        env = self.env_manager.wrapper.create_environment(
            buildings=building_ids,
            start_time=start_time,
            end_time=end_time,
            reward_function=None
        )
        
        return env
    
    def _train_on_building(self, building_id: str, building_idx: int) -> Dict[str, Any]:
        """
        Train agent on a single building.
        
        Args:
            building_id: Building ID
            building_idx: Index of building in sequence
            
        Returns:
            Training results for this building
        """
        # Create environment for this building
        env = self._create_environment([building_id])
        
        # Training loop
        episode_rewards = []
        episode_metrics = []
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(1, self.episodes + 1):
            episode_reward, metrics = self._run_episode(env, episode)
            episode_rewards.append(episode_reward)
            episode_metrics.append(metrics)
            
            # Update best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                
                # Save best model
                if self.save_best:
                    model_path = self.output_dir / 'models' / 'sequential' / f"{self.agent.name}_{building_id}_best.pkl"
                    save_model(
                        self.agent,
                        model_path,
                        metadata={
                            'building_id': building_id,
                            'episode': episode,
                            'reward': episode_reward,
                            'type': 'best'
                        }
                    )
            else:
                patience_counter += 1
            
            # Logging
            if episode % 10 == 0 or episode == self.episodes:
                self.logger.info(
                    f"  Episode {episode}/{self.episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Best: {best_reward:.2f}"
                )
            
            # Early stopping check
            if self.early_stopping_config and self.early_stopping_config.get('enabled', False):
                patience = self.early_stopping_config.get('patience', 50)
                if patience_counter >= patience:
                    self.logger.info(f"  [EARLY STOP] No improvement for {patience} episodes")
                    break
            
            # Periodic checkpoint
            if episode % self.save_frequency == 0:
                try:
                    checkpoint_path = self.output_dir / 'models' / 'sequential' / f"{self.agent.name}_{building_id}_ep{episode}.pkl"
                    save_model(
                        self.agent,
                        checkpoint_path,
                        metadata={
                            'building_id': building_id,
                            'episode': episode,
                            'reward': episode_reward,
                            'type': 'checkpoint'
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"  [WARN] Failed to save checkpoint: {e}")
        
        # Save final model
        if self.save_final and len(episode_rewards) > 0:
            try:
                final_path = self.output_dir / 'models' / 'sequential' / f"{self.agent.name}_{building_id}_final.pkl"
                save_model(
                    self.agent,
                    final_path,
                    metadata={
                        'building_id': building_id,
                        'episodes': len(episode_rewards),
                        'final_reward': episode_rewards[-1],
                        'best_reward': best_reward,
                        'type': 'final'
                    }
                )
            except Exception as e:
                self.logger.warning(f"  [WARN] Failed to save final model: {e}")
        
        # Return results
        return {
            'building_id': building_id,
            'episodes_completed': len(episode_rewards),
            'rewards': episode_rewards,
            'best_reward': best_reward if best_reward != float('-inf') else 0.0,
            'final_reward': episode_rewards[-1] if episode_rewards else 0.0,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'metrics': episode_metrics
        }
    
    def _run_episode(self, env, episode: int) -> Tuple[float, Dict]:
        """
        Run a single training episode.
        
        Args:
            env: Environment instance
            episode: Episode number
            
        Returns:
            Tuple of (total_reward, episode_metrics)
        """
        citylearn_obs = env.reset()
        obs = self.obs_adapter.adapt(citylearn_obs)
        done = False
        total_reward = 0
        step = 0
        
        self.agent.is_training = True
        
        while not done:
            action = self.agent.act(obs, deterministic=False)
            citylearn_action = self.action_adapter.adapt(action)
            
            result = env.step(citylearn_action)
            if len(result) == 5:
                next_citylearn_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_citylearn_obs, reward, done, info = result
            
            next_obs = self.obs_adapter.adapt(next_citylearn_obs)
            
            if hasattr(self.agent, 'learn'):
                self.agent.learn(obs, action, reward, next_obs, done)
            
            total_reward += sum(reward) if isinstance(reward, (list, np.ndarray)) else float(reward)
            obs = next_obs
            step += 1
        
        if hasattr(self.agent, 'learn_episode'):
            self.agent.learn_episode()
        
        metrics = {
            'episode': episode,
            'steps': step,
            'reward': total_reward
        }
        
        return total_reward, metrics
    
    def _create_average_model(self):
        """
        Create average model from all building models.
        Only works if agent supports state averaging.
        """
        self.logger.info("[AVERAGE] Creating average model from all buildings")
        
        # Load all best models
        states = []
        for building_id in self.building_ids:
            model_path = self.output_dir / 'models' / 'sequential' / f"{self.agent.name}_{building_id}_best.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    states.append(model_data.get('agent_state'))
        
        if not states or not all(states):
            self.logger.warning("[AVERAGE] Could not create average model - missing states")
            return
        
        # Average the states (if algorithm supports it)
        if hasattr(self.agent.algorithm, 'average_states'):
            avg_state = self.agent.algorithm.average_states(states)
            
            # Save averaged model
            avg_path = self.output_dir / 'models' / 'sequential' / f"{self.agent.name}_average.pkl"
            model_data = {
                'agent_state': avg_state,
                'metadata': {
                    'type': 'average',
                    'buildings': self.building_ids,
                    'source': 'sequential_training'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(avg_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"[AVERAGE] Average model saved to {avg_path}")
        else:
            self.logger.warning("[AVERAGE] Algorithm does not support state averaging")
    
    def _compute_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all buildings."""
        all_rewards = [r['best_reward'] for r in all_results]
        
        return {
            'total_buildings': len(all_results),
            'avg_best_reward': np.mean(all_rewards),
            'std_best_reward': np.std(all_rewards),
            'min_best_reward': np.min(all_rewards),
            'max_best_reward': np.max(all_rewards),
            'total_episodes': sum(r['episodes_completed'] for r in all_results)
        }


# ============================================================================
# Parallel Training Mode
# ============================================================================

class RoundRobinTrainer:
    """
    Round-robin training mode: cycle through buildings one step at a time.
    
    Features:
    - Cycles through buildings (one step per building)
    - Separate environments maintain temporal structure
    - Single shared model learns from all buildings
    """
    
    def __init__(
        self,
        agent,
        env_manager,
        config,
        output_dir: Path,
        logger_instance: logging.Logger
    ):
        """
        Initialize parallel trainer.
        
        Args:
            agent: Agent instance to train
            env_manager: Environment manager
            config: Benchmark configuration
            output_dir: Output directory for models
            logger_instance: Logger instance
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger_instance
        
        # Training settings
        self.episodes = config.training.episodes
        self.building_ids = config.training.buildings.ids
        self.save_frequency = config.training.save_frequency
        self.save_best = config.training.save_best
        self.save_final = config.training.save_final
        
        # Parallel settings
        self.shared_buffer = config.advanced.get('parallel', {}).get('shared_replay_buffer', False)
        
        # Early stopping
        self.early_stopping_config = config.training.early_stopping
        
        # Results tracking
        self.results = {}
        
        # Get central_agent setting from environment
        central_agent = getattr(config.environment, 'central_agent', True)
        
        # Observation/action adapters
        self.obs_adapter = ObservationAdapter()
        self.action_adapter = ActionAdapter(central_agent=central_agent)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute round-robin training across all buildings.
        
        Returns:
            Training results dictionary
        """
        self.logger.info("[ROUND_ROBIN] Starting round-robin training mode")
        self.logger.info(f"[ROUND_ROBIN] Buildings: {self.building_ids}")
        self.logger.info(f"[ROUND_ROBIN] Total episodes: {self.episodes}")
        
        # Track training time
        training_start_time = time.time()
        
        # Set seed
        set_random_seeds(self.config.seed, "parallel training")
        
        # Create separate environment for each building
        start_time, end_time = self.env_manager.wrapper.select_simulation_period()
        envs = {}
        for building_id in self.building_ids:
            envs[building_id] = self.env_manager.wrapper.create_environment(
                buildings=[building_id],
                start_time=start_time,
                end_time=end_time,
                reward_function=None
            )
        
        # Initialize tracking per building
        current_obs = {}
        episode_steps = {bid: 0 for bid in self.building_ids}
        episode_rewards_per_building = {bid: [] for bid in self.building_ids}
        building_done = {bid: False for bid in self.building_ids}
        building_episode_reward = {bid: 0.0 for bid in self.building_ids}
        completed_episodes_per_building = {bid: 0 for bid in self.building_ids}
        
        # Reset all environments
        for building_id in self.building_ids:
            citylearn_obs = envs[building_id].reset()
            current_obs[building_id] = self.obs_adapter.adapt(citylearn_obs)
        
        # Training metrics
        best_avg_reward = float('-inf')
        patience_counter = 0
        total_steps = 0
        
        self.agent.is_training = True
        
        # Progress bar for round-robin training
        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=self.episodes,
                desc="[ROUND_ROBIN] Min Episodes Completed",
                unit="episode",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        except ImportError:
            pbar = None
            self.logger.info("[ROUND_ROBIN] Progress bar not available (install tqdm)")
        
        last_min_episodes = 0
        
        # Training loop - continue until all buildings complete target episodes
        while min(completed_episodes_per_building.values()) < self.episodes:
            # Round-robin through buildings
            for building_id in self.building_ids:
                # Skip if this building is done with current episode
                if building_done[building_id]:
                    continue
                
                # Get action for current building
                obs = current_obs[building_id]
                action = self.agent.act(obs, deterministic=False)
                citylearn_action = self.action_adapter.adapt(action)
                
                # Step environment
                citylearn_result = envs[building_id].step(citylearn_action)
                
                # Parse result
                if len(citylearn_result) == 5:
                    next_citylearn_obs, reward, terminated, truncated, info = citylearn_result
                    done = terminated or truncated
                else:
                    next_citylearn_obs, reward, done, info = citylearn_result
                
                next_obs = self.obs_adapter.adapt(next_citylearn_obs)
                reward_scalar = float(reward) if np.isscalar(reward) else float(np.mean(reward))
                
                # Learn from this transition
                if hasattr(self.agent, 'learn'):
                    self.agent.learn(obs, action, reward_scalar, next_obs, done)
                
                # Update state
                building_episode_reward[building_id] += reward_scalar
                episode_steps[building_id] += 1
                total_steps += 1
                
                if done:
                    # Episode finished for this building
                    episode_rewards_per_building[building_id].append(building_episode_reward[building_id])
                    completed_episodes_per_building[building_id] += 1
                    
                    # Check if this building needs more episodes
                    if completed_episodes_per_building[building_id] < self.episodes:
                        # Reset for next episode
                        citylearn_obs = envs[building_id].reset()
                        current_obs[building_id] = self.obs_adapter.adapt(citylearn_obs)
                        building_episode_reward[building_id] = 0.0
                        episode_steps[building_id] = 0
                    else:
                        # This building is done training
                        building_done[building_id] = True
                else:
                    # Continue episode
                    current_obs[building_id] = next_obs
            
            # Logging every N steps
            if total_steps % 1000 == 0:
                min_episodes = min(completed_episodes_per_building.values())
                
                # Update progress bar
                if pbar and min_episodes > last_min_episodes:
                    pbar.update(min_episodes - last_min_episodes)
                    last_min_episodes = min_episodes
                
                avg_rewards = [np.mean(episode_rewards_per_building[bid][-10:]) if episode_rewards_per_building[bid] else 0 
                              for bid in self.building_ids]
                avg_reward = np.mean(avg_rewards)
                
                self.logger.info(
                    f"  Steps: {total_steps} | "
                    f"Min Episodes: {min_episodes}/{self.episodes} | "
                    f"Avg Reward: {avg_reward:.2f}"
                )
                
                # Save best model based on average reward across buildings
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                    
                    if self.save_best:
                        model_path = self.output_dir / 'models' / 'round_robin' / f"{self.agent.name}_best.pkl"
                        save_model(
                            self.agent,
                            model_path,
                            metadata={
                                'buildings': self.building_ids,
                                'steps': total_steps,
                                'avg_reward': avg_reward,
                                'type': 'best'
                            }
                        )
                else:
                    patience_counter += 1
                
                # Early stopping check
                if self.early_stopping_config and self.early_stopping_config.get('enabled', False):
                    patience = self.early_stopping_config.get('patience', 50)
                    if patience_counter >= patience:
                        self.logger.info(f"  [EARLY STOP] No improvement for {patience} checks")
                        break
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Save final model
        if self.save_final:
            final_path = self.output_dir / 'models' / 'round_robin' / f"{self.agent.name}_final.pkl"
            save_model(
                self.agent,
                final_path,
                metadata={
                    'buildings': self.building_ids,
                    'episodes_per_building': completed_episodes_per_building,
                    'total_steps': total_steps,
                    'type': 'final'
                }
            )
        
        # Compile results - format compatible with visualizer
        # Convert to dict structure like sequential mode
        buildings_results = {}
        for building_id in self.building_ids:
            buildings_results[building_id] = {
                'episodes_completed': completed_episodes_per_building[building_id],
                'rewards': episode_rewards_per_building[building_id],
                'best_reward': max(episode_rewards_per_building[building_id]) if episode_rewards_per_building[building_id] else 0,
                'final_reward': episode_rewards_per_building[building_id][-1] if episode_rewards_per_building[building_id] else 0,
                'avg_reward': np.mean(episode_rewards_per_building[building_id]) if episode_rewards_per_building[building_id] else 0,
            }
        
        all_rewards = []
        for building_id in self.building_ids:
            all_rewards.extend(episode_rewards_per_building[building_id])
        
        training_duration = time.time() - training_start_time
        self.results = {
            'mode': 'round_robin',
            'buildings': buildings_results,  # Dict structure for visualizer
            'total_steps': total_steps,
            'best_avg_reward': best_avg_reward,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'training_time': training_duration,
        }
        
        self.logger.info("[ROUND_ROBIN] Round-robin training complete")
        return self.results
    
    def _run_episode(self, env, episode: int) -> Tuple[float, Dict]:
        """Run a single training episode."""
        citylearn_obs = env.reset()
        obs = self.obs_adapter.adapt(citylearn_obs)
        done = False
        total_reward = 0
        step = 0
        
        self.agent.is_training = True
        
        while not done:
            action = self.agent.act(obs, deterministic=False)
            citylearn_action = self.action_adapter.adapt(action)
            
            result = env.step(citylearn_action)
            if len(result) == 5:
                next_citylearn_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_citylearn_obs, reward, done, info = result
            
            next_obs = self.obs_adapter.adapt(next_citylearn_obs)
            
            if hasattr(self.agent, 'learn'):
                self.agent.learn(obs, action, reward, next_obs, done)
            
            total_reward += sum(reward) if isinstance(reward, (list, np.ndarray)) else float(reward)
            obs = next_obs
            step += 1
        
        if hasattr(self.agent, 'learn_episode'):
            self.agent.learn_episode()
        
        metrics = {
            'episode': episode,
            'steps': step,
            'reward': total_reward
        }
        
        return total_reward, metrics


# ============================================================================
# True Parallel Training Mode (Multi-Head)
# ============================================================================

class ParallelTrainer:
    """
    Parallel training mode: all buildings in ONE environment (multi-head).
    
    Features:
    - Single environment with all buildings
    - Agent sees concatenated observations from all buildings
    - Multi-head architecture outputs different action per building
    - Learns joint multi-building policy
    """
    
    def __init__(
        self,
        agent,
        env_manager,
        config,
        output_dir: Path,
        logger_instance: logging.Logger
    ):
        """
        Initialize parallel trainer.
        
        Args:
            agent: Agent instance to train (must support multi-head)
            env_manager: Environment manager
            config: Benchmark configuration
            output_dir: Output directory for models
            logger_instance: Logger instance
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger_instance
        
        # Training settings
        self.episodes = config.training.episodes
        self.building_ids = config.training.buildings.ids
        self.save_frequency = config.training.save_frequency
        self.save_best = config.training.save_best
        self.save_final = config.training.save_final
        
        # Early stopping
        self.early_stopping_config = config.training.early_stopping
        
        # Results tracking
        self.results = {}
        
        # Get central_agent setting from environment (config.environment is a DICT)
        central_agent = config.environment.get('central_agent', True) if isinstance(config.environment, dict) else getattr(config.environment, 'central_agent', True)
        
        # Observation/action adapters
        self.obs_adapter = ObservationAdapter()
        self.action_adapter = ActionAdapter(central_agent=central_agent)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute parallel training.
        
        Returns:
            Training results dictionary
        """
        self.logger.info("[PARALLEL] Starting parallel training mode")
        self.logger.info(f"[PARALLEL] Buildings: {self.building_ids}")
        self.logger.info(f"[PARALLEL] Total episodes: {self.episodes}")
        
        # Set seed
        set_random_seeds(self.config.seed, "parallel training")
        
        # Track training time
        training_start_time = time.time()
        
        # Create multi-building environment
        start_time, end_time = self.env_manager.wrapper.select_simulation_period()
        env = self.env_manager.wrapper.create_environment(
            buildings=self.building_ids,
            start_time=start_time,
            end_time=end_time,
            reward_function=None
        )
        
        # Training loop 
        episode_rewards = []
        best_reward = float('-inf')
        patience_counter = 0
        
        try:
            from tqdm import tqdm
            episode_iterator = tqdm(
                range(1, self.episodes + 1),
                desc="[PARALLEL] Training Episodes",
                unit="episode",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        except ImportError:
            episode_iterator = range(1, self.episodes + 1)
            self.logger.info("[PARALLEL] Progress bar not available (install tqdm)")
        
        for episode in episode_iterator:
            # Run episode
            episode_reward, metrics = self._run_episode(env, episode)
            episode_rewards.append(episode_reward)
            
            # Update best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                
                # Save best model
                if self.save_best:
                    model_path = self.output_dir / 'models' / 'parallel' / f"{self.agent.name}_best.pkl"
                    save_model(
                        self.agent,
                        model_path,
                        metadata={
                            'buildings': self.building_ids,
                            'episode': episode,
                            'reward': episode_reward,
                            'type': 'best'
                        }
                    )
            else:
                patience_counter += 1
            
            # Logging
            if episode % 10 == 0 or episode == self.episodes:
                self.logger.info(
                    f"  Episode {episode}/{self.episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Best: {best_reward:.2f}"
                )
            
            # Early stopping check
            if self.early_stopping_config and self.early_stopping_config.get('enabled', False):
                patience = self.early_stopping_config.get('patience', 50)
                if patience_counter >= patience:
                    self.logger.info(f"  [EARLY STOP] No improvement for {patience} episodes")
                    break
            
            # Periodic checkpoint
            if episode % self.save_frequency == 0:
                checkpoint_path = self.output_dir / 'models' / 'parallel' / f"{self.agent.name}_ep{episode}.pkl"
                save_model(
                    self.agent,
                    checkpoint_path,
                    metadata={
                        'buildings': self.building_ids,
                        'episode': episode,
                        'reward': episode_reward,
                        'type': 'checkpoint'
                    }
                )
        
        # Save final model
        if self.save_final:
            final_path = self.output_dir / 'models' / 'parallel' / f"{self.agent.name}_final.pkl"
            save_model(
                self.agent,
                final_path,
                metadata={
                    'buildings': self.building_ids,
                    'episodes': len(episode_rewards),
                    'final_reward': episode_rewards[-1],
                    'best_reward': best_reward,
                    'type': 'final'
                }
            )
        
        # Compile results
        training_duration = time.time() - training_start_time
        self.results = {
            'mode': 'parallel',
            'buildings': self.building_ids,
            'episodes_completed': len(episode_rewards),
            'rewards': episode_rewards,
            'best_reward': best_reward,
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'training_time': training_duration,
        }
        
        self.logger.info("[PARALLEL] Parallel training complete")
        return self.results
    
    def _run_episode(self, env, episode: int) -> Tuple[float, Dict]:
        """Run a single training episode."""
        citylearn_obs = env.reset()
        obs = self.obs_adapter.adapt(citylearn_obs)
        done = False
        total_reward = 0
        step = 0
        
        self.agent.is_training = True
        
        while not done:
            action = self.agent.act(obs, deterministic=False)
            citylearn_action = self.action_adapter.adapt(action)
            
            result = env.step(citylearn_action)
            
            if len(result) == 5:
                next_citylearn_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_citylearn_obs, reward, done, info = result
            
            next_obs = self.obs_adapter.adapt(next_citylearn_obs)
            
            if hasattr(self.agent, 'learn'):
                self.agent.learn(obs, action, reward, next_obs, done)
            
            total_reward += sum(reward) if isinstance(reward, (list, np.ndarray)) else float(reward)
            obs = next_obs
            step += 1
        
        if hasattr(self.agent, 'learn_episode'):
            self.agent.learn_episode()
        
        metrics = {
            'episode': episode,
            'steps': step,
            'reward': total_reward
        }
        
        return total_reward, metrics


# ============================================================================
# Training Mode Factory
# ============================================================================

def create_trainer(mode: str, agent, env_manager, config, output_dir: Path, logger_instance: logging.Logger):
    """
    Factory function to create appropriate trainer.
    
    Args:
        mode: Training mode ('sequential', 'round_robin', 'parallel', or 'both')
        agent: Agent instance
        env_manager: Environment manager
        config: Benchmark configuration
        output_dir: Output directory
        logger_instance: Logger instance
        
    Returns:
        Trainer instance
    """
    if mode == 'sequential':
        return SequentialTrainer(agent, env_manager, config, output_dir, logger_instance)
    elif mode == 'round_robin':
        return RoundRobinTrainer(agent, env_manager, config, output_dir, logger_instance)
    elif mode == 'parallel':
        return ParallelTrainer(agent, env_manager, config, output_dir, logger_instance)
    elif mode == 'both':
        return BothTrainer(agent, env_manager, config, output_dir, logger_instance)
    else:
        raise ValueError(f"Unknown training mode: {mode}")


class BothTrainer:
    """
    Both training mode: run sequential first, then parallel.
    """
    
    def __init__(self, agent, env_manager, config, output_dir, logger_instance):
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.output_dir = output_dir
        self.logger = logger_instance
    
    def train(self) -> Dict[str, Any]:
        """Execute both training modes."""
        self.logger.info("[BOTH] Starting both training modes")
        
        # Sequential first
        self.logger.info("[BOTH] Phase 1: Sequential training")
        seq_trainer = SequentialTrainer(
            self.agent, self.env_manager, self.config,
            self.output_dir, self.logger
        )
        seq_results = seq_trainer.train()
        
        # Then parallel
        self.logger.info("[BOTH] Phase 2: Parallel training")
        par_trainer = ParallelTrainer(
            self.agent, self.env_manager, self.config,
            self.output_dir, self.logger
        )
        par_results = par_trainer.train()
        
        return {
            'mode': 'both',
            'sequential': seq_results,
            'parallel': par_results
        }