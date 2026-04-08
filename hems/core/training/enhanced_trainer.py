"""
Enhanced Training Module
Handles comprehensive training with analytics, model checkpointing, and convergence monitoring.
"""

import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
import copy

from hems.agents.legacy_adapter import create_agent


class EnhancedTrainer:
    """
    Enhanced training module with analytics, checkpointing, and convergence monitoring.
    """
    
    def __init__(self, config, env_manager, experiment_dir: Path, logger):
        """
        Initialize enhanced trainer.
        
        Args:
            config: Simulation configuration
            env_manager: Environment manager
            experiment_dir: Experiment directory
            logger: Logger instance
        """
        self.config = config
        self.env_manager = env_manager
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        # Create training directories
        self.models_dir = experiment_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.training_logs_dir = experiment_dir / 'logs' / 'training'
        self.training_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 50)
        self.convergence_threshold = getattr(config, 'convergence_threshold', 0.01)
        self.checkpoint_frequency = getattr(config, 'checkpoint_frequency', 10)
        
    def run_enhanced_training(self) -> Dict[str, Any]:
        """
        Run enhanced training for all configured agents.
        
        Returns:
            Training results with analytics and model paths
        """
        self.logger.info("Starting enhanced training phase...")
        
        training_results = {
            'agent_results': {},
            'training_summary': {},
            'best_models': {},
            'training_analytics': {}
        }
        
        # Get training environment
        env = self.env_manager.get_training_environment()
        
        for agent_name in self.config.agents_to_evaluate:
            self.logger.info(f"Training agent: {agent_name}")
            
            try:
                # Train agent with enhanced analytics
                agent_results = self._train_single_agent(env, agent_name)
                training_results['agent_results'][agent_name] = agent_results
                
                # Extract best model path
                if agent_results.get('best_model_path'):
                    training_results['best_models'][agent_name] = agent_results['best_model_path']
                
            except Exception as e:
                self.logger.error(f"Training failed for agent {agent_name}: {str(e)}")
                training_results['agent_results'][agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate training summary
        training_results['training_summary'] = self._generate_training_summary(training_results)
        
        self.logger.info("Enhanced training phase completed")
        return training_results
    


    def _train_single_agent(self, env, agent_name: str) -> Dict[str, Any]:
        """
        Train a single agent with enhanced analytics and error handling.
        
        Args:
            env: Training environment
            agent_name: Name of the agent to train
            
        Returns:
            Training results for the agent
        """
        start_time = time.time()
        
        try:
            # Create agent
            agent = create_agent(agent_name, env, self.config)
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_name}: {e}")
            return {
                'status': 'failed',
                'error': f"Agent creation failed: {str(e)}",
                'training_time': 0,
                'total_episodes': 0
            }
        
        # Initialize training analytics
        training_analytics = {
            'episode_rewards': [],
            'episode_losses': [],
            'convergence_metrics': {},
            'training_time': 0,
            'best_performance': float('-inf'),
            'convergence_episode': 0,
            'exploration_schedule': [],
            'memory_usage': [],
            'gradient_norms': [],
            'learning_rate_schedule': []
        }
        
        best_performance = float('-inf')
        best_model_state = None
        best_model_path = None
        episodes_without_improvement = 0
        episodes_completed = 0
        
        # Training loop with enhanced error handling
        for episode in tqdm(range(self.config.train_episodes), desc=f"Training {agent_name}"):
            episode_start = time.time()
            
            try:
                # Reset environment
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    observations, info = reset_result
                else:
                    observations = reset_result
                    info = {}
                agent.reset()
                episode_reward = 0
                episode_loss = 0
                step_count = 0
                
                done = False
                max_steps = 1000  # Safety limit
                
                while not done and step_count < max_steps:
                    try:
                        # Agent action
                        actions = agent.act(observations)
                        
                        # Ensure actions are in correct format
                        if isinstance(actions, (list, np.ndarray)):
                            # Handle list actions properly for RBC
                            if agent_name == 'rbc' and isinstance(actions, list):
                                # Convert list to numpy array for RBC
                                actions = np.array(actions, dtype=np.float32)
                        
                        # Environment step - handle both old and new gym formats
                        step_result = env.step(actions)
                        
                        if len(step_result) == 5:
                            # New gym format: (obs, reward, terminated, truncated, info)
                            next_observations, rewards, terminated, truncated, info = step_result
                            done = terminated or truncated
                        elif len(step_result) == 4:
                            # Old gym format: (obs, reward, done, info)
                            next_observations, rewards, done, info = step_result
                        else:
                            raise ValueError(f"Unexpected step result format: {len(step_result)} values")
                        
                        # Agent learning (if applicable)
                        if hasattr(agent, 'learn') and callable(getattr(agent, 'learn')):
                            try:
                                learn_info = agent.learn(
                                    observations, actions, rewards, next_observations, done
                                )
                                if learn_info and isinstance(learn_info, dict) and 'loss' in learn_info:
                                    episode_loss += learn_info['loss']
                            except Exception as learn_error:
                                # Log learning errors but continue training
                                if step_count < 5:  # Only log first few errors
                                    self.logger.warning(f"Learning step failed for {agent_name}: {learn_error}")
                        
                        # Update state
                        observations = next_observations
                        
                        # Handle rewards properly
                        if isinstance(rewards, (list, np.ndarray)):
                            reward_value = sum(rewards)
                        else:
                            reward_value = float(rewards)
                        
                        episode_reward += reward_value
                        step_count += 1
                        
                    except Exception as step_error:
                        self.logger.error(f"Step error in episode {episode} for {agent_name}: {step_error}")
                        break
                
                # Record analytics
                training_analytics['episode_rewards'].append(episode_reward)
                training_analytics['episode_losses'].append(episode_loss / max(step_count, 1))
                
                # Record exploration rate (if available)
                try:
                    if hasattr(agent, 'exploration_rate'):
                        training_analytics['exploration_schedule'].append(agent.exploration_rate)
                    elif hasattr(agent, 'epsilon'):
                        training_analytics['exploration_schedule'].append(agent.epsilon)
                    elif hasattr(agent, 'algorithm') and hasattr(agent.algorithm, 'epsilon'):
                        training_analytics['exploration_schedule'].append(agent.algorithm.epsilon)
                except:
                    pass  # Ignore if exploration rate not available
                
                # Record learning rate (if available)
                try:
                    if hasattr(agent, 'learning_rate'):
                        training_analytics['learning_rate_schedule'].append(agent.learning_rate)
                    elif hasattr(agent, 'lr'):
                        training_analytics['learning_rate_schedule'].append(agent.lr)
                    elif hasattr(agent, 'algorithm') and hasattr(agent.algorithm, 'learning_rate'):
                        training_analytics['learning_rate_schedule'].append(agent.algorithm.learning_rate)
                except:
                    pass  # Ignore if learning rate not available
                
                # Check for best performance
                if episode_reward > best_performance:
                    best_performance = episode_reward
                    training_analytics['best_performance'] = best_performance
                    episodes_without_improvement = 0
                    
                    # Save best model (with error handling)
                    try:
                        best_model_path = self._save_best_model(agent, agent_name, episode, episode_reward)
                        if hasattr(agent, '__dict__'):
                            best_model_state = copy.deepcopy(agent.__dict__)
                    except Exception as save_error:
                        self.logger.warning(f"Failed to save best model for {agent_name}: {save_error}")
                else:
                    episodes_without_improvement += 1
                
                # Check for convergence
                if self._check_convergence(training_analytics, episode):
                    self.logger.info(f"Agent {agent_name} converged at episode {episode}")
                    training_analytics['convergence_episode'] = episode
                    break
                
                # Early stopping
                if episodes_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping for {agent_name} at episode {episode}")
                    break
                
                # Periodic checkpointing
                if episode % self.checkpoint_frequency == 0 and episode > 0:
                    try:
                        self._save_checkpoint(agent, agent_name, episode)
                    except Exception as checkpoint_error:
                        self.logger.warning(f"Failed to save checkpoint for {agent_name}: {checkpoint_error}")
                
                episodes_completed = episode + 1
                
            except Exception as episode_error:
                self.logger.error(f"Episode {episode} failed for {agent_name}: {episode_error}")
                # Continue to next episode instead of failing completely
                episodes_completed = episode
                continue
        
        # Calculate final metrics
        training_analytics['training_time'] = time.time() - start_time
        
        if training_analytics['episode_rewards']:
            training_analytics.update(
                self._calculate_convergence_metrics(training_analytics['episode_rewards'])
            )
        
        return {
            'status': 'completed' if episodes_completed > 0 else 'failed',
            'agent_name': agent_name,
            'final_performance': best_performance,
            'total_episodes': episodes_completed,
            'training_analytics': training_analytics,
            'best_model_path': best_model_path,
            'convergence_episode': training_analytics.get('convergence_episode', 0)
        }
    
    def _save_best_model(self, agent, agent_name: str, episode: int, performance: float) -> Optional[str]:
        """
        Save the best model checkpoint with proper error handling and verification.
        
        Args:
            agent: Agent instance to save
            agent_name: Name of the agent
            episode: Current episode number
            performance: Performance metric value
            
        Returns:
            Path to saved model if successful, None if failed
        """
        try:
            # Create model filename with timestamp
            timestamp = int(time.time())
            model_filename = f"best_{agent_name}_{episode}_{timestamp}.pkl"
            model_path = self.models_dir / model_filename
            
            # Model metadata
            metadata = {
                'agent_name': agent_name,
                'episode': episode,
                'performance': performance,
                'timestamp': timestamp,
                'config': self.config.to_dict()
            }
            
            # Prepare model data
            model_data = {
                'agent_state': agent,
                'metadata': metadata
            }
            
            # Attempt to save model with verification
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                # Verify file was actually written and has content
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file was not created: {model_path}")
                
                file_size = model_path.stat().st_size
                if file_size == 0:
                    raise ValueError(f"Model file is empty: {model_path}")
                
                # Test if file can be read back
                try:
                    with open(model_path, 'rb') as f:
                        test_load = pickle.load(f)
                    if 'agent_state' not in test_load or 'metadata' not in test_load:
                        raise ValueError("Saved model data structure is invalid")
                except Exception as read_error:
                    raise ValueError(f"Saved model cannot be read back: {read_error}")
                
                # Save metadata separately (JSON format for easier inspection)
                metadata_path = self.models_dir / f"metadata_{agent_name}_{episode}_{timestamp}.json"
                try:
                    import json
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as json_error:
                    # Metadata save failure shouldn't fail the whole operation
                    self.logger.warning(f"Failed to save metadata for {agent_name}: {json_error}")
                
                # Success logging with file size
                self.logger.info(f"✅ Successfully saved model for {agent_name} at episode {episode}")
                self.logger.info(f"   File: {model_path} ({file_size:,} bytes)")
                return str(model_path)
                
            except (pickle.PickleError, pickle.PicklingError) as pickle_error:
                # Handle pickle-specific errors (common with PyTorch models)
                self.logger.error(f"❌ Pickle serialization failed for {agent_name}: {pickle_error}")
                
                # Try alternative serialization for specific agent types
                if agent_name in ['dqn', 'sac']:
                    self.logger.info(f"Attempting PyTorch-specific save for {agent_name}")
                    return self._save_pytorch_model(agent, agent_name, episode, performance)
                
                return None
                
            except (IOError, OSError) as io_error:
                # Handle file system errors
                self.logger.error(f"❌ File system error saving model for {agent_name}: {io_error}")
                return None
                
        except Exception as general_error:
            # Catch any other unexpected errors
            self.logger.error(f"❌ Unexpected error saving model for {agent_name}: {general_error}")
            return None

    def _save_pytorch_model(self, agent, agent_name: str, episode: int, performance: float) -> Optional[str]:
        """
        Alternative save method for PyTorch-based agents (DQN, SAC).
        
        Args:
            agent: PyTorch-based agent
            agent_name: Name of the agent  
            episode: Current episode number
            performance: Performance metric
            
        Returns:
            Path to saved model if successful, None if failed
        """
        try:
            import torch
            
            timestamp = int(time.time())
            model_filename = f"best_{agent_name}_{episode}_{timestamp}.pth"
            model_path = self.models_dir / model_filename
            
            # Extract PyTorch state dict instead of full agent
            if hasattr(agent, 'q_network'):  # DQN
                model_state = {
                    'q_network_state_dict': agent.q_network.state_dict(),
                    'target_network_state_dict': agent.target_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'config': agent.config,
                    'episode': episode,
                    'performance': performance
                }
            elif hasattr(agent, 'policy'):  # SAC  
                model_state = {
                    'policy_state_dict': agent.policy.state_dict(),
                    'config': agent.config,
                    'episode': episode,
                    'performance': performance
                }
            else:
                # Fallback: try to get any state dict
                model_state = {
                    'agent_config': getattr(agent, 'config', {}),
                    'episode': episode,
                    'performance': performance
                }
            
            torch.save(model_state, model_path)
            
            # Verify save
            if model_path.exists() and model_path.stat().st_size > 0:
                self.logger.info(f"✅ Successfully saved PyTorch model for {agent_name}: {model_path}")
                return str(model_path)
            else:
                self.logger.error(f"❌ PyTorch model save verification failed for {agent_name}")
                return None
                
        except Exception as torch_error:
            self.logger.error(f"❌ PyTorch model save failed for {agent_name}: {torch_error}")
            return None
    
    def _save_checkpoint(self, agent, agent_name: str, episode: int):
        """Save periodic checkpoint."""
        checkpoint_dir = self.models_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{agent_name}_ep{episode}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(agent, f)
    
    def _check_convergence(self, analytics: Dict[str, Any], current_episode: int) -> bool:
        """
        Check if training has converged.
        
        Args:
            analytics: Training analytics
            current_episode: Current episode number
            
        Returns:
            True if converged, False otherwise
        """
        if current_episode < 20:  # Need minimum episodes
            return False
        
        recent_rewards = analytics['episode_rewards'][-10:]
        
        if len(recent_rewards) < 10:
            return False
        
        # Check if variance in recent rewards is below threshold
        reward_variance = np.var(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        # Coefficient of variation as convergence metric
        if reward_mean != 0:
            cv = np.sqrt(reward_variance) / abs(reward_mean)
            return cv < self.convergence_threshold
        
        return False
    
    def _calculate_convergence_metrics(self, episode_rewards: List[float]) -> Dict[str, float]:
        """Calculate convergence-related metrics."""
        if not episode_rewards:
            return {}
        
        rewards = np.array(episode_rewards)
        
        # Moving average convergence
        window_size = min(10, len(rewards) // 4)
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        else:
            moving_avg = rewards
        
        # Trend analysis
        episodes = np.arange(len(rewards))
        if len(episodes) > 1:
            trend_slope = np.polyfit(episodes, rewards, 1)[0]
        else:
            trend_slope = 0
        
        return {
            'final_mean_reward': float(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)),
            'reward_variance': float(np.var(rewards)),
            'reward_std': float(np.std(rewards)),
            'trend_slope': float(trend_slope),
            'improvement_rate': float((rewards[-1] - rewards[0]) / len(rewards) if len(rewards) > 1 else 0),
            'stability_index': float(1.0 / (1.0 + np.std(rewards[-10:]) if len(rewards) >= 10 else np.std(rewards)))
        }
    
    def _generate_training_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        summary = {
            'total_agents': len(training_results['agent_results']),
            'successful_agents': 0,
            'failed_agents': 0,
            'convergence_summary': {},
            'performance_ranking': [],
            'training_efficiency': {}
        }
        
        agent_performances = []
        
        for agent_name, results in training_results['agent_results'].items():
            if results.get('status') == 'completed':
                summary['successful_agents'] += 1
                
                # Performance ranking
                performance = results.get('final_performance', float('-inf'))
                agent_performances.append((agent_name, performance))
                
                # Convergence info
                analytics = results.get('training_analytics', {})
                summary['convergence_summary'][agent_name] = {
                    'converged': analytics.get('convergence_episode', 0) > 0,
                    'convergence_episode': analytics.get('convergence_episode', 'Not converged'),
                    'best_performance': analytics.get('best_performance', 0),
                    'training_time': analytics.get('training_time', 0),
                    'total_episodes': results.get('total_episodes', 0)
                }
                
                # Training efficiency
                training_time = analytics.get('training_time', 1)
                episodes = results.get('total_episodes', 1)
                summary['training_efficiency'][agent_name] = {
                    'time_per_episode': training_time / episodes,
                    'performance_per_time': performance / training_time,
                    'episodes_to_best': analytics.get('convergence_episode', episodes)
                }
            else:
                summary['failed_agents'] += 1
        
        # Sort agents by performance
        agent_performances.sort(key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = [agent[0] for agent in agent_performances]
        
        return summary