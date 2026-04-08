#====================
#hems/algorithms/dqn
#====================

"""
Deep Q-Network (DQN) algorithm with dueling architecture and double DQN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional
from .base import BaseAlgorithm

# Import utilities (assuming they exist in the utils module)
try:
    from hems.utils.utils import ReplayBuffer, ObservationNormalizer
except ImportError:
    # Fallback if utils not available during refactor
    pass


class DuelingDQN(nn.Module):
    """Dueling DQN neural network with shared feature layers and separate value/advantage streams.

    Implements the dueling architecture from Wang et al. (2016), which decomposes
    Q-values into a state-value function V(s) and an advantage function A(s, a).
    Supports both single-head (one output) and multi-head (per-building outputs)
    configurations.

    Attributes:
        n_heads: Number of output heads. When 1, uses standard dueling architecture.
            When >1, creates separate value and advantage heads per building.
        shared: Sequential module containing the shared feature extraction layers.
        value_head: Linear layer for state-value estimation (single-head mode).
        advantage_head: Linear layer for advantage estimation (single-head mode).
        value_heads: ModuleList of per-building value heads (multi-head mode).
        advantage_heads: ModuleList of per-building advantage heads (multi-head mode).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], n_heads: int = 1):
        """Initializes the DuelingDQN network.

        Args:
            input_dim: Dimensionality of the input observation vector.
            output_dim: Number of discrete actions (output neurons per head).
            hidden_layers: List of hidden layer sizes for the shared feature network.
            n_heads: Number of output heads. Use 1 for standard mode, >1 for
                multi-building parallel mode with separate heads per building.
        """
        super().__init__()
        
        self.n_heads = n_heads
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        if n_heads == 1:
            # Single-head: standard dueling architecture
            self.value_head = nn.Linear(prev_dim, 1)
            self.advantage_head = nn.Linear(prev_dim, output_dim)
        else:
            # Multi-head: separate value and advantage for each building
            self.value_heads = nn.ModuleList([nn.Linear(prev_dim, 1) for _ in range(n_heads)])
            self.advantage_heads = nn.ModuleList([nn.Linear(prev_dim, output_dim) for _ in range(n_heads)])
    
    def forward(self, x):
        """Computes Q-values using the dueling architecture.

        Passes input through shared layers, then splits into value and advantage
        streams. Q-values are computed as Q(s,a) = V(s) + (A(s,a) - mean(A(s,.))).

        Args:
            x: Input observation tensor of shape (batch_size, input_dim).

        Returns:
            In single-head mode: Tensor of Q-values with shape
                (batch_size, n_actions).
            In multi-head mode: List of Q-value tensors, one per head, each
                with shape (batch_size, n_actions).
        """
        shared_features = self.shared(x)
        
        if self.n_heads == 1:
            # Single-head forward
            value = self.value_head(shared_features)
            advantage = self.advantage_head(shared_features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # Multi-head forward: return list of Q-values per head
            q_values_per_head = []
            for i in range(self.n_heads):
                value = self.value_heads[i](shared_features)
                advantage = self.advantage_heads[i](shared_features)
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                q_values_per_head.append(q_values)
            return q_values_per_head


class DQNAlgorithm(BaseAlgorithm):
    """Deep Q-Network algorithm with dueling architecture and double DQN.

    Implements a DQN agent that uses a dueling network architecture, double
    DQN target computation, experience replay, epsilon-greedy exploration
    with linear decay, and observation normalization. Supports both sequential
    (single-building) and parallel (shared-weight multi-building) modes.

    The agent discretizes the continuous action space into a fixed number of
    bins and selects among them. In parallel mode, each building's observation
    is processed independently through a shared network.

    Attributes:
        device: Torch device (CPU or CUDA) used for computation.
        obs_dim: Observation dimensionality for a single building.
        n_buildings: Number of buildings in the environment.
        multi_head: Whether to use multi-head parallel mode.
        n_bins: Number of discrete action bins.
        discrete_actions: Array of evenly spaced action values in [-1, 1].
        n_actions: Number of discrete actions (same as n_bins).
        q_network: Online dueling DQN network.
        target_network: Target dueling DQN network for stable Q-value estimation.
        optimizer: Adam optimizer for the Q-network.
        replay_buffer: Experience replay buffer.
        obs_normalizer: Running observation normalizer.
        gamma: Discount factor for future rewards.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate after decay.
        epsilon_decay_steps: Number of steps over which epsilon decays linearly.
        target_update_freq: Steps between target network updates.
        batch_size: Mini-batch size for training.
        train_start_steps: Minimum replay buffer size before training begins.
        total_steps: Total environment steps taken so far.
        training_losses: History of training loss values.
        epsilon_history: History of epsilon values during training.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """Initializes the DQN algorithm.

        Args:
            env: CityLearn environment instance providing observation_space
                and buildings attributes.
            config: Algorithm configuration dictionary. Supported keys include:
                - use_gpu (bool): Whether to use GPU if available.
                - multi_head (bool): Enable multi-head parallel mode.
                - action_bins (int): Number of discrete action bins (default 31).
                - hidden_layers (list[int]): Hidden layer sizes (default [256, 256]).
                - learning_rate (float): Adam learning rate (default 3e-4).
                - buffer_size (int): Replay buffer capacity (default 300000).
                - gamma (float): Discount factor (default 0.99).
                - epsilon_start (float): Initial epsilon (default 1.0).
                - epsilon_end (float): Final epsilon (default 0.08).
                - epsilon_decay_steps (int): Steps for linear decay (default 150000).
                - target_update_freq (int): Target network update interval (default 1000).
                - batch_size (int): Training batch size (default 128).
                - train_start_steps (int): Minimum buffer size before training (default 1000).

        Raises:
            ImportError: If ReplayBuffer or ObservationNormalizer cannot be
                created from hems.utils.utils.
        """
        super().__init__(env, config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', getattr(config, 'use_gpu', False)) else 'cpu')
        
        # Environment dimensions
        self.obs_dims = [s.shape[0] for s in env.observation_space]
        # CHANGED: Use single building obs_dim for flexible building counts
        self.obs_dim = self.obs_dims[0]  # Single building observation size
        self.n_buildings = len(env.buildings)
        
        # Check if multi-head mode (true parallel with multiple buildings)
        # NOTE: Shared-weight parallel processes each building separately through same network
        self.multi_head = config.get('multi_head', False) and self.n_buildings > 1
        
        # Action discretization
        self.n_bins = config.get('action_bins', 31)
        self.discrete_actions = np.linspace(-1.0, 1.0, self.n_bins, dtype=np.float32)
        self.n_actions = self.n_bins  # Actions per building
        
        # Network input dimension: ALWAYS single building observation for shared-weight architecture
        # In parallel mode, we process each building separately through the same network
        self.network_input_dim = self.obs_dim
        
        # Networks - ALWAYS single-head now (shared weights)
        hidden_layers = config.get('hidden_layers', [256, 256])
        self.q_network = DuelingDQN(self.network_input_dim, self.n_actions, hidden_layers, n_heads=1).to(self.device)
        self.target_network = DuelingDQN(self.network_input_dim, self.n_actions, hidden_layers, n_heads=1).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        lr = config.get('learning_rate', 3e-4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        buffer_size = config.get('buffer_size', 300_000)
        try:
            self.replay_buffer = ReplayBuffer(buffer_size, self.network_input_dim, device=self.device)
        except Exception as e:
            raise ImportError(f"Failed to create ReplayBuffer: {e}. Make sure hems.utils.utils is available.")
        
        # Observation normalization
        try:
            self.obs_normalizer = ObservationNormalizer(self.network_input_dim)
        except Exception as e:
            raise ImportError(f"Failed to create ObservationNormalizer: {e}. Make sure hems.utils.utils is available.")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.08)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 150_000)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.batch_size = config.get('batch_size', 128)
        self.train_start_steps = config.get('train_start_steps', 1000)
        
        # Training state
        self.total_steps = 0
        self.training_losses = []
        self.epsilon_history = []
        
        mode_str = f"Parallel (shared-weight)" if self.multi_head else "Sequential/Single"
        print(f"DQN Algorithm initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mode: {mode_str}")
        print(f"  Obs dim per building: {self.obs_dim}, Action bins: {self.n_bins}")
        print(f"  Buildings: {self.n_buildings}")
        print(f"  Network input dim: {self.network_input_dim} (single building)")
        print(f"  Network: {hidden_layers}")

    
    def _flatten_observations(self, observations: List[List[float]]) -> np.ndarray:
        """Flattens and concatenates observations from all buildings into a single vector.

        Args:
            observations: List of per-building observation lists, where each
                inner list contains float feature values for one building.

        Returns:
            A 1-D numpy float32 array containing all building observations
            concatenated sequentially.
        """
        flattened = []
        for obs in observations:
            flattened.extend(obs)
        return np.array(flattened, dtype=np.float32)
    
    def _update_normalizer_if_needed(self, obs_flat: np.ndarray):
        """Updates or recreates the observation normalizer if the dimension changes.

        When switching between buildings with different observation sizes (e.g.,
        during sequential training), the normalizer must be recreated to match
        the new dimensionality.

        Args:
            obs_flat: Flattened observation array whose length determines the
                expected normalizer dimension.
        """
        if not hasattr(self, 'obs_normalizer') or self.obs_normalizer is None:
            self.obs_normalizer = ObservationNormalizer(len(obs_flat))
        elif self.obs_normalizer.dim != len(obs_flat):
            # Dimension changed - recreate normalizer
            self.obs_normalizer = ObservationNormalizer(len(obs_flat))
    
    def _update_replay_buffer_if_needed(self, obs_flat: np.ndarray):
        """Updates or recreates the replay buffer if the observation dimension changes.

        When the observation dimensionality changes (e.g., switching buildings),
        the existing replay buffer becomes incompatible and must be recreated.
        Note that recreating the buffer discards all previously stored experiences.

        Args:
            obs_flat: Flattened observation array whose length determines the
                expected buffer observation dimension.
        """
        obs_dim = len(obs_flat)
        if not hasattr(self, 'replay_buffer') or self.replay_buffer is None:
            buffer_size = self.config.get('buffer_size', 300_000)
            self.replay_buffer = ReplayBuffer(buffer_size, obs_dim, device=self.device)
        elif self.replay_buffer.obs_dim != obs_dim:
            # Dimension changed - recreate buffer (lose old experiences)
            buffer_size = self.config.get('buffer_size', 300_000)
            self.replay_buffer = ReplayBuffer(buffer_size, obs_dim, device=self.device)
    
    def _epsilon_schedule(self) -> float:
        """Calculates the current epsilon value using a linear decay schedule.

        Epsilon decays linearly from ``epsilon_start`` to ``epsilon_end`` over
        ``epsilon_decay_steps`` total environment steps, then remains constant
        at ``epsilon_end``.

        Returns:
            Current epsilon value for epsilon-greedy exploration.
        """
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
    
    def _idx_to_actions(self, action_idx: int) -> List[float]:
        """Converts a discrete action index to a list of per-building continuous actions.

        Maps the index to its corresponding value in ``discrete_actions`` and
        broadcasts it to all buildings. In sequential mode (1 building), returns
        a single-element list. In parallel mode, returns the same action value
        repeated for each building.

        Args:
            action_idx: Index into the ``discrete_actions`` array, in range
                [0, n_bins).

        Returns:
            List of float action values, one per building, all set to the
            same discrete action value.
        """
        action_value = float(self.discrete_actions[action_idx])
        # Return list of actions, one per building
        # For sequential: n_buildings=1, returns [action]
        # For parallel: n_buildings>1, returns [action, action, ...]
        return [action_value] * self.n_buildings
    
    def _actions_to_idx(self, actions: List[List[float]]) -> int:
        """Converts a nested list of building actions back to a discrete action index.

        Extracts the first action value from the nested action structure and
        finds the closest matching index in ``discrete_actions``.

        Args:
            actions: Nested list of actions. Accepts two formats:
                - Multi-head: [[a1], [a2], ...] where each inner list has one action.
                - Single-head: [[a1, a2, ...]] where all actions are in one list.
                In both cases, only the first action value is used.

        Returns:
            Index of the closest discrete action in ``discrete_actions``.
        """
        # Handle multi-head format: [[action1], [action2], ...]
        if len(actions) > 0 and isinstance(actions[0], list):
            if len(actions[0]) == 1:
                # Multi-head format
                action_val = float(actions[0][0])
            else:
                # Single-head format [[action1, action2, ...]]
                action_val = float(actions[0][0])
        else:
            action_val = 0.0
        
        # Find closest discrete action
        action_idx = np.argmin(np.abs(self.discrete_actions - action_val))
    
        return int(action_idx)
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Selects actions for all buildings using an epsilon-greedy policy.

        In multi-head (parallel) mode, each building's observation is processed
        independently through the shared Q-network. In single-head (sequential)
        mode, observations are flattened and a single action is broadcast to
        all buildings.

        Args:
            observations: List of per-building observation lists, where
                ``observations[i]`` is a list of float features for building i.
            deterministic: If True, always selects the greedy (highest Q-value)
                action with no exploration. If False, uses epsilon-greedy.

        Returns:
            Actions in centralized format: a list containing one inner list of
            float action values, i.e. ``[[a1, a2, ...]]`` where each value
            is in [-1, 1].
        """
        # Get current number of buildings
        n_current_buildings = len(observations)
        
        if self.multi_head:
            # Parallel shared-weight mode: process each building separately through network
            actions = []
            
            for i in range(n_current_buildings):
                # Get single building observation
                building_obs = np.array(observations[i], dtype=np.float32)
                
                # Normalize
                self._update_normalizer_if_needed(building_obs)
                self.obs_normalizer.update(building_obs)
                obs_normalized = self.obs_normalizer.normalize(building_obs)
                
                # Get action for this building
                if not deterministic and np.random.random() < self._epsilon_schedule():
                    # Random action for this building
                    action_idx = np.random.randint(self.n_actions)
                else:
                    # Greedy action for this building
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                        q_values = self.q_network(obs_tensor)
                        action_idx = q_values.argmax().item()
                
                action_value = float(self.discrete_actions[action_idx])
                actions.append(action_value)
        else:
            # Sequential/single building mode: same action for all buildings
            # Flatten and normalize observations
            obs_flat = self._flatten_observations(observations)
            self._update_normalizer_if_needed(obs_flat)
            self.obs_normalizer.update(obs_flat)
            obs_normalized = self.obs_normalizer.normalize(obs_flat)
            
            if not deterministic and np.random.random() < self._epsilon_schedule():
                action_idx = np.random.randint(self.n_actions)
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    action_idx = q_values.argmax().item()
            
            action_value = float(self.discrete_actions[action_idx])
            actions = [action_value] * n_current_buildings
    
        # Track epsilon
        if not deterministic:
            self.epsilon_history.append(self._epsilon_schedule())
    
        
        return [actions]
    
    def learn(self, obs, actions, reward, next_obs, done) -> Optional[Dict[str, Any]]:
        """Stores a transition in the replay buffer and performs a training step.

        In multi-head (parallel) mode, the transition is decomposed into
        per-building experiences stored separately. In single-head (sequential)
        mode, observations are flattened into a single vector. A gradient
        update is performed once the replay buffer reaches ``train_start_steps``,
        and the target network is updated every ``target_update_freq`` steps.

        Args:
            obs: Current observations as a list of per-building observation lists.
            actions: Actions taken, in centralized format ``[[a1, a2, ...]]``.
            reward: Reward received. Can be a scalar (applied to all buildings
                in parallel mode) or a list of per-building rewards.
            next_obs: Next observations, same format as ``obs``.
            done: Whether the episode has ended (bool).

        Returns:
            Dictionary of training statistics from ``get_training_stats()``.
        """
        if self.multi_head:
            # Parallel mode: store separate experience for each building
            n_buildings = len(obs)
            
            # Handle reward - can be scalar or list
            if np.isscalar(reward):
                rewards = [float(reward)] * n_buildings
            else:
                rewards = [float(r) for r in reward]
            
            # Extract actions (format is [[a1, a2, ...]])
            if isinstance(actions, list) and len(actions) > 0:
                if isinstance(actions[0], list):
                    building_actions = actions[0]  # [[a1, a2, ...]] -> [a1, a2, ...]
                else:
                    building_actions = actions
            else:
                building_actions = [0.0] * n_buildings
            
            # Store experience for each building
            for i in range(n_buildings):
                # Single building observation
                building_obs = np.array(obs[i], dtype=np.float32)
                building_next_obs = np.array(next_obs[i], dtype=np.float32)
                
                # Update normalizer
                self._update_normalizer_if_needed(building_obs)
                self.obs_normalizer.update(building_obs)
                self.obs_normalizer.update(building_next_obs)
                
                # Normalize
                obs_norm = self.obs_normalizer.normalize(building_obs)
                next_obs_norm = self.obs_normalizer.normalize(building_next_obs)
                
                # Convert action to index
                action_val = float(building_actions[i])
                action_idx = np.argmin(np.abs(self.discrete_actions - action_val))
                
                # Store in replay buffer
                self.replay_buffer.push(obs_norm, action_idx, rewards[i], next_obs_norm, done)
            
            self.total_steps += 1
        else:
            # Sequential mode: flatten observations and store single experience
            obs_flat = self._flatten_observations(obs)
            next_obs_flat = self._flatten_observations(next_obs)
            
            # Update normalizer and replay buffer dimensions if needed
            self._update_normalizer_if_needed(obs_flat)
            self._update_replay_buffer_if_needed(obs_flat)
            
            # Update normalizer statistics
            self.obs_normalizer.update(obs_flat)
            self.obs_normalizer.update(next_obs_flat)
            
            # Normalize observations
            obs_norm = self.obs_normalizer.normalize(obs_flat)
            next_obs_norm = self.obs_normalizer.normalize(next_obs_flat)
            
            # Convert actions to index
            action_idx = self._actions_to_idx(actions)
            
            # Store experience
            reward_scalar = float(reward) if np.isscalar(reward) else float(np.mean(reward))
            self.replay_buffer.push(obs_norm, action_idx, reward_scalar, next_obs_norm, done)
            
            self.total_steps += 1
        
        # Train if enough experience
        if len(self.replay_buffer) >= self.train_start_steps and self.total_steps % 1 == 0:
            self._train_step()
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return self.get_training_stats()
    
    def _train_step(self):
        """Performs one gradient update step using a mini-batch from the replay buffer.

        Samples a batch of transitions, computes Q-value predictions for the
        taken actions, calculates double DQN targets using the target network,
        and updates the Q-network via MSE loss with gradient clipping (max norm 10).

        The loss is appended to ``training_losses`` for tracking. Does nothing
        if the replay buffer contains fewer samples than ``batch_size``.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # Standard single-head training (used for both sequential and parallel modes)
        q_values = self.q_network(obs_batch)
        q_values_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN target
        with torch.no_grad():
            next_q_values = self.q_network(next_obs_batch)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            next_q_values_target = self.target_network(next_obs_batch)
            next_q_values_selected = next_q_values_target.gather(1, next_actions).squeeze(1)
            
            targets = reward_batch + (1 - done_batch) * self.gamma * next_q_values_selected
        
        # Compute loss
        total_loss = nn.functional.mse_loss(q_values_selected, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Track loss
        self.training_losses.append(total_loss.item())
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Returns a dictionary of current training statistics.

        Returns:
            Dictionary containing:
                - total_steps: Total environment steps taken.
                - buffer_size: Current number of transitions in the replay buffer.
                - avg_loss: Average training loss over the last 100 steps.
                - current_epsilon: Current exploration epsilon value.
                - losses: Full list of training loss values.
                - epsilon_history: Full list of historical epsilon values.
        """
        return {
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'current_epsilon': self._epsilon_schedule(),
            'losses': self.training_losses.copy(),
            'epsilon_history': self.epsilon_history.copy()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Serializes the full algorithm state for checkpointing.

        Captures Q-network weights, target network weights, optimizer state,
        current epsilon, step count, buffer size, and normalizer statistics
        (if available).

        Returns:
            Dictionary containing all serializable state, with keys:
                - q_network: Q-network state dict.
                - target_network: Target network state dict.
                - optimizer: Optimizer state dict.
                - epsilon: Current epsilon value.
                - steps_done: Total training steps completed.
                - buffer_size: Current replay buffer size.
                - normalizer (optional): Dict with mean, std, and count.
        """
        state = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self._epsilon_schedule(),  # Current epsilon value
            'steps_done': self.total_steps,
            'buffer_size': len(self.replay_buffer),
        }
        
        # Include normalizer state if it exists
        if hasattr(self, 'normalizer') and self.normalizer is not None:
            state['normalizer'] = {
                'mean': self.normalizer.mean,
                'std': self.normalizer.std,
                'count': self.normalizer.count
            }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Restores the algorithm state from a previously saved checkpoint.

        Loads Q-network and target network weights, optimizer state, epsilon
        value, step count, and normalizer statistics. After loading, runs a
        verification forward pass to confirm the model is functional.

        Args:
            state: Dictionary produced by ``get_state()`` containing model
                weights and training state. Missing keys are silently skipped.

        Raises:
            Exception: If loading any component fails (e.g., architecture
                mismatch in state dicts). The exception is logged with a
                traceback before being re-raised.
        """
        try:
            # Load Q-network weights
            if 'q_network' in state and state['q_network'] is not None:
                self.q_network.load_state_dict(state['q_network'])
                print(f"[DQN.load_state] Loaded Q-network weights")
            
            # Load target network weights
            if 'target_network' in state and state['target_network'] is not None:
                if hasattr(self, 'target_network'):
                    self.target_network.load_state_dict(state['target_network'])
                    print(f"[DQN.load_state] Loaded target network weights")
            
            # Load optimizer state
            if 'optimizer' in state and state['optimizer'] is not None:
                if hasattr(self, 'optimizer'):
                    self.optimizer.load_state_dict(state['optimizer'])
                    print(f"[DQN.load_state] Loaded optimizer state")
            
            # Load epsilon (for exploration)
            if 'epsilon' in state:
                # Set both start and end to loaded epsilon to keep it fixed
                self.epsilon_start = state['epsilon']
                self.epsilon_end = state['epsilon']
                print(f"[DQN.load_state] Set epsilon to {state['epsilon']:.4f}")
            
            # Load training step count
            if 'steps_done' in state:
                self.total_steps = state['steps_done']
                print(f"[DQN.load_state] Loaded total_steps: {self.total_steps}")
            
            # Load normalizer state if it exists
            if 'normalizer' in state and hasattr(self, 'normalizer') and self.normalizer is not None:
                norm_state = state['normalizer']
                self.normalizer.mean = norm_state['mean']
                self.normalizer.std = norm_state['std']
                self.normalizer.count = norm_state['count']
                print(f"[DQN.load_state] Loaded normalizer state")
            
            print(f"[DQN.load_state] Successfully loaded complete model state")
            
            # Verify model loaded by checking Q-values
            import torch
            test_input = torch.randn(1, self.network_input_dim).to(self.device)
            with torch.no_grad():
                test_output = self.q_network(test_input)
            print(f"[DQN.load_state] Verification - Q-value range: [{test_output.min().item():.2f}, {test_output.max().item():.2f}]")
            
        except Exception as e:
            print(f"[DQN.load_state] Error loading state: {e}")
            import traceback
            traceback.print_exc()
            raise

    
    def get_info(self) -> Dict[str, Any]:
        """Returns metadata and hyperparameters describing this DQN algorithm.

        Extends the base class info with DQN-specific details including
        the algorithm description, trainability flag, and key hyperparameters.

        Returns:
            Dictionary containing algorithm metadata with keys from
            ``BaseAlgorithm.get_info()`` plus 'description', 'trainable',
            and 'parameters' (a nested dict of hyperparameter values).
        """
        info = super().get_info()
        info.update({
            'description': 'Deep Q-Network with dueling architecture',
            'trainable': True,
            'parameters': {
                'action_bins': self.n_bins,
                'buffer_size': self.config.get('buffer_size', 300_000),
                'learning_rate': self.config.get('learning_rate', 3e-4),
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end
            }
        })
        return info