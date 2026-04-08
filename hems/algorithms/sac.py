#========================
# hems/algorithms/sac.py
#========================
"""
Soft Actor-Critic (SAC) algorithm implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseAlgorithm

try:
    from hems.utils.utils import ObservationNormalizer
except ImportError:
    pass


class ContinuousReplayBuffer:
    """Fixed-size circular replay buffer for continuous action spaces.

    Stores experience tuples (observation, action, reward, next_observation, done)
    as pre-allocated GPU/CPU tensors for efficient sampling during SAC training.
    Overwrites oldest transitions once capacity is reached.

    Attributes:
        capacity: Maximum number of transitions the buffer can hold.
        obs_dim: Dimensionality of observation vectors.
        action_dim: Dimensionality of action vectors.
        device: Torch device where buffer tensors are allocated.
        position: Index at which the next transition will be written.
        size: Current number of stored transitions.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device):
        """Initializes the replay buffer with pre-allocated tensors.

        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Dimensionality of observation vectors.
            action_dim: Dimensionality of action vectors.
            device: Torch device (CPU or CUDA) for tensor storage.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate tensors
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, obs, action, reward, next_obs, done):
        """Stores a single transition in the buffer.

        Writes the transition at the current position and advances the circular
        pointer. If the buffer is full, the oldest transition is overwritten.

        Args:
            obs: Current observation as a numpy array or list of shape ``(obs_dim,)``.
            action: Action taken as a numpy array or list of shape ``(action_dim,)``.
            reward: Scalar reward received.
            next_obs: Next observation as a numpy array or list of shape ``(obs_dim,)``.
            done: Whether the episode terminated after this transition.
        """
        self.observations[self.position] = torch.FloatTensor(obs).to(self.device)
        self.actions[self.position] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.position] = reward
        self.next_observations[self.position] = torch.FloatTensor(next_obs).to(self.device)
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Samples a random batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of ``(observations, actions, rewards, next_observations, dones)``
            tensors, each with batch dimension equal to ``batch_size``.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        """Returns the current number of transitions stored in the buffer."""
        return self.size


class GaussianPolicy(nn.Module):
    """Stochastic Gaussian policy network for continuous action spaces.

    Outputs a mean and log-standard-deviation for a diagonal Gaussian
    distribution over actions. Uses a shared feature extractor followed
    by separate linear heads for mean and log-std.

    Attributes:
        log_std_min: Lower clamp for log standard deviation.
        log_std_max: Upper clamp for log standard deviation.
        shared: Shared feature-extraction layers (Linear + ReLU stack).
        mean_head: Linear head producing action means.
        log_std_head: Linear head producing action log standard deviations.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: List[int],
                 log_std_min: float = -20, log_std_max: float = 2):
        """Initializes the Gaussian policy network.

        Args:
            obs_dim: Dimensionality of the observation input.
            action_dim: Dimensionality of the action output.
            hidden_layers: List of hidden layer sizes for the shared feature
                extractor.
            log_std_min: Minimum value for clamped log standard deviation.
            log_std_max: Maximum value for clamped log standard deviation.
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, obs):
        """Computes the mean and clamped log standard deviation of the policy.

        Args:
            obs: Observation tensor of shape ``(batch_size, obs_dim)``.

        Returns:
            Tuple of ``(mean, log_std)`` tensors, each of shape
            ``(batch_size, action_dim)``.
        """
        shared_features = self.shared(obs)
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs):
        """Samples an action using the reparameterization trick.

        Draws from the Gaussian distribution, applies ``tanh`` squashing,
        and computes the corrected log probability accounting for the
        tanh transformation.

        Args:
            obs: Observation tensor of shape ``(batch_size, obs_dim)``.

        Returns:
            Tuple of ``(action, log_prob, mean)`` where:
                - ``action``: Squashed action in ``[-1, 1]`` of shape
                  ``(batch_size, action_dim)``.
                - ``log_prob``: Log probability of the sampled action of shape
                  ``(batch_size, 1)``.
                - ``mean``: Raw (pre-squash) mean of shape
                  ``(batch_size, action_dim)``.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class QNetwork(nn.Module):
    """State-action value (Q) network for the SAC critic.

    Concatenates observation and action vectors, then passes them through
    a fully connected network with ReLU activations to produce a scalar
    Q-value estimate. Two instances are used in SAC for clipped double-Q
    learning.

    Attributes:
        network: Sequential MLP mapping ``(obs, action)`` to a scalar Q-value.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: List[int]):
        """Initializes the Q-network.

        Args:
            obs_dim: Dimensionality of the observation input.
            action_dim: Dimensionality of the action input.
            hidden_layers: List of hidden layer sizes for the MLP.
        """
        super().__init__()

        layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs, action):
        """Computes the Q-value for an observation-action pair.

        Args:
            obs: Observation tensor of shape ``(batch_size, obs_dim)``.
            action: Action tensor of shape ``(batch_size, action_dim)``.

        Returns:
            Q-value tensor of shape ``(batch_size, 1)``.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class SACAlgorithm(BaseAlgorithm):
    """Soft Actor-Critic (SAC) algorithm with automatic entropy tuning.

    Implements the SAC algorithm with twin Q-networks, a Gaussian stochastic
    policy, and optional automatic entropy coefficient tuning. Supports both
    sequential (single-building) and parallel (shared-weight multi-building)
    training modes.

    Reference:
        Haarnoja et al. (2018) "Soft Actor-Critic Algorithms and Applications"

    Attributes:
        device: Torch device used for computation.
        obs_dim: Observation dimensionality for a single building.
        n_buildings: Number of buildings in the environment.
        multi_head: Whether parallel shared-weight mode is active.
        policy: Gaussian policy (actor) network.
        q1: First Q-network (critic).
        q2: Second Q-network (critic).
        q1_target: Target network for q1.
        q2_target: Target network for q2.
        alpha: Entropy temperature coefficient (learnable or fixed).
        replay_buffer: Experience replay buffer.
        obs_normalizer: Running normalizer for observations.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        batch_size: Mini-batch size for training updates.
        total_steps: Total environment steps taken so far.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """Initializes the SAC algorithm.

        Args:
            env: CityLearn-compatible environment with ``observation_space``
                and ``buildings`` attributes.
            config: Configuration dictionary. Supported keys include:
                - ``use_gpu`` (bool): Whether to use CUDA if available.
                - ``multi_head`` (bool): Enable shared-weight parallel mode.
                - ``hidden_layers`` (List[int]): Hidden layer sizes.
                - ``auto_entropy_tuning`` (bool): Enable learnable alpha.
                - ``alpha`` (float): Fixed entropy coefficient (if auto is off).
                - ``alpha_lr`` (float): Learning rate for alpha optimizer.
                - ``lr_policy`` (float): Policy optimizer learning rate.
                - ``lr_q`` (float): Q-network optimizer learning rate.
                - ``buffer_size`` (int): Replay buffer capacity.
                - ``gamma`` (float): Discount factor.
                - ``tau`` (float): Soft update coefficient.
                - ``batch_size`` (int): Training batch size.
                - ``train_start_steps`` (int): Steps before training begins.
                - ``update_frequency`` (int): Steps between training updates.

        Raises:
            ImportError: If ``ObservationNormalizer`` cannot be created.
        """
        super().__init__(env, config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', False) else 'cpu')
        
        # Environment dimensions
        self.obs_dims = [s.shape[0] for s in env.observation_space]
        self.obs_dim = self.obs_dims[0]  # Single building observation size
        self.n_buildings = len(env.buildings)
        
        # Check if multi-head mode (parallel with multiple buildings)
        self.multi_head = config.get('multi_head', False) and self.n_buildings > 1
        
        # Network input dimension: ALWAYS single building observation for shared-weight architecture
        # In parallel mode, we process each building separately through the same network
        self.network_input_dim = self.obs_dim
        self.action_output_dim = 1  # Single action per building
        
        hidden_layers = config.get('hidden_layers', [256, 256])
        
        # Policy network
        self.policy = GaussianPolicy(self.network_input_dim, self.action_output_dim, hidden_layers).to(self.device)
        
        # Q networks (twin delayed)
        self.q1 = QNetwork(self.network_input_dim, self.action_output_dim, hidden_layers).to(self.device)
        self.q2 = QNetwork(self.network_input_dim, self.action_output_dim, hidden_layers).to(self.device)
        
        # Target Q networks
        self.q1_target = QNetwork(self.network_input_dim, self.action_output_dim, hidden_layers).to(self.device)
        self.q2_target = QNetwork(self.network_input_dim, self.action_output_dim, hidden_layers).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = config.get('auto_entropy_tuning', True)
        if self.auto_entropy_tuning:
            self.target_entropy = -self.action_output_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get('alpha_lr', 3e-4))
        else:
            self.alpha = config.get('alpha', 0.2)
        
        # Optimizers
        lr_policy = config.get('lr_policy', 3e-4)
        lr_q = config.get('lr_q', 3e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr_q)
        
        # Replay buffer
        buffer_size = config.get('buffer_size', 300_000)
        self.replay_buffer = ContinuousReplayBuffer(buffer_size, self.network_input_dim, 
                                                   self.action_output_dim, device=self.device)
        
        # Observation normalizer
        try:
            self.obs_normalizer = ObservationNormalizer(self.network_input_dim)
        except Exception as e:
            raise ImportError(f"Failed to create ObservationNormalizer: {e}")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        self.train_start_steps = config.get('train_start_steps', 1000)
        self.update_frequency = config.get('update_frequency', 1)
        
        # Training state
        self.total_steps = 0
        self.policy_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.alpha_losses = []
        self.alpha_values = []
        
        mode_str = "Parallel (shared-weight)" if self.multi_head else "Sequential/Single"
        print(f"SAC Algorithm initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mode: {mode_str}")
        print(f"  Buildings: {self.n_buildings}")
        print(f"  Obs dim per building: {self.obs_dim}")
        print(f"  Network input dim: {self.network_input_dim} (single building)")
        print(f"  Action output dim: {self.action_output_dim}")
        print(f"  Auto entropy tuning: {self.auto_entropy_tuning}")
        print(f"  Network: {hidden_layers}")
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Selects actions for all buildings using the Gaussian policy network.

        In parallel (shared-weight) mode, each building's observation is
        processed independently through the same policy network. In
        sequential/single mode, the first building's observation is used
        and the resulting action is broadcast to all buildings.

        Args:
            observations: List of per-building observation vectors, each of
                length ``obs_dim``.
            deterministic: If True, uses the tanh-squashed mean action
                instead of sampling from the policy distribution.

        Returns:
            Nested list of shape ``[[a1, a2, ...]]`` containing one action
            per building, wrapped in a single outer list (centralized format).
        """
        n_current_buildings = len(observations)
        
        if self.multi_head:
            # Parallel shared-weight mode: process each building separately through network
            actions = []
            
            for i in range(n_current_buildings):
                # Get single building observation
                building_obs = np.array(observations[i], dtype=np.float32)
                
                # Normalize
                self.obs_normalizer.update(building_obs)
                obs_normalized = self.obs_normalizer.normalize(building_obs)
                
                # Get action for this building
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                    if deterministic:
                        mean, _ = self.policy(obs_tensor)
                        action = torch.tanh(mean)
                    else:
                        action, _, _ = self.policy.sample(obs_tensor)
                    
                    action_value = action.item()
                    actions.append(action_value)
        else:
            # Sequential/single building mode: same action for all buildings
            # Use first building observation
            building_obs = np.array(observations[0], dtype=np.float32)
            
            self.obs_normalizer.update(building_obs)
            obs_normalized = self.obs_normalizer.normalize(building_obs)
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                if deterministic:
                    mean, _ = self.policy(obs_tensor)
                    action = torch.tanh(mean)
                else:
                    action, _, _ = self.policy.sample(obs_tensor)
                
                action_value = action.item()
                actions = [action_value] * n_current_buildings
        
        return [actions]
    
    def learn(self, obs, actions, reward, next_obs, done) -> Optional[Dict[str, Any]]:
        """Stores a transition and optionally performs a training update.

        In parallel mode, each building's experience is stored as a separate
        transition in the replay buffer. In sequential mode, only the first
        building's observation is used. A training step is triggered once the
        buffer exceeds ``train_start_steps`` and the step count aligns with
        ``update_frequency``.

        Args:
            obs: List of per-building observation vectors.
            actions: Actions in centralized format ``[[a1, a2, ...]]`` or
                flat list ``[a1, a2, ...]``.
            reward: Scalar reward or list of per-building rewards.
            next_obs: List of per-building next-observation vectors.
            done: Whether the episode has terminated.

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
                self.obs_normalizer.update(building_obs)
                self.obs_normalizer.update(building_next_obs)
                
                # Normalize
                obs_norm = self.obs_normalizer.normalize(building_obs)
                next_obs_norm = self.obs_normalizer.normalize(building_next_obs)
                
                # Action
                action_val = float(building_actions[i]) if i < len(building_actions) else 0.0
                action_array = np.array([action_val], dtype=np.float32)
                
                # Store in replay buffer
                self.replay_buffer.push(obs_norm, action_array, rewards[i], next_obs_norm, done)
            
            self.total_steps += 1
        else:
            # Sequential mode: use single building observation
            building_obs = np.array(obs[0], dtype=np.float32)
            building_next_obs = np.array(next_obs[0], dtype=np.float32)
            
            # Update normalizer
            self.obs_normalizer.update(building_obs)
            self.obs_normalizer.update(building_next_obs)
            
            # Normalize observations
            obs_norm = self.obs_normalizer.normalize(building_obs)
            next_obs_norm = self.obs_normalizer.normalize(building_next_obs)
            
            # Convert actions to scalar
            if isinstance(actions, list) and len(actions) > 0:
                if isinstance(actions[0], list):
                    action_val = float(actions[0][0])
                else:
                    action_val = float(actions[0])
            else:
                action_val = 0.0
            
            action_array = np.array([action_val], dtype=np.float32)
            
            # Store experience
            reward_scalar = float(reward) if np.isscalar(reward) else float(np.mean(reward))
            self.replay_buffer.push(obs_norm, action_array, reward_scalar, next_obs_norm, done)
            
            self.total_steps += 1
        
        # Train if enough experience
        if len(self.replay_buffer) >= self.train_start_steps and self.total_steps % self.update_frequency == 0:
            self._train_step()
        
        return self.get_training_stats()
    
    def _train_step(self):
        """Performs one full SAC training update.

        Executes the following steps in order:
            1. Samples a mini-batch from the replay buffer.
            2. Updates both Q-networks using clipped double-Q Bellman targets.
            3. Updates the policy to maximize expected Q minus entropy penalty.
            4. Updates the entropy coefficient alpha (if auto-tuning is on).
            5. Soft-updates both target Q-networks toward the online networks.

        No-ops if the replay buffer contains fewer samples than ``batch_size``.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # Ensure action_batch has correct shape
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)
        
        # ========== Update Q networks ==========
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.policy.sample(next_obs_batch)
            
            # Compute target Q values
            q1_next_target = self.q1_target(next_obs_batch, next_actions)
            q2_next_target = self.q2_target(next_obs_batch, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Add entropy term
            if self.auto_entropy_tuning:
                alpha = self.alpha.detach()
            else:
                alpha = self.alpha
            
            next_q_value = min_q_next_target - alpha * next_log_probs
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_value
        
        # Q1 loss
        q1_value = self.q1(obs_batch, action_batch)
        q1_loss = F.mse_loss(q1_value, target_q_value)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        # Q2 loss
        q2_value = self.q2(obs_batch, action_batch)
        q2_loss = F.mse_loss(q2_value, target_q_value)
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # ========== Update policy ==========
        new_actions, log_probs, _ = self.policy.sample(obs_batch)
        q1_new = self.q1(obs_batch, new_actions)
        q2_new = self.q2(obs_batch, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        if self.auto_entropy_tuning:
            alpha = self.alpha
        else:
            alpha = self.alpha
        
        policy_loss = (alpha * log_probs - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # ========== Update alpha (entropy coefficient) ==========
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            self.alpha_losses.append(alpha_loss.item())
            self.alpha_values.append(self.alpha.item())
        
        # ========== Soft update target networks ==========
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # Track losses
        self.policy_losses.append(policy_loss.item())
        self.q1_losses.append(q1_loss.item())
        self.q2_losses.append(q2_loss.item())
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Performs Polyak averaging to update target network parameters.

        Each target parameter is updated as:
        ``target = tau * source + (1 - tau) * target``

        Args:
            source: Online network whose parameters are copied from.
            target: Target network whose parameters are updated in-place.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Returns recent training statistics averaged over the last 100 updates.

        Returns:
            Dictionary with keys ``total_steps``, ``buffer_size``,
            ``avg_policy_loss``, ``avg_q1_loss``, ``avg_q2_loss``, and
            optionally ``avg_alpha`` and ``avg_alpha_loss`` when automatic
            entropy tuning is enabled.
        """
        stats = {
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0.0,
            'avg_q1_loss': np.mean(self.q1_losses[-100:]) if self.q1_losses else 0.0,
            'avg_q2_loss': np.mean(self.q2_losses[-100:]) if self.q2_losses else 0.0,
        }
        
        if self.auto_entropy_tuning and self.alpha_values:
            stats['avg_alpha'] = np.mean(self.alpha_values[-100:])
            stats['avg_alpha_loss'] = np.mean(self.alpha_losses[-100:]) if self.alpha_losses else 0.0
        
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        """Serializes the full algorithm state for checkpointing.

        Returns:
            Dictionary containing state dicts for all networks, optimizers,
            the step counter, and (if applicable) the learned entropy
            coefficient and its optimizer state.
        """
        state = {
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }
        
        if self.auto_entropy_tuning:
            state['log_alpha'] = self.log_alpha.detach().cpu().numpy()
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Restores algorithm state from a previously saved checkpoint.

        Gracefully handles partial checkpoints by loading only the keys
        that are present in the state dictionary.

        Args:
            state: Dictionary produced by ``get_state()``, containing network
                and optimizer state dicts.

        Raises:
            Exception: If any state dict fails to load (e.g., shape mismatch).
        """
        try:
            if 'policy' in state:
                self.policy.load_state_dict(state['policy'])
            if 'q1' in state:
                self.q1.load_state_dict(state['q1'])
            if 'q2' in state:
                self.q2.load_state_dict(state['q2'])
            if 'q1_target' in state:
                self.q1_target.load_state_dict(state['q1_target'])
            if 'q2_target' in state:
                self.q2_target.load_state_dict(state['q2_target'])
            
            if 'policy_optimizer' in state:
                self.policy_optimizer.load_state_dict(state['policy_optimizer'])
            if 'q1_optimizer' in state:
                self.q1_optimizer.load_state_dict(state['q1_optimizer'])
            if 'q2_optimizer' in state:
                self.q2_optimizer.load_state_dict(state['q2_optimizer'])
            
            if 'total_steps' in state:
                self.total_steps = state['total_steps']
            
            if self.auto_entropy_tuning and 'log_alpha' in state:
                self.log_alpha.data = torch.FloatTensor(state['log_alpha']).to(self.device)
                self.alpha = self.log_alpha.exp()
                if 'alpha_optimizer' in state:
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
            
            print(f"[SAC] Successfully loaded model state")
            
        except Exception as e:
            print(f"[SAC] Error loading state: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Returns metadata and hyperparameters for this SAC instance.

        Extends the base class info with SAC-specific fields including
        description, trainability flag, and all key hyperparameters.

        Returns:
            Dictionary with algorithm metadata and a nested ``parameters``
            dictionary containing buffer size, learning rates, discount
            factor, soft-update rate, and entropy tuning settings.
        """
        info = super().get_info()
        info.update({
            'description': 'Soft Actor-Critic with automatic entropy tuning',
            'trainable': True,
            'parameters': {
                'buffer_size': self.config.get('buffer_size', 300_000),
                'lr_policy': self.config.get('lr_policy', 3e-4),
                'lr_q': self.config.get('lr_q', 3e-4),
                'gamma': self.gamma,
                'tau': self.tau,
                'auto_entropy_tuning': self.auto_entropy_tuning,
                'target_entropy': self.target_entropy if self.auto_entropy_tuning else None
            }
        })
        return info