"""Memory-augmented DQN algorithm (CB-DQN).

Chen-Bu-inspired DQN variant that augments observations with a sliding
window of past observation-action pairs. Originally proposed for peer-to-peer
energy trading in microgrids, adapted here for residential battery control.

Reference:
    Chen & Bu, "Realistic peer-to-peer energy trading model for microgrids
    using deep reinforcement learning", IEEE PES ISGT Europe, 2019.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import deque
from .dqn import DQNAlgorithm


class MemDQNAlgorithm(DQNAlgorithm):
    """DQN variant with history-augmented phi features.

    Extends DQNAlgorithm by constructing a phi vector that concatenates
    the current observation with a sliding window of n past
    (observation, action) pairs:

        phi(t) = [obs_{t-n}, a_{t-n}, ..., obs_{t-1}, a_{t-1}, obs_t]

    This gives the agent access to temporal context without recurrent
    networks, at the cost of a larger input dimension.

    Args:
        env: CityLearn environment instance.
        config: Algorithm configuration. Supports all DQN keys plus:
            history_length (int): Number of past timesteps to include.
                Defaults to 1.
    """

    def __init__(self, env, config: Dict[str, Any]):
        # Store history configuration before calling parent init
        self.history_length = config.get('history_length', 1)  # Default to 1 for backward compatibility
        
        # Call parent init first
        super().__init__(env, config)
        
        # Store original dimensions
        self._obs_dim_single = self.obs_dim
        self._act_bins = self.n_bins
        
        # Initialize history buffers as deques with maxlen
        self._obs_history = deque(maxlen=self.history_length)
        self._action_history = deque(maxlen=self.history_length)
        
        # Calculate new phi dimension
        # phi = [obs_{t-n}, action_{t-n}, ..., obs_{t-1}, action_{t-1}, obs_t]
        # = history_length * (obs_dim + n_buildings) + obs_dim
        phi_in = self.history_length * (self._obs_dim_single + self.n_buildings) + self._obs_dim_single
        
        # UPDATE: Set obs_dim to phi dimension for all components
        self.obs_dim = phi_in
        
        # Rebuild networks with phi input size
        hidden_layers = config.get('hidden_layers', [256, 256])
        from .dqn import DuelingDQN
        self.q_network = DuelingDQN(phi_in, self.n_actions, hidden_layers).to(self.device)
        self.target_network = DuelingDQN(phi_in, self.n_actions, hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # CRITICAL: Recreate buffer and normalizer with phi dimension
        buffer_size = config.get('buffer_size', 300_000)
        
        # Get the class types from parent to ensure compatibility
        buffer_class = type(self.replay_buffer)
        normalizer_class = type(self.obs_normalizer)
        
        # Recreate with correct dimensions
        self.replay_buffer = buffer_class(buffer_size, phi_in, device=self.device)
        self.obs_normalizer = normalizer_class(phi_in)
        
        print(f"DQN Phi reconfigured:")
        print(f"  Original obs dim: {self._obs_dim_single}")
        print(f"  History length: {self.history_length}")
        print(f"  Phi dimension: {phi_in}")
        print(f"  Buffer and normalizer updated")

    def _initialize_history_with_zeros(self):
        """Initialize history buffers with zero vectors."""
        zero_obs = np.zeros(self._obs_dim_single, dtype=np.float32)
        zero_action = np.zeros(self.n_buildings, dtype=np.float32)
        
        # Fill history with zeros up to history_length
        for _ in range(self.history_length):
            self._obs_history.append(zero_obs.copy())
            self._action_history.append(zero_action.copy())

    def _phi(self, curr_obs_flat: np.ndarray) -> np.ndarray:
        """Build the phi feature vector from history and current observation.

        Constructs phi(t) = [obs_{t-n}, a_{t-n}, ..., obs_{t-1}, a_{t-1}, obs_t]
        by concatenating historical (obs, action) pairs with the current
        observation. Pads with zeros if the history buffer is not yet full.

        Args:
            curr_obs_flat: Current flattened observation vector.

        Returns:
            Concatenated phi vector of dimension
            (history_length * (obs_dim + n_buildings) + obs_dim).
        """
        # If history is not full, pad with zeros
        if len(self._obs_history) < self.history_length:
            self._initialize_history_with_zeros()
        
        phi_components = []
        
        # Add historical obs-action pairs
        for i in range(self.history_length):
            if i < len(self._obs_history):
                phi_components.append(self._obs_history[i])
                phi_components.append(self._action_history[i])
            else:
                # Pad with zeros if not enough history
                phi_components.append(np.zeros(self._obs_dim_single, dtype=np.float32))
                phi_components.append(np.zeros(self.n_buildings, dtype=np.float32))
        
        # Add current observation
        phi_components.append(curr_obs_flat)
        
        return np.concatenate(phi_components, axis=0)

    def _update_history(self, obs_flat: np.ndarray, action_vec: np.ndarray):
        """Append the latest observation and action to the history buffers.

        Args:
            obs_flat: Flattened observation vector at current timestep.
            action_vec: Action vector applied at current timestep.
        """
        self._obs_history.append(obs_flat.copy())
        self._action_history.append(action_vec.copy())

    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Select actions using epsilon-greedy policy over phi features.

        Builds the phi vector from history, normalizes it, and selects
        an action via epsilon-greedy (or greedy if deterministic).

        Args:
            observations: Per-building observation lists from the environment.
            deterministic: If True, always pick the greedy action (no exploration).

        Returns:
            Nested list of actions, one sub-list per building.
        """
        # Build current observation
        curr_obs_flat = self._flatten_observations(observations)
        
        # Build phi vector from history and current observation
        phi = self._phi(curr_obs_flat)
        
        # Only normalize phi if we have training data
        if self.obs_normalizer.count > 0:
            phi_norm = self.obs_normalizer.normalize(phi)
        else:
            phi_norm = phi

        # Epsilon-greedy over φ(t)
        if not deterministic and np.random.random() < self._epsilon_schedule():
            action_idx = np.random.randint(self.n_actions)
        else:
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(phi_norm).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax().item()

        actions = self._idx_to_actions(action_idx)
        action_vec = np.array(actions, dtype=np.float32)
        
        # Update history with current observation and chosen action
        self._update_history(curr_obs_flat, action_vec)
        
        if not deterministic:
            self.epsilon_history.append(self._epsilon_schedule())
        
        return [actions]

    def learn(self, obs, actions, reward, next_obs, done) -> Optional[Dict[str, Any]]:
        """Store a transition in the replay buffer and train on a minibatch.

        Converts observations to phi vectors before storing. For the replay
        buffer, phi is approximated using available history (exact replay of
        historical context is not stored).

        Args:
            obs: Current observations (per-building lists).
            actions: Actions taken (per-building lists).
            reward: Scalar reward or per-building rewards.
            next_obs: Next observations after the action.
            done: Whether the episode has ended.

        Returns:
            Training statistics dict, or None if training has not started.
        """
        # Convert observations to flat arrays
        obs_flat = self._flatten_observations(obs)
        next_flat = self._flatten_observations(next_obs)
        
        # Create approximate phi vectors for replay buffer
        # (In practice, you might want to store actual phi vectors in buffer)
        phi_obs = self._create_phi_for_replay(obs_flat, actions[0] if len(actions) > 0 else None, is_current=False)
        phi_next = self._create_phi_for_replay(next_flat, actions[0] if len(actions) > 0 else None, is_current=True)
        
        # Update normalizer with phi vectors
        self.obs_normalizer.update(phi_obs)
        self.obs_normalizer.update(phi_next)
        
        # Normalize phi vectors
        phi_obs_norm = self.obs_normalizer.normalize(phi_obs)
        phi_next_norm = self.obs_normalizer.normalize(phi_next)

        action_idx = self._actions_to_idx(actions)
        r = float(reward) if np.isscalar(reward) else float(np.mean(reward))
        self.replay_buffer.push(phi_obs_norm, action_idx, r, phi_next_norm, done)

        self.total_steps += 1
        if len(self.replay_buffer) >= self.train_start_steps:
            self._train_step()
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return self.get_training_stats()

    def _create_phi_for_replay(self, obs_flat: np.ndarray, action_vec: Optional[List[float]], is_current: bool) -> np.ndarray:
        """Create an approximate phi vector for replay buffer storage.

        Since the full observation-action history is not stored per transition,
        this method approximates phi using the available history buffers for
        the current timestep and zeros otherwise.

        Args:
            obs_flat: Flattened observation at the timestep.
            action_vec: Action taken at the timestep, or None.
            is_current: If True, use actual history buffers; otherwise pad
                with zeros.

        Returns:
            Approximate phi vector for replay storage.
        """
        phi_components = []
        
        # For simplicity, use zeros for historical components in replay
        # A more sophisticated approach would store the actual phi vectors
        for _ in range(self.history_length):
            if is_current and len(self._obs_history) > 0:
                # Use some actual history if available
                hist_idx = min(len(self._obs_history) - 1, _)
                phi_components.append(self._obs_history[hist_idx])
                phi_components.append(self._action_history[hist_idx])
            else:
                # Use zeros as approximation
                phi_components.append(np.zeros(self._obs_dim_single, dtype=np.float32))
                if action_vec is not None:
                    phi_components.append(np.array(action_vec, dtype=np.float32))
                else:
                    phi_components.append(np.zeros(self.n_buildings, dtype=np.float32))
        
        # Add current observation
        phi_components.append(obs_flat)
        
        return np.concatenate(phi_components, axis=0)

    def reset(self):
        """Reset the algorithm state, including history buffers."""
        if hasattr(super(), 'reset'):
            super().reset()
        
        # Clear history buffers
        self._obs_history.clear()
        self._action_history.clear()

    def get_info(self) -> Dict[str, Any]:
        """Get algorithm information including history configuration."""
        info = super().get_info()
        info['parameters']['history_length'] = self.history_length
        info['parameters']['phi_dimension'] = self.obs_dim
        info['description'] = f'DQN with phi features (history_length={self.history_length})'
        return info
