"""Multi-Policy Proximal Policy Optimization (MP-PPO) for energy management.

This module implements an exploratory RL variant that combines a Transformer-based
probabilistic forecaster (using Student-t output heads) with PPO policy optimization.
The forecaster produces latent representations of predicted future energy states,
which are concatenated with raw observations to form augmented inputs for the
actor-critic policy network. This coupling allows the policy to condition its
decisions on anticipated future conditions (e.g., expected net load over a 24-hour
horizon).

Key components:
    - ``TinyTransformer``: Encoder-decoder Transformer that produces Student-t
      distribution parameters and a latent vector for each forecast horizon.
    - ``ActorCritic``: Two-headed MLP (policy + value) operating on
      observation-augmented-with-forecast-latent inputs.
    - ``RollingReplay``: Circular experience buffer with a lookback mechanism
      for retrospectively filling forecast target values (Algorithm 1).
    - ``MPPPO``: Core agent orchestrating action selection, experience storage,
      GAE advantage computation, and joint predictor/policy training (Algorithm 2).
    - ``MPPPOAlgorithm``: Integration wrapper conforming to the HEMS
      ``BaseAlgorithm`` interface, handling CityLearn observation parsing,
      action discretization, and episode lifecycle.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import pickle
import collections
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# -------------------------- Observation Normalization -------------------------- #

class RunningMeanStd:
    """Welford-style running mean and variance for observation normalization.

    This is an engineering addition (not part of the core MP-PPO paper) that
    stabilizes training by keeping observations zero-centered with unit variance.
    Uses the parallel/batch variant of Welford's online algorithm to merge
    batch statistics into running accumulators.

    Attributes:
        mean: Running mean vector of shape ``(shape,)``.
        var: Running variance vector of shape ``(shape,)``.
        count: Pseudo-count of observations seen so far (initialized to 1e-4
            to avoid division-by-zero on first normalization call).
    """

    def __init__(self, shape):
        """Initialize running statistics with zero mean and unit variance.

        Args:
            shape: Dimensionality of the observation vector.
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        """Update running statistics with a new batch of observations.

        Args:
            x: Array of shape ``(batch_size, shape)`` containing new
                observations.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Merge batch moments into running accumulators using Chan's method.

        Args:
            batch_mean: Mean of the incoming batch.
            batch_var: Variance of the incoming batch.
            batch_count: Number of samples in the incoming batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        """Normalize observations to zero mean and unit variance.

        Args:
            x: Raw observation array, broadcastable to ``(*, shape)``.

        Returns:
            Normalized observation array with the same shape as ``x``.
        """
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

# -------------------------- Positional Encoding -------------------------- #

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for Transformer sequence inputs.

    Adds deterministic sine/cosine position signals to input embeddings so the
    Transformer can distinguish token positions without recurrence. The encoding
    is precomputed at initialization and registered as a non-trainable buffer.

    Attributes:
        pe: Precomputed encoding tensor of shape ``(1, max_len, d_model)``.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize the positional encoding buffer.

        Args:
            d_model: Embedding dimensionality (must be even).
            max_len: Maximum sequence length to precompute encodings for.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of the same shape with positional encodings added.
        """
        return x + self.pe[:, :x.size(1)]

# -------------------------- Predictor (Transformer + Student-t) -------------------------- #

class StudentTHead(nn.Module):
    """Output head that parameterizes an independent Student-t distribution.

    For each forecast horizon step, predicts three parameters:
        - ``nu`` (degrees of freedom): controls tail heaviness, constrained > 1
          via ``1 + softplus``.
        - ``xi`` (location): unconstrained center of the distribution.
        - ``sigma`` (scale): positive scale via ``1e-5 + softplus``.

    This allows the predictor to express calibrated uncertainty about future
    energy values, with heavier tails than a Gaussian when appropriate.
    """

    def __init__(self, d_in: int):
        """Initialize the Student-t output head.

        Args:
            d_in: Dimensionality of the input features (typically ``d_model``).
        """
        super().__init__()
        self.out = nn.Linear(d_in, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Student-t distribution parameters from decoder features.

        Args:
            x: Decoder output of shape ``(batch, horizon, d_in)``.

        Returns:
            Tuple of ``(nu, xi, sigma)``, each of shape ``(batch, horizon)``.
        """
        y = self.out(x)
        nu, xi, sigma = y[..., 0], y[..., 1], y[..., 2]
        nu    = 1.0 + F.softplus(nu)
        sigma = 1e-5 + F.softplus(sigma)
        return nu, xi, sigma

def student_t_nll(y_true: torch.Tensor, nu: torch.Tensor, xi: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute the mean negative log-likelihood under independent Student-t distributions.

    Used as the training loss for the probabilistic forecaster. Each horizon
    step is treated as an independent Student-t variate parameterized by
    ``(nu, xi, sigma)``.

    Args:
        y_true: Ground-truth future values of shape ``(batch, horizon)``.
        nu: Degrees-of-freedom parameter (> 1), shape ``(batch, horizon)``.
        xi: Location parameter, shape ``(batch, horizon)``.
        sigma: Scale parameter (> 0), shape ``(batch, horizon)``.

    Returns:
        Scalar tensor containing the mean NLL across all batch elements and
        horizon steps.
    """
    z = (y_true - xi) / sigma
    c = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
    return -(c - ((nu + 1) / 2) * torch.log1p(z.pow(2) / nu)).mean()

class TinyTransformer(nn.Module):
    """Lightweight encoder-decoder Transformer for probabilistic forecasting.

    Encodes a historical context window and decodes future horizon queries to
    produce Student-t distribution parameters for each forecast step. The
    mean-pooled decoder output serves as a dense latent representation that
    is fed to the PPO actor-critic alongside raw observations.

    Architecture:
        1. Linear projection from ``in_dim`` features to ``d_model``.
        2. Sinusoidal positional encoding.
        3. Standard Transformer encoder over historical context.
        4. Standard Transformer decoder over future horizon queries attending
           to encoded history.
        5. ``StudentTHead`` producing ``(nu, xi, sigma)`` per horizon step.
        6. Mean-pooled decoder output as a latent summary vector.

    Attributes:
        h: Forecast horizon length.
        ctx: Historical context window length.
        d_model: Model embedding dimensionality.
    """

    def __init__(self, d_model=32, nhead=2, num_layers=4, ff=32, horizon=24, ctx=48, in_dim=5):
        """Initialize the Transformer forecaster.

        Args:
            d_model: Hidden dimensionality for embeddings and attention.
            nhead: Number of attention heads.
            num_layers: Number of encoder and decoder layers (each).
            ff: Feed-forward inner dimensionality in Transformer layers.
            horizon: Number of future timesteps to forecast.
            ctx: Number of historical timesteps in the context window.
            in_dim: Dimensionality of per-timestep input features.
        """
        super().__init__()
        self.h = horizon
        self.ctx = ctx
        self.d_model = d_model

        self.inp = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=ctx + horizon)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers)
        self.head = StudentTHead(d_model)

    def forward(self, hist_series: torch.Tensor, tgt_queries: torch.Tensor):
        """Run the encoder-decoder forward pass.

        Args:
            hist_series: Historical feature sequences of shape
                ``(batch, ctx, in_dim)``.
            tgt_queries: Future query features of shape
                ``(batch, horizon, in_dim)``.

        Returns:
            Tuple of ``(nu, xi, sigma, latent)`` where:
                - ``nu``: Degrees of freedom, shape ``(batch, horizon)``.
                - ``xi``: Location parameters, shape ``(batch, horizon)``.
                - ``sigma``: Scale parameters, shape ``(batch, horizon)``.
                - ``latent``: Mean-pooled decoder output, shape
                  ``(batch, d_model)``, used as augmented input to the policy.
        """
        src = self.pos_enc(self.inp(hist_series))
        tgt = self.pos_enc(self.inp(tgt_queries))
        mem = self.enc(src)
        dec = self.dec(tgt, mem)
        nu, xi, sigma = self.head(dec)
        latent = dec.mean(dim=1)
        return nu, xi, sigma, latent

# -------------------------- PPO core with invalid-action masking -------------------------- #

class MaskedCategorical(torch.distributions.Categorical):
    """Categorical distribution with invalid-action masking.

    Extends PyTorch's ``Categorical`` by zeroing out the probability of
    invalid actions. A boolean mask indicates which actions are valid; logits
    for invalid actions are shifted to approximately negative infinity (-1e9)
    before the softmax normalization inside ``Categorical``.
    """

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        """Initialize the masked categorical distribution.

        Args:
            logits: Unnormalized log-probabilities of shape ``(batch, act_dim)``.
            mask: Boolean tensor of shape ``(batch, act_dim)`` where ``True``
                marks valid actions. If ``None``, all actions are considered
                valid.
        """
        if mask is not None:
            logits = logits + torch.where(mask, torch.zeros_like(logits), torch.full_like(logits, -1e9))
        super().__init__(logits=logits)

class ActorCritic(nn.Module):
    """Two-headed MLP actor-critic network for PPO.

    Takes augmented observations (raw observation concatenated with the
    predictor's latent vector) and produces both action logits (policy head)
    and a scalar state-value estimate (value head). Each head is a 2-layer
    MLP with 64 hidden units and Tanh activations.

    Attributes:
        pi: Policy network producing unnormalized action logits.
        v: Value network producing a scalar state-value estimate.
    """

    def __init__(self, obs_dim: int, act_dim: int, pred_latent_dim: int = 32):
        """Initialize the actor-critic network.

        Args:
            obs_dim: Dimensionality of the raw observation vector.
            act_dim: Number of discrete actions.
            pred_latent_dim: Dimensionality of the predictor latent vector
                (defaults to 32, matching ``TinyTransformer.d_model``).
        """
        super().__init__()
        hid = 64
        self.pi = nn.Sequential(nn.Linear(obs_dim + pred_latent_dim, hid), nn.Tanh(),
                                nn.Linear(hid, hid), nn.Tanh(),
                                nn.Linear(hid, act_dim))
        self.v  = nn.Sequential(nn.Linear(obs_dim + pred_latent_dim, hid), nn.Tanh(),
                                nn.Linear(hid, hid), nn.Tanh(),
                                nn.Linear(hid, 1))

    def forward(self, obs_aug: torch.Tensor):
        """Compute policy logits and value estimate.

        Args:
            obs_aug: Augmented observation tensor of shape
                ``(batch, obs_dim + pred_latent_dim)``.

        Returns:
            Tuple of ``(logits, value)`` where ``logits`` has shape
            ``(batch, act_dim)`` and ``value`` has shape ``(batch,)``.
        """
        logits = self.pi(obs_aug)
        value  = self.v(obs_aug).squeeze(-1)
        return logits, value

# -------------------------- Rolling Replay Buffer (Algorithm 1) -------------------------- #

class RollingReplay:
    """Circular experience replay buffer with retrospective forecast targets.

    Implements Algorithm 1 from the paper. Stores standard RL transition data
    (observations, actions, rewards, dones, log-probs, values, action masks)
    alongside Transformer inputs (historical series, target queries) and
    future forecast targets. The buffer overwrites the oldest entries when
    full, enabling continual learning.

    The key mechanism is ``write_forecast_value``, which retrospectively fills
    in ground-truth future values for past transitions once those values become
    available (i.e., after the agent has stepped forward in time).

    Attributes:
        N: Buffer capacity (maximum number of transitions).
        h: Forecast horizon length.
        ctx: Historical context window length.
        in_dim: Per-timestep feature dimensionality.
        pos: Current write position in the circular buffer.
        episode_pos: Steps elapsed in the current episode.
        full: Whether the buffer has wrapped around at least once.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, h: int, ctx: int, in_dim: int):
        """Initialize pre-allocated numpy arrays for the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Dimensionality of observation vectors.
            act_dim: Number of discrete actions (for mask storage).
            h: Forecast horizon length.
            ctx: Historical context window length.
            in_dim: Per-timestep feature dimensionality for Transformer inputs.
        """
        self.N = capacity
        self.h, self.ctx, self.in_dim = h, ctx, in_dim
        self.pos = 0
        self.episode_pos = 0
        self.full = False

        self.obs = np.zeros((self.N, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.N,), dtype=np.int64)
        self.rew = np.zeros((self.N,), dtype=np.float32)
        self.done = np.zeros((self.N,), dtype=np.float32)
        self.logp = np.zeros((self.N,), dtype=np.float32)
        self.val = np.zeros((self.N + 1,), dtype=np.float32)
        self.mask = np.ones((self.N, act_dim), dtype=bool)

        self.hist_series = np.zeros((self.N, ctx, in_dim), dtype=np.float32)
        self.tgt_queries = np.zeros((self.N, h,   in_dim), dtype=np.float32)
        self.y_future = np.zeros((self.N, h), dtype=np.float32)
        self.dones_flag = np.zeros((self.N,), dtype=np.bool_)

    def add(self, obs, act, rew, done, logp, val, mask, hist_series, tgt_queries):
        """Store a single transition in the buffer at the current write position.

        Args:
            obs: Observation vector of shape ``(obs_dim,)``.
            act: Discrete action index.
            rew: Scalar reward.
            done: Whether the episode terminated after this transition.
            logp: Log-probability of the selected action under the current policy.
            val: Value estimate for this state.
            mask: Boolean action mask of shape ``(act_dim,)`` or ``None``.
            hist_series: Historical feature array of shape ``(ctx, in_dim)``.
            tgt_queries: Future query feature array of shape ``(h, in_dim)``.
        """
        i = self.pos
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = float(done)
        self.logp[i] = logp
        self.val[i] = val
        self.mask[i] = mask if mask is not None else True
        self.hist_series[i] = hist_series
        self.tgt_queries[i] = tgt_queries
        if done:
            self.dones_flag[i] = True

        self.pos = (self.pos + 1) % self.N
        self.episode_pos = 0 if done else (self.episode_pos + 1)
        self.full = self.full or (self.pos == 0)

    def write_forecast_value(self, v_scalar: float):
        """Algorithm 1: Rolling Replay Buffer Transition Update
        
        This implements the lookback mechanism from the paper to fill future values
        in the buffer. When a transition is added at position t, we need to update
        the forecast targets for previous transitions that expected to see this value.
        
        Detailed explanation:
        1. pos = current position - 1 (the transition we just added)
        2. episode_pos = how many steps into current episode
        3. lookback = how many past transitions to update (max = horizon h)
        
        The complex logic handles three cases:
        
        Case A: Early in episode (episode_pos <= h)
          - We're still building up the first h predictions
          - lookback = h (try to fill all horizon steps)
          
        Case B: Middle of episode (episode_pos > h, but episode_pos % l in [0, h])
          - Special boundary condition near multiples of episode length
          - Adjust lookback to avoid wraparound issues
          
        Case C: Normal case (episode_pos > h)
          - Standard sliding window
          - lookback = h
        
        Example with h=3, episode of length 5:
        Step 0: Add transition, fill [] (nothing to backfill)
        Step 1: Add transition, fill [t0_future[0]=v1] 
        Step 2: Add transition, fill [t1_future[0]=v2, t0_future[1]=v2]
        Step 3: Add transition, fill [t2_future[0]=v3, t1_future[1]=v3, t0_future[2]=v3]
        Step 4: Add transition, fill [t3_future[0]=v4, t2_future[1]=v4, t1_future[2]=v4]
        
        This ensures each transition has h future values for training the predictor.
        """
        pos = (self.pos - 1) % self.N
        l = self.episode_pos + 1 if self.episode_pos > 0 else 1
        lookback = self.h
        
        # Handle boundary conditions at episode start/wraparound
        if (self.episode_pos % l) in range(0, self.h + 1):
            if self.episode_pos > self.h:
                lookback = 1 + (self.episode_pos % l)
        
        # Fill future values for previous transitions
        for i in range(lookback):
            j = (pos - i) % self.N
            if j < 0: 
                break
            if i < self.h:
                self.y_future[j, i] = float(v_scalar)

    def size(self) -> int:
        """Return the number of valid transitions currently stored.

        Returns:
            The full capacity if the buffer has wrapped, otherwise the
            current write position.
        """
        return self.N if self.full else self.pos

# -------------------------- Temporal Feature Extraction -------------------------- #

def create_temporal_features(
    timestep: int,
    ctx: int = 48,
    consumption_history: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create temporal features for historical context.
    
    Features are normalized to [0, 1] range:
    - hour: [0, 24) / 23.0 -> [0, 1]
    - day: [0, 7) / 6.0 -> [0, 1]
    - week: [0, 4) / 3.0 -> [0, 1]
    - month: [0, 12) / 11.0 -> [0, 1]
    """
    features = np.zeros((ctx, 5), dtype=np.float32)
    
    for i in range(ctx):
        t = timestep - (ctx - 1 - i)
        t = max(0, t)
        
        hour = (t % 24) / 23.0
        day = ((t // 24) % 7) / 6.0
        week = ((t // (24 * 7)) % 4) / 3.0
        month = ((t // (24 * 30)) % 12) / 11.0
        
        consumption = 0.0
        if consumption_history is not None and i < len(consumption_history):
            consumption = float(consumption_history[i])
        
        features[i] = [hour, day, week, month, consumption]
    
    return features

def create_future_temporal_features(timestep: int, horizon: int = 24) -> np.ndarray:
    """Create temporal features for future horizon.
    
    Features are normalized to [0, 1] range for consistency.
    """
    features = np.zeros((horizon, 5), dtype=np.float32)
    
    for i in range(horizon):
        t = timestep + i + 1
        
        hour = (t % 24) / 23.0
        day = ((t // 24) % 7) / 6.0
        week = ((t // (24 * 7)) % 4) / 3.0
        month = ((t // (24 * 30)) % 12) / 11.0
        
        features[i] = [hour, day, week, month, 0.0]
    
    return features

# -------------------------- MP-PPO Agent -------------------------- #

class MPPPO:
    """Core Multi-Policy PPO agent combining a Transformer forecaster with PPO.

    This class owns the neural network modules (``TinyTransformer`` predictor
    and ``ActorCritic`` policy), their optimizers, and the ``RollingReplay``
    buffer. It provides the main training loop (Algorithm 2) which alternates
    between updating the predictor on Student-t NLL and updating the policy
    via clipped PPO with GAE advantages.

    The "multi-policy" aspect refers to the forecaster's latent vector
    augmenting the policy input, effectively conditioning the policy on
    predicted future states. This creates a family of implicit sub-policies
    parameterized by the forecast context.

    Attributes:
        obs_dim: Raw observation dimensionality.
        act_dim: Number of discrete actions.
        horizon: Forecast horizon in timesteps.
        ctx: Historical context window in timesteps.
        predictor: ``TinyTransformer`` forecasting module.
        ac: ``ActorCritic`` policy and value network.
        buf: ``RollingReplay`` experience buffer.
        total_updates: Number of ``learn()`` calls completed so far.
        metrics_history: Dictionary of per-update loss histories.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: Dict[str, Any]):
        """Initialize the MP-PPO agent, networks, optimizers, and buffer.

        Args:
            obs_dim: Dimensionality of the raw observation vector.
            act_dim: Number of discrete actions available.
            config: Configuration dictionary. Supported keys:

                - ``horizon`` (int): Forecast horizon, default 24.
                - ``ctx`` (int): Context window length, default 48.
                - ``d_model`` (int): Transformer embedding dim, default 32.
                - ``nhead`` (int): Number of attention heads, default 2.
                - ``layers`` (int): Transformer layer count, default 4.
                - ``ff`` (int): Feed-forward hidden size, default 32.
                - ``in_dim`` (int): Per-timestep feature dim, default 5.
                - ``gamma`` (float): Discount factor, default 0.99.
                - ``lam`` (float): GAE lambda, default 0.95.
                - ``clip_ratio`` (float): PPO clip ratio, default 0.2.
                - ``ent_coef`` (float): Entropy bonus coefficient,
                  default 0.01.
                - ``vf_coef`` (float): Value loss coefficient, default 0.5.
                - ``lr_policy`` (float): Policy learning rate, default 3e-4.
                - ``lr_pred`` (float): Predictor learning rate, default 3e-4.
                - ``train_iters`` (int): PPO epochs per learn call,
                  default 10.
                - ``minibatch`` (int): Minibatch size, default 2048.
                - ``buffer_steps`` (int): Replay capacity, default 8000.
                - ``normalize_obs`` (bool): Enable observation normalization,
                  default False.
                - ``ent_coef_decay`` (bool): Enable entropy coefficient
                  decay, default False.
                - ``pred_updates_per_iter`` (int): Predictor gradient steps
                  per minibatch, default 10.
                - ``device`` (str): Torch device, default ``"cuda"`` if
                  available else ``"cpu"``.
                - ``seed`` (int or None): Random seed, default None.

        Raises:
            ValueError: If ``obs_dim``, ``act_dim``, or any architecture
                parameter has an invalid value (non-positive, or ``d_model``
                not divisible by ``nhead``).
        """
        # Input validation
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
        
        horizon = config.get('horizon', 24)
        ctx = config.get('ctx', 48)
        d_model = config.get('d_model', 32)
        nhead = config.get('nhead', 2)
        layers = config.get('layers', 4)
        minibatch = config.get('minibatch', 2048)
        buffer_steps = config.get('buffer_steps', 8000)
        
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if ctx <= 0:
            raise ValueError(f"ctx must be positive, got {ctx}")
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model must be positive and even for positional encoding, got {d_model}")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got d_model={d_model}, nhead={nhead}")
        if layers <= 0:
            raise ValueError(f"layers must be positive, got {layers}")
        if minibatch <= 0:
            raise ValueError(f"minibatch must be positive, got {minibatch}")
        if buffer_steps <= 0:
            raise ValueError(f"buffer_steps must be positive, got {buffer_steps}")
        
        seed = config.get('seed', None)
        if seed is not None:
            self._set_seed(seed)
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.ctx = ctx
        self.in_dim = config.get('in_dim', 5)
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.ff = config.get('ff', 32)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.lr_policy = config.get('lr_policy', 3e-4)
        self.lr_pred = config.get('lr_pred', 3e-4)
        self.train_iters = config.get('train_iters', 10)
        self.minibatch = minibatch
        self.buffer_steps = buffer_steps
        
        # Engineering additions (NOT in paper, disabled by default for paper faithfulness)
        self.normalize_obs = config.get('normalize_obs', False)
        self.ent_coef_decay = config.get('ent_coef_decay', False)
        self.pred_updates_per_iter = config.get('pred_updates_per_iter', 10)
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.predictor = TinyTransformer(d_model=self.d_model, nhead=self.nhead, num_layers=self.layers,
                                         ff=self.ff, horizon=self.horizon, ctx=self.ctx, in_dim=self.in_dim).to(self.device)
        self.ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, pred_latent_dim=self.d_model).to(self.device)

        self.opt_policy = optim.Adam(self.ac.parameters(), lr=self.lr_policy)
        self.opt_pred   = optim.AdamW(self.predictor.parameters(), lr=self.lr_pred)

        self.buf = RollingReplay(capacity=self.buffer_steps, obs_dim=obs_dim, act_dim=act_dim,
                                 h=self.horizon, ctx=self.ctx, in_dim=self.in_dim)

        if self.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_rms = None

        self._last_value = 0.0
        self.total_updates = 0
        self.ent_coef_init = self.ent_coef
        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'predictor_nll': []
        }
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility across all RNG backends.

        Args:
            seed: Integer seed applied to Python, NumPy, and PyTorch (CPU
                and CUDA) random number generators.
        """
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: Optional[np.ndarray],
            hist_series: np.ndarray, tgt_queries: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict[str, float]]:
        """Select an action given the current observation and forecast context.

        Runs the predictor to obtain a latent summary of predicted future
        states, concatenates it with the (optionally normalized) observation,
        and samples (or greedily selects) an action from the masked policy
        distribution. Inference runs with gradients disabled.

        Args:
            obs: Raw observation vector of shape ``(obs_dim,)``.
            mask: Boolean action-validity mask of shape ``(act_dim,)``, or
                ``None`` if all actions are valid.
            hist_series: Historical feature array of shape ``(ctx, in_dim)``.
            tgt_queries: Future query features of shape ``(horizon, in_dim)``.
            deterministic: If ``True``, select the highest-probability action
                instead of sampling.

        Returns:
            Tuple of ``(action_index, info_dict)`` where ``info_dict``
            contains ``"logp"`` (log-probability of the chosen action) and
            ``"value"`` (state-value estimate).
        """
        self.ac.eval(); self.predictor.eval()
        
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        hist_t = torch.as_tensor(hist_series, dtype=torch.float32, device=self.device).unsqueeze(0)
        tgt_t  = torch.as_tensor(tgt_queries, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, _, latent = self.predictor(hist_t, tgt_t)
        obs_aug = torch.cat([obs_t, latent], dim=-1)
        logits, value = self.ac(obs_aug)

        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0) if mask is not None else None
        dist = MaskedCategorical(logits=logits, mask=mask_t)
        a = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)

        self._last_value = float(value.item())
        return int(a.item()), {"logp": float(logp.item()), "value": float(value.item())}

    def store(self, obs, act, rew, done, logp, val, mask, hist_series, tgt_queries, s_tilde_scalar: float):
        """Store a transition and retrospectively update forecast targets.

        Adds the transition to the rolling replay buffer, then calls the
        lookback mechanism to fill in ground-truth future values for
        previously stored transitions that anticipated this timestep.

        Args:
            obs: Observation vector of shape ``(obs_dim,)``.
            act: Discrete action index taken.
            rew: Scalar reward received.
            done: Whether the episode ended.
            logp: Log-probability of the action under the policy at action
                time.
            val: Value estimate at action time.
            mask: Boolean action mask of shape ``(act_dim,)`` or ``None``.
            hist_series: Historical features of shape ``(ctx, in_dim)``.
            tgt_queries: Future query features of shape ``(horizon, in_dim)``.
            s_tilde_scalar: Observed scalar value (e.g., net load) used to
                retrospectively fill forecast targets for prior transitions.
        """
        self.buf.add(obs, act, rew, done, logp, val, mask, hist_series, tgt_queries)
        self.buf.write_forecast_value(s_tilde_scalar)

    def _gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float):
        """Compute Generalized Advantage Estimation (GAE-lambda).

        Args:
            rewards: Reward array of shape ``(T,)``.
            values: Value estimates of shape ``(T+1,)`` (includes bootstrap
                value at position ``T``).
            dones: Done flags of shape ``(T,)`` (1.0 if terminal, else 0.0).
            gamma: Discount factor.
            lam: GAE lambda for bias-variance trade-off.

        Returns:
            Tuple of ``(advantages, returns)`` each of shape ``(T,)``.
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        returns = adv + values[:-1]
        return adv, returns

    def learn(self) -> Dict[str, float]:
        """Run the MP-PPO training loop (Algorithm 2 from the paper).

        Performs multiple epochs of joint predictor and policy optimization
        over the current replay buffer contents. Each epoch uses circular
        minibatch sampling (Eq. 12): a random start position is chosen and
        minibatches wrap around the buffer.

        For each minibatch:
            1. The predictor is updated ``pred_updates_per_iter`` times on
               Student-t NLL against retrospectively filled forecast targets.
            2. The predictor's latent output (detached) is concatenated with
               observations to form augmented inputs for the actor-critic.
            3. The policy is updated via clipped PPO loss with entropy bonus
               and value function loss.

        After training, the buffer is reset (``pos = 0``, ``full = False``).

        Returns:
            Dictionary of mean training statistics with keys
            ``"loss/policy"``, ``"loss/value"``, ``"loss/entropy"``,
            ``"loss/predictor_nll"``, and ``"ent_coef"``. Returns an empty
            dict if the buffer is empty.
        """
        N = self.buf.size()
        if N == 0:
            return {}

        obs   = torch.as_tensor(self.buf.obs[:N], dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(self.buf.act[:N], dtype=torch.long,    device=self.device)
        rew   = torch.as_tensor(self.buf.rew[:N], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(self.buf.done[:N], dtype=torch.float32, device=self.device)
        logp0 = torch.as_tensor(self.buf.logp[:N], dtype=torch.float32, device=self.device)
        val   = torch.as_tensor(self.buf.val[:N+1], dtype=torch.float32, device=self.device)

        hist  = torch.as_tensor(self.buf.hist_series[:N], dtype=torch.float32, device=self.device)
        tgtq  = torch.as_tensor(self.buf.tgt_queries[:N], dtype=torch.float32, device=self.device)
        ytr   = torch.as_tensor(self.buf.y_future[:N], dtype=torch.float32, device=self.device)

        if self.obs_rms is not None:
            self.obs_rms.update(self.buf.obs[:N])
            obs_normalized = torch.as_tensor(
                self.obs_rms.normalize(self.buf.obs[:N]), 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            obs_normalized = obs

        adv_np, ret_np = self._gae(rew.cpu().numpy(), val.cpu().numpy(), done.cpu().numpy(),
                                   self.gamma, self.lam)
        adv = torch.as_tensor((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8), dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

        if self.ent_coef_decay:
            self.ent_coef = max(0.0, self.ent_coef_init * (1.0 - self.total_updates / 1000.0))

        policy_losses = []
        value_losses = []
        entropies = []
        pred_losses = []

        # Paper Algorithm 2 with circular sampling (Eq. 12)
        for epoch in range(self.train_iters):
            start_pos = np.random.randint(0, max(1, N))
            
            for offset in range(0, N, self.minibatch):
                actual_start = (start_pos + offset) % N
                batch_size = min(self.minibatch, N)
                
                mb_indices = np.array([
                    (actual_start + i) % N for i in range(batch_size)
                ], dtype=np.int64)
                
                mb = torch.as_tensor(mb_indices, dtype=torch.long, device=self.device)

                # Update predictor (paper: one update per minibatch)
                for _ in range(self.pred_updates_per_iter):
                    nu, xi, sigma, _ = self.predictor(hist[mb], tgtq[mb])
                    nll = student_t_nll(ytr[mb], nu, xi, sigma)
                    self.opt_pred.zero_grad(set_to_none=True)
                    nll.backward()
                    nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                    self.opt_pred.step()
                    pred_losses.append(nll.item())

                # Get latent representation for policy update
                with torch.no_grad():
                    _, _, _, latent = self.predictor(hist[mb], tgtq[mb])
                obs_aug = torch.cat([obs_normalized[mb], latent], dim=-1)

                # Update policy
                logits, v = self.ac(obs_aug)
                dist = MaskedCategorical(logits=logits, mask=None)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp0[mb])

                pg1 = ratio * adv[mb]
                pg2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[mb]
                pg_loss = -torch.min(pg1, pg2).mean()

                v_loss = 0.5 * (ret[mb] - v).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt_policy.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt_policy.step()

                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(ent.item())

        self.buf.pos = 0
        self.buf.full = False
        self.total_updates += 1
        
        stats = {
            "loss/policy": np.mean(policy_losses),
            "loss/value": np.mean(value_losses),
            "loss/entropy": np.mean(entropies),
            "loss/predictor_nll": np.mean(pred_losses),
            "ent_coef": self.ent_coef,
        }
        
        self.metrics_history['policy_loss'].append(stats['loss/policy'])
        self.metrics_history['value_loss'].append(stats['loss/value'])
        self.metrics_history['entropy'].append(stats['loss/entropy'])
        self.metrics_history['predictor_nll'].append(stats['loss/predictor_nll'])
        
        return stats

    def save(self, path: str):
        """Save the full agent state to a checkpoint file.

        Persists predictor and actor-critic weights, optimizer states,
        update count, and (if enabled) observation normalization statistics.

        Args:
            path: File path for the checkpoint (typically ``*.pt``).
        """
        save_dict = {
            'predictor': self.predictor.state_dict(),
            'ac': self.ac.state_dict(),
            'opt_policy': self.opt_policy.state_dict(),
            'opt_pred': self.opt_pred.state_dict(),
            'total_updates': self.total_updates
        }
        if self.obs_rms is not None:
            save_dict['obs_rms_mean'] = self.obs_rms.mean
            save_dict['obs_rms_var'] = self.obs_rms.var
            save_dict['obs_rms_count'] = self.obs_rms.count
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load agent state from a checkpoint file.

        Restores predictor and actor-critic weights, optimizer states,
        update count, and observation normalization statistics (if present
        in the checkpoint and normalization is enabled).

        Args:
            path: File path to the checkpoint to load.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.ac.load_state_dict(checkpoint['ac'])
        self.opt_policy.load_state_dict(checkpoint['opt_policy'])
        self.opt_pred.load_state_dict(checkpoint['opt_pred'])
        self.total_updates = checkpoint.get('total_updates', 0)
        if self.obs_rms is not None and 'obs_rms_mean' in checkpoint:
            self.obs_rms.mean = checkpoint['obs_rms_mean']
            self.obs_rms.var = checkpoint['obs_rms_var']
            self.obs_rms.count = checkpoint['obs_rms_count']

    def pretrain_predictor(self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                          epochs: int = 100, batch_size: int = 64) -> Dict[str, float]:
        """Pre-train the Transformer forecaster on a supervised dataset.

        Trains the predictor in isolation (without the policy) by minimizing
        the Student-t NLL on provided (history, query, target) triples. This
        can warm-start the forecaster before online RL training begins.

        Args:
            dataset: List of ``(hist, tgt, y)`` tuples where:

                - ``hist``: Historical features, shape ``(ctx, in_dim)``.
                - ``tgt``: Future query features, shape ``(horizon, in_dim)``.
                - ``y``: Ground-truth future values, shape ``(horizon,)``.
            epochs: Number of full passes over the dataset.
            batch_size: Number of samples per gradient step.

        Returns:
            Dictionary with ``"pretrain_epochs"`` and ``"final_nll"`` keys.
        """
        self.predictor.train()
        opt = optim.AdamW(self.predictor.parameters(), lr=self.lr_pred)
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(dataset))
            epoch_loss = 0.0
            
            for start_idx in range(0, len(dataset), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                hist_batch = []
                tgt_batch = []
                y_batch = []
                
                for idx in batch_indices:
                    hist, tgt, y = dataset[idx]
                    hist_batch.append(hist)
                    tgt_batch.append(tgt)
                    y_batch.append(y)
                
                hist = torch.as_tensor(np.array(hist_batch), dtype=torch.float32, device=self.device)
                tgt = torch.as_tensor(np.array(tgt_batch), dtype=torch.float32, device=self.device)
                y = torch.as_tensor(np.array(y_batch), dtype=torch.float32, device=self.device)
                
                nu, xi, sigma, _ = self.predictor(hist, tgt)
                nll = student_t_nll(y, nu, xi, sigma)
                
                opt.zero_grad(set_to_none=True)
                nll.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                opt.step()
                
                epoch_loss += nll.item()
            
            avg_loss = epoch_loss / max(1, len(dataset) // batch_size)
            if (epoch + 1) % 10 == 0:
                print(f"Pretrain Epoch {epoch+1}/{epochs}, NLL: {avg_loss:.4f}")
        
        return {"pretrain_epochs": epochs, "final_nll": avg_loss}

# ==================== HEMS FRAMEWORK INTEGRATION ==================== #

from .base import BaseAlgorithm

class MPPPOAlgorithm(BaseAlgorithm):
    """HEMS framework adapter for the MP-PPO agent.

    Bridges the ``MPPPO`` core agent with the ``BaseAlgorithm`` interface
    expected by the HEMS runner. Handles:

    - Inferring observation and action dimensions from the CityLearn
      environment.
    - Discretizing continuous action spaces into ``action_bins`` uniformly
      spaced values in [-1, 1].
    - Auto-detecting observation indices for load, PV generation, and
      battery SoC from environment metadata (with configurable fallbacks).
    - Maintaining a rolling consumption history for temporal feature
      construction.
    - Converting between the HEMS per-building observation/action format
      and the flat vectors expected by ``MPPPO``.
    - Optionally loading a pre-trained predictor checkpoint at init time.

    Attributes:
        obs_dim: Total flattened observation dimensionality.
        act_dim: Number of discrete actions.
        n_buildings: Number of buildings in the environment.
        agent: Underlying ``MPPPO`` instance.
        consumption_history: Deque tracking recent net-load values for
            temporal feature construction.
        obs_indices: Dictionary mapping ``"load"``, ``"pv"``, ``"soc"``
            to their positions in the observation vector.
        timestep: Current timestep within the episode.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """Initialize the HEMS-integrated MP-PPO algorithm.

        Args:
            env: CityLearn environment instance. Must expose
                ``observation_space``, ``action_space``, and ``buildings``
                attributes.
            config: Algorithm configuration dictionary. In addition to
                all keys accepted by ``MPPPO.__init__``, supports:

                - ``action_bins`` (int): Number of discretization bins for
                  continuous action spaces, default 31.
                - ``obs_index_load`` (int): Fallback observation index for
                  non-shiftable load, default 10.
                - ``obs_index_pv`` (int): Fallback observation index for
                  solar generation, default 11.
                - ``obs_index_soc`` (int): Fallback observation index for
                  battery SoC, default 15.
                - ``pretrained_model_path`` (str or None): Path to a
                  pre-trained predictor checkpoint to load.
        """
        super().__init__(env, config)
        
        self.obs_dims = [s.shape[0] for s in env.observation_space]
        self.obs_dim = sum(self.obs_dims)
        self.n_buildings = len(env.buildings)
        
        if hasattr(env.action_space[0], 'n'):
            self.act_dim = env.action_space[0].n
        else:
            self.n_bins = config.get('action_bins', 31)
            self.discrete_actions = np.linspace(-1.0, 1.0, self.n_bins, dtype=np.float32)
            self.act_dim = self.n_bins
        
        config['in_dim'] = 5
        self.agent = MPPPO(self.obs_dim, self.act_dim, config)
        
        self.consumption_history = collections.deque(maxlen=self.agent.ctx)
        for _ in range(self.agent.ctx):
            self.consumption_history.append(0.0)
        
        self.obs_indices = self._parse_observation_indices(config)
        
        self.last_info = {}
        self.last_obs = None
        self.last_hist = None
        self.last_tgt = None
        self.timestep = 0
        
        pretrained_path = config.get('pretrained_model_path', None)
        if pretrained_path:
            from pathlib import Path
            if Path(pretrained_path).exists():
                checkpoint = torch.load(pretrained_path, map_location=self.agent.device, weights_only=False)
                
                if 'predictor' in checkpoint:
                    if 'obs_indices' in checkpoint:
                        saved_indices = checkpoint['obs_indices']
                        if saved_indices != self.obs_indices:
                            print(f"WARNING: Observation index mismatch!")
                            print(f"  Pretrained with: {saved_indices}")
                            print(f"  Current config:  {self.obs_indices}")
                            print(f"  Using pretrained indices to maintain consistency")
                            self.obs_indices = saved_indices
                    
                    self.agent.predictor.load_state_dict(checkpoint['predictor'])
                    print(f"Loaded pretrained predictor from {pretrained_path}")
                    
                    if 'epoch' in checkpoint:
                        print(f"  Trained for {checkpoint['epoch']} epochs")
                    if 'loss' in checkpoint:
                        print(f"  Best loss: {checkpoint['loss']:.4f}")
                else:
                    print(f"Warning: 'predictor' key not found in checkpoint")
            else:
                print(f"Warning: Pretrained model file not found at {pretrained_path}")
    
    def _parse_observation_indices(self, config: Dict[str, Any]) -> Dict[str, int]:
        """Determine observation vector indices for load, PV, and SoC.

        Attempts auto-detection by scanning ``env.observation_names`` for
        known substrings (e.g., ``"non_shiftable_load"``, ``"solar_gen"``).
        Falls back to manually configured index values if auto-detection
        fails.

        Args:
            config: Configuration dictionary containing fallback keys
                ``obs_index_load``, ``obs_index_pv``, ``obs_index_soc``.

        Returns:
            Dictionary with keys ``"load"``, ``"pv"``, ``"soc"`` mapped to
            their integer positions in the observation vector (or ``None``
            if not found).
        """
        if hasattr(self.env, 'observation_names') and len(self.env.observation_names) > 0:
            obs_names = self.env.observation_names[0]
            indices = {'load': None, 'pv': None, 'soc': None}
            
            for i, name in enumerate(obs_names):
                name_lower = name.lower()
                if 'non_shiftable_load' in name_lower or 'non shiftable load' in name_lower:
                    indices['load'] = i
                elif 'solar_gen' in name_lower or 'solar generation' in name_lower:
                    indices['pv'] = i
                elif 'electrical_storage_soc' in name_lower or 'battery' in name_lower:
                    indices['soc'] = i
            
            if indices['load'] is not None and indices['pv'] is not None:
                print(f"Auto-detected observation indices: load={indices['load']}, pv={indices['pv']}, soc={indices['soc']}")
                return indices
            else:
                import warnings
                warnings.warn(f"Could not auto-detect all required observation indices. Found: {indices}")
        
        # Fallback to config defaults
        indices = {
            'load': config.get('obs_index_load', 10),
            'pv': config.get('obs_index_pv', 11),
            'soc': config.get('obs_index_soc', 15),
        }
        print(f"Using configured observation indices: {indices}")
        return indices
    
    def _compute_action_mask(self, obs: np.ndarray) -> np.ndarray:
        """Compute a boolean action-validity mask based on battery SoC.

        Disables charging actions when the battery is nearly full (SoC >= 0.95)
        and discharging actions when nearly empty (SoC <= 0.05). Only applies
        when the action space has been discretized (``discrete_actions``
        attribute exists).

        Args:
            obs: Observation vector of shape ``(obs_dim,)``.

        Returns:
            Boolean array of shape ``(act_dim,)`` where ``True`` indicates
            a valid action.
        """
        mask = np.ones(self.act_dim, dtype=bool)
        
        soc_idx = self.obs_indices.get('soc')
        if soc_idx is None:
            import warnings
            warnings.warn("SOC index not available, returning all-valid mask")
            return mask
            
        if soc_idx >= len(obs):
            import warnings
            warnings.warn(f"SOC index {soc_idx} out of range for observation size {len(obs)}, returning all-valid mask")
            return mask
        
        soc = obs[soc_idx]
        
        if hasattr(self, 'discrete_actions'):
            for i, action in enumerate(self.discrete_actions):
                if action > 0.05 and soc >= 0.95:
                    mask[i] = False
                elif action < -0.05 and soc <= 0.05:
                    mask[i] = False
        
        return mask
                
    def _extract_net_load(self, obs: np.ndarray) -> float:
        """Extract net load (consumption minus PV generation) from an observation.

        Computes ``obs[load_idx] - obs[pv_idx]``. Returns 0.0 with a
        warning if the required indices are unavailable or out of range.

        Args:
            obs: Observation vector of shape ``(obs_dim,)``.

        Returns:
            Scalar net load value (positive means net consumption).
        """
        load_idx = self.obs_indices['load']
        pv_idx = self.obs_indices['pv']
        
        if load_idx is None or pv_idx is None:
            import warnings
            warnings.warn(f"Load or PV index not available (load={load_idx}, pv={pv_idx}), returning 0.0")
            return 0.0
        
        if len(obs) <= max(load_idx, pv_idx):
            import warnings
            warnings.warn(f"Observation size {len(obs)} too small for indices (load={load_idx}, pv={pv_idx}), returning 0.0")
            return 0.0
        
        load = obs[load_idx]
        pv = obs[pv_idx]
        return float(load - pv)
                
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Select actions for all buildings given their observations.

        Flattens the first building's observations, constructs temporal
        features for the historical context and future horizon, computes
        an action mask based on battery SoC, and delegates to the core
        ``MPPPO.act`` method. The discrete action index is mapped back to
        a continuous value in [-1, 1] if the action space was discretized.

        Args:
            observations: Per-building observation lists. Currently only
                the first building (index 0) is used.
            deterministic: If ``True``, select the greedy action.

        Returns:
            Nested list of actions in HEMS centralized format
            ``[[action_value]]``.
        """
        obs_flat = np.array(observations[0], dtype=np.float32)
        
        net_load = self._extract_net_load(obs_flat)
        self.consumption_history.append(net_load)
        
        hist_series = create_temporal_features(
            self.timestep, 
            self.agent.ctx,
            consumption_history=np.array(self.consumption_history)
        )
        tgt_queries = create_future_temporal_features(self.timestep, self.agent.horizon)
        
        mask = self._compute_action_mask(obs_flat)
        
        action_idx, info = self.agent.act(obs_flat, mask=mask, 
                                         hist_series=hist_series, 
                                         tgt_queries=tgt_queries, 
                                         deterministic=deterministic)
        
        self.last_info = info
        self.last_obs = obs_flat
        self.last_hist = hist_series  
        self.last_tgt = tgt_queries
        self.timestep += 1
        
        if hasattr(self, 'discrete_actions'):
            continuous_action = self.discrete_actions[action_idx]
            return [[float(continuous_action)]]
        else:
            return [[float(action_idx)]]
    
    def store_transition(self, obs, action, reward, next_obs, done, **kwargs):
        """Store an environment transition into the replay buffer.

        Converts the HEMS-format action back to a discrete index, extracts
        the current net load from ``next_obs`` for forecast target
        back-filling, and delegates to ``MPPPO.store``. On episode
        termination, resets the timestep counter and consumption history.

        Args:
            obs: Observations at the current step (unused; cached
                ``last_obs`` from ``act`` is used instead).
            action: Action taken, in HEMS nested-list format.
            reward: Scalar reward received.
            next_obs: Observations at the next step, used to extract the
                realized net load for forecast targets.
            done: Whether the episode has terminated.
            **kwargs: Additional keyword arguments (ignored).
        """
        if hasattr(self, 'last_info') and self.last_obs is not None:
            if isinstance(action[0], list):
                action_val = action[0][0]
            else:
                action_val = action[0]
            
            if hasattr(self, 'discrete_actions'):
                action_idx = np.argmin(np.abs(self.discrete_actions - action_val))
            else:
                action_idx = int(action_val)
            
            next_obs_array = np.array(next_obs[0], dtype=np.float32) if isinstance(next_obs, list) else next_obs
            current_net_load = self._extract_net_load(next_obs_array)
            
            mask = self._compute_action_mask(self.last_obs)
            
            self.agent.store(
                obs=self.last_obs,
                act=action_idx,
                rew=float(reward),
                done=bool(done),
                logp=self.last_info['logp'],
                val=self.last_info['value'],
                mask=mask,
                hist_series=self.last_hist,
                tgt_queries=self.last_tgt,
                s_tilde_scalar=float(current_net_load)
            )
            
            if done:
                self.timestep = 0
                self.consumption_history.clear()
                for _ in range(self.agent.ctx):
                    self.consumption_history.append(0.0)
    
    def learn(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Trigger a learning update on the underlying MP-PPO agent.

        Args:
            *args: Passed through (unused by ``MPPPO.learn``).
            **kwargs: Passed through (unused by ``MPPPO.learn``).

        Returns:
            Training statistics dictionary from ``MPPPO.learn``, or
            ``None`` if the buffer was empty.
        """
        return self.agent.learn()

    def reset(self):
        """Reset episode state, timestep counter, and consumption history."""
        super().reset()
        self.timestep = 0
        self.consumption_history.clear()
        for _ in range(self.agent.ctx):
            self.consumption_history.append(0.0)

    def get_training_stats(self) -> Dict[str, Any]:
        """Return current training statistics from the underlying agent.

        Returns:
            Dictionary containing algorithm name, update count, buffer
            utilization, and per-update loss histories.
        """
        return {
            'algorithm': 'MP-PPO',
            'total_updates': self.agent.total_updates,
            'buffer_size': self.agent.buf.size(),
            'buffer_capacity': self.agent.buffer_steps,
            'buffer_full': self.agent.buf.full,
            'metrics_history': self.agent.metrics_history,
        }

    def save_model(self, path: str):
        """Save the agent checkpoint to disk.

        Args:
            path: Destination file path for the checkpoint.
        """
        self.agent.save(path)

    def pretrain_forecaster(self, dataset, epochs: int = 100):
        """Pre-train the Transformer forecaster on a supervised dataset.

        Convenience wrapper around ``MPPPO.pretrain_predictor``.

        Args:
            dataset: List of ``(hist, tgt, y)`` tuples for supervised
                training. See ``MPPPO.pretrain_predictor`` for details.
            epochs: Number of training epochs.

        Returns:
            Dictionary with ``"pretrain_epochs"`` and ``"final_nll"`` keys.
        """
        return self.agent.pretrain_predictor(dataset, epochs)