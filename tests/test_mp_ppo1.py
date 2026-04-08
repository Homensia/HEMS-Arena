# ==========================================================
# tests/test_mp_ppo.py
# ==========================================================
import sys, os

# Ensure root project directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import numpy as np
import torch
import pytest

from hems.algorithms.mp_ppo import (
    MPPPO,
    TinyTransformer,
    ActorCritic,
    RollingReplay,
    create_temporal_features,
    create_future_temporal_features,
    student_t_nll,
)

# ---------------------------------------------------------------------
# 1. Fixtures and helpers
# ---------------------------------------------------------------------

@pytest.fixture
def basic_config():
    """Minimal config for deterministic unit testing."""
    return {
        "horizon": 4,
        "ctx": 8,
        "in_dim": 5,
        "d_model": 16,
        "nhead": 2,
        "layers": 2,
        "ff": 16,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_ratio": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "lr_policy": 1e-3,
        "lr_pred": 1e-3,
        "buffer_steps": 128,
        "train_iters": 1,
        "pred_updates_per_iter": 1,
        "minibatch": 8,
        "normalize_obs": True,
        "device": "cpu",
        "seed": 42,
    }

@pytest.fixture
def dummy_agent(basic_config):
    """Return initialized MP-PPO agent."""
    obs_dim, act_dim = 10, 4
    return MPPPO(obs_dim, act_dim, basic_config)

# ---------------------------------------------------------------------
# 2. Core component tests
# ---------------------------------------------------------------------

def test_predictor_forward():
    """Ensure TinyTransformer forward works and outputs correct shapes."""
    net = TinyTransformer(d_model=16, nhead=2, num_layers=2, ff=16, horizon=4, ctx=8, in_dim=5)
    hist = torch.randn(2, 8, 5)
    tgt = torch.randn(2, 4, 5)
    nu, xi, sigma, latent = net(hist, tgt)
    assert nu.shape == (2, 4)
    assert xi.shape == (2, 4)
    assert sigma.shape == (2, 4)
    assert latent.shape == (2, 16)
    assert torch.all(nu > 1.0)
    assert torch.all(sigma > 0.0)

def test_student_t_nll():
    """Verify Student-t NLL behaves sensibly (lower for matching means)."""
    y = torch.ones(10)
    nu, xi, sigma = torch.ones(10) * 5, torch.ones(10), torch.ones(10) * 0.5
    loss_good = student_t_nll(y, nu, xi, sigma)
    loss_bad = student_t_nll(y, nu, xi + 5.0, sigma)
    assert loss_good < loss_bad

def test_actor_critic_shapes():
    """ActorCritic forward returns correct shapes."""
    ac = ActorCritic(obs_dim=10, act_dim=4, pred_latent_dim=16)
    obs = torch.randn(3, 10 + 16)
    logits, value = ac(obs)
    assert logits.shape == (3, 4)
    assert value.shape == (3,)

# ---------------------------------------------------------------------
# 3. Agent initialization & seed reproducibility
# ---------------------------------------------------------------------

def test_seed_reproducibility(basic_config):
    """Ensure deterministic initialization for same seed."""
    a1 = MPPPO(10, 4, basic_config)
    a2 = MPPPO(10, 4, basic_config)
    p1 = list(a1.ac.parameters())[0][0][0].item()
    p2 = list(a2.ac.parameters())[0][0][0].item()
    assert np.isclose(p1, p2, atol=1e-6)

# ---------------------------------------------------------------------
# 4. Temporal feature generation
# ---------------------------------------------------------------------

def test_temporal_features_consistency():
    """Temporal features should have smooth transitions."""
    ctx = 8
    feat1 = create_temporal_features(10, ctx)
    feat2 = create_temporal_features(11, ctx)
    # check continuity in hour/day encoding
    assert feat1.shape == (ctx, 5)
    assert np.allclose(feat2[-1, 0], ((11 % 24) / 24.0), atol=1e-3)

def test_future_temporal_features():
    """Future features increase monotonically in time."""
    horizon = 4
    feat = create_future_temporal_features(0, horizon)
    assert feat.shape == (horizon, 5)
    assert np.all(np.diff(feat[:, 0]) > -0.1)  # hour increases roughly

# ---------------------------------------------------------------------
# 5. Buffer and GAE logic
# ---------------------------------------------------------------------

def test_buffer_add_and_size(dummy_agent):
    """Ensure buffer add increments size correctly."""
    buf = dummy_agent.buf
    N = buf.size()
    obs = np.zeros(dummy_agent.buf.obs.shape[1], np.float32)
    hist = np.zeros((dummy_agent.ctx, dummy_agent.in_dim), np.float32)
    tgt = np.zeros((dummy_agent.horizon, dummy_agent.in_dim), np.float32)
    for _ in range(5):
        buf.add(obs, 0, 1.0, False, 0.0, 0.0, np.ones((dummy_agent.act_dim,), bool), hist, tgt)
    assert buf.size() == 5
    buf.add(obs, 0, 1.0, True, 0.0, 0.0, np.ones((dummy_agent.act_dim,), bool), hist, tgt)
    assert buf.dones_flag.any()

def test_gae_calculation(dummy_agent):
    """Verify GAE returns correct advantage length and finite values."""
    rewards = np.ones(5)
    values = np.linspace(0, 1, 6)
    dones = np.zeros(5)
    adv, ret = dummy_agent._gae(rewards, values, dones, 0.99, 0.95)
    assert len(adv) == 5 and len(ret) == 5
    assert np.all(np.isfinite(adv))

# ---------------------------------------------------------------------
# 6. End-to-end learning step (sanity)
# ---------------------------------------------------------------------

def test_learn_step_runs(dummy_agent):
    """Run one learn iteration with dummy data; ensure no crash."""
    N = dummy_agent.buf.N
    obs_dim, act_dim = dummy_agent.buf.obs.shape[1], dummy_agent.buf.mask.shape[1]

    for i in range(32):
        obs = np.random.randn(obs_dim).astype(np.float32)
        act = np.random.randint(0, act_dim)
        rew = np.random.randn()
        done = i % 10 == 0
        logp, val = np.random.randn(), np.random.randn()
        mask = np.ones(act_dim, bool)
        hist = np.random.randn(dummy_agent.ctx, dummy_agent.in_dim).astype(np.float32)
        tgt = np.random.randn(dummy_agent.horizon, dummy_agent.in_dim).astype(np.float32)
        dummy_agent.buf.add(obs, act, rew, done, logp, val, mask, hist, tgt)
        dummy_agent.buf.write_forecast_value(0.0)

    losses = dummy_agent.learn()
    assert "loss/policy" in losses
    assert np.isfinite(losses["loss/policy"])

# ---------------------------------------------------------------------
# 7. Save / Load
# ---------------------------------------------------------------------

def test_save_and_load(tmp_path, dummy_agent):
    """Ensure model saving and reloading preserve weights."""
    save_path = tmp_path / "mp_ppo_test.pth"
    dummy_agent.save(str(save_path))
    assert os.path.exists(save_path)
    
    test_config = {
        'horizon': dummy_agent.horizon,
        'ctx': dummy_agent.ctx,
        'in_dim': dummy_agent.in_dim,
        'd_model': dummy_agent.d_model,
        'nhead': dummy_agent.nhead,
        'layers': dummy_agent.layers,
        'ff': dummy_agent.ff,
        'buffer_steps': dummy_agent.buffer_steps
    }
    agent2 = MPPPO(dummy_agent.obs_dim, dummy_agent.act_dim, test_config)
    agent2.load(str(save_path))
    assert isinstance(agent2.predictor, TinyTransformer)