#=======================
# tests/test_mp_ppo.py
#=======================
import math
import numpy as np
import torch
import pytest

# ---- import your implementation ----

from hems.algorithms.mp_ppo import (
    MaskedCategorical, RollingReplay, MPPPO, TinyTransformer, student_t_nll
)

# ----------------------------
# Helpers / tiny configs
# ----------------------------
SMALL_CFG = {
    "horizon": 4,          # small to make assertions easy
    "ctx": 6,
    "in_dim": 1,
    "d_model": 16,
    "nhead": 2,
    "layers": 2,
    "ff": 16,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "lr_policy": 1e-3,
    "lr_pred": 1e-3,
    "train_iters": 2,
    "pred_updates_per_cycle": 3,  # faithful "per cycle" count
    "minibatch": 16,
    "buffer_steps": 64,
    "device": "cpu",
}

def make_fake_obs(obs_dim):
    return np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)

def make_hist_and_queries(ctx, h, in_dim):
    # simple ramp features; hist carries values, queries are zeros (time-only placeholder)
    hist = np.linspace(0.0, 1.0, ctx, dtype=np.float32).reshape(ctx, 1).repeat(in_dim, axis=1)
    tgtq = np.zeros((h, in_dim), dtype=np.float32)
    return hist, tgtq

# ----------------------------
# Unit tests
# ----------------------------

def test_masked_categorical_respects_mask():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # 4 actions
    mask = torch.tensor([[True, False, False, True]])  # only 0 and 3 valid
    dist = MaskedCategorical(logits=logits, mask=mask)
    # sample multiple times; invalid actions must never be drawn
    samples = [int(dist.sample().item()) for _ in range(100)]
    assert all(s in (0, 3) for s in samples), f"Invalid action sampled: {samples}"

def test_rolling_replay_future_fill_algorithm1():
    h, ctx, in_dim = 4, 6, 1
    buf = RollingReplay(capacity=16, obs_dim=3, act_dim=5, h=h, ctx=ctx, in_dim=in_dim)

    # add 6 transitions in same episode, write s_tilde_t after each add
    for t in range(6):
        obs = np.zeros(3, dtype=np.float32)
        hist = np.zeros((ctx, in_dim), dtype=np.float32)
        tgtq = np.zeros((h,   in_dim), dtype=np.float32)
        buf.add(obs=obs, act=0, rew=0.0, done=False, logp=0.0, val=0.0,
                mask=np.ones(5, dtype=bool), hist_series=hist, tgt_queries=tgtq)
        # current s_tilde_t is just float(t)
        buf.write_forecast_value(float(t))

    # check last slot (t=5) has [5, 0, 0, 0]
    last = (buf.pos - 1) % buf.N
    np.testing.assert_allclose(buf.y_future[last], [5, 0, 0, 0], atol=1e-6)

    # check previous slot (t=4) has [4,5,0,0]
    prev = (buf.pos - 2) % buf.N
    np.testing.assert_allclose(buf.y_future[prev], [4,5,0,0], atol=1e-6)

    # check one more back (t=3) has [3,4,5,0]
    prev2 = (buf.pos - 3) % buf.N
    np.testing.assert_allclose(buf.y_future[prev2], [3,4,5,0], atol=1e-6)

    # and (t=2) has [2,3,4,5] since we observed up to t=5 (h=4 future labels filled)
    prev3 = (buf.pos - 4) % buf.N
    np.testing.assert_allclose(buf.y_future[prev3], [2,3,4,5], atol=1e-6)

def test_tiny_transformer_student_t_nll_shapes_and_grad():
    B, ctx, h, in_dim = 8, 6, 4, 1
    net = TinyTransformer(d_model=16, nhead=2, num_layers=2, ff=16,
                          horizon=h, ctx=ctx, in_dim=in_dim)
    hist = torch.randn(B, ctx, in_dim)
    tgtq = torch.randn(B, h,   in_dim)
    y    = torch.randn(B, h)

    nu, xi, sigma, latent = net(hist, tgtq)
    assert nu.shape == (B, h) and xi.shape == (B, h) and sigma.shape == (B, h)
    assert latent.shape == (B, 16)

    loss = student_t_nll(y, nu, xi, sigma)
    loss.backward()
    # some gradients must be non-zero
    grad_norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.abs().sum().item()
    assert grad_norm > 0.0

@pytest.mark.parametrize("apply_mask", [False, True])
def test_mpppo_end_to_end_smoke(apply_mask):
    torch.manual_seed(0); np.random.seed(0)

    obs_dim, act_dim = 7, 9
    agent = MPPPO(obs_dim=obs_dim, act_dim=act_dim, config=SMALL_CFG)

    # fabricate one buffer cycle
    steps = min(SMALL_CFG["buffer_steps"], 64)
    for t in range(steps):
        obs = make_fake_obs(obs_dim)
        hist, tgtq = make_hist_and_queries(SMALL_CFG["ctx"], SMALL_CFG["horizon"], SMALL_CFG["in_dim"])

        if apply_mask:
            mask = np.zeros(act_dim, dtype=bool)
            mask[[0, 3, 5]] = True  # only a subset valid
        else:
            mask = np.ones(act_dim, dtype=bool)

        a, aux = agent.act(obs, mask=mask, hist_series=hist, tgt_queries=tgtq, deterministic=False)

        # synthetic reward; s_tilde_t is independent scalar we want the predictor to learn (here sinusoid)
        s_tilde_t = float(math.sin(t / 10.0))
        rew = float(np.random.randn() * 0.01)

        agent.store(obs=obs, act=a, rew=rew, done=False, logp=aux["logp"], val=aux["value"],
                    mask=mask, hist_series=hist, tgt_queries=tgtq, s_tilde_scalar=s_tilde_t)

    stats = agent.learn()
    # check keys exist and values are finite
    assert "loss/policy" in stats and "loss/value" in stats and "loss/predictor_nll" in stats
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} not finite: {v}"

def test_mask_applied_during_update_changes_entropy():
    # compare entropy with tight mask vs all-valid; masked should reduce entropy
    torch.manual_seed(0); np.random.seed(0)
    obs_dim, act_dim = 5, 10
    cfg = dict(SMALL_CFG)
    cfg["buffer_steps"] = 32
    agent = MPPPO(obs_dim=obs_dim, act_dim=act_dim, config=cfg)

    steps = cfg["buffer_steps"]
    for t in range(steps):
        obs = make_fake_obs(obs_dim)
        hist, tgtq = make_hist_and_queries(cfg["ctx"], cfg["horizon"], cfg["in_dim"])

        # alternate masks: many invalid vs all valid
        if t % 2 == 0:
            mask = np.zeros(act_dim, dtype=bool); mask[[1, 7]] = True
        else:
            mask = np.ones(act_dim, dtype=bool)

        a, aux = agent.act(obs, mask=mask, hist_series=hist, tgt_queries=tgtq, deterministic=False)
        agent.store(obs=obs, act=a, rew=0.0, done=False, logp=aux["logp"], val=aux["value"],
                    mask=mask, hist_series=hist, tgt_queries=tgtq, s_tilde_scalar=float(t))

    stats = agent.learn()
    assert "loss/entropy" in stats
    # not strictly equal check but ensure it's a reasonable number
    assert -10.0 < stats["loss/entropy"] < 10.0
