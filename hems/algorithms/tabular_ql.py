# hems/algorithms/tql.py
from __future__ import annotations

import warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseAlgorithm

try:
    from citylearn.agents.q_learning import TabularQLearning
    from citylearn.wrappers import TabularQLearningWrapper
    CITYLEARN_AVAILABLE = True
except Exception:
    CITYLEARN_AVAILABLE = False
    warnings.warn(
        "CityLearn TabularQLearning not available. Falling back to internal RealQAgent."
    )


# ------------------------ Utilities ------------------------

def _to_scalar_reward(r) -> float:
    if isinstance(r, (list, tuple, np.ndarray)):
        return float(r[0]) if len(r) > 0 else 0.0
    return float(r)


# ------------------ Picklable fallback agent ----------------

class RealQAgent:
    """Simple, picklable Q-learning agent with:
    - Action discretization from env bounds
    - Online state binning (min/max) and reward normalization
    - Epsilon & LR decay
    """

    def __init__(
        self,
        actions: Optional[List[float]] = None,
        epsilon: float = 0.2,
        lr: float = 0.2,
        gamma: float = 0.99,
        seed: int = 42,
        q_table: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None,
        n_state_features: int = 3,
        n_bins_per_feat: int = 12,
        eps_min: float = 0.01,
        eps_decay_episodes: int = 60,
        lr_min: float = 0.02,
        lr_decay_episodes: int = 60,
        reward_norm_beta: float = 0.01,
    ):
        self.actions = list(actions) if actions is not None else [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
        self.epsilon = float(epsilon)
        self.learning_rate = float(lr)
        self.discount_factor = float(gamma)
        self.q: Dict[Tuple[int, ...], Dict[int, float]] = q_table if q_table is not None else {}
        self.last_state: Optional[Tuple[int, ...]] = None
        self.last_action: Optional[int] = None
        self._rng = np.random.default_rng(int(seed))

        # Binning/normalization params
        self.n_state_features = int(n_state_features)
        self.n_bins_per_feat = int(n_bins_per_feat)
        self._feat_min = np.full(self.n_state_features, np.inf, dtype=np.float32)
        self._feat_max = np.full(self.n_state_features, -np.inf, dtype=np.float32)

        # Reward normalization (EMA)
        self._r_mean = 0.0
        self._r_var = 1.0
        self._r_beta = float(reward_norm_beta)

        # Schedules
        self._episode = 0
        self._eps_min = float(eps_min)
        self._eps_decay_episodes = max(1, int(eps_decay_episodes))
        self._lr_min = float(lr_min)
        self._lr_decay_episodes = max(1, int(lr_decay_episodes))

    # ---------- helpers ----------
    def _safe_obs(self, obs_list: List[List[float]]) -> np.ndarray:
        if not obs_list or not obs_list[0]:
            return np.zeros(self.n_state_features, dtype=np.float32)
        arr = np.asarray(obs_list[0], dtype=np.float32)
        if arr.size < self.n_state_features:
            pad = np.zeros(self.n_state_features - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[: self.n_state_features]

    def _update_running_minmax(self, x: np.ndarray) -> None:
        self._feat_min = np.minimum(self._feat_min, x)
        self._feat_max = np.maximum(self._feat_max, x)

    def _discretize_feats(self, x: np.ndarray) -> Tuple[int, ...]:
        # normalize to [0,1] via running min/max; avoid zero-div
        span = np.maximum(self._feat_max - self._feat_min, 1e-6)
        z = (x - self._feat_min) / span
        z = np.clip(z, 0.0, 1.0)
        bins = np.floor(z * self.n_bins_per_feat).astype(int)
        bins = np.clip(bins, 0, self.n_bins_per_feat)  # right edge case
        return tuple(int(b) for b in bins.tolist())

    def _state_key(self, observations: List[List[float]]) -> Tuple[int, ...]:
        x = self._safe_obs(observations)
        self._update_running_minmax(x)
        return self._discretize_feats(x)

    # ---------- policy ----------
    def predict(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        key = self._state_key(observations)
        if key not in self.q:
            self.q[key] = {i: 0.0 for i in range(len(self.actions))}
        if not deterministic and self._rng.random() < self.epsilon:
            a_idx = int(self._rng.integers(0, len(self.actions)))
        else:
            qvals = self.q[key]
            a_idx = max(qvals, key=qvals.get)
        self.last_state, self.last_action = key, a_idx
        return [[float(self.actions[a_idx])]]

    # ---------- learning ----------
    def _norm_reward(self, r: float) -> float:
        # online mean/var (EMA)
        delta = r - self._r_mean
        self._r_mean += self._r_beta * delta
        self._r_var = (1 - self._r_beta) * (self._r_var + self._r_beta * delta * delta)
        std = max(1e-6, np.sqrt(self._r_var))
        return float((r - self._r_mean) / std)

    def learn(self, reward: float, next_observations: List[List[float]], done: bool) -> None:
        if self.last_state is None or self.last_action is None:
            return
        r = self._norm_reward(float(reward))
        next_key = self._state_key(next_observations)
        if next_key not in self.q:
            self.q[next_key] = {i: 0.0 for i in range(len(self.actions))}
        q_now = self.q[self.last_state][self.last_action]
        if done:
            target = r
        else:
            target = r + self.discount_factor * max(self.q[next_key].values())
        self.q[self.last_state][self.last_action] = q_now + self.learning_rate * (target - q_now)

    # ---------- schedules ----------
    def episode_end(self) -> None:
        self._episode += 1
        # linear decay
        frac_e = min(1.0, self._episode / self._eps_decay_episodes)
        frac_l = min(1.0, self._episode / self._lr_decay_episodes)
        self.epsilon = max(self._eps_min, (1.0 - frac_e) * self.epsilon + frac_e * self._eps_min)
        self.learning_rate = max(self._lr_min, (1.0 - frac_l) * self.learning_rate + frac_l * self._lr_min)

    # ---------- serialization ----------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "q": [[list(k), v] for k, v in self.q.items()],
            "n_state_features": self.n_state_features,
            "n_bins_per_feat": self.n_bins_per_feat,
            "feat_min": self._feat_min.tolist(),
            "feat_max": self._feat_max.tolist(),
            "r_mean": self._r_mean,
            "r_var": self._r_var,
            "episode": self._episode,
            "eps_min": self._eps_min,
            "eps_decay_episodes": self._eps_decay_episodes,
            "lr_min": self._lr_min,
            "lr_decay_episodes": self._lr_decay_episodes,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any], seed: int = 42) -> "RealQAgent":
        q_restored: Dict[Tuple[int, ...], Dict[int, float]] = {}
        for key_list, v in sd.get("q", []):
            q_restored[tuple(int(x) for x in key_list)] = {int(ai): float(qv) for ai, qv in v.items()}
        agent = cls(
            actions=[float(a) for a in sd.get("actions", [])] or None,
            epsilon=float(sd.get("epsilon", 0.2)),
            lr=float(sd.get("learning_rate", 0.2)),
            gamma=float(sd.get("discount_factor", 0.99)),
            seed=int(seed),
            q_table=q_restored,
            n_state_features=int(sd.get("n_state_features", 3)),
            n_bins_per_feat=int(sd.get("n_bins_per_feat", 12)),
            eps_min=float(sd.get("eps_min", 0.01)),
            eps_decay_episodes=int(sd.get("eps_decay_episodes", 60)),
            lr_min=float(sd.get("lr_min", 0.02)),
            lr_decay_episodes=int(sd.get("lr_decay_episodes", 60)),
        )
        agent._feat_min = np.asarray(sd.get("feat_min", [0, 0, 0]), dtype=np.float32)
        agent._feat_max = np.asarray(sd.get("feat_max", [1, 1, 1]), dtype=np.float32)
        agent._r_mean = float(sd.get("r_mean", 0.0))
        agent._r_var = float(sd.get("r_var", 1.0))
        agent._episode = int(sd.get("episode", 0))
        return agent


# -------------------- Main Algorithm wrapper --------------------

class TabularQLearningAlgorithm(BaseAlgorithm):
    """TQL with CityLearn backend if possible, else RealQAgent fallback.
       Fully picklable. Discretizes actions from env bounds for fallback.
    """

    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env, config)

        # hyper-params
        self.epsilon = float(config.get("epsilon", 0.2))
        self.learning_rate = float(config.get("learning_rate", 0.2))
        self.discount_factor = float(config.get("discount_factor", 0.99))
        self.random_seed = int(config.get("random_seed", 42))
        self._uses_citylearn_agent: bool = False
        self._episodes = 0

        # discretization settings for fallback
        self.n_state_features = int(config.get("n_state_features", 3))
        self.n_bins_per_feat = int(config.get("n_bins_per_feat", 12))
        self.act_bins = int(config.get("act_bins", 31))  # align with DQN default

        # schedules for fallback
        self.eps_min = float(config.get("eps_min", 0.01))
        self.eps_decay_episodes = int(config.get("eps_decay_episodes", 60))
        self.lr_min = float(config.get("lr_min", 0.02))
        self.lr_decay_episodes = int(config.get("lr_decay_episodes", 60))
        self.reward_norm_beta = float(config.get("reward_norm_beta", 0.01))

        np.random.seed(self.random_seed)

        # Try CityLearn agent first
        if CITYLEARN_AVAILABLE:
            try:
                wrapped_env = TabularQLearningWrapper(
                    env,
                    default_observation_bin_size=int(config.get("obs_bins", 20)),
                    default_action_bin_size=int(config.get("act_bins", self.act_bins)),
                )
                # Monkey-patch missing attribute expected by some code paths
                if not hasattr(wrapped_env, "observation_names") and hasattr(env, "observation_names"):
                    setattr(wrapped_env, "observation_names", getattr(env, "observation_names"))

                self.agent = TabularQLearning(
                    env=wrapped_env,
                    epsilon=self.epsilon,
                    learning_rate=self.learning_rate,
                    discount_factor=self.discount_factor,
                    random_seed=self.random_seed,
                )
                self._uses_citylearn_agent = True
            except Exception as e:
                print(f"[TQL] CityLearn creation failed: {e}. Using fallback RealQAgent.")
                self.agent = self._build_fallback_agent(env, config)
        else:
            self.agent = self._build_fallback_agent(env, config)

    # ------------ fallback construction ------------
    def _build_fallback_agent(self, env, config) -> RealQAgent:
        # infer discrete actions from env action_space if Box
        actions = config.get("actions")
        if actions is None:
            try:
                low = float(np.asarray(env.action_space.low).squeeze()[0])
                high = float(np.asarray(env.action_space.high).squeeze()[0])
                # ensure 0.0 included
                grid = np.linspace(low, high, self.act_bins, dtype=np.float32)
                # deduplicate + sort to be safe
                actions = sorted(set([float(x) for x in grid.tolist() + [0.0]]))
            except Exception:
                actions = None  # fallback to RealQAgent defaults

        return RealQAgent(
            actions=actions,
            epsilon=self.epsilon,
            lr=self.learning_rate,
            gamma=self.discount_factor,
            seed=self.random_seed,
            n_state_features=self.n_state_features,
            n_bins_per_feat=self.n_bins_per_feat,
            eps_min=self.eps_min,
            eps_decay_episodes=self.eps_decay_episodes,
            lr_min=self.lr_min,
            lr_decay_episodes=self.lr_decay_episodes,
            reward_norm_beta=self.reward_norm_beta,
        )

    # ---------------- Runner API ----------------
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        try:
            return self.agent.predict(observations, deterministic=deterministic)
        except Exception as e:
            print(f"[TQL] act error -> neutral action: {e}")
            return [[0.0]]

    def learn(self, obs, actions, reward, next_obs, done) -> Optional[Dict[str, Any]]:
        try:
            r = _to_scalar_reward(reward)
            if hasattr(self.agent, "learn") and callable(self.agent.learn):
                self.agent.learn(r, next_obs, bool(done))
            stats = {
                "epsilon": float(getattr(self.agent, "epsilon", self.epsilon)),
                "learning_rate": float(getattr(self.agent, "learning_rate", self.learning_rate)),
                "q_table_size": int(len(self.agent.q)) if isinstance(self.agent, RealQAgent) else 0,
                "reward": float(r),
            }
            # decay schedules at episode end
            if bool(done) and isinstance(self.agent, RealQAgent):
                self.agent.episode_end()
                self._episodes += 1
            return stats
        except Exception as e:
            return {"error": str(e), "reward": 0.0}

    def get_training_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": getattr(self.agent, "epsilon", self.epsilon),
            "learning_rate": getattr(self.agent, "learning_rate", self.learning_rate),
            "q_table_size": len(self.agent.q) if isinstance(self.agent, RealQAgent) else 0,
            "episodes": self._episodes,
        }

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "description": "Tabular Q-Learning with CityLearn backend or enhanced RealQAgent fallback (picklable).",
            "trainable": True,
            "parameters": {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "backend": "citylearn" if self._uses_citylearn_agent else "real_q",
                "act_bins": self.act_bins,
                "n_state_features": self.n_state_features,
                "n_bins_per_feat": self.n_bins_per_feat,
            }
        })
        return info

    # --------------- Pickle safety ---------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # Replace agent with snapshot
        if self._uses_citylearn_agent:
            snap = {
                "impl": "citylearn",
                "epsilon": getattr(self.agent, "epsilon", self.epsilon),
                "learning_rate": getattr(self.agent, "learning_rate", self.learning_rate),
                "discount_factor": getattr(self.agent, "discount_factor", self.discount_factor),
                "q": getattr(self.agent, "q", None) or getattr(self.agent, "q_table", None),
            }
            state["agent"] = snap
        else:
            state["agent"] = {"impl": "real_q", "state": self.agent.state_dict()}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        blob = state.get("agent", {})
        if blob.get("impl") == "real_q":
            self.agent = RealQAgent.from_state_dict(blob.get("state", {}), seed=self.random_seed)
            self._uses_citylearn_agent = False
        else:
            # Recreate a usable agent even if we can't rebuild CityLearn agent without env
            self.agent = RealQAgent(
                actions=None,
                epsilon=float(blob.get("epsilon", self.epsilon)),
                lr=float(blob.get("learning_rate", self.learning_rate)),
                gamma=float(blob.get("discount_factor", self.discount_factor)),
                seed=self.random_seed,
                n_state_features=self.n_state_features,
                n_bins_per_feat=self.n_bins_per_feat,
                eps_min=self.eps_min,
                eps_decay_episodes=self.eps_decay_episodes,
                lr_min=self.lr_min,
                lr_decay_episodes=self.lr_decay_episodes,
                reward_norm_beta=self.reward_norm_beta,
            )
            self._uses_citylearn_agent = False


    def __getstate__(self):
        # 1) copie de l'état ET purge des refs lourdes (env, wrappers, loggers…)
        state = _purge_heavy_refs(self.__dict__.copy())

        # 2) remplace l'agent par un snapshot compact et picklable
        if getattr(self, "_uses_citylearn_agent", False):
            # CityLearn agent → ne JAMAIS pickle l'agent réel (il capture l'env)
            snapshot = {
                "impl": "citylearn",
                "epsilon": float(getattr(self.agent, "epsilon", self.epsilon)),
                "learning_rate": float(getattr(self.agent, "learning_rate", self.learning_rate)),
                "discount_factor": float(getattr(self.agent, "discount_factor", self.discount_factor)),
                # Optionnel: on n’essaie PAS de sauver une q-table interne ici.
                "q": None,
            }
            state["agent"] = snapshot
        else:
            # RealQAgent → safe : on sérialise un state_dict léger
            try:
                agent_state = self.agent.state_dict()
            except Exception:
                agent_state = None
            state["agent"] = {"impl": "real_q", "state": agent_state}

        # sécurité: s'assurer qu'il n’y a plus d'attributs *_env* restants
        for k in [k for k in list(state.keys()) if "env" in k.lower() or "wrapper" in k.lower()]:
            state.pop(k, None)

        return state

    def __setstate__(self, state):
        # on restaure le dict brut
        self.__dict__.update(state)
        blob = state.get("agent", {})
        if blob.get("impl") == "real_q":
            # on reconstruit un RealQAgent utilisable
            self.agent = RealQAgent.from_state_dict(blob.get("state", {}) or {}, seed=self.random_seed)
            self._uses_citylearn_agent = False
        else:
            # CityLearn snapshot → on reconstruit un RealQAgent par défaut (sans env)
            self.agent = RealQAgent(
                actions=None,
                epsilon=float(blob.get("epsilon", self.epsilon)),
                lr=float(blob.get("learning_rate", self.learning_rate)),
                gamma=float(blob.get("discount_factor", self.discount_factor)),
                seed=self.random_seed,
                n_state_features=self.n_state_features,
                n_bins_per_feat=self.n_bins_per_feat,
                eps_min=self.eps_min,
                eps_decay_episodes=self.eps_decay_episodes,
                lr_min=self.lr_min,
                lr_decay_episodes=self.lr_decay_episodes,
                reward_norm_beta=self.reward_norm_beta,
            )
            self._uses_citylearn_agent = False  
            
def _purge_heavy_refs(d: dict) -> dict:
    """Remove environment/wrapper/logger-like refs from a __dict__ copy."""
    keys = list(d.keys())
    for k in keys:
        kl = k.lower()
        if any(w in kl for w in ("env", "wrapper", "logger", "visual", "dataset")):
            d.pop(k, None)
    return d