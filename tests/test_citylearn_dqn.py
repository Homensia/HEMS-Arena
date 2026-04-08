"""
DQN Diagnostic Test for CityLearn environment.
Runs 100 steps and prints actions, observations, and SoC evolution.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper
from config import SimulationConfig  # keep your existing SimulationConfig class

# ============================
# Simple DQN-like test agent
# ============================

class SimpleDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),  # output between [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a = self.forward(obs_t)
        return a.numpy()


# =================================
# Main diagnostic test
# =================================

def run_citylearn_dqn_test():
    print("=" * 80)
    print("CITYLEARN DQN TEST — Battery, PV, and Observation Dynamics")
    print("=" * 80)

    # ---- Config ----
    cfg = SimulationConfig(
        building_count=1,
        simulation_days=7,
        dataset_name='akrem_phase_all',
        random_seed=42,
        use_gpu=False
    )
    wrapper = CityLearnWrapper(cfg)

    # Select building and period
    buildings = wrapper.select_buildings()
    start, end = wrapper.select_simulation_period()

    # Create environment
    env = wrapper.create_environment(buildings, start, end)
    print(f"\nEnvironment ready with {len(env.buildings)} building(s)")
    print(f"Simulation period: {start} → {end} ({end-start} steps)\n")

    # ---- Agent ----
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].shape[0]
    agent = SimpleDQN(obs_dim, act_dim)
    print(f"Agent initialized: obs_dim={obs_dim}, act_dim={act_dim}\n")

    # ---- Run simulation ----
    obs, _ = env.reset()
    episode_rewards = []
    print("-" * 80)
    print(f"{'Step':<5} {'Action':<10} {'Reward':<10} {'SoC':<10} {'PV':<10} {'NetLoad':<12} {'Price':<10}")
    print("-" * 80)

    for t in range(100):
        action = agent.act(obs[0])  # one building
        obs_next, reward, done, truncated, info = env.step([action])

        # gather building info
        b = env.buildings[0]
        soc = getattr(b.electrical_storage, "soc", [None])[-1]
        pv = getattr(b, "solar_generation", [0])[-1]
        net = getattr(b, "net_electricity_consumption", [0])[-1]
        price = getattr(b.pricing, "electricity_pricing", [0])[-1]

        episode_rewards.append(float(np.mean(reward)))

        print(f"{t:<5} {action[0]:<10.3f} {np.mean(reward):<10.3f} {soc:<10.3f} {pv:<10.3f} {net:<12.3f} {price:<10.3f}")

        obs = obs_next
        if done or truncated:
            break

    print("-" * 80)
    print(f"Mean Reward: {np.mean(episode_rewards):.3f}")
    soc_values = np.array(env.buildings[0].electrical_storage.soc)
    print(f"SoC Range: {soc_values.min():.3f} – {soc_values.max():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_citylearn_dqn_test()
