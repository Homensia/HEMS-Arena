"""
CityLearn RL Integration Test
------------------------------
Validates the new CityLearnWrapper end-to-end with:
 - Dynamic tariffs (price signal)
 - PV generation & battery storage
 - Simple DQN-like agent interacting with the environment

Run:
    python3 -m hems.core.test_citylearn_rl
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Allow imports if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from hems.core.config import SimulationConfig

from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper


# ============================================================
# 1. Simple DQN-like agent (no training, just inference + noise)
# ============================================================

class SimpleDQNAgent:
    """Minimal DQN-like agent for quick environment sanity checks."""

    def __init__(self, obs_dim: int, act_dim: int, epsilon: float = 0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = epsilon

    def act(self, obs):
        """Choose an action with epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return [[np.random.uniform(-1, 1)]]
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_t)
        action = torch.tanh(q_values).numpy()[0]
        return [[float(action[0])]]

    def learn(self, *args, **kwargs):
        """No-op (we're only testing env dynamics)."""
        pass


# ============================================================
# 2. Test runner for CityLearnWrapper
# ============================================================

def run_citylearn_rl_test():
    print("=" * 80)
    print("CITYLEARN RL TEST — Environment, Battery, PV, Tariff, Agent Interaction")
    print("=" * 80)

    # ---- Config setup ----
    cfg = SimulationConfig(
        building_count=1,
        simulation_days=5,
        dataset_name='akrem_phase_all',
        random_seed=42,
        use_gpu=False
    )

    # ---- Environment setup ----
    wrapper = CityLearnWrapper(cfg)
    buildings = wrapper.select_buildings()
    start, end = wrapper.select_simulation_period()
    env = wrapper.create_environment(buildings, start, end, use_custom_battery=True)
    b = env.buildings[0]
    print("\n--- Battery diagnostic ---")
    print("Has electrical_storage:", hasattr(b, 'electrical_storage'))
    if hasattr(b, 'electrical_storage'):
        print("Capacity:", getattr(b.electrical_storage, 'capacity', None))
        print("Power rating:", getattr(b.electrical_storage, 'power_rating', None))
        print("Initial SoC:", getattr(b.electrical_storage, 'soc_init', None))


    print(f"Environment created with {len(env.buildings)} building(s)")
    print(f"Simulation period: {start} → {end} ({end - start + 1} steps)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # ---- Agent setup ----
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_dim = len(obs[0])
    action_space = env.action_space[0] if isinstance(env.action_space, list) else env.action_space
    act_dim = action_space.shape[0]

    agent = SimpleDQNAgent(obs_dim, act_dim)
    print(f"Agent initialized: obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- Data containers ----
    steps = min(168, env.time_steps)  # simulate ~1 week
    soc_hist, price_hist, pv_hist, act_hist, rew_hist = [], [], [], [], []

    # ---- Interaction loop ----
    for t in range(steps):
        action = agent.act(obs[0])
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, done, trunc, info = step_result
        else:
            next_obs, reward, done, info = step_result

        # Collect info from building
        building = env.buildings[0]
        soc = getattr(building.electrical_storage, 'soc', [0])[-1]
        price = getattr(building.pricing, 'electricity_pricing', [0])[-1]
        pv = getattr(building, 'solar_generation', [0])[-1]

        soc_hist.append(float(soc))
        price_hist.append(float(price))
        pv_hist.append(float(pv))
        act_hist.append(float(action[0][0]))
        rew_hist.append(float(reward[0] if isinstance(reward, list) else reward))

        obs = next_obs
        if done:
            break

    # ---- Summary ----
    print("\n--- Simulation Summary ---")
    print(f"Total steps: {len(rew_hist)}")
    print(f"Average reward: {np.mean(rew_hist):.3f}")
    print(f"SOC mean: {np.mean(soc_hist):.3f}")
    print(f"Price mean: {np.mean(price_hist):.3f}")
    print(f"PV mean: {np.mean(pv_hist):.3f}")
    print(f"Action mean: {np.mean(act_hist):.3f}")
    print(f"Reward range: [{min(rew_hist):.2f}, {max(rew_hist):.2f}]")

    # ---- Plot (optional) ----
    try:
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(price_hist, label="Electricity Price [€/kWh]")
        axs[1].plot(pv_hist, label="PV Generation [kW]")
        axs[2].plot(soc_hist, label="Battery SoC [0–1]")
        axs[3].plot(act_hist, label="Agent Action [-1=discharge, +1=charge]")

        axs[0].set_title("CityLearn RL Test — Tariff, PV, SoC, Actions")
        for ax in axs:
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(Plot skipped: {e})")


# ============================================================
# 3. Main entry
# ============================================================

if __name__ == "__main__":
    run_citylearn_rl_test()
