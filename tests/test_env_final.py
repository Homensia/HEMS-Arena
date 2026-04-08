"""
COMPREHENSIVE CITYLEARN ENVIRONMENT TEST
Tests: Battery, Observations, Tariff, Rewards, Agent Interaction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper

class SimulationConfig:
    def __init__(self, **kwargs):
        # Defaults
        self.dataset_name = 'citylearn_challenge_2022_phase_all'
        self.building_count = 1
        self.simulation_days = 7
        self.random_seed = 42
        self.building_id = None
        self.tql_config = {}
        self.device = 'cpu'
        # Override with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class SimpleDQN:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def get_action(self, obs):
        # Simple rule-based policy
        obs_array = np.array(obs)
        soc_idx = min(20, len(obs_array) - 1)
        price_idx = min(19, len(obs_array) - 1)
        
        soc = obs_array[soc_idx]
        price = obs_array[price_idx]
        
        # Simple strategy
        if price < 0.25 and soc < 0.7:
            return 0.6  # Charge
        elif price > 0.3 and soc > 0.4:
            return -0.6  # Discharge
        else:
            return 0.0


class MPCAgent:
    """Model Predictive Control agent for battery management."""
    
    def __init__(self, obs_dim, act_dim, capacity=6.4, nominal_power=3.2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.nominal_power = nominal_power
        self.efficiency = 0.9
    
    def get_action(self, obs):
        """MPC strategy: charge at low prices, discharge at high prices."""
        obs_array = np.array(obs)
        
        # Extract features (adjust indices if needed)
        soc = obs_array[min(20, len(obs_array) - 1)]
        price = obs_array[min(19, len(obs_array) - 1)]
        hour = obs_array[2] if len(obs_array) > 2 else 12
        
        # Predict if price will increase (simple heuristic based on hour)
        # Typically: low at night (0-6), high in evening (17-21)
        price_will_rise = (6 <= hour < 17)
        
        # MPC decision logic
        if price < 0.25:  # Very low price
            if soc < 0.95:
                return 0.8  # Aggressive charge
            else:
                return 0.0
        elif price < 0.35:  # Low price
            if soc < 0.7 and price_will_rise:
                return 0.6  # Moderate charge
            else:
                return 0.0
        elif price > 0.45:  # High price
            if soc > 0.3:
                return -0.8  # Aggressive discharge
            else:
                return 0.0
        elif price > 0.35:  # Moderate price
            if soc > 0.6 and not price_will_rise:
                return -0.4  # Light discharge
            else:
                return 0.0
        else:
            return 0.0  # Hold


class ImprovedAgent:
    """Improved rule-based agent with hysteresis."""
    
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.last_action = 0.0
    
    def get_action(self, obs):
        obs_array = np.array(obs)
        soc = obs_array[min(20, len(obs_array) - 1)]
        price = obs_array[min(19, len(obs_array) - 1)]
        hour = obs_array[2] if len(obs_array) > 2 else 12
        
        # Hysteresis: don't change action too quickly
        action = self.last_action * 0.3
        
        # Strong signals
        if price < 0.24 and soc < 0.8:
            action += 0.7  # Strong charge signal
        elif price > 0.5 and soc > 0.25:
            action -= 0.7  # Strong discharge signal
        
        # Moderate signals
        elif price < 0.28 and soc < 0.5:
            action += 0.3
        elif price > 0.4 and soc > 0.5:
            action -= 0.3
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action
        
        return action


def test_environment(agent_type='mpc'):
    print("="*80)
    print("CITYLEARN ENVIRONMENT COMPREHENSIVE TEST")
    print("="*80)
    
    # Setup
    cfg = SimulationConfig(building_count=1, simulation_days=7, random_seed=42, use_gpu=False)
    wrapper = CityLearnWrapper(cfg)
    buildings = wrapper.select_buildings()
    start, end = wrapper.select_simulation_period()
    env = wrapper.create_environment(buildings, start, end, use_custom_battery=True)
    
    print(f"\nEnvironment ready with {len(buildings)} building(s)")
    print(f"Simulation period: {start} → {end} ({end - start + 1} steps)")
    
    # Initialize agent
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_dim = len(obs[0])
    
    if agent_type == 'mpc':
        agent = MPCAgent(obs_dim, 1)
        print(f"Agent: MPC (Model Predictive Control)")
    elif agent_type == 'improved':
        agent = ImprovedAgent(obs_dim, 1)
        print(f"Agent: Improved Rule-Based")
    else:
        agent = SimpleDQN(obs_dim, 1)
        print(f"Agent: Simple Rule-Based")
    
    epsilon = 0.1  # Low exploration for MPC
    
    print(f"obs_dim={obs_dim}, act_dim=1")
    
    # Get building reference
    b = env.buildings[0] if hasattr(env, 'buildings') else env.env.buildings[0]
    
    # DEBUG: Print full observation on first step to find indices
    print("\n[DEBUG] First observation structure:")
    first_obs = obs[0] if isinstance(obs, list) else obs
    print(f"Observation length: {len(first_obs)}")
    print(f"First 10 values: {first_obs[:10]}")
    
    # Try to identify SoC and Price indices
    soc_init = env.get_battery_soc() if hasattr(env, 'get_battery_soc') else 0.5
    price_init = b.pricing.electricity_pricing[0] if hasattr(b, 'pricing') else 0.22
    
    soc_idx = None
    price_idx = None
    for i, val in enumerate(first_obs):
        if abs(val - soc_init) < 0.01:
            if soc_idx is None:
                soc_idx = i
                print(f"Found SoC at index {i}: {val}")
        if abs(val - price_init) < 0.01:
            if price_idx is None:
                price_idx = i
                print(f"Found Price at index {i}: {val}")
    
    if soc_idx is None:
        print(f"WARNING: Could not find SoC in observations (looking for ~{soc_init})")
        soc_idx = 20  # fallback
    if price_idx is None:
        print(f"WARNING: Could not find Price in observations (looking for ~{price_init})")
        price_idx = 19  # fallback
    
    print(f"Using SoC index: {soc_idx}, Price index: {price_idx}\n")
    
    # Data collection
    data = {
        'step': [],
        'action': [],
        'reward': [],
        'soc': [],
        'pv': [],
        'net_load': [],
        'price': [],
        'obs_hour': [],
        'obs_price': [],
        'obs_soc': []
    }
    
    print("\n" + "-"*80)
    print(f"{'Step':<6} {'Action':<10} {'Reward':<10} {'SoC':<10} {'PV':<10} {'NetLoad':<12} {'Price':<10}")
    print("-"*80)
    
    # Run 100 steps
    for step in range(100):
        # Agent action
        obs_array = obs[0] if isinstance(obs, list) else obs
        action_val = agent.get_action(obs_array)
        
        # Add exploration noise
        if np.random.rand() < epsilon:
            action_val += np.random.uniform(-0.3, 0.3)
            action_val = np.clip(action_val, -1, 1)
        
        # Environment step
        next_obs, reward, done, trunc, info = env.step([[action_val]])
        
        # Extract data
        if hasattr(env, 'get_battery_soc'):
            soc = env.get_battery_soc()
        else:
            storage = b.electrical_storage
            soc = storage.soc[b.time_step - 1] if b.time_step > 0 else 0.0
        
        pv = b.solar_generation[b.time_step - 1] if hasattr(b, 'solar_generation') else 0.0
        net_load = b.net_electricity_consumption[b.time_step - 1] if hasattr(b, 'net_electricity_consumption') else 0.0
        price = b.pricing.electricity_pricing[b.time_step - 1] if hasattr(b, 'pricing') else 0.0
        
        reward_val = reward[0] if isinstance(reward, list) else reward
        
        # Store data
        data['step'].append(step)
        data['action'].append(action_val)
        data['reward'].append(reward_val)
        data['soc'].append(soc)
        data['pv'].append(pv)
        data['net_load'].append(net_load)
        data['price'].append(price)
        
        # Extract observation features - use found indices
        obs_array = next_obs[0] if isinstance(next_obs, list) else next_obs
        data['obs_hour'].append(obs_array[2] if len(obs_array) > 2 else 0)
        data['obs_price'].append(obs_array[price_idx] if len(obs_array) > price_idx else 0)
        data['obs_soc'].append(obs_array[soc_idx] if len(obs_array) > soc_idx else 0)
        
        # Print
        print(f"{step:<6} {action_val:<10.3f} {reward_val:<10.3f} {soc:<10.3f} {pv:<10.3f} {net_load:<12.3f} {price:<10.3f}")
        
        obs = next_obs
        
        if done or trunc:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    data_arrays = {k: np.array(v) for k, v in data.items()}
    
    # 1. Battery analysis
    print("\n1. BATTERY DYNAMICS:")
    soc_arr = data_arrays['soc']
    action_arr = data_arrays['action']
    print(f"   SoC range: [{soc_arr.min():.4f}, {soc_arr.max():.4f}]")
    print(f"   SoC mean: {soc_arr.mean():.4f}")
    print(f"   SoC std: {soc_arr.std():.4f}")
    print(f"   Utilization: {(soc_arr.max() - soc_arr.min()):.4f}")
    
    # Check if actions correlate with SoC changes
    soc_changes = np.diff(soc_arr)
    if len(soc_changes) > 0:
        # Filter out saturated states (SoC > 0.98 or < 0.02)
        not_saturated = (soc_arr[:-1] < 0.98) & (soc_arr[:-1] > 0.02)
        
        if np.sum(not_saturated) > 10:
            filtered_actions = action_arr[:-1][not_saturated]
            filtered_changes = soc_changes[not_saturated]
            corr = np.corrcoef(filtered_actions, filtered_changes)[0,1] if len(filtered_actions) > 1 else 0
            print(f"   Action-SoC correlation (non-saturated): {corr:.4f}")
        else:
            corr = np.corrcoef(action_arr[:-1], soc_changes)[0,1]
            print(f"   Action-SoC correlation (all): {corr:.4f}")
        
        # Better check: do positive actions increase SoC (when not full)?
        pos_actions = (action_arr[:-1] > 0.3) & (soc_arr[:-1] < 0.98)
        neg_actions = (action_arr[:-1] < -0.3) & (soc_arr[:-1] > 0.02)
        pos_soc_changes = soc_changes[pos_actions]
        neg_soc_changes = soc_changes[neg_actions]
        
        if len(pos_soc_changes) > 0:
            pos_avg = np.mean(pos_soc_changes)
            print(f"   Avg SoC change after CHARGE (not full): {pos_avg:+.6f}")
        if len(neg_soc_changes) > 0:
            neg_avg = np.mean(neg_soc_changes)
            print(f"   Avg SoC change after DISCHARGE (not empty): {neg_avg:+.6f}")
        
        battery_works = (len(pos_soc_changes) > 0 and np.mean(pos_soc_changes) > 0.001) or \
                       (len(neg_soc_changes) > 0 and np.mean(neg_soc_changes) < -0.001)
        
        if battery_works:
            print("   ✓ Battery responding to actions")
        else:
            print("   ✗ Battery NOT responding properly")
    
    # 2. Tariff analysis
    print("\n2. TARIFF (PRICE) DYNAMICS:")
    price_arr = data_arrays['price']
    print(f"   Price range: [{price_arr.min():.4f}, {price_arr.max():.4f}]")
    print(f"   Price mean: {price_arr.mean():.4f}")
    print(f"   Price std: {price_arr.std():.4f}")
    print(f"   Unique prices: {len(np.unique(price_arr.round(4)))}")
    if price_arr.std() > 0.01:
        print("   ✓ Dynamic pricing (varies over time)")
    else:
        print("   ✗ Static pricing (fixed)")
    
    # 3. PV generation
    print("\n3. PV GENERATION:")
    pv_arr = data_arrays['pv']
    print(f"   PV range: [{pv_arr.min():.4f}, {pv_arr.max():.4f}]")
    print(f"   PV mean: {pv_arr.mean():.4f}")
    print(f"   PV varies: {len(np.unique(pv_arr.round(4)))} unique values")
    if len(np.unique(pv_arr.round(4))) > 10:
        print("   ✓ PV shows realistic variation")
    else:
        print("   ⚠ PV may be static or limited")
    
    # 4. Reward analysis
    print("\n4. REWARD FUNCTION:")
    reward_arr = data_arrays['reward']
    print(f"   Reward range: [{reward_arr.min():.4f}, {reward_arr.max():.4f}]")
    print(f"   Reward mean: {reward_arr.mean():.4f}")
    print(f"   Reward std: {reward_arr.std():.4f}")
    if reward_arr.std() > 0.1:
        print("   ✓ Rewards vary (informative)")
    else:
        print("   ⚠ Rewards mostly constant")
    
    # 5. Observation-Action coherence
    print("\n5. OBSERVATION-ACTION COHERENCE:")
    obs_soc = data_arrays['obs_soc']
    actual_soc = data_arrays['soc']
    
    # Check if obs SoC matches actual SoC
    if len(obs_soc) > 0 and len(actual_soc) > 0:
        soc_diff = np.mean(np.abs(obs_soc - actual_soc))
        print(f"   Obs SoC vs Actual SoC diff: {soc_diff:.6f}")
        if soc_diff < 0.01:
            print("   ✓ Observations match environment state")
        else:
            print("   ⚠ Observation mismatch with environment")
    
    obs_price = data_arrays['obs_price']
    actual_price = data_arrays['price']
    if len(obs_price) > 0 and len(actual_price) > 0:
        price_diff = np.mean(np.abs(obs_price - actual_price))
        print(f"   Obs Price vs Actual Price diff: {price_diff:.6f}")
        if price_diff < 0.01:
            print("   ✓ Price observations accurate")
        else:
            print("   ⚠ Price observation mismatch")
    
    # 6. Time progression
    print("\n6. TIME PROGRESSION:")
    obs_hour = data_arrays['obs_hour']
    print(f"   Hour range: [{obs_hour.min():.1f}, {obs_hour.max():.1f}]")
    hour_changes = np.sum(np.abs(np.diff(obs_hour)) > 0.5)
    print(f"   Hour changes: {hour_changes}")
    if hour_changes > len(obs_hour) * 0.8:
        print("   ✓ Time progresses normally")
    else:
        print("   ⚠ Time may not be progressing")
    
    # 7. Net load analysis
    print("\n7. NET LOAD:")
    net_load_arr = data_arrays['net_load']
    print(f"   Net load range: [{net_load_arr.min():.4f}, {net_load_arr.max():.4f}]")
    print(f"   Net load mean: {net_load_arr.mean():.4f}")
    print(f"   Net load varies: {len(np.unique(net_load_arr.round(4)))} unique values")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    checks = {
        'Battery responds': battery_works if 'battery_works' in locals() else False,
        'SoC utilization': (soc_arr.max() - soc_arr.min()) > 0.1,
        'Dynamic pricing': price_arr.std() > 0.01,
        'PV variation': len(np.unique(pv_arr.round(4))) > 5,
        'Reward varies': reward_arr.std() > 0.1,
        'Obs-state match': (soc_diff < 0.05 and price_diff < 0.05) if 'soc_diff' in locals() and 'price_diff' in locals() else True,
        'Time progresses': hour_changes > len(obs_hour) * 0.5 if 'hour_changes' in locals() else True
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")
    
    print(f"\n{'='*80}")
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("✓ Environment is FULLY FUNCTIONAL")
    elif passed >= total * 0.7:
        print(f"⚠ MOSTLY WORKING ({passed}/{total})")
        print("✓ Environment is usable with minor issues")
    else:
        print(f"✗ CRITICAL ISSUES ({passed}/{total})")
        print("✗ Environment needs fixes")
    print("="*80)


if __name__ == "__main__":
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'mpc'
    test_environment(agent_type)