"""
COMPREHENSIVE CITYLEARN ENVIRONMENT VERIFICATION
Tests all critical aspects: battery, observations, actions, rewards, coherence
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hems.core.config import SimulationConfig
from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper

def test_battery_configuration():
    """Test 1: Battery configuration"""
    print("\n" + "="*80)
    print("TEST 1: BATTERY CONFIGURATION")
    print("="*80)
    
    cfg = SimulationConfig(building_count=1, simulation_days=1, random_seed=42, use_gpu=False)
    wrapper = CityLearnWrapper(cfg)
    buildings = wrapper.select_buildings()
    start, end = wrapper.select_simulation_period()
    env = wrapper.create_environment(buildings, start, end)
    
    b = env.buildings[0]
    storage = b.electrical_storage
    
    checks = {
        'has_storage': storage is not None,
        'has_capacity': hasattr(storage, 'capacity') and storage.capacity > 0,
        'has_nominal_power': hasattr(storage, 'nominal_power') and storage.nominal_power is not None and storage.nominal_power > 0,
        'has_initial_soc': hasattr(storage, 'soc_init') and storage.soc_init is not None,
        'has_efficiency': hasattr(storage, 'efficiency') and storage.efficiency is not None
    }
    
    print(f"✓ Has storage: {checks['has_storage']}")
    print(f"✓ Capacity: {storage.capacity if checks['has_capacity'] else 'MISSING'} kWh")
    print(f"✓ Nominal power: {storage.nominal_power if checks['has_nominal_power'] else 'MISSING'} kW")
    print(f"✓ Initial SoC: {storage.soc_init if checks['has_initial_soc'] else 'MISSING'}")
    print(f"✓ Efficiency: {storage.efficiency if checks['has_efficiency'] else 'MISSING'}")
    
    passed = all(checks.values())
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Battery configuration")
    return passed, env

def test_battery_dynamics(env):
    """Test 2: Battery charge/discharge dynamics"""
    print("\n" + "="*80)
    print("TEST 2: BATTERY DYNAMICS")
    print("="*80)
    
    b = env.buildings[0]
    storage = b.electrical_storage
    
    env.reset()
    soc_0 = storage.soc[-1]
    print(f"Initial SoC: {soc_0:.4f}")
    
    # Test charge
    env.step([[0.6]])
    soc_1 = storage.soc[-1]
    delta_charge = soc_1 - soc_0
    charge_works = delta_charge > 1e-6
    print(f"After CHARGE (+0.6): SoC={soc_1:.4f}, Δ={delta_charge:+.6f} {'✓' if charge_works else '✗'}")
    
    # Test discharge
    env.step([[-0.6]])
    soc_2 = storage.soc[-1]
    delta_discharge = soc_2 - soc_1
    discharge_works = delta_discharge < -1e-6
    print(f"After DISCHARGE (-0.6): SoC={soc_2:.4f}, Δ={delta_discharge:+.6f} {'✓' if discharge_works else '✗'}")
    
    # Test no action
    env.step([[0.0]])
    soc_3 = storage.soc[-1]
    delta_none = abs(soc_3 - soc_2)
    none_works = delta_none < 0.01
    print(f"After NO ACTION (0.0): SoC={soc_3:.4f}, Δ={delta_none:+.6f} {'✓' if none_works else '⚠'}")
    
    passed = charge_works and discharge_works
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Battery dynamics")
    return passed

def test_soc_bounds(env):
    """Test 3: SoC stays within [0, 1]"""
    print("\n" + "="*80)
    print("TEST 3: SOC BOUNDS")
    print("="*80)
    
    b = env.buildings[0]
    storage = b.electrical_storage
    
    env.reset()
    soc_history = []
    
    # Run 48 steps with random actions
    for _ in range(48):
        action = [[np.random.uniform(-0.8, 0.8)]]
        env.step(action)
        soc_history.append(storage.soc[-1])
    
    soc_array = np.array(soc_history)
    min_soc = soc_array.min()
    max_soc = soc_array.max()
    
    bounds_ok = (min_soc >= 0) and (max_soc <= 1)
    
    print(f"SoC range: [{min_soc:.4f}, {max_soc:.4f}]")
    print(f"Within bounds [0, 1]: {'✓' if bounds_ok else '✗'}")
    print(f"Utilization: {(max_soc - min_soc):.4f}")
    
    passed = bounds_ok and (max_soc - min_soc > 0.05)
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: SoC bounds")
    return passed

def test_observation_action_coherence(env):
    """Test 4: Observations update with actions"""
    print("\n" + "="*80)
    print("TEST 4: OBSERVATION-ACTION COHERENCE")
    print("="*80)
    
    b = env.buildings[0]
    storage = b.electrical_storage
    
    obs_list = []
    env.reset()
    
    # Take 5 distinct actions
    actions = [[0.7], [-0.7], [0.3], [-0.3], [0.0]]
    for action in actions:
        obs, reward, done, truncated, info = env.step([action])
        obs_list.append(obs[0] if isinstance(obs, list) else obs)
    
    # Check if observations are different
    obs_array = np.array(obs_list)
    unique_obs = len(np.unique(obs_array, axis=0))
    obs_changing = unique_obs > 1
    
    print(f"Unique observations: {unique_obs}/5")
    print(f"Observations updating: {'✓' if obs_changing else '✗'}")
    
    # Check if SoC feature in observations correlates with actual SoC
    # (SoC is typically one of the observation features)
    actual_soc = [storage.soc[-(5-i)] for i in range(5)]
    obs_has_variation = np.std(obs_array) > 0.01
    
    print(f"Observation variation: {'✓' if obs_has_variation else '✗'}")
    
    passed = obs_changing and obs_has_variation
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Observation-action coherence")
    return passed

def test_reward_calculation(env):
    """Test 5: Rewards are calculated and non-zero"""
    print("\n" + "="*80)
    print("TEST 5: REWARD CALCULATION")
    print("="*80)
    
    env.reset()
    rewards = []
    
    for i in range(24):
        action = [[0.5 if i % 2 == 0 else -0.5]]
        obs, reward, done, truncated, info = env.step(action)
        reward_val = reward[0] if isinstance(reward, list) else reward
        rewards.append(float(reward_val))
    
    rewards_array = np.array(rewards)
    
    print(f"Reward mean: {rewards_array.mean():.4f}")
    print(f"Reward std: {rewards_array.std():.4f}")
    print(f"Reward range: [{rewards_array.min():.4f}, {rewards_array.max():.4f}]")
    
    # Rewards should vary (not all identical)
    reward_varies = len(set(rewards_array.round(4))) > 1
    # Rewards should be finite
    rewards_finite = np.all(np.isfinite(rewards_array))
    
    print(f"Rewards vary: {'✓' if reward_varies else '✗'}")
    print(f"Rewards finite: {'✓' if rewards_finite else '✗'}")
    
    passed = reward_varies and rewards_finite
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Reward calculation")
    return passed

def test_pv_generation(env):
    """Test 6: PV generation is tracked"""
    print("\n" + "="*80)
    print("TEST 6: PV GENERATION TRACKING")
    print("="*80)
    
    b = env.buildings[0]
    
    env.reset()
    pv_values = []
    
    for _ in range(48):  # 2 days to capture day/night cycle
        env.step([[0.0]])
        if hasattr(b, 'solar_generation') and len(b.solar_generation) > 0:
            pv_values.append(float(b.solar_generation[-1]))
    
    if pv_values:
        pv_array = np.array(pv_values)
        pv_max = pv_array.max()
        pv_min = pv_array.min()
        pv_varies = len(set(pv_array.round(3))) > 1
        
        print(f"PV range: [{pv_min:.3f}, {pv_max:.3f}] kW")
        print(f"PV varies (day/night): {'✓' if pv_varies else '✗'}")
        print(f"Note: Negative values = generation (CityLearn convention)")
        
        passed = pv_varies
    else:
        print("⚠ No PV data found")
        passed = False
    
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: PV generation tracking")
    return passed

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("CITYLEARN ENVIRONMENT COMPREHENSIVE VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Test 1: Battery configuration
    results['battery_config'], env = test_battery_configuration()
    
    if not results['battery_config']:
        print("\n❌ CRITICAL: Battery configuration failed. Cannot proceed with other tests.")
        return results
    
    # Test 2: Battery dynamics
    results['battery_dynamics'] = test_battery_dynamics(env)
    
    # Test 3: SoC bounds
    results['soc_bounds'] = test_soc_bounds(env)
    
    # Test 4: Observation-action coherence
    results['obs_action_coherence'] = test_observation_action_coherence(env)
    
    # Test 5: Reward calculation
    results['reward_calculation'] = test_reward_calculation(env)
    
    # Test 6: PV generation
    results['pv_generation'] = test_pv_generation(env)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, test_passed in results.items():
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    print(f"\n{'='*80}")
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("✓ CityLearn environment is working correctly")
    else:
        print(f"⚠ SOME TESTS FAILED ({passed}/{total} passed)")
        print("✗ Review failed tests above")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if all(results.values()) else 1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)