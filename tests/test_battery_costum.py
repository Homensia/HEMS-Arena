"""
Test CityLearn with CUSTOM BATTERY simulator.
This bypasses CityLearn's frozen battery data and implements real battery dynamics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hems.core.config import SimulationConfig
from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper

print("="*80)
print("CUSTOM BATTERY TEST")
print("="*80)

# Create environment with custom battery enabled
cfg = SimulationConfig(building_count=1, simulation_days=1, random_seed=42, use_gpu=False)
wrapper = CityLearnWrapper(cfg)
buildings = wrapper.select_buildings()
start, end = wrapper.select_simulation_period()

# KEY: use_custom_battery=True enables our custom battery simulator
env = wrapper.create_environment(buildings, start, end, use_custom_battery=True)

print("✓ Environment created with CUSTOM BATTERY simulator")
print(f"  Buildings: {buildings}")
print(f"  Period: {start} to {end} ({end - start + 1} steps)")

# Reset environment
obs = env.reset()
print(f"\n✓ Environment reset")
print(f"  Initial SoC: {env.get_battery_soc():.6f}")

# Test actions
print("\n--- TESTING BATTERY DYNAMICS ---")

test_actions = [
    (0.6, "CHARGE (60%)"),
    (0.6, "CHARGE (60%)"),
    (0.6, "CHARGE (60%)"),
    (-0.4, "DISCHARGE (40%)"),
    (-0.4, "DISCHARGE (40%)"),
    (0.0, "NO ACTION"),
    (0.8, "MAX CHARGE"),
    (-0.8, "MAX DISCHARGE"),
]

soc_history = [env.get_battery_soc()]

for i, (action_val, desc) in enumerate(test_actions):
    soc_before = env.get_battery_soc()
    
    obs, reward, done, trunc, info = env.step([[action_val]])
    
    soc_after = env.get_battery_soc()
    delta = soc_after - soc_before
    
    status = "✓" if abs(delta) > 1e-6 or action_val == 0 else "✗"
    print(f"{status} Step {i+1}: {desc:20s} | SoC: {soc_before:.4f} → {soc_after:.4f} (Δ={delta:+.6f})")
    
    soc_history.append(soc_after)
    
    if done or trunc:
        break

# Summary
soc_array = np.array(soc_history)
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Steps completed: {len(soc_history) - 1}")
print(f"SoC range: [{soc_array.min():.4f}, {soc_array.max():.4f}]")
print(f"SoC mean: {soc_array.mean():.4f}")
print(f"SoC utilization: {(soc_array.max() - soc_array.min()):.4f}")
print(f"Unique SoC values: {len(np.unique(soc_array.round(6)))}")

if soc_array.max() - soc_array.min() > 0.05:
    print("\n✅ SUCCESS: Custom battery is working!")
    print("   Battery responds to charge/discharge actions")
    print("   SoC changes dynamically based on actions")
else:
    print("\n⚠ WARNING: Battery utilization is low")
    print(f"   Range: {soc_array.max() - soc_array.min():.6f}")

print("\n" + "="*80)
print("FULL SOC HISTORY")
print("="*80)
print(f"SoC values: {[f'{x:.4f}' for x in soc_history]}")

# Compare to CityLearn's internal SoC
b = env.env.env.buildings[0]  # Unwrap to get original environment
if hasattr(b, 'electrical_storage') and b.electrical_storage is not None:
    citylearn_soc = b.electrical_storage.soc
    print(f"\nCityLearn's SoC array (first 10): {citylearn_soc[:10]}")
    print("Note: CityLearn's array is updated by our custom simulator")

print("\n="*80)