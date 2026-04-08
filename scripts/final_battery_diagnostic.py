"""
FINAL DIAGNOSTIC: CityLearn Battery Investigation
This script will help identify why the battery SoC is stuck at 0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper
from hems.core.config import SimulationConfig


def print_section(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

print_section("CITYLEARN BATTERY ROOT CAUSE ANALYSIS")

# Create environment
cfg = SimulationConfig(building_count=1, simulation_days=1, random_seed=42, use_gpu=False)
wrapper = CityLearnWrapper(cfg)
buildings = wrapper.select_buildings()
start, end = wrapper.select_simulation_period()
env = wrapper.create_environment(buildings, start, end)

b = env.buildings[0]
storage = b.electrical_storage

# ==============================================
# DIAGNOSTIC 1: Check all power-related attributes
# ==============================================
print_section("1. POWER ATTRIBUTES")
power_attrs = ['nominal_power', 'power_rating', 'power', 'max_power', 
               'capacity', 'capacity_history']
for attr in power_attrs:
    val = getattr(storage, attr, 'NOT_FOUND')
    if hasattr(val, '__len__') and len(val) > 3:
        print(f"{attr}: {type(val).__name__}(length={len(val)})")
    else:
        print(f"{attr}: {val}")

# ==============================================
# DIAGNOSTIC 2: Check SoC initialization
# ==============================================
print_section("2. SOC INITIALIZATION")
soc_attrs = ['soc_init', 'initial_soc', '_soc_init', 'soc', '_soc']
for attr in soc_attrs:
    val = getattr(storage, attr, 'NOT_FOUND')
    if hasattr(val, '__len__') and len(val) > 0:
        print(f"{attr}: {type(val).__name__}(length={len(val)}, values={val[:5]})")
    else:
        print(f"{attr}: {val}")

# ==============================================
# DIAGNOSTIC 3: Test action processing
# ==============================================
print_section("3. ACTION PROCESSING TEST")

print("Calling env.reset()...")
env.reset()
soc_0 = storage.soc[-1] if hasattr(storage, 'soc') and len(storage.soc) > 0 else 0.0
print(f"SoC after reset: {soc_0:.6f}")

actions_to_test = [
    (0.78125, "MAX CHARGE"),
    (-0.78125, "MAX DISCHARGE"),
    (0.5, "MEDIUM CHARGE"),
    (-0.5, "MEDIUM DISCHARGE"),
    (0.0, "NO ACTION")
]

soc_history = [soc_0]
for action_val, description in actions_to_test:
    obs, reward, done, trunc, info = env.step([[action_val]])
    soc_new = storage.soc[-1] if hasattr(storage, 'soc') and len(storage.soc) > 0 else 0.0
    delta = soc_new - soc_history[-1]
    soc_history.append(soc_new)
    
    status = "✓" if abs(delta) > 1e-6 else "✗"
    print(f"{status} {description:20s} (action={action_val:+.3f}): "
          f"SoC={soc_new:.6f}, Δ={delta:+.6f}, reward={reward}")

# ==============================================
# DIAGNOSTIC 4: Check if SoC can change at all
# ==============================================
print_section("4. SOC MUTABILITY TEST")

# Try to directly modify SoC
original_soc = storage.soc[-1]
print(f"Original SoC: {original_soc:.6f}")

try:
    storage.soc[-1] = 0.75
    print(f"After setting to 0.75: {storage.soc[-1]:.6f}")
    modified = abs(storage.soc[-1] - 0.75) < 0.001
    print(f"SoC is modifiable: {'YES ✓' if modified else 'NO ✗'}")
    
    # Take an action and see if it overwrites our change
    env.step([[0.0]])
    print(f"After step with 0 action: {storage.soc[-1]:.6f}")
    
    if abs(storage.soc[-1] - 0.75) > 0.01:
        print("⚠ WARNING: SoC was overwritten after step() - likely reading from CSV!")
    
except Exception as e:
    print(f"ERROR: Cannot modify SoC - {e}")

# ==============================================
# DIAGNOSTIC 5: Check for data replay mode
# ==============================================
print_section("5. DATA REPLAY MODE CHECK")

# Check if storage has a data source
if hasattr(storage, 'data') or hasattr(storage, '_data'):
    print("⚠ Storage has 'data' attribute - might be in replay mode")
    data_attr = storage.data if hasattr(storage, 'data') else storage._data
    print(f"Data type: {type(data_attr)}")
    if hasattr(data_attr, 'shape'):
        print(f"Data shape: {data_attr.shape}")

# Check if building has inactive_actions flag
if hasattr(b, 'inactive_actions'):
    print(f"Building inactive_actions: {b.inactive_actions}")
    if b.inactive_actions and 'electrical_storage' in b.inactive_actions:
        print("⚠ CRITICAL: electrical_storage action is INACTIVE!")

# Check environment-level flags
if hasattr(env, 'central_agent'):
    print(f"Central agent mode: {env.central_agent}")

# ==============================================
# DIAGNOSTIC 6: Extended simulation
# ==============================================
print_section("6. EXTENDED SIMULATION (48 STEPS)")

env.reset()
soc_extended = []
actions_extended = []

for i in range(48):
    action_val = 0.6 if i % 2 == 0 else -0.6
    actions_extended.append(action_val)
    env.step([[action_val]])
    soc_extended.append(storage.soc[-1])

soc_array = np.array(soc_extended)
action_array = np.array(actions_extended)

print(f"SoC statistics:")
print(f"  Min: {soc_array.min():.6f}")
print(f"  Max: {soc_array.max():.6f}")
print(f"  Mean: {soc_array.mean():.6f}")
print(f"  Std: {soc_array.std():.6f}")
print(f"  Range: {soc_array.max() - soc_array.min():.6f}")
print(f"  Unique values: {len(np.unique(soc_array.round(6)))}")

# Check correlation between actions and SoC changes
soc_changes = np.diff(soc_array)
if len(soc_changes) > 0:
    correlation = np.corrcoef(actions_extended[:-1], soc_changes)[0,1]
    print(f"  Action-SoC correlation: {correlation:.4f}")
    
    if abs(correlation) < 0.1:
        print("  ✗ PROBLEM: Actions not correlated with SoC changes!")
    else:
        print("  ✓ Actions affect SoC")

# ==============================================
# FINAL DIAGNOSIS
# ==============================================
print_section("FINAL DIAGNOSIS")

soc_changes_total = abs(soc_array.max() - soc_array.min())

if soc_changes_total < 0.01:
    print("❌ CRITICAL PROBLEM: Battery SoC is NOT responding to actions")
    print("\nPossible causes:")
    print("  1. Battery actions are inactive/disabled in schema")
    print("  2. CityLearn is in data replay mode (reading SoC from CSV)")
    print("  3. Power rating is 0 or incorrectly configured")
    print("  4. Bug in CityLearn version 2.3.1")
    print("\nRECOMMENDED ACTIONS:")
    print("  → Check schema for 'inactive_actions' field")
    print("  → Verify CityLearn version: pip show citylearn")
    print("  → Try with a different dataset")
    print("  → Check CityLearn GitHub issues for known bugs")
elif soc_changes_total < 0.1:
    print("⚠ WARNING: Battery SoC changes are very small")
    print(f"  Range: {soc_changes_total:.6f}")
    print("\nPossible causes:")
    print("  1. Power rating too low relative to capacity")
    print("  2. Timestep duration incorrect")
    print("  3. Action scaling issue")
else:
    print("✅ SUCCESS: Battery is responding to actions!")
    print(f"  SoC range: {soc_changes_total:.4f}")
    print("  The battery is now working correctly.")

print("\n" + "="*80)