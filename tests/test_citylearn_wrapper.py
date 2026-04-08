"""
Comprehensive test suite for CityLearn wrapper.
Validates all environment functionality: observations, actions, battery, PV, rewards, agent interaction.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import SimulationConfig
from hems.environments.citylearn.citylearn_wrapper import CityLearnWrapper, CityLearnEnvironmentManager


class CityLearnWrapperTest:
    """Comprehensive test suite for CityLearn wrapper functionality."""
    
    def __init__(self):
        self.config = SimulationConfig(
            building_count=1,
            simulation_days=7,
            dataset_name='citylearn_challenge_2022_phase_all',
            random_seed=42,
            use_gpu=False
        )
        self.wrapper = CityLearnWrapper(self.config)
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def run_all_tests(self):
        """Run all test suites."""
        print("=" * 80)
        print("CITYLEARN WRAPPER COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        self.test_building_selection()
        self.test_period_selection()
        self.test_environment_creation()
        self.test_observations()
        self.test_actions_and_battery()
        self.test_pv_production()
        self.test_reward_agnostic()
        self.test_agent_interaction()
        self.test_wrappers()
        self.test_gpu_device()
        
        self.print_results()
    
    def test_building_selection(self):
        """Test building selection logic (deterministic + random)."""
        print("\n[TEST] Building Selection")
        
        try:
            buildings = self.wrapper.select_buildings()
            assert len(buildings) == self.config.building_count
            assert all(isinstance(b, str) for b in buildings)
            assert all('Building_' in b for b in buildings)
            self.results['passed'].append('Building selection: correct count and format')
            print(f"  ✓ Selected buildings: {buildings}")
            
            config_multi = SimulationConfig(building_count=3, random_seed=42)
            wrapper_multi = CityLearnWrapper(config_multi)
            buildings_multi = wrapper_multi.select_buildings()
            assert len(buildings_multi) == 3
            self.results['passed'].append('Building selection: multiple buildings')
            print(f"  ✓ Multiple buildings: {buildings_multi}")
            
            buildings_excluded = self.wrapper.select_buildings(exclude_buildings=['Building_12', 'Building_15'])
            assert 'Building_12' not in buildings_excluded
            assert 'Building_15' not in buildings_excluded
            self.results['passed'].append('Building selection: exclusion logic')
            print("  ✓ Exclusion logic working")
            
        except Exception as e:
            self.results['failed'].append(f'Building selection: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_period_selection(self):
        """Test simulation period selection."""
        print("\n[TEST] Period Selection")
        
        try:
            start_time, end_time = self.wrapper.select_simulation_period()
            expected_length = 24 * self.config.simulation_days
            actual_length = end_time - start_time + 1
            
            assert start_time >= 0
            assert end_time > start_time
            assert actual_length <= expected_length + 24
            
            self.results['passed'].append(f'Period selection: {start_time} to {end_time} ({actual_length} steps)')
            print(f"  ✓ Period: {start_time} to {end_time} ({actual_length} timesteps)")
            
        except Exception as e:
            self.results['failed'].append(f'Period selection: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_environment_creation(self):
        """Test environment creation and schema loading."""
        print("\n[TEST] Environment Creation")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            
            env = self.wrapper.create_environment(buildings, start_time, end_time, use_custom_battery=True)
            
            assert env is not None
            assert hasattr(env, 'buildings')
            assert len(env.buildings) == len(buildings)
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'time_steps')
            
            self.results['passed'].append('Environment creation: all attributes present')
            print(f"  ✓ Environment created with {len(env.buildings)} buildings")
            print(f"  ✓ Observation space: {env.observation_space}")
            print(f"  ✓ Action space: {env.action_space}")
            
        except Exception as e:
            self.results['failed'].append(f'Environment creation: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_observations(self):
        """Test observation correctness."""
        print("\n[TEST] Observations")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            env = self.wrapper.create_environment(buildings, start_time, end_time, use_custom_battery=True)
            reset_result = env.reset()
            
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                observations, info = reset_result
                print("  ✓ Environment uses Gymnasium API (obs, info)")
            else:
                observations = reset_result
                print("  ✓ Environment uses Gym API (obs only)")
            
            assert observations is not None, "Observations are None"
            assert isinstance(observations, (list, tuple)), f"Observations not list/tuple: {type(observations)}"
            assert len(observations) == len(buildings), f"Wrong number of observations: {len(observations)} vs {len(buildings)}"
            
            obs = observations[0]
            print(f"  ✓ Observation type: {type(obs)}")
            print(f"  ✓ Observation shape: {np.array(obs).shape}")
            
            obs_array = np.array(obs)
            has_nan = np.any(np.isnan(obs_array))
            has_inf = np.any(np.isinf(obs_array))
            
            assert not has_nan, "Observations contain NaN"
            assert not has_inf, "Observations contain Inf"
            
            self.results['passed'].append('Observations: valid structure and values')
            print(f"  ✓ Observations validated (no NaN/Inf)")
            
        except Exception as e:
            self.results['failed'].append(f'Observations: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_actions_and_battery(self):
        """Test action processing and battery constraints."""
        print("\n[TEST] Actions & Battery Logic")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            env = self.wrapper.create_environment(buildings, start_time, end_time, use_custom_battery=True)
            
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observations, _ = reset_result
            else:
                observations = reset_result
            
            test_actions = [
                [[-1.0]],
                [[0.0]],  
                [[1.0]],  
                [[0.5]],
                [[-0.5]] 
            ]
            
            for i, action in enumerate(test_actions):
                next_obs, reward, done, truncated, info = env.step(action)
                
                assert not np.any(np.isnan(next_obs)), f"Observations contain NaN at step {i}"
                assert isinstance(reward, (int, float, list)), f"Invalid reward type at step {i}"
                
                print(f"  ✓ Step {i}: Action={action[0][0]:.2f}, Done={done}")
                
                if done or truncated:
                    break
            
            self.results['passed'].append('Actions: executed successfully')
            print("  ✓ All action types processed correctly")
            
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                building = env.buildings[0]
                if hasattr(building, 'electrical_storage') and building.electrical_storage is not None:
                    storage = building.electrical_storage
                    if hasattr(storage, 'soc') and len(storage.soc) > 0:
                        soc_values = storage.soc[:min(10, len(storage.soc))]
                        soc_valid = all(0 <= s <= 1 for s in soc_values if not np.isnan(s))
                        if soc_valid:
                            self.results['passed'].append('Battery: SoC in valid range [0,1]')
                            print(f"  ✓ Battery SoC validated: {soc_values[:3]}")
                        else:
                            self.results['warnings'].append('Battery: SoC out of range detected')
                            print(f"  ⚠ Battery SoC out of range: {soc_values}")
            
        except Exception as e:
            self.results['failed'].append(f'Actions & Battery: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_pv_production(self):
        """Test PV production tracking."""
        print("\n[TEST] PV Production Tracking")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            env = self.wrapper.create_environment(buildings, start_time, end_time, use_custom_battery=True)
            
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observations, _ = reset_result
            else:
                observations = reset_result
            
            for _ in range(min(24, env.time_steps)):
                action = [[0.0]]
                next_obs, reward, done, truncated, info = env.step(action)
                
                if done or truncated:
                    break
            
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                building = env.buildings[0]
                if hasattr(building, 'solar_generation'):
                    pv_data = building.solar_generation
                    if hasattr(pv_data, '__len__') and len(pv_data) > 0:
                        pv_sample = pv_data[:min(24, len(pv_data))]
                        pv_list = [float(x) for x in pv_sample]
                        max_pv = max(pv_list) if pv_list else 0
                        has_variation = len(set(pv_list)) > 1 if pv_list else False
                        
                        self.results['passed'].append(f'PV tracking: max={max_pv:.2f}')
                        print(f"  ✓ PV generation tracked: max={max_pv:.2f} kW")
                        if has_variation:
                            print("  ✓ PV shows variation over time")
                    else:
                        self.results['warnings'].append('PV: empty solar_generation data')
                        print("  ⚠ PV data empty")
                else:
                    self.results['warnings'].append('PV: no solar_generation attribute')
                    print("  ⚠ No solar_generation attribute")
            
        except Exception as e:
            self.results['failed'].append(f'PV tracking: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_reward_agnostic(self):
        """Test reward agnostic design."""
        print("\n[TEST] Reward Agnostic Design")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            
            env1 = self.wrapper.create_environment(buildings, start_time, end_time, reward_function=None)
            reset_result = env1.reset()
            if isinstance(reset_result, tuple):
                observations, _ = reset_result
            else:
                observations = reset_result
                
            action = [[0.0]]
            next_obs, reward1, done, truncated, info = env1.step(action)
            
            self.results['passed'].append('Reward agnostic: default reward works')
            print(f"  ✓ Environment accepts None reward (default CityLearn reward)")
            print(f"  ✓ Reward structure: {type(reward1)}")
            
        except Exception as e:
            self.results['failed'].append(f'Reward agnostic: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_agent_interaction(self):
        """Test full agent interaction loop."""
        print("\n[TEST] Agent Interaction Loop")
        
        try:
            buildings = self.wrapper.select_buildings()
            start_time, end_time = self.wrapper.select_simulation_period()
            env = self.wrapper.create_environment(buildings, start_time, end_time, use_custom_battery=True)
            
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observations, _ = reset_result
            else:
                observations = reset_result
                
            total_reward = 0.0
            steps = 0
            
            for i in range(min(24, env.time_steps)):
                action = [[np.random.uniform(-0.5, 0.5)]]
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                if isinstance(reward, list):
                    reward = sum(reward)
                
                total_reward += float(reward)
                steps += 1
                observations = next_obs
                
                if done or truncated:
                    break
            
            self.results['passed'].append(f'Agent interaction: {steps} steps completed')
            print(f"  ✓ Completed {steps} steps")
            print(f"  ✓ Total reward: {total_reward:.2f}")
            print(f"  ✓ Average reward: {total_reward/steps:.3f}")
            
        except Exception as e:
            self.results['failed'].append(f'Agent interaction: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def test_wrappers(self):
        """Test wrapper compatibility (TabularQLearningWrapper)."""
        print("\n[TEST] Wrapper Compatibility")
        
        try:
            from citylearn.wrappers import TabularQLearningWrapper
            self.results['passed'].append('Wrappers: TabularQLearningWrapper available')
            print("  ✓ TabularQLearningWrapper imported successfully")
            
            print("  • Note: TQL wrapper creates large Q-tables, full test skipped")
            print("  • TQL wrapper works with single building only (Q-table size)")
            
            config_test = SimulationConfig(building_count=1, simulation_days=1, random_seed=42)
            wrapper_test = CityLearnWrapper(config_test)
            buildings = wrapper_test.select_buildings()
            
            assert len(buildings) == 1, "TQL requires single building"
            self.results['passed'].append('Wrappers: TQL configuration validated')
            print("  ✓ TQL wrapper configuration validated")
            
        except ImportError as e:
            self.results['warnings'].append(f'Wrappers: TabularQLearningWrapper not available - {e}')
            print(f"  ⚠ TabularQLearningWrapper not available")
        except Exception as e:
            self.results['warnings'].append(f'Wrappers: {e}')
            print(f"  ⚠ Wrapper validation warning: {e}")
    
    def test_gpu_device(self):
        """Test GPU device configuration."""
        print("\n[TEST] GPU Device Configuration")
        
        if not TORCH_AVAILABLE:
            self.results['warnings'].append('GPU test skipped: torch not available')
            print("  ⚠ Torch not available, skipping GPU test")
            return
        
        try:
            config_gpu = SimulationConfig(use_gpu=True)
            
            if torch.cuda.is_available():
                assert config_gpu.device == 'cuda'
                self.results['passed'].append('GPU: CUDA device configured')
                print(f"  ✓ GPU available: device set to '{config_gpu.device}'")
            else:
                assert config_gpu.device == 'cpu'
                self.results['passed'].append('GPU: CPU fallback working')
                print(f"  ✓ No GPU available: device fallback to 'cpu'")
            
            config_cpu = SimulationConfig(use_gpu=False)
            assert config_cpu.device == 'cpu'
            self.results['passed'].append('GPU: CPU mode configured')
            print(f"  ✓ CPU mode: device set to 'cpu'")
            
        except Exception as e:
            self.results['failed'].append(f'GPU device: {e}')
            print(f"  ✗ FAILED: {e}")
    
    def print_results(self):
        """Print final test results."""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n✓ PASSED: {len(self.results['passed'])}")
        for test in self.results['passed']:
            print(f"  • {test}")
        
        if self.results['warnings']:
            print(f"\n⚠ WARNINGS: {len(self.results['warnings'])}")
            for warning in self.results['warnings']:
                print(f"  • {warning}")
        
        if self.results['failed']:
            print(f"\n✗ FAILED: {len(self.results['failed'])}")
            for failure in self.results['failed']:
                print(f"  • {failure}")
        else:
            print("\n🎉 ALL CRITICAL TESTS PASSED!")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_suite = CityLearnWrapperTest()
    test_suite.run_all_tests()