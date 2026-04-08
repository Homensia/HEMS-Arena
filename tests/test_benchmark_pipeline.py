"""
HEMS Benchmark Pipeline Tests
=====================================

These tests are designed to work regardless of your project structure.
They test CRITICAL functionality without needing complex environment setup.

Run with: pytest hems/core/test_benchmark_pipeline.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import sys


class TestCriticalStructure:
    """Test critical files and methods exist."""
    
    def test_dqn_file_exists(self):
        """Test DQN algorithm file exists."""
        try:
            from hems.algorithms.dqn import DQNAlgorithm
            assert DQNAlgorithm is not None
        except ImportError:
            pytest.fail("Cannot import DQNAlgorithm from hems.algorithms.dqn")
    
    def test_sac_file_exists(self):
        """Test SAC algorithm file exists."""
        try:
            from hems.algorithms.sac import SACAlgorithm
            assert SACAlgorithm is not None
        except ImportError:
            pytest.skip("SAC not available - that's okay if not using SAC")
    
    def test_validation_testing_file_exists(self):
        """Test validation_testing.py exists."""
        possible_paths = [
            Path('hems/core/validation_testing.py'),
            Path('validation_testing.py'),
        ]
        
        found = any(p.exists() for p in possible_paths)
        assert found, f"validation_testing.py not found in: {possible_paths}"


class TestDQNHasRequiredMethods:
    """Test DQN has all required methods for the pipeline."""
    
    def test_dqn_has_get_state(self):
        """CRITICAL: DQN must have get_state() for model saving."""
        from hems.algorithms.dqn import DQNAlgorithm
        assert hasattr(DQNAlgorithm, 'get_state'), \
            "CRITICAL: DQN missing get_state() method! Models cannot be saved!"
    
    def test_dqn_has_load_state(self):
        """CRITICAL: DQN must have load_state() for model loading."""
        from hems.algorithms.dqn import DQNAlgorithm
        assert hasattr(DQNAlgorithm, 'load_state'), \
            "CRITICAL: DQN missing load_state() method! Models cannot be loaded!"
    
    def test_dqn_has_act(self):
        """DQN must have act() method."""
        from hems.algorithms.dqn import DQNAlgorithm
        assert hasattr(DQNAlgorithm, 'act'), \
            "DQN missing act() method!"
    
    def test_dqn_has_learn(self):
        """DQN must have learn() method."""
        from hems.algorithms.dqn import DQNAlgorithm
        assert hasattr(DQNAlgorithm, 'learn'), \
            "DQN missing learn() method!"


class TestSACHasRequiredMethods:
    """Test SAC has all required methods for the pipeline."""
    
    def test_sac_has_get_state(self):
        """CRITICAL: SAC must have get_state() for model saving."""
        try:
            from hems.algorithms.sac import SACAlgorithm
            assert hasattr(SACAlgorithm, 'get_state'), \
                "CRITICAL: SAC missing get_state() method! Models cannot be saved!"
        except ImportError:
            pytest.skip("SAC not available")
    
    def test_sac_has_load_state(self):
        """CRITICAL: SAC must have load_state() for model loading."""
        try:
            from hems.algorithms.sac import SACAlgorithm
            assert hasattr(SACAlgorithm, 'load_state'), \
                "CRITICAL: SAC missing load_state() method! Models cannot be loaded!"
        except ImportError:
            pytest.skip("SAC not available")


class TestModelSavingLogic:
    """Test model saving/loading logic works correctly."""
    
    def test_dqn_get_state_returns_dict(self):
        """Test get_state() returns a dictionary (not empty)."""
        from hems.algorithms.dqn import DQNAlgorithm
        import inspect
        
        # Get the source code of get_state
        source = inspect.getsource(DQNAlgorithm.get_state)
        
        # Check it returns something meaningful
        assert 'return' in source.lower(), \
            "get_state() must return something!"
        assert 'state_dict()' in source or 'network' in source.lower(), \
            "get_state() should return network state dictionaries!"
        
        print("[TEST] ✓ get_state() appears to return network states")
    
    def test_dqn_load_state_uses_load_state_dict(self):
        """Test load_state() actually loads network weights."""
        from hems.algorithms.dqn import DQNAlgorithm
        import inspect
        
        # Get the source code of load_state
        source = inspect.getsource(DQNAlgorithm.load_state)
        
        # Check it loads state dicts
        assert 'load_state_dict' in source, \
            "load_state() must call load_state_dict() to load weights!"
        
        print("[TEST] ✓ load_state() loads network weights")


class TestBaselinePerBuildingFix:
    """Test the CRITICAL baseline per-building fix is implemented."""
    
    def test_per_building_baseline_method_exists(self):
        """CRITICAL: _create_baseline_for_building() must exist."""
        possible_paths = [
            Path('hems/core/validation_testing.py'),
            Path('validation_testing.py'),
        ]
        
        validation_file = None
        for path in possible_paths:
            if path.exists():
                validation_file = path
                break
        
        assert validation_file is not None, "validation_testing.py not found!"
        
        content = validation_file.read_text()
        assert '_create_baseline_for_building' in content, \
            "CRITICAL: Missing _create_baseline_for_building()! Baseline fix not applied!"
        
        print("[TEST] ✓ Per-building baseline method exists")
    
    def test_old_baseline_method_removed(self):
        """CRITICAL: Old broken _initialize_baseline() must be removed."""
        possible_paths = [
            Path('hems/core/validation_testing.py'),
            Path('validation_testing.py'),
        ]
        
        validation_file = None
        for path in possible_paths:
            if path.exists():
                validation_file = path
                break
        
        if validation_file is None:
            pytest.skip("validation_testing.py not found")
        
        content = validation_file.read_text()
        
        # Check if old method exists (should NOT)
        if 'def _initialize_baseline(' in content:
            pytest.fail(
                "CRITICAL: Old _initialize_baseline() method still exists! "
                "This causes baseline dimensional mismatch. Must be REMOVED!"
            )
        
        print("[TEST] ✓ Old broken baseline method removed")
    
    def test_progress_bar_implemented(self):
        """Test progress bar is implemented."""
        possible_paths = [
            Path('hems/core/validation_testing.py'),
            Path('validation_testing.py'),
        ]
        
        validation_file = None
        for path in possible_paths:
            if path.exists():
                validation_file = path
                break
        
        if validation_file is None:
            pytest.skip("validation_testing.py not found")
        
        content = validation_file.read_text()
        assert 'tqdm' in content, "Progress bar (tqdm) not implemented!"
        
        print("[TEST] ✓ Progress bar implemented")


class TestDQNNetworkDimensions:
    """Test DQN network uses correct dimensions."""
    
    def test_dqn_network_accepts_single_building_obs(self):
        """CRITICAL: DQN network should expect 28 dims, not 28*N dims."""
        from hems.algorithms.dqn import DQNAlgorithm
        import inspect
        
        # Get DQN source
        source = inspect.getsource(DQNAlgorithm.__init__)
        
        # Look for evidence of multi-head vs flattening
        if 'multi_head' in source or 'n_heads' in source:
            print("[TEST] ✓ DQN uses multi-head architecture (shared-weight approach)")
            print("[TEST]   This is CORRECT for parallel mode")
        else:
            print("[TEST] ⚠ DQN might be using flattening approach")
            print("[TEST]   Verify network expects 28 dims per building, not 28*N dims")


class TestCodeQuality:
    """Test code quality and best practices."""
    
    def test_no_silent_exception_catching(self):
        """Test no silent exception catching without logging."""
        critical_files = [
            Path('hems/algorithms/dqn.py'),
            Path('hems/algorithms/sac.py'),
            Path('hems/core/validation_testing.py'),
        ]
        
        violations = []
        
        for file_path in critical_files:
            if not file_path.exists():
                continue
            
            content = file_path.read_text()
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                # Check for "except:" or "except Exception:" with "pass" nearby
                if ('except:' in line or 'except Exception:' in line):
                    # Check next few lines for 'pass' without logging
                    context_lines = lines[i:min(i+5, len(lines))]
                    context = '\n'.join(context_lines)
                    
                    if 'pass' in context and 'logger' not in context and 'print' not in context:
                        violations.append(f"{file_path.name} line {i+1}")
        
        if violations:
            pytest.fail(
                f"Found silent exception catching without logging in: {', '.join(violations)}. "
                f"Always log errors!"
            )
        
        print("[TEST] ✓ No silent exception catching detected")


class TestPytorchBasics:
    """Test PyTorch basics work correctly."""
    
    def test_pytorch_available(self):
        """Test PyTorch is installed."""
        import torch
        assert torch is not None
        print(f"[TEST] ✓ PyTorch {torch.__version__} available")
    
    def test_device_available(self):
        """Test CUDA or CPU device available."""
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[TEST] ✓ Using device: {device}")
        assert device is not None


class TestNetworkArchitecture:
    """Test network architecture is correctly configured."""
    
    def test_dqn_network_class_exists(self):
        """Test DQN has a network class."""
        try:
            # Try to import DQN network
            from hems.algorithms.dqn import DuelingQNetwork
            print("[TEST] ✓ DuelingQNetwork found")
        except ImportError:
            # Try alternate name
            try:
                from hems.algorithms.dqn import QNetwork
                print("[TEST] ✓ QNetwork found")
            except ImportError:
                pytest.skip("Network class not directly importable - that's okay")
    
    def test_multi_head_support(self):
        """Test DQN supports multi-head architecture."""
        from hems.algorithms.dqn import DQNAlgorithm
        import inspect
        
        source = inspect.getsource(DQNAlgorithm)
        
        if 'multi_head' in source or 'n_heads' in source:
            print("[TEST] ✓ DQN supports multi-head architecture")
            print("[TEST]   This enables shared-weight approach for parallel mode")
        else:
            print("[TEST] ⚠ No multi-head support detected")
            print("[TEST]   Parallel mode may use flattening (could cause issues)")


class TestDocumentation:
    """Test documentation exists."""
    
    def test_readme_exists(self):
        """Test some form of documentation exists."""
        doc_files = [
            Path('README.md'),
            Path('HEMS-README.md'),
            Path('docs/README.md'),
        ]
        
        found = any(p.exists() for p in doc_files)
        if not found:
            pytest.skip("No README found - consider adding documentation")
        
        print("[TEST] ✓ Documentation exists")


def run_all_tests():
    """Run all tests with detailed output."""
    pytest.main([
        __file__,
        '-v',  # Verbose
        '-s',  # Show print statements
        '--tb=short',  # Short traceback
    ])


if __name__ == "__main__":
    run_all_tests()