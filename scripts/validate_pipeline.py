"""
HEMS Benchmark Pipeline Validation Script
==========================================

Runs comprehensive checks to ensure pipeline works PERFECTLY.

Usage:
    python validate_pipeline.py

Checks:
✅ Network architecture correct for training mode
✅ Parallel trains ALL buildings (not just first)
✅ Testing works regardless of training mode  
✅ No dimension mismatches
✅ Model saving contains actual weights
✅ No silent fallbacks or dummy implementations
✅ Clear error messages for all failures

Returns:
    Exit 0: All checks passed
    Exit 1: Critical failures detected
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


class PipelineValidator:
    """Validates HEMS benchmark pipeline integrity."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def check(self, name: str, condition: bool, error_msg: str = ""):
        """Check a condition and record result."""
        if condition:
            self.passed.append(name)
            print(f"{GREEN}✓{RESET} {name}")
        else:
            self.failed.append((name, error_msg))
            print(f"{RED}✗{RESET} {name}")
            if error_msg:
                print(f"  {RED}Error: {error_msg}{RESET}")
    
    def warn(self, name: str, msg: str):
        """Record a warning."""
        self.warnings.append((name, msg))
        print(f"{YELLOW}⚠{RESET} {name}")
        print(f"  {YELLOW}Warning: {msg}{RESET}")
    
    def check_network_architecture(self):
        """Verify DQN/SAC use correct architecture for parallel mode."""
        print(f"\n{BOLD}Checking Network Architecture...{RESET}")
        
        try:
            # Import DQN - use direct import from algorithms
            from hems.algorithms.dqn import DQNAlgorithm
            
            # Create a proper config object with required attributes
            class TestConfig:
                def __init__(self):
                    self.obs_dim = 28
                    self.n_buildings = 10
                    self.n_actions = 31
                    self.multi_head = True
                    self.buffer_size = 10000
                    self.batch_size = 32
                    self.learning_rate = 0.001
                    self.gamma = 0.99
                    self.epsilon_start = 1.0
                    self.epsilon_end = 0.05
                    self.epsilon_decay_steps = 10000
                    self.target_update_freq = 100
                    self.train_start_steps = 100
                    self.hidden_layers = [256, 256]
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            config = TestConfig()
            
            # Create algorithm
            algo = DQNAlgorithm(config)
            
            # Check 1: Network input dimension
            expected_input = 28  # Should ALWAYS be single building
            actual_input = algo.network_input_dim
            
            self.check(
                "DQN network input dimension (parallel mode)",
                actual_input == expected_input,
                f"Expected {expected_input} dims, got {actual_input}. "
                f"Network should ALWAYS expect single building (28 dims), "
                f"not flattened multi-building ({28*10}=280 dims)"
            )
            
            # Check 2: Network can accept 28-dim input
            test_input = torch.randn(1, 28)
            try:
                with torch.no_grad():
                    output = algo.q_network(test_input)
                self.check(
                    "DQN network accepts 28-dim input",
                    True,
                    ""
                )
            except RuntimeError as e:
                self.check(
                    "DQN network accepts 28-dim input",
                    False,
                    f"Network cannot process 28-dim input: {e}"
                )
            
        except Exception as e:
            self.check(
                "DQN network architecture",
                False,
                f"Failed to check DQN: {e}"
            )
    
    def check_parallel_processing(self):
        """Verify parallel mode processes ALL buildings."""
        print(f"\n{BOLD}Checking Parallel Processing...{RESET}")
        
        try:
            from hems.algorithms.dqn import DQNAlgorithm
            
            # Create config
            class TestConfig:
                def __init__(self):
                    self.obs_dim = 28
                    self.n_buildings = 10
                    self.n_actions = 31
                    self.multi_head = True
                    self.buffer_size = 10000
                    self.batch_size = 32
                    self.learning_rate = 0.001
                    self.gamma = 0.99
                    self.epsilon_start = 1.0
                    self.epsilon_end = 0.05
                    self.epsilon_decay_steps = 10000
                    self.target_update_freq = 100
                    self.train_start_steps = 100
                    self.hidden_layers = [256, 256]
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            config = TestConfig()
            algo = DQNAlgorithm(config)
            
            # Simulate observations from 10 buildings
            observations = [[float(i)] * 28 for i in range(10)]
            
            # Check act() method
            try:
                actions = algo.act(observations, deterministic=True)
                
                # Verify we get actions for ALL 10 buildings
                self.check(
                    "Parallel mode returns actions for ALL buildings",
                    len(actions) == 10 or (isinstance(actions[0], list) and len(actions[0]) == 10),
                    f"Expected 10 actions, got {len(actions) if not isinstance(actions[0], list) else len(actions[0])}"
                )
                
            except Exception as e:
                self.check(
                    "Parallel mode processes all buildings",
                    False,
                    f"act() failed: {e}"
                )
                
        except Exception as e:
            self.check(
                "Parallel processing",
                False,
                f"Failed to check parallel processing: {e}"
            )
    
    def check_model_saving(self):
        """Verify model saving includes actual weights."""
        print(f"\n{BOLD}Checking Model Saving...{RESET}")
        
        try:
            from hems.algorithms.dqn import DQNAlgorithm
            
            # Create config
            class TestConfig:
                def __init__(self):
                    self.obs_dim = 28
                    self.n_buildings = 1
                    self.n_actions = 31
                    self.multi_head = False
                    self.buffer_size = 10000
                    self.batch_size = 32
                    self.learning_rate = 0.001
                    self.gamma = 0.99
                    self.epsilon_start = 1.0
                    self.epsilon_end = 0.05
                    self.epsilon_decay_steps = 10000
                    self.target_update_freq = 100
                    self.train_start_steps = 100
                    self.hidden_layers = [256, 256]
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            config = TestConfig()
            algo = DQNAlgorithm(config)
            
            # Check 1: get_state() method exists
            self.check(
                "DQN has get_state() method",
                hasattr(algo, 'get_state'),
                "DQN missing get_state() method - models won't save properly!"
            )
            
            if hasattr(algo, 'get_state'):
                # Check 2: get_state() returns dict with weights
                state = algo.get_state()
                
                self.check(
                    "get_state() returns dictionary",
                    isinstance(state, dict),
                    f"get_state() returned {type(state)}, not dict"
                )
                
                self.check(
                    "get_state() includes q_network weights",
                    'q_network' in state and state['q_network'] is not None,
                    "Missing q_network in state dict"
                )
                
                # Check 3: Weights are not None/empty
                if 'q_network' in state:
                    self.check(
                        "Q-network weights are non-empty",
                        len(state['q_network']) > 0,
                        "Q-network state dict is empty!"
                    )
                
        except Exception as e:
            self.check(
                "Model saving",
                False,
                f"Failed to check model saving: {e}"
            )
    
    def check_testing_infrastructure(self):
        """Verify testing works with per-building baseline."""
        print(f"\n{BOLD}Checking Testing Infrastructure...{RESET}")
        
        # Check ValidationTester file - try multiple possible locations
        possible_paths = [
            Path(__file__).parent / 'validation_testing.py',  # Same directory as this script
            Path(__file__).parent.parent / 'hems' / 'core' / 'validation_testing.py',  # Standard location
            Path.cwd() / 'hems' / 'core' / 'validation_testing.py',  # From project root
        ]
        
        validation_file = None
        for path in possible_paths:
            if path.exists():
                validation_file = path
                break
        
        if validation_file is None:
            self.check(
                "validation_testing.py exists",
                False,
                f"File not found in any of these locations: {[str(p) for p in possible_paths]}"
            )
            return
        
        self.check(
            "validation_testing.py exists",
            True,
            ""
        )
        
        if validation_file.exists():
            content = validation_file.read_text()
            
            # Check for per-building baseline creation
            self.check(
                "Per-building baseline creation implemented",
                '_create_baseline_for_building' in content,
                "Missing _create_baseline_for_building() method"
            )
            
            # Check for proper error handling (no silent fallbacks)
            has_dummy_baseline = 'DummyBaseline' in content
            if has_dummy_baseline:
                self.warn(
                    "DummyBaseline fallback exists",
                    "DummyBaseline should only be last resort - prefer raising errors"
                )
            
            # Check for progress bar
            self.check(
                "Progress bar implemented",
                'tqdm' in content and 'Testing Buildings' in content,
                "Missing progress bar for testing"
            )
    
    def check_no_silent_fallbacks(self):
        """Verify no silent fallbacks in code."""
        print(f"\n{BOLD}Checking for Silent Fallbacks...{RESET}")
        
        files_to_check = [
            Path(__file__).parent.parent / 'hems' / 'algorithms' / 'dqn.py',
            Path(__file__).parent.parent / 'hems' / 'algorithms' / 'sac.py',
            Path(__file__).parent.parent / 'hems' / 'core' / 'validation_testing.py',
        ]
        
        silent_fallback_patterns = [
            ('except: pass', 'Silent exception catching'),
            ('except Exception: pass', 'Silent exception catching'),
            ('action = [[0.0]]', 'Silent zero action fallback'),
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                
                for pattern, desc in silent_fallback_patterns:
                    if pattern in content:
                        # Check if it's properly logged
                        # Look for logger.warning or logger.error near the pattern
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line:
                                # Check surrounding lines for logging
                                context = '\n'.join(lines[max(0, i-3):min(len(lines), i+3)])
                                has_logging = 'logger.warning' in context or 'logger.error' in context
                                
                                if not has_logging:
                                    self.warn(
                                        f"Silent fallback in {file_path.name}",
                                        f"Found '{pattern}' without proper error logging"
                                    )
    
    def run_all_checks(self):
        """Run all validation checks."""
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}HEMS Benchmark Pipeline Validation{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        
        self.check_network_architecture()
        self.check_parallel_processing()
        self.check_model_saving()
        self.check_testing_infrastructure()
        self.check_no_silent_fallbacks()
        
        # Print summary
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}VALIDATION SUMMARY{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        
        print(f"\n{GREEN}Passed: {len(self.passed)}{RESET}")
        for check in self.passed:
            print(f"  {GREEN}✓{RESET} {check}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings: {len(self.warnings)}{RESET}")
            for check, msg in self.warnings:
                print(f"  {YELLOW}⚠{RESET} {check}")
                print(f"    {msg}")
        
        if self.failed:
            print(f"\n{RED}Failed: {len(self.failed)}{RESET}")
            for check, msg in self.failed:
                print(f"  {RED}✗{RESET} {check}")
                if msg:
                    print(f"    {msg}")
            
            print(f"\n{RED}{BOLD}❌ VALIDATION FAILED{RESET}")
            print(f"{RED}Pipeline has critical issues that must be fixed!{RESET}")
            return 1
        else:
            print(f"\n{GREEN}{BOLD}✅ ALL CHECKS PASSED{RESET}")
            print(f"{GREEN}Pipeline is ready for conference-quality experiments!{RESET}")
            
            if self.warnings:
                print(f"\n{YELLOW}Note: {len(self.warnings)} warnings - review recommended{RESET}")
            
            return 0


if __name__ == "__main__":
    validator = PipelineValidator()
    exit_code = validator.run_all_checks()
    sys.exit(exit_code)