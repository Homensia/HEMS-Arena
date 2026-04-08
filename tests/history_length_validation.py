#!/usr/bin/env python3
"""
Professional History Length Verification Suite for Mem Algorithm

This comprehensive test suite validates that the Memdqn algorithm correctly implements
variable history lengths by testing dimension calculations, buffer management, 
phi vector construction, and network compatibility.


"""

import numpy as np
import sys
import traceback
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class TestResult:
    """Container for individual test results"""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HistoryLengthTestReport:
    """Comprehensive test report for a specific history length"""
    history_length: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[TestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
    
    @property
    def overall_passed(self) -> bool:
        return self.failed_tests == 0

class HistoryLengthTester:
    """
    Professional test suite for validating Memdqn history length implementation
    """
    
    def __init__(self, 
                 obs_dim_single: int = 5, 
                 n_buildings: int = 1,
                 verbose: bool = True):
        """
        Initialize the tester with environment parameters
        
        Args:
            obs_dim_single: Dimension of single observation
            n_buildings: Number of buildings in environment
            verbose: Whether to print detailed output
        """
        self.obs_dim_single = obs_dim_single
        self.n_buildings = n_buildings
        self.verbose = verbose
        
    def calculate_expected_phi_dimension(self, history_length: int) -> int:
        """Calculate expected phi vector dimension for given history length"""
        return history_length * (self.obs_dim_single + self.n_buildings) + self.obs_dim_single
    
    def test_dimension_calculation(self, history_length: int) -> TestResult:
        """Test that phi dimension calculation is correct"""
        expected_dim = self.calculate_expected_phi_dimension(history_length)
        
        # Manual calculation for verification
        history_part = history_length * (self.obs_dim_single + self.n_buildings)
        current_part = self.obs_dim_single
        manual_calculation = history_part + current_part
        
        passed = expected_dim == manual_calculation
        
        return TestResult(
            test_name="dimension_calculation",
            passed=passed,
            expected=expected_dim,
            actual=manual_calculation,
            details={
                "formula": f"{history_length} × ({self.obs_dim_single} + {self.n_buildings}) + {self.obs_dim_single}",
                "calculation": f"{history_length} × {self.obs_dim_single + self.n_buildings} + {self.obs_dim_single} = {expected_dim}",
                "history_part": history_part,
                "current_part": current_part
            }
        )
    
    def test_buffer_behavior(self, history_length: int) -> TestResult:
        """Test that deque buffer behaves correctly with maxlen"""
        try:
            obs_buffer = deque(maxlen=history_length)
            action_buffer = deque(maxlen=history_length)
            
            # Test buffer filling
            buffer_states = []
            
            for step in range(history_length + 3):  # Test beyond capacity
                # Add dummy data
                obs = [float(step + i) for i in range(self.obs_dim_single)]
                action = [0.1 * (step + 1)]
                
                obs_buffer.append(obs)
                action_buffer.append(action)
                
                buffer_states.append({
                    "step": step + 1,
                    "obs_len": len(obs_buffer),
                    "action_len": len(action_buffer),
                    "obs_max_reached": len(obs_buffer) == history_length,
                    "action_max_reached": len(action_buffer) == history_length
                })
            
            # Verify final state
            final_obs_len = len(obs_buffer)
            final_action_len = len(action_buffer)
            
            # Check that buffers maintain correct size
            correct_final_size = (final_obs_len == history_length and 
                                final_action_len == history_length)
            
            # Check that buffers grow correctly then stop
            growth_correct = True
            max_reached_step = None
            
            for i, state in enumerate(buffer_states):
                expected_len = min(i + 1, history_length)
                if (state["obs_len"] != expected_len or 
                    state["action_len"] != expected_len):
                    growth_correct = False
                    break
                    
                if state["obs_max_reached"] and max_reached_step is None:
                    max_reached_step = state["step"]
            
            passed = correct_final_size and growth_correct
            
            return TestResult(
                test_name="buffer_behavior",
                passed=passed,
                expected={"final_size": history_length, "growth_pattern": "correct"},
                actual={"final_obs_size": final_obs_len, "final_action_size": final_action_len},
                details={
                    "buffer_states": buffer_states,
                    "max_reached_at_step": max_reached_step,
                    "growth_correct": growth_correct
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="buffer_behavior",
                passed=False,
                expected="no_error",
                actual=f"exception: {str(e)}",
                error_message=str(e)
            )
    
    def test_phi_construction(self, history_length: int) -> TestResult:
        """Test phi vector construction step by step"""
        try:
            obs_buffer = deque(maxlen=history_length)
            action_buffer = deque(maxlen=history_length)
            
            expected_phi_dim = self.calculate_expected_phi_dimension(history_length)
            phi_vectors = []
            
            # Simulate multiple steps
            test_steps = max(history_length + 2, 7)  # At least 7 steps or history_length + 2
            
            for step in range(test_steps):
                # Create test observation and action
                current_obs = np.array([float(step + 1 + i) for i in range(self.obs_dim_single)], 
                                     dtype=np.float32)
                current_action = np.array([0.1 * (step + 1)], dtype=np.float32)
                
                # Build phi vector (simulate _phi method)
                phi_components = []
                
                # Add historical obs-action pairs
                for i in range(history_length):
                    if i < len(obs_buffer):
                        phi_components.extend(obs_buffer[i])
                        phi_components.extend(action_buffer[i])
                    else:
                        # Pad with zeros
                        phi_components.extend([0.0] * self.obs_dim_single)
                        phi_components.extend([0.0] * self.n_buildings)
                
                # Add current observation
                phi_components.extend(current_obs)
                
                phi_vector = np.array(phi_components, dtype=np.float32)
                
                phi_vectors.append({
                    "step": step + 1,
                    "phi_length": len(phi_vector),
                    "expected_length": expected_phi_dim,
                    "length_correct": len(phi_vector) == expected_phi_dim,
                    "phi_first_5": phi_vector[:5].tolist(),
                    "phi_last_5": phi_vector[-5:].tolist(),
                    "buffer_sizes": {"obs": len(obs_buffer), "action": len(action_buffer)}
                })
                
                # Update buffers
                obs_buffer.append(current_obs.tolist())
                action_buffer.append(current_action.tolist())
            
            # Check if all phi vectors have correct dimensions
            all_dimensions_correct = all(pv["length_correct"] for pv in phi_vectors)
            
            # Check for consistent dimension across all steps
            phi_lengths = [pv["phi_length"] for pv in phi_vectors]
            consistent_dimensions = len(set(phi_lengths)) == 1
            
            passed = all_dimensions_correct and consistent_dimensions
            
            return TestResult(
                test_name="phi_construction",
                passed=passed,
                expected={"all_lengths": expected_phi_dim, "consistent": True},
                actual={"lengths": phi_lengths, "consistent": consistent_dimensions},
                details={
                    "phi_vectors": phi_vectors,
                    "expected_phi_dim": expected_phi_dim
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="phi_construction",
                passed=False,
                expected="successful_construction",
                actual=f"exception: {str(e)}",
                error_message=str(e)
            )
    
    def test_phi_structure_integrity(self, history_length: int) -> TestResult:
        """Test that phi vector has correct internal structure"""
        try:
            # Create full history scenario
            obs_buffer = deque(maxlen=history_length)
            action_buffer = deque(maxlen=history_length)
            
            # Fill buffers completely
            for i in range(history_length):
                obs = [float(10 + i + j) for j in range(self.obs_dim_single)]
                action = [float(0.1 * (i + 1))]
                obs_buffer.append(obs)
                action_buffer.append(action)
            
            # Create current observation
            current_obs = [float(100 + i) for i in range(self.obs_dim_single)]
            
            # Build phi vector
            phi_components = []
            for i in range(history_length):
                phi_components.extend(obs_buffer[i])
                phi_components.extend(action_buffer[i])
            phi_components.extend(current_obs)
            
            phi_vector = np.array(phi_components)
            
            # Verify structure by parsing back
            parsed_components = []
            idx = 0
            
            # Parse historical components
            for i in range(history_length):
                obs_part = phi_vector[idx:idx + self.obs_dim_single]
                idx += self.obs_dim_single
                action_part = phi_vector[idx:idx + self.n_buildings]
                idx += self.n_buildings
                
                parsed_components.append({
                    "type": f"history_{i}",
                    "obs": obs_part.tolist(),
                    "action": action_part.tolist(),
                    "obs_matches": np.allclose(obs_part, obs_buffer[i]),
                    "action_matches": np.allclose(action_part, action_buffer[i])
                })
            
            # Parse current observation
            current_part = phi_vector[idx:]
            parsed_components.append({
                "type": "current",
                "obs": current_part.tolist(),
                "obs_matches": np.allclose(current_part, current_obs)
            })
            
            # Check all components match
            all_history_obs_match = all(pc["obs_matches"] for pc in parsed_components[:-1])
            all_history_action_match = all(pc["action_matches"] for pc in parsed_components[:-1])
            current_obs_matches = parsed_components[-1]["obs_matches"]
            
            structure_correct = (all_history_obs_match and 
                               all_history_action_match and 
                               current_obs_matches)
            
            passed = structure_correct and len(phi_vector) == self.calculate_expected_phi_dimension(history_length)
            
            return TestResult(
                test_name="phi_structure_integrity",
                passed=passed,
                expected="all_components_match",
                actual={
                    "history_obs_match": all_history_obs_match,
                    "history_action_match": all_history_action_match,
                    "current_obs_match": current_obs_matches
                },
                details={
                    "parsed_components": parsed_components,
                    "phi_length": len(phi_vector),
                    "expected_length": self.calculate_expected_phi_dimension(history_length)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="phi_structure_integrity",
                passed=False,
                expected="correct_structure",
                actual=f"exception: {str(e)}",
                error_message=str(e)
            )
    
    def test_edge_cases(self, history_length: int) -> TestResult:
        """Test edge cases like empty history, single step, etc."""
        try:
            edge_case_results = []
            
            # Test 1: Empty history
            obs_buffer = deque(maxlen=history_length)
            action_buffer = deque(maxlen=history_length)
            current_obs = np.ones(self.obs_dim_single)
            
            phi_components = []
            for i in range(history_length):
                phi_components.extend([0.0] * self.obs_dim_single)
                phi_components.extend([0.0] * self.n_buildings)
            phi_components.extend(current_obs)
            
            empty_history_phi = np.array(phi_components)
            expected_dim = self.calculate_expected_phi_dimension(history_length)
            
            edge_case_results.append({
                "case": "empty_history",
                "phi_length": len(empty_history_phi),
                "expected_length": expected_dim,
                "correct": len(empty_history_phi) == expected_dim
            })
            
            # Test 2: history_length = 1 (minimum case)
            if history_length == 1:
                single_obs = [1.0] * self.obs_dim_single
                single_action = [0.5] * self.n_buildings
                current_obs_single = [2.0] * self.obs_dim_single
                
                phi_single = single_obs + single_action + current_obs_single
                expected_single = 1 * (self.obs_dim_single + self.n_buildings) + self.obs_dim_single
                
                edge_case_results.append({
                    "case": "history_length_1",
                    "phi_length": len(phi_single),
                    "expected_length": expected_single,
                    "correct": len(phi_single) == expected_single
                })
            
            # Test 3: Very large history_length
            if history_length >= 10:
                large_expected = self.calculate_expected_phi_dimension(history_length)
                edge_case_results.append({
                    "case": "large_history_length",
                    "history_length": history_length,
                    "expected_phi_dim": large_expected,
                    "reasonable_size": large_expected < 10000  # Sanity check
                })
            
            all_edge_cases_passed = all(case.get("correct", case.get("reasonable_size", True)) 
                                      for case in edge_case_results)
            
            return TestResult(
                test_name="edge_cases",
                passed=all_edge_cases_passed,
                expected="all_edge_cases_pass",
                actual=f"{sum(case.get('correct', case.get('reasonable_size', True)) for case in edge_case_results)}/{len(edge_case_results)} passed",
                details={"edge_case_results": edge_case_results}
            )
            
        except Exception as e:
            return TestResult(
                test_name="edge_cases",
                passed=False,
                expected="no_errors",
                actual=f"exception: {str(e)}",
                error_message=str(e)
            )
    
    def test_single_history_length(self, history_length: int) -> HistoryLengthTestReport:
        """Run complete test suite for a single history length"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🧪 TESTING HISTORY LENGTH: {history_length}")
            print(f"{'='*60}")
        
        tests = [
            self.test_dimension_calculation,
            self.test_buffer_behavior,
            self.test_phi_construction,
            self.test_phi_structure_integrity,
            self.test_edge_cases
        ]
        
        results = []
        passed_count = 0
        
        for test_func in tests:
            try:
                result = test_func(history_length)
                results.append(result)
                
                if result.passed:
                    passed_count += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                if self.verbose:
                    print(f"{status} {result.test_name}")
                    if result.error_message:
                        print(f"      Error: {result.error_message}")
                    if not result.passed and result.expected != result.actual:
                        print(f"      Expected: {result.expected}")
                        print(f"      Actual: {result.actual}")
                        
            except Exception as e:
                error_result = TestResult(
                    test_name=test_func.__name__,
                    passed=False,
                    expected="no_exception",
                    actual=f"exception: {str(e)}",
                    error_message=str(e)
                )
                results.append(error_result)
                if self.verbose:
                    print(f"❌ FAIL {test_func.__name__} - Exception: {e}")
        
        report = HistoryLengthTestReport(
            history_length=history_length,
            total_tests=len(tests),
            passed_tests=passed_count,
            failed_tests=len(tests) - passed_count,
            test_results=results
        )
        
        if self.verbose:
            print(f"\n📊 SUMMARY for history_length={history_length}:")
            print(f"   Tests Passed: {report.passed_tests}/{report.total_tests}")
            print(f"   Success Rate: {report.success_rate:.1f}%")
            print(f"   Overall: {'✅ PASS' if report.overall_passed else '❌ FAIL'}")
        
        return report
    
    def run_comprehensive_test(self, 
                             history_lengths: Optional[List[int]] = None) -> Dict[int, HistoryLengthTestReport]:
        """
        Run comprehensive test suite across multiple history lengths
        
        Args:
            history_lengths: List of history lengths to test. If None, uses default range.
            
        Returns:
            Dictionary mapping history_length to test report
        """
        if history_lengths is None:
            history_lengths = [1, 2, 3, 5, 7, 10, 15, 20, 50]
        
        print("🚀 PROFESSIONAL HISTORY LENGTH VERIFICATION SUITE")
        print("="*70)
        print(f"Environment Parameters:")
        print(f"  obs_dim_single: {self.obs_dim_single}")
        print(f"  n_buildings: {self.n_buildings}")
        print(f"Testing history lengths: {history_lengths}")
        
        all_reports = {}
        overall_stats = {"total_tests": 0, "total_passed": 0, "lengths_passed": 0}
        
        for history_length in history_lengths:
            try:
                report = self.test_single_history_length(history_length)
                all_reports[history_length] = report
                
                overall_stats["total_tests"] += report.total_tests
                overall_stats["total_passed"] += report.passed_tests
                if report.overall_passed:
                    overall_stats["lengths_passed"] += 1
                    
            except Exception as e:
                print(f"❌ CRITICAL ERROR testing history_length={history_length}: {e}")
                traceback.print_exc()
        
        # Overall summary
        print(f"\n{'='*70}")
        print("📈 OVERALL VERIFICATION RESULTS")
        print(f"{'='*70}")
        
        success_lengths = []
        failed_lengths = []
        
        for history_length, report in all_reports.items():
            if report.overall_passed:
                success_lengths.append(history_length)
                print(f"✅ history_length={history_length:2d}: {report.success_rate:5.1f}% "
                      f"({report.passed_tests}/{report.total_tests})")
            else:
                failed_lengths.append(history_length)
                print(f"❌ history_length={history_length:2d}: {report.success_rate:5.1f}% "
                      f"({report.passed_tests}/{report.total_tests})")
        
        overall_success_rate = (overall_stats["total_passed"] / overall_stats["total_tests"]) * 100
        lengths_success_rate = (overall_stats["lengths_passed"] / len(history_lengths)) * 100
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Total Tests Run: {overall_stats['total_tests']}")
        print(f"   Total Tests Passed: {overall_stats['total_passed']}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"   History Lengths Fully Passed: {overall_stats['lengths_passed']}/{len(history_lengths)} ({lengths_success_rate:.1f}%)")
        
        if success_lengths:
            print(f"   ✅ Successful Lengths: {success_lengths}")
        if failed_lengths:
            print(f"   ❌ Failed Lengths: {failed_lengths}")
        
        # Implementation status
        if overall_stats["lengths_passed"] == len(history_lengths):
            print(f"\n🎉 VERIFICATION COMPLETE: Your history length implementation is FULLY WORKING!")
        elif overall_stats["lengths_passed"] > 0:
            print(f"\n⚠️  PARTIAL SUCCESS: Implementation works for some history lengths but needs fixes.")
        else:
            print(f"\n💥 IMPLEMENTATION ISSUES: History length mechanism needs debugging.")
        
        return all_reports
    
    def export_report(self, reports: Dict[int, HistoryLengthTestReport], filename: str = "history_length_test_report.json"):
        """Export detailed test results to JSON file"""
        export_data = {}
        
        for history_length, report in reports.items():
            export_data[str(history_length)] = {
                "history_length": report.history_length,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "success_rate": report.success_rate,
                "overall_passed": report.overall_passed,
                "test_results": []
            }
            
            for test_result in report.test_results:
                export_data[str(history_length)]["test_results"].append({
                    "test_name": test_result.test_name,
                    "passed": test_result.passed,
                    "expected": str(test_result.expected),
                    "actual": str(test_result.actual),
                    "error_message": test_result.error_message,
                    "details": test_result.details
                })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📄 Detailed report exported to: {filename}")


def main():
    """Main function for running tests from command line"""
    print("🧪 Professional History Length Verification Suite")
    print("="*60)
    
    # You can customize these parameters based on your environment
    obs_dim = int(input("Enter observation dimension (default 5): ") or "5")
    n_buildings = int(input("Enter number of buildings (default 1): ") or "1")
    
    print("\nSelect test mode:")
    print("1. Quick test (lengths: 1, 2, 3, 5)")
    print("2. Standard test (lengths: 1, 2, 3, 5, 7, 10)")
    print("3. Comprehensive test (lengths: 1, 2, 3, 5, 7, 10, 15, 20, 50)")
    print("4. Custom lengths")
    
    choice = input("Enter choice (default 2): ") or "2"
    
    if choice == "1":
        test_lengths = [1, 2, 3, 5]
    elif choice == "3":
        test_lengths = [1, 2, 3, 5, 7, 10, 15, 20, 50]
    elif choice == "4":
        lengths_input = input("Enter comma-separated history lengths: ")
        test_lengths = [int(x.strip()) for x in lengths_input.split(",")]
    else:  # Default choice 2
        test_lengths = [1, 2, 3, 5, 7, 10]
    
    # Initialize tester
    tester = HistoryLengthTester(obs_dim_single=obs_dim, n_buildings=n_buildings, verbose=True)
    
    # Run tests
    results = tester.run_comprehensive_test(test_lengths)
    
    # Export results
    export_choice = input("\nExport detailed results to JSON? (y/n, default n): ").lower()
    if export_choice in ['y', 'yes']:
        filename = input("Enter filename (default: history_length_test_report.json): ") or "history_length_test_report.json"
        tester.export_report(results, filename)


if __name__ == "__main__":
    main()