"""
hems/scenarios/runner.py
Scenario execution engine with robust error handling and reporting.
"""

import time
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

from .base import BaseScenario, ScenarioValidator
from .config import ScenarioConfig
from hems.core.runner import HEMSRunner


class ScenarioRunner:
    """
    High-level runner for executing HEMS scenarios.
    
    Provides robust execution, error handling, and result reporting.
    """
    
    def __init__(self):
        """Initialize scenario runner."""
        self.execution_stats = {}
    
    def execute_scenario(self, scenario: BaseScenario) -> Dict[str, Any]:
        """
        Execute a single scenario with comprehensive error handling.
        
        Args:
            scenario: Scenario to execute
            
        Returns:
            Dictionary with execution results and metadata
        """
        start_time = time.time()
        scenario_name = scenario.config.name
        
        print(f"\n🚀 Executing Scenario: {scenario_name}")
        print(f"📋 Description: {scenario.config.description}")
        print(f"🔬 Scientific Purpose: {scenario.config.scientific_purpose}")
        print(f"📊 Category: {scenario.config.category}")
        
        # Prepare result structure
        result = {
            'scenario_name': scenario_name,
            'scenario_config': scenario.config.to_dict(),
            'execution_metadata': {
                'start_time': start_time,
                'status': 'running'
            }
        }
        
        try:
            # Validate scenario
            print("🔍 Validating scenario configuration...")
            is_valid, issues = ScenarioValidator.validate_scenario_config(scenario.config)
            
            if not is_valid:
                raise ValueError(f"Invalid scenario configuration: {issues}")
            
            # Check compatibility
            is_compatible, warnings = ScenarioValidator.validate_scenario_compatibility(scenario.config)
            if warnings:
                print("⚠️ Compatibility warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            # Convert to simulation config
            print("🔧 Converting to simulation configuration...")
            sim_config = scenario.config.to_simulation_config()
            
            # Create output directory
            output_dir = Path(sim_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save scenario configuration
            scenario.save_config(str(output_dir / 'scenario_config.json'))
            
            # Execute with HEMS runner
            print("🏃‍♂️ Executing with HEMS runner...")
            hems_runner = HEMSRunner(sim_config)
            
            # Run EDA if requested
            eda_results = None
            if scenario.config.perform_eda:
                print("📊 Performing Exploratory Data Analysis...")
                eda_results = hems_runner.run_eda()
            
            # Run agent comparison
            print("🤖 Running agent comparison...")
            comparison_results = hems_runner.run_comparison()
            
            # Print summary
            hems_runner.print_summary(comparison_results)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update result
            result.update({
                'status': 'success',
                'eda_results': eda_results,
                'comparison_results': comparison_results,
                'execution_metadata': {
                    'start_time': start_time,
                    'end_time': time.time(),
                    'execution_time_seconds': execution_time,
                    'status': 'completed',
                    'output_directory': str(output_dir),
                    'warnings': warnings
                }
            })
            
            # Store execution stats
            self.execution_stats[scenario_name] = {
                'execution_time': execution_time,
                'status': 'success',
                'timestamp': start_time
            }
            
            print(f"\n✅ Scenario '{scenario_name}' completed successfully!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"📁 Results saved to: {output_dir}")
            
            return result
            
        except Exception as e:
            # Handle execution errors
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            result.update({
                'status': 'error',
                'error': str(e),
                'error_trace': error_trace,
                'execution_metadata': {
                    'start_time': start_time,
                    'end_time': time.time(),
                    'execution_time_seconds': execution_time,
                    'status': 'failed'
                }
            })
            
            # Store execution stats
            self.execution_stats[scenario_name] = {
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e),
                'timestamp': start_time
            }
            
            print(f"\n❌ Scenario '{scenario_name}' failed!")
            print(f"💥 Error: {e}")
            print(f"⏱️ Failed after: {execution_time:.2f} seconds")
            
            # Print detailed error in debug mode
            import os
            if os.getenv('HEMS_DEBUG', '').lower() in ('1', 'true'):
                print(f"🐛 Full error trace:\n{error_trace}")
            
            return result
    
    def execute_scenario_by_name(self, scenario_name: str, **overrides) -> Dict[str, Any]:
        """
        Execute scenario by name with optional parameter overrides.
        
        Args:
            scenario_name: Name of scenario to execute
            **overrides: Parameter overrides
            
        Returns:
            Execution results
        """
        from .registry import ScenarioRegistry
        
        # Get base scenario
        scenario = ScenarioRegistry.get_scenario(scenario_name)
        
        # Apply overrides if provided
        if overrides:
            print(f"🔧 Applying parameter overrides: {overrides}")
            
            # Create custom scenario with overrides
            config_dict = scenario.config.to_dict()
            config_dict.update(overrides)
            
            scenario = ScenarioRegistry.create_custom_scenario(
                name=f"{scenario_name}_custom",
                **config_dict
            )
        
        return self.execute_scenario(scenario)
    
    def execute_category(self, category: str) -> Dict[str, Any]:
        """
        Execute all scenarios in a category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary with results from all scenarios
        """
        from .registry import ScenarioRegistry
        
        scenarios = ScenarioRegistry.get_scenarios_by_category(category)
        
        if not scenarios:
            raise ValueError(f"No scenarios found in category: {category}")
        
        print(f"🎯 Executing {len(scenarios)} scenarios in category: {category}")
        
        results = {}
        successful = 0
        failed = 0
        
        for name, scenario in scenarios.items():
            print(f"\n{'='*80}")
            try:
                result = self.execute_scenario(scenario)
                results[name] = result
                
                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Critical error executing {name}: {e}")
                results[name] = {
                    'status': 'critical_error',
                    'error': str(e),
                    'scenario_name': name
                }
                failed += 1
        
        print(f"\n🏁 Category '{category}' execution completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(scenarios)}")
        
        return {
            'category': category,
            'summary': {
                'total_scenarios': len(scenarios),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(scenarios) if scenarios else 0
            },
            'results': results
        }
    
    def dry_run(self, scenario: BaseScenario) -> Dict[str, Any]:
        """
        Perform a dry run of scenario without execution.
        
        Args:
            scenario: Scenario to validate
            
        Returns:
            Validation results
        """
        print(f"🔍 Dry run for scenario: {scenario.config.name}")
        
        # Validate configuration
        is_valid, issues = ScenarioValidator.validate_scenario_config(scenario.config)
        is_compatible, warnings = ScenarioValidator.validate_scenario_compatibility(scenario.config)
        
        # Check simulation config conversion
        try:
            sim_config = scenario.config.to_simulation_config()
            conversion_success = True
            conversion_error = None
        except Exception as e:
            conversion_success = False
            conversion_error = str(e)
        
        # Estimate execution time
        estimated_time = self._estimate_execution_time(scenario.config)
        
        result = {
            'scenario_name': scenario.config.name,
            'validation': {
                'is_valid': is_valid,
                'issues': issues,
                'is_compatible': is_compatible,
                'warnings': warnings
            },
            'conversion': {
                'success': conversion_success,
                'error': conversion_error
            },
            'estimates': {
                'execution_time_minutes': estimated_time,
                'output_size_mb': self._estimate_output_size(scenario.config)
            },
            'cli_command': scenario.get_cli_command()
        }
        
        # Print dry run results
        print(f"📋 Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        if issues:
            print("🚨 Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        if warnings:
            print("⚠️ Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print(f"⏱️ Estimated execution time: {estimated_time:.1f} minutes")
        print(f"💾 Estimated output size: {result['estimates']['output_size_mb']:.1f} MB")
        
        return result
    
    def _estimate_execution_time(self, config: ScenarioConfig) -> float:
        """
        Estimate execution time in minutes.
        
        Args:
            config: Scenario configuration
            
        Returns:
            Estimated time in minutes
        """
        base_time = 2.0  # Base time in minutes
        
        # Time factors
        building_factor = config.building_count * 0.5
        day_factor = config.simulation_days * 0.1
        agent_factor = len(config.agents_to_evaluate) * 1.0
        
        # Training time for RL agents
        rl_agents = [a for a in config.agents_to_evaluate if a in ['dqn', 'sac', 'tql']]
        training_factor = len(rl_agents) * config.train_episodes * 0.01
        
        # EDA factor
        eda_factor = 2.0 if config.perform_eda else 0.0
        
        # Environment factor
        env_factor = 0.5 if config.environment_type == 'dummy' else 1.0
        
        total_time = (base_time + building_factor + day_factor + 
                     agent_factor + training_factor + eda_factor) * env_factor
        
        return max(1.0, total_time)  # Minimum 1 minute
    
    def _estimate_output_size(self, config: ScenarioConfig) -> float:
        """
        Estimate output size in MB.
        
        Args:
            config: Scenario configuration
            
        Returns:
            Estimated size in MB
        """
        base_size = 10.0  # Base size in MB
        
        # Size factors
        time_factor = config.simulation_days * config.building_count * 0.1
        agent_factor = len(config.agents_to_evaluate) * 5.0
        plot_factor = 20.0 if config.save_plots else 0.0
        eda_factor = 50.0 if config.perform_eda else 0.0
        
        total_size = base_size + time_factor + agent_factor + plot_factor + eda_factor
        
        return max(5.0, total_size)  # Minimum 5 MB
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        if not self.execution_stats:
            return {'message': 'No scenarios executed yet'}
        
        total_executions = len(self.execution_stats)
        successful = sum(1 for s in self.execution_stats.values() if s['status'] == 'success')
        failed = total_executions - successful
        
        total_time = sum(s['execution_time'] for s in self.execution_stats.values())
        avg_time = total_time / total_executions if total_executions > 0 else 0
        
        return {
            'total_executions': total_executions,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_executions if total_executions > 0 else 0,
            'total_execution_time_seconds': total_time,
            'average_execution_time_seconds': avg_time,
            'scenarios': self.execution_stats
        }


class BatchScenarioRunner:
    """
    Runner for executing multiple scenarios in batch with advanced features.
    """
    
    def __init__(self, max_parallel: int = 1, continue_on_error: bool = True):
        """
        Initialize batch runner.
        
        Args:
            max_parallel: Maximum parallel executions (future enhancement)
            continue_on_error: Whether to continue if a scenario fails
        """
        self.max_parallel = max_parallel
        self.continue_on_error = continue_on_error
        self.runner = ScenarioRunner()
    
    def run_scenarios(self, scenario_names: list, **common_overrides) -> Dict[str, Any]:
        """
        Run multiple scenarios with common overrides.
        
        Args:
            scenario_names: List of scenario names to run
            **common_overrides: Common parameter overrides for all scenarios
            
        Returns:
            Batch execution results
        """
        print(f"🚀 Starting batch execution of {len(scenario_names)} scenarios")
        if common_overrides:
            print(f"🔧 Common overrides: {common_overrides}")
        
        results = {}
        start_time = time.time()
        
        for i, scenario_name in enumerate(scenario_names, 1):
            print(f"\n📊 Progress: {i}/{len(scenario_names)} scenarios")
            
            try:
                result = self.runner.execute_scenario_by_name(scenario_name, **common_overrides)
                results[scenario_name] = result
                
                if result['status'] != 'success' and not self.continue_on_error:
                    print(f"🛑 Stopping batch execution due to failure in {scenario_name}")
                    break
                    
            except Exception as e:
                print(f"❌ Critical error in {scenario_name}: {e}")
                results[scenario_name] = {
                    'status': 'critical_error',
                    'error': str(e),
                    'scenario_name': scenario_name
                }
                
                if not self.continue_on_error:
                    print("🛑 Stopping batch execution due to critical error")
                    break
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        
        print(f"\n🏁 Batch execution completed!")
        print(f"✅ Successful: {successful}/{len(results)}")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        
        return {
            'batch_summary': {
                'total_scenarios': len(scenario_names),
                'executed_scenarios': len(results),
                'successful': successful,
                'failed': len(results) - successful,
                'total_time_seconds': total_time,
                'common_overrides': common_overrides
            },
            'results': results
        }