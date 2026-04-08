"""
hems/scenarios/__main__.py
Main CLI entry point for HEMS scenarios system.
"""

#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# Add project root to path if needed
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hems.scenarios.runner import ScenarioRunner, BatchScenarioRunner
from hems.scenarios.registry import ScenarioRegistry, ScenarioCollection, StandardCollections


def create_parser():
    """Create argument parser for scenarios CLI."""
    parser = argparse.ArgumentParser(
        description='HEMS Scientific Scenarios CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 HEMS Scenarios System - Scientific Benchmarking and Evaluation

EXAMPLES:
  # List all scenarios
  python -m hems.scenarios list
  
  # List scenarios by category
  python -m hems.scenarios list benchmark
  
  # Show detailed scenario information
  python -m hems.scenarios info benchmark_basic
  
  # Run a scenario
  python -m hems.scenarios run benchmark_basic
  
  # Run all scenarios in a category
  python -m hems.scenarios suite benchmark
  
  # Get equivalent CLI command for a scenario
  python -m hems.scenarios cli benchmark_basic
  
  # Run scenario with parameter overrides
  python -m hems.scenarios run test_quick --buildings 2 --days 14 --gpu
  
  # Dry run (validation only)
  python -m hems.scenarios dry-run benchmark_basic
  
  # Run multiple scenarios in batch
  python -m hems.scenarios batch test_quick test_robustness --days 10
  
  # Search scenarios
  python -m hems.scenarios search battery
  
  # Export scenario configurations
  python -m hems.scenarios export benchmark scenarios_backup.json

CATEGORIES:
  benchmark    - Performance benchmarking scenarios
  research     - Scientific research scenarios  
  testing      - Development and testing scenarios
  development  - Algorithm development scenarios
"""
    )
    
    # Main command
    parser.add_argument('command', 
                       choices=['list', 'info', 'run', 'suite', 'cli', 'dry-run', 
                               'batch', 'search', 'export', 'import', 'stats'],
                       help='Command to execute')
    
    # Target (scenario name, category, or query)
    parser.add_argument('target', nargs='*',
                       help='Scenario name(s), category, or search query')
    
    # Scenario parameter overrides
    override_group = parser.add_argument_group('Scenario Overrides')
    override_group.add_argument('--buildings', type=int, metavar='N',
                               help='Override number of buildings')
    override_group.add_argument('--days', type=int, metavar='N',
                               help='Override simulation days')
    override_group.add_argument('--agents', nargs='+', metavar='AGENT',
                               choices=['baseline', 'rbc', 'dqn', 'tql', 'sac'],
                               help='Override agents to evaluate')
    override_group.add_argument('--train-episodes', type=int, metavar='N',
                               help='Override training episodes')
    override_group.add_argument('--tariff', type=str, metavar='TYPE',
                               choices=['hp_hc', 'tempo', 'standard', 'default'],
                               help='Override tariff type')
    override_group.add_argument('--dataset-type', type=str, metavar='TYPE',
                               choices=['original', 'synthetic', 'dummy'],
                               help='Override dataset type')
    override_group.add_argument('--environment-type', type=str, metavar='TYPE',
                               choices=['citylearn', 'dummy'],
                               help='Override environment type')
    override_group.add_argument('--experiment-name', type=str, metavar='NAME',
                               help='Override experiment name')
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument('--gpu', action='store_true',
                           help='Enable GPU acceleration')
    exec_group.add_argument('--no-eda', action='store_true',
                           help='Disable exploratory data analysis')
    exec_group.add_argument('--no-plots', action='store_true',
                           help='Disable plot generation')
    exec_group.add_argument('--seed', type=int, metavar='N',
                           help='Override random seed')
    
    # Batch execution options
    batch_group = parser.add_argument_group('Batch Execution')
    batch_group.add_argument('--continue-on-error', action='store_true',
                            help='Continue batch execution if scenario fails')
    batch_group.add_argument('--max-parallel', type=int, default=1,
                            help='Maximum parallel executions (future)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-file', type=str, metavar='FILE',
                             help='Output file for export commands')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Quiet output (errors only)')
    
    return parser


def collect_overrides(args):
    """Collect parameter overrides from command line arguments."""
    overrides = {}
    
    # Basic overrides
    if args.buildings is not None:
        overrides['building_count'] = args.buildings
    if args.days is not None:
        overrides['simulation_days'] = args.days
    if args.agents is not None:
        overrides['agents_to_evaluate'] = args.agents
    if args.train_episodes is not None:
        overrides['train_episodes'] = args.train_episodes
    if args.tariff is not None:
        overrides['tariff_type'] = args.tariff
    if args.dataset_type is not None:
        overrides['dataset_type'] = args.dataset_type
    if args.environment_type is not None:
        overrides['environment_type'] = args.environment_type
    if args.experiment_name is not None:
        overrides['experiment_name'] = args.experiment_name
    if args.seed is not None:
        overrides['random_seed'] = args.seed
    
    # Boolean overrides
    if args.gpu:
        overrides['use_gpu'] = True
    if args.no_eda:
        overrides['perform_eda'] = False
    if args.no_plots:
        overrides['save_plots'] = False
    
    return overrides


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect parameter overrides
    overrides = collect_overrides(args)
    
    # Create runner
    runner = ScenarioRunner()
    
    try:
        # Execute commands
        if args.command == 'list':
            execute_list_command(args)
        
        elif args.command == 'info':
            execute_info_command(args)
        
        elif args.command == 'run':
            execute_run_command(args, runner, overrides)
        
        elif args.command == 'suite':
            execute_suite_command(args, runner)
        
        elif args.command == 'cli':
            execute_cli_command(args, overrides)
        
        elif args.command == 'dry-run':
            execute_dry_run_command(args, runner, overrides)
        
        elif args.command == 'batch':
            execute_batch_command(args, overrides)
        
        elif args.command == 'search':
            execute_search_command(args)
        
        elif args.command == 'export':
            execute_export_command(args)
        
        elif args.command == 'import':
            execute_import_command(args)
        
        elif args.command == 'stats':
            execute_stats_command()
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def execute_list_command(args):
    """Execute list command."""
    if args.target:
        # List scenarios in specific category
        category = args.target[0]
        scenarios = ScenarioRegistry.list_scenarios(category)
        if scenarios:
            print(f"\n📂 {category.upper()} scenarios:")
            for name, desc in scenarios.items():
                print(f"  🎯 {name}")
                print(f"     {desc}")
        else:
            print(f"❌ No scenarios found in category: {category}")
            print(f"Available categories: {', '.join(ScenarioRegistry.list_categories())}")
    else:
        # List all scenarios grouped by category
        print("\n🎯 HEMS SCENARIOS CATALOG")
        print("=" * 60)
        
        categories = ScenarioRegistry.list_categories()
        for category in categories:
            scenarios = ScenarioRegistry.list_scenarios(category)
            print(f"\n📂 {category.upper()} SCENARIOS:")
            for name, desc in scenarios.items():
                print(f"  🎯 {name}: {desc}")
        
        total = sum(len(ScenarioRegistry.list_scenarios(cat)) for cat in categories)
        print(f"\nTotal scenarios: {total}")
        print("\nUsage: python -m hems.scenarios run <scenario_name>")


def execute_info_command(args):
    """Execute info command."""
    if not args.target:
        print("❌ Error: Scenario name required for 'info' command")
        sys.exit(1)
    
    scenario_name = args.target[0]
    try:
        scenario = ScenarioRegistry.get_scenario(scenario_name)
        print(scenario.info())
    except ValueError as e:
        print(f"❌ {e}")
        # Suggest similar scenarios
        all_scenarios = ScenarioRegistry.list_scenarios()
        suggestions = [name for name in all_scenarios.keys() if scenario_name.lower() in name.lower()]
        if suggestions:
            print(f"💡 Did you mean: {', '.join(suggestions[:3])}")


def execute_run_command(args, runner, overrides):
    """Execute run command."""
    if not args.target:
        print("❌ Error: Scenario name required for 'run' command")
        sys.exit(1)
    
    scenario_name = args.target[0]
    
    print(f"🚀 Running scenario: {scenario_name}")
    if overrides:
        print(f"🔧 Parameter overrides: {overrides}")
    
    try:
        result = runner.execute_scenario_by_name(scenario_name, **overrides)
        
        if result['status'] == 'success':
            print(f"\n✅ Scenario '{scenario_name}' completed successfully!")
            exec_time = result['execution_metadata']['execution_time_seconds']
            print(f"⏱️ Execution time: {exec_time:.2f} seconds")
            print(f"📁 Results: {result['execution_metadata']['output_directory']}")
        else:
            print(f"\n❌ Scenario '{scenario_name}' failed!")
            print(f"💥 Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Failed to execute scenario: {e}")
        sys.exit(1)


def execute_suite_command(args, runner):
    """Execute suite command."""
    if not args.target:
        print("❌ Error: Category name required for 'suite' command")
        sys.exit(1)
    
    category = args.target[0]
    
    try:
        result = runner.execute_category(category)
        
        summary = result['summary']
        print(f"\n🏁 Suite '{category}' execution summary:")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📊 Success rate: {summary['success_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Failed to execute suite: {e}")
        sys.exit(1)


def execute_cli_command(args, overrides):
    """Execute CLI command."""
    if not args.target:
        print("❌ Error: Scenario name required for 'cli' command")
        sys.exit(1)
    
    scenario_name = args.target[0]
    
    try:
        scenario = ScenarioRegistry.get_scenario(scenario_name)
        
        # Apply overrides if provided
        if overrides:
            config_dict = scenario.config.to_dict()
            config_dict.update(overrides)
            scenario = ScenarioRegistry.create_custom_scenario(
                name=f"{scenario_name}_custom",
                **config_dict
            )
        
        print("🖥️ Equivalent CLI command:")
        print(scenario.get_cli_command())
        
    except Exception as e:
        print(f"❌ Error generating CLI command: {e}")


def execute_dry_run_command(args, runner, overrides):
    """Execute dry run command."""
    if not args.target:
        print("❌ Error: Scenario name required for 'dry-run' command")
        sys.exit(1)
    
    scenario_name = args.target[0]
    
    try:
        scenario = ScenarioRegistry.get_scenario(scenario_name)
        
        # Apply overrides if provided
        if overrides:
            config_dict = scenario.config.to_dict()
            config_dict.update(overrides)
            scenario = ScenarioRegistry.create_custom_scenario(
                name=f"{scenario_name}_dryrun",
                **config_dict
            )
        
        result = runner.dry_run(scenario)
        
        if result['validation']['is_valid']:
            print("✅ Dry run completed successfully!")
        else:
            print("❌ Validation failed!")
            
    except Exception as e:
        print(f"❌ Dry run failed: {e}")


def execute_batch_command(args, overrides):
    """Execute batch command."""
    if not args.target:
        print("❌ Error: Scenario names required for 'batch' command")
        sys.exit(1)
    
    scenario_names = args.target
    
    batch_runner = BatchScenarioRunner(
        max_parallel=args.max_parallel,
        continue_on_error=args.continue_on_error
    )
    
    try:
        result = batch_runner.run_scenarios(scenario_names, **overrides)
        
        summary = result['batch_summary']
        print(f"\n🏁 Batch execution summary:")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏱️ Total time: {summary['total_time_seconds']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Batch execution failed: {e}")
        sys.exit(1)


def execute_search_command(args):
    """Execute search command."""
    if not args.target:
        print("❌ Error: Search query required")
        sys.exit(1)
    
    query = ' '.join(args.target)
    matches = ScenarioRegistry.search_scenarios(query)
    
    if matches:
        print(f"\n🔍 Found {len(matches)} scenarios matching '{query}':")
        for name, desc in matches.items():
            print(f"  🎯 {name}: {desc}")
    else:
        print(f"❌ No scenarios found matching '{query}'")


def execute_export_command(args):
    """Execute export command."""
    if not args.target:
        print("❌ Error: Category required for export")
        sys.exit(1)
    
    category = args.target[0]
    output_file = args.output_file or f"{category}_scenarios.json"
    
    try:
        ScenarioRegistry.export_scenarios(output_file, category)
        print(f"✅ Exported {category} scenarios to {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")


def execute_import_command(args):
    """Execute import command."""
    if not args.target:
        print("❌ Error: JSON file required for import")
        sys.exit(1)
    
    json_file = args.target[0]
    
    try:
        ScenarioRegistry.import_scenarios(json_file)
        print(f"✅ Imported scenarios from {json_file}")
    except Exception as e:
        print(f"❌ Import failed: {e}")


def execute_stats_command():
    """Execute stats command."""
    stats = ScenarioRegistry.get_registry_stats()
    
    print("\n📊 SCENARIO REGISTRY STATISTICS")
    print("=" * 40)
    print(f"Total scenarios: {stats['total_scenarios']}")
    print(f"Categories: {len(stats['categories'])}")
    
    for category, scenarios in stats['category_counts'].items():
        print(f"  📂 {category}: {scenarios} scenarios")
    
    print(f"\nAll scenarios: {', '.join(stats['scenario_list'])}")


if __name__ == '__main__':
    main()