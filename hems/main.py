#!/usr/bin/env python3
"""
HEMS (Home Energy Management System) Simulation Environment
A comprehensive, modular framework for evaluating AI agents in energy management tasks.

This is the main entry point for the HEMS simulation environment.
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from hems.core.benchmark_runner import BenchmarkRunner
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from hems.environments.citylearn.hems_environments import HEMSEnvironment
from hems.core.config import SimulationConfig
from hems.core.runner import HEMSRunner
from hems.utils.utils import setup_logger


def main():
    """Main entry point for HEMS simulation."""
    parser = argparse.ArgumentParser(description='HEMS Simulation Environment')


    # add --benchmark-config option
    
    parser.add_argument('--benchmark-config', type=str, default=None,
                       help='Path to YAML/JSON benchmark configuration file')

    # Basic simulation parameters
    parser.add_argument('--buildings', type=int, default=1, 
                       help='Number of buildings to simulate (1-15)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to simulate (1-365)')
    parser.add_argument('--building-id', type=str, default=None,
                       help='Specific building ID (e.g., Building_1)')
    
    # Agent selection
    parser.add_argument('--agents', nargs='+', 
                       default=['baseline', 'rbc', 'dqn'],
                       choices=['baseline', 'rbc', 'tql', 'sac', 'dqn', 'mp_ppo', 'mpc_forecast', 'ambitious_engineers', 'Chen_Bu_p2p', 'custom'],
                       help='Agents to evaluate')
    
    # Training parameters
    parser.add_argument('--train-episodes', type=int, default=100,
                       help='Training episodes for RL agents')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training (if available)')
    
    # Data and evaluation
    parser.add_argument('--tariff', type=str, default='hp_hc',
                       choices=['default', 'hp_hc', 'tempo', 'standard'],
                       help='Electricity tariff type')
    parser.add_argument('--eda', action='store_true',
                       help='Perform exploratory data analysis')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots and results')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    # In the argument parser section
    parser.add_argument('--environment', choices=['citylearn', 'dummy', 'synthetic'], default='citylearn',
                   help='Environment type')
                       

    #dataset selection
    
    # Dataset configuration
    parser.add_argument('--dataset-type', type=str, default='original',
                    choices=['original', 'synthetic'],
                    help='Dataset type to use')
    parser.add_argument('--synthetic-dataset', type=str, default='demo_1',
                    help='Synthetic dataset name (when using synthetic dataset type)')
    parser.add_argument('--datasets-root', type=str, default='datasets',
                    help='Root directory for synthetic datasets')

    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('HEMS', log_dir=args.output_dir)
    logger.info(f"Starting HEMS simulation with args: {vars(args)}")

    # AmbitiousEngineers mode selection
    parser.add_argument(
        '--ae-mode',
        choices=['dp_only', 'phase1', 'phase23'],
        default=None,
        help="AmbitiousEngineers mode: 'dp_only' (baseline), 'phase1' (CMA-ES single-agent), 'phase23' (multi-agent)"
    )

    parser.add_argument(
        '--ae-weights',
        type=str,
        default=None,
        help='Path to pretrained AmbitiousEngineers weights (.npy file)'
    )

    parser.add_argument(
        '--fidelity',
        action='store_true',
        help="Reproduce AmbitiousEngineers original setup (Phase-1 dataset, 5 buildings, correct metrics)"
    )
    # If benchmark config is provided, use benchmark runner
    if args.benchmark_config:
        print(f"🚀 Running benchmark from config: {args.benchmark_config}")
        runner = BenchmarkRunner(args.benchmark_config)
        results = runner.run()
        return results

    
    # Create simulation configuration
    config = SimulationConfig(
        building_count=args.buildings,
        simulation_days=args.days,
        building_id=args.building_id,
        agents_to_evaluate=args.agents,
        train_episodes=args.train_episodes,
        use_gpu=args.gpu,
        tariff_type=args.tariff,
        random_seed=args.seed,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        perform_eda=args.eda,
        environment_type=getattr(args, 'environment', 'citylearn'),
        
    )
    

    # AmbitiousEngineers config handling
    ae_mode = getattr(args, 'ae_mode', None)

    if not hasattr(config, 'ambitious_engineers_config') or config.ambitious_engineers_config is None:
        config.ambitious_engineers_config = {}

    if ae_mode is not None:
        config.ambitious_engineers_config['mode'] = ae_mode
        config.ambitious_engineers_config['train_mode'] = False  # Always False for inference
        
        # Pass weights path if provided
        if hasattr(args, 'ae_weights') and args.ae_weights:
            if ae_mode == 'phase1':
                config.ambitious_engineers_config['phase1_weights_path'] = args.ae_weights
            elif ae_mode == 'phase23':
                config.ambitious_engineers_config['phase23_weights_path'] = args.ae_weights
    else:
        config.ambitious_engineers_config['mode'] = 'dp_only'
        config.ambitious_engineers_config['train_mode'] = False

    # Fidelity mode overrides
    if getattr(args, 'fidelity', False):
        config.dataset_type = 'original'
        config.dataset_name = 'citylearn_challenge_2022_phase_1'
        config.building_count = max(5, getattr(config, 'building_count', 5))
        config.environment_type = 'citylearn'

    print(f"[AE FLAGS] ae_mode={config.ambitious_engineers_config.get('mode')} | "
        f"train_mode={config.ambitious_engineers_config.get('train_mode')} | "
        f"fidelity={getattr(args,'fidelity',False)}")

    if hasattr(args, 'dataset_type'):
        config.dataset_type = args.dataset_type
    if hasattr(args, 'synthetic_dataset'):
        config.synthetic_dataset_name = args.synthetic_dataset
    if hasattr(args, 'datasets_root'):
        config.datasets_root = args.datasets_root

    
    # Create and run simulation
    runner = HEMSRunner(config)
    
    try:
        # Run EDA if requested
        if args.eda:
            logger.info("Performing Exploratory Data Analysis...")
            runner.run_eda()
        
        # Run agent comparison
        logger.info("Running agent comparison...")
        results = runner.run_comparison()
        
        # Print summary
        runner.print_summary(results)
        
        logger.info(f"Simulation completed successfully. Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()