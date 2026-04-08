"""
Run testing phase only on pre-trained models.

This module allows running just the testing phase without retraining,
using previously saved models from an experiment directory.

Usage:
    python3 -m hems.core.run_testing_only <experiment_dir> <config.yaml>
    
Example:
    python3 -m hems.core.run_testing_only DQN_Sequential_0_20251120_112930 benchmark_configs/example_simple_sequential0.yaml
"""

import sys
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

# Local imports
from hems.core.yaml_loader import YAMLConfigLoader
from hems.core.validation_testing import ValidationTester
from hems.environments.citylearn.citylearn_wrapper import CityLearnEnvironmentManager
from hems.agents.legacy_adapter import create_agent


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging for testing-only mode."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('TestingOnlyRunner')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'testing_only_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized - testing only mode")
    logger.info(f"Log file: {log_file}")
    
    return logger


def load_training_results(experiment_dir: Path, logger: logging.Logger):
    """Load training results from disk."""
    results_file = experiment_dir / 'results' / 'training_results.json'
    
    if not results_file.exists():
        logger.warning(f"Training results not found at {results_file}")
        return {}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    logger.info(f"[LOAD] Loaded training results from {results_file}")
    return results


def run_testing_only(experiment_dir: str, config_path: str):
    """
    Run testing phase only using pre-trained models.
    
    Args:
        experiment_dir: Path to existing experiment directory (e.g., DQN_Sequential_0_20251120_112930)
        config_path: Path to benchmark config YAML file
    """
    experiment_dir = Path(experiment_dir)
    config_path = Path(config_path)
    
    # Validate paths
    if not experiment_dir.exists():
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
        print(f"       Make sure you're in the right directory or use absolute path")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    
    logger.info("=" * 80)
    logger.info("TESTING ONLY MODE - Loading Pre-Trained Models")
    logger.info("=" * 80)
    logger.info(f"Experiment Directory: {experiment_dir}")
    logger.info(f"Config File: {config_path}")
    logger.info("")
    
    # Load config
    logger.info("[LOAD] Loading benchmark configuration...")
    try:
        yaml_loader = YAMLConfigLoader()
        benchmark_config = yaml_loader.load(str(config_path))
        logger.info("[OK] Configuration loaded")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load config: {e}")
        raise
    
    # Load training results
    logger.info("[LOAD] Loading training results...")
    training_results = load_training_results(experiment_dir, logger)
    
    # Initialize environment
    logger.info("[INIT] Initializing environment manager...")
    try:
        sim_config = yaml_loader.to_simulation_config(benchmark_config)
        env_manager = CityLearnEnvironmentManager(sim_config)
        logger.info("[OK] Environment manager initialized")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize environment: {e}")
        raise
    
    # Initialize agents
    logger.info("[INIT] Initializing agents...")
    agents = {}
    enabled_agents = [a for a in benchmark_config.agents if a.get('enabled', True)]
    
    for agent_config in enabled_agents:
        agent_name = agent_config['name']
        logger.info(f"[INIT] Initializing agent: {agent_name}")
        
        try:
            # Create agent with single building for initialization
            first_building = benchmark_config.training.buildings.ids[0]
            start_time, end_time = env_manager.wrapper.select_simulation_period()
            env = env_manager.wrapper.create_environment(
                buildings=[first_building],
                start_time=start_time,
                end_time=end_time
            )
            
            agent = create_agent(
                agent_type=agent_name,
                env=env,
                config=sim_config
            )
            
            # Load trained model
            model_path = experiment_dir / 'models' / 'sequential' / f"{agent_name}_average.pkl"
            if not model_path.exists():
                # Try best model from first building
                model_path = experiment_dir / 'models' / 'sequential' / f"{agent_name}_{first_building}_best.pkl"
            
            if model_path.exists():
                logger.info(f"[LOAD] Loading model from {model_path}")
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    logger.info(f"[DEBUG] Model data keys: {model_data.keys()}")
                    
                    if model_data.get('agent_state') and hasattr(agent.algorithm, 'load_state'):
                        agent.algorithm.load_state(model_data['agent_state'])
                        logger.info(f"[OK] Model loaded successfully")
                    else:
                        if not model_data.get('agent_state'):
                            logger.warning(f"[WARN] No 'agent_state' in model file")
                        if not hasattr(agent.algorithm, 'load_state'):
                            logger.warning(f"[WARN] Algorithm does not have 'load_state' method")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning(f"[WARN] No saved model found at {model_path}")
            
            agents[agent_name] = agent
            logger.info(f"[OK] Agent ready: {agent_name}")
            
        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize agent {agent_name}: {e}")
            raise
    
    # Run testing phase
    logger.info("\n[PHASE] TESTING")
    logger.info("=" * 80)
    
    testing_results = {}
    
    for agent_name, agent in agents.items():
        logger.info(f"\n[TEST] Testing agent: {agent_name}")
        logger.info("-" * 80)
        
        try:
            tester = ValidationTester(
                agent=agent,
                env_manager=env_manager,
                config=benchmark_config,
                phase='testing',
                output_dir=experiment_dir,
                logger_instance=logger
            )
            
            # Set training results if available
            if training_results.get(agent_name):
                tester.set_training_results(training_results[agent_name])
            
            # Run testing (model already loaded above)
            results = tester.run(model_path=None)
            testing_results[agent_name] = results
            
            logger.info(f"[OK] Testing complete for {agent_name}")
            
        except Exception as e:
            logger.error(f"[FAIL] Testing failed for {agent_name}: {e}")
            logger.exception("Full traceback:")
            testing_results[agent_name] = {'error': str(e)}
    
    # Save testing results
    logger.info("\n[SAVE] Saving testing results...")
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    testing_file = results_dir / 'testing_results.json'
    with open(testing_file, 'w') as f:
        json.dump(testing_results, f, indent=2, default=str)
    logger.info(f"[OK] Testing results saved to {testing_file}")
    
    # Run evaluation
    logger.info("\n[PHASE] EVALUATION")
    logger.info("=" * 80)
    
    try:
        # Import here to avoid import errors
        eval_dir = Path(__file__).parent / 'benchmark_evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        try:
            from benchmark_evaluator import BenchmarkEvaluator
        finally:
            if str(eval_dir) in sys.path:
                sys.path.remove(str(eval_dir))
        
        evaluator = BenchmarkEvaluator(
            experiment_dir=experiment_dir,
            logger_instance=logger
        )
        
        evaluation_results = evaluator.evaluate(
            benchmark_config=benchmark_config,
            training_results=training_results,
            validation_results={},
            testing_results=testing_results
        )
        
        logger.info("[OK] Evaluation complete")
        
    except Exception as e:
        logger.error(f"[FAIL] Evaluation failed: {e}")
        logger.exception("Full traceback:")
        evaluation_results = {}
    
    # Run visualization
    logger.info("\n[PHASE] VISUALIZATION")
    logger.info("=" * 80)
    
    try:
        # Import here to avoid import errors
        eval_dir = Path(__file__).parent / 'benchmark_evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        try:
            from benchmark_visualizer import BenchmarkVisualizer
        finally:
            if str(eval_dir) in sys.path:
                sys.path.remove(str(eval_dir))
        
        visualizer = BenchmarkVisualizer(
            experiment_dir=experiment_dir,
            logger_instance=logger
        )
        
        visualizer.generate_all(evaluation_results)
        
        logger.info("[OK] Visualization complete")
        
    except Exception as e:
        logger.error(f"[FAIL] Visualization failed: {e}")
        logger.exception("Full traceback:")
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PHASE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved in: {experiment_dir}")
    logger.info("")


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python3 -m hems.core.run_testing_only <experiment_dir> <config.yaml>")
        print()
        print("Example:")
        print("  python3 -m hems.core.run_testing_only DQN_Sequential_0_20251120_112930 benchmark_configs/example_simple_sequential0.yaml")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    config_path = sys.argv[2]
    
    run_testing_only(experiment_dir, config_path)


if __name__ == '__main__':
    main()