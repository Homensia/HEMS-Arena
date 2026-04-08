#==============================
#hems/core/benchmark_runner.py
#==============================
"""
Benchmark Runner - Main Orchestrator for HEMS Benchmarks 

This is the main entry point that coordinates:
1. Configuration loading
2. Training phase (sequential/parallel/both)
3. Validation phase (optional)
4. Testing phase
5. Evaluation and visualization

Key Features:
- Clean separation of concerns
- Proper seed management per phase
- Comprehensive logging
- Error handling and recovery
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging(output_dir: Path, benchmark_name: str) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        output_dir: Output directory
        benchmark_name: Name of benchmark
        
    Returns:
        Logger instance
    """
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('BenchmarkRunner')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'benchmark_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for benchmark: {benchmark_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """
    Main orchestrator for HEMS benchmark execution.
    
    Coordinates all phases:
    1. Load configuration
    2. Initialize components
    3. Run training
    4. Run validation (optional)
    5. Run testing
    6. Generate evaluation and visualizations
    """
    
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        
        # Load configuration
        from hems.core.yaml_loader import YAMLConfigLoader
        self.yaml_loader = YAMLConfigLoader()
        self.benchmark_config = self.yaml_loader.load(self.config_path)
        
        # Setup output directory
        self.output_dir = Path(self.benchmark_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.experiment_dir, self.benchmark_config.name)
        
        # Log configuration
        self._log_configuration()
        
        # Check building overlap
        self._check_building_overlap()
        
        # Initialize components (will be done in run())
        self.env_manager = None
        self.agents = {}
        self.training_results = {}
        self.validation_results = {}
        self.testing_results = {}
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment identifier from the benchmark name and timestamp.

        Returns:
            A string in the format ``{benchmark_name}_{YYYYMMDD_HHMMSS}``.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.benchmark_config.name}_{timestamp}"
    
    def _log_configuration(self):
        """Log benchmark configuration details to the logger.

        Outputs the benchmark name, description, author, tags, seed,
        output directory, training/validation/testing settings, and
        the list of enabled agents.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"BENCHMARK: {self.benchmark_config.name}")
        self.logger.info("=" * 80)
        self.logger.info(f"Description: {self.benchmark_config.description}")
        if self.benchmark_config.author:
            self.logger.info(f"Author: {self.benchmark_config.author}")
        if self.benchmark_config.tags:
            self.logger.info(f"Tags: {', '.join(self.benchmark_config.tags)}")
        self.logger.info(f"Seed: {self.benchmark_config.seed}")
        self.logger.info(f"Output: {self.experiment_dir}")
        self.logger.info("")
        
        # Training config
        self.logger.info(f"Training Mode: {self.benchmark_config.training.mode}")
        self.logger.info(f"Training Episodes: {self.benchmark_config.training.episodes}")
        self.logger.info(f"Training Buildings: {self.benchmark_config.training.buildings.ids}")
        
        # Validation config
        if self.benchmark_config.validation.enabled:
            self.logger.info(f"Validation: Enabled ({self.benchmark_config.validation.episodes} episodes)")
            if self.benchmark_config.validation.buildings:
                self.logger.info(f"Validation Buildings: {self.benchmark_config.validation.buildings.ids}")
        else:
            self.logger.info("Validation: Disabled")
        
        # Testing config
        if self.benchmark_config.testing.enabled:
            self.logger.info(f"Testing: Enabled ({self.benchmark_config.testing.episodes} episodes)")
            if self.benchmark_config.testing.buildings:
                self.logger.info(f"Testing Buildings: {self.benchmark_config.testing.buildings.ids}")
        else:
            self.logger.info("Testing: Disabled")
        
        # Agents
        enabled_agents = [a['name'] for a in self.benchmark_config.agents if a.get('enabled', True)]
        self.logger.info(f"Agents: {', '.join(enabled_agents)}")
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def _check_building_overlap(self):
        """Check for building overlap across train, validation, and test sets.

        Uses the YAML loader's validation utility to detect whether any
        buildings appear in more than one phase (train/val, train/test,
        val/test). Logs a warning if overlap is found, or an OK message
        if proper separation is maintained.
        """
        overlap = self.yaml_loader.validate_building_overlap(self.benchmark_config)
        
        if overlap['has_overlap']:
            self.logger.warning("[WARN] Building overlap detected:")
            if overlap['train_val_overlap']:
                self.logger.warning(f"  Train/Val overlap: {overlap['train_val_overlap']}")
            if overlap['train_test_overlap']:
                self.logger.warning(f"  Train/Test overlap: {overlap['train_test_overlap']}")
            if overlap['val_test_overlap']:
                self.logger.warning(f"  Val/Test overlap: {overlap['val_test_overlap']}")
        else:
            self.logger.info("[OK] No building overlap - proper separation maintained")
    
    def _initialize_components(self):
        """Initialize the environment manager and all enabled agents.

        Converts the benchmark configuration to a ``SimulationConfig``,
        creates the ``CityLearnEnvironmentManager``, and delegates agent
        initialization to ``_initialize_agents``.

        Raises:
            Exception: If environment manager creation fails.
        """
        self.logger.info("\n[INIT] Initializing components...")
        
        # Convert to SimulationConfig for compatibility
        self.sim_config = self.yaml_loader.to_simulation_config(self.benchmark_config)
        
        # Initialize environment manager (assume it exists)
        # This will use the existing environment infrastructure
        try:
            from hems.environments.citylearn.citylearn_wrapper import CityLearnEnvironmentManager
            self.env_manager = CityLearnEnvironmentManager(self.sim_config)
            self.logger.info("[OK] Environment manager initialized")
        except Exception as e:
            self.logger.error(f"[FAIL] Failed to initialize environment: {e}")
            raise
        
        # Initialize agents
        self._initialize_agents()
        
        self.logger.info("[OK] Component initialization complete\n")
    
    def _initialize_agents(self):
        """Initialize all enabled agents from the benchmark configuration.

        For each enabled agent, creates a CityLearn environment appropriate
        to the training mode (sequential, parallel, or round-robin) and
        instantiates the agent via the legacy adapter factory.

        Raises:
            Exception: If any agent fails to initialize.
        """
        from hems.agents.legacy_adapter import create_agent
        
        enabled_agents = [a for a in self.benchmark_config.agents if a.get('enabled', True)]
        
        for agent_config in enabled_agents:
            agent_name = agent_config['name']
            self.logger.info(f"[INIT] Initializing agent: {agent_name}")
            
            try:
                # Determine environment based on training mode
                if self.benchmark_config.training.mode == 'sequential':
                    # Sequential: use first building only
                    first_building = self.benchmark_config.training.buildings.ids[0]
                    start_time, end_time = self.env_manager.wrapper.select_simulation_period()
                    env = self.env_manager.wrapper.create_environment(
                        buildings=[first_building],
                        start_time=start_time,
                        end_time=end_time
                    )
                elif self.benchmark_config.training.mode == 'parallel':
                    # Parallel (multi-head): use all buildings, enable multi-head for DQN
                    start_time, end_time = self.env_manager.wrapper.select_simulation_period()
                    env = self.env_manager.wrapper.create_environment(
                        buildings=self.benchmark_config.training.buildings.ids,
                        start_time=start_time,
                        end_time=end_time
                    )
                    # Enable multi-head for DQN
                    if agent_name == 'dqn':
                        self.sim_config.dqn_config['multi_head'] = True
                else:
                    # Round-robin: use first building
                    first_building = self.benchmark_config.training.buildings.ids[0]
                    start_time, end_time = self.env_manager.wrapper.select_simulation_period()
                    env = self.env_manager.wrapper.create_environment(
                        buildings=[first_building],
                        start_time=start_time,
                        end_time=end_time
                    )
                
                # Create agent using legacy adapter
                agent = create_agent(
                    agent_type=agent_name,
                    env=env,
                    config=self.sim_config
                )
                self.agents[agent_name] = agent
                self.logger.info(f"[OK] Agent initialized: {agent_name}")
                
            except Exception as e:
                self.logger.error(f"[FAIL] Failed to initialize agent {agent_name}: {e}")
                raise
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete benchmark pipeline end-to-end.

        Runs all phases in order: component initialization, training,
        validation (if enabled), testing (if enabled), evaluation,
        visualization, and result persistence.

        Returns:
            A dict containing the full evaluation results produced by
            the ``BenchmarkEvaluator``.

        Raises:
            Exception: If any critical phase (initialization, training,
                evaluation) fails. The exception is logged with a full
                traceback before being re-raised.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING BENCHMARK EXECUTION")
        self.logger.info("=" * 80 + "\n")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Phase 1: Training
            self.logger.info("\n[PHASE 1] TRAINING")
            self.logger.info("=" * 80)
            self.training_results = self._run_training()
            
            # Phase 2: Validation (optional)
            if self.benchmark_config.validation.enabled:
                self.logger.info("\n[PHASE 2] VALIDATION")
                self.logger.info("=" * 80)
                self.validation_results = self._run_validation()
            
            # Phase 3: Testing
            if self.benchmark_config.testing.enabled:
                self.logger.info("\n[PHASE 3] TESTING")
                self.logger.info("=" * 80)
                self.testing_results = self._run_testing()
            
            # Phase 4: Evaluation
            self.logger.info("\n[PHASE 4] EVALUATION")
            self.logger.info("=" * 80)
            evaluation_results = self._run_evaluation()
            
            # Phase 5: Visualization
            self.logger.info("\n[PHASE 5] VISUALIZATION")
            self.logger.info("=" * 80)
            self._run_visualization(evaluation_results)
            
            # Phase 6: Save results
            self._save_results(evaluation_results)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("BENCHMARK EXECUTION COMPLETE")
            self.logger.info("=" * 80 + "\n")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"\n[FAIL] Benchmark execution failed: {e}")
            self.logger.exception("Full traceback:")
            raise
    
    def _run_training(self) -> Dict[str, Any]:
        """Execute the training phase for all registered agents.

        Creates a trainer for each agent based on the configured training
        mode (sequential, parallel, or both) and runs training. Errors
        for individual agents are captured without aborting the remaining
        agents.

        Returns:
            A dict mapping agent names to result dicts, each containing
            a ``'status'`` key (``'completed'`` or ``'failed'``) and
            either ``'results'`` or ``'error'``.
        """
        from hems.core.training_modes import create_trainer
        
        all_results = {}
        
        for agent_name, agent in self.agents.items():
            self.logger.info(f"\n[TRAIN] Training agent: {agent_name}")
            self.logger.info("-" * 80)
            
            try:
                # Create trainer
                trainer = create_trainer(
                    mode=self.benchmark_config.training.mode,
                    agent=agent,
                    env_manager=self.env_manager,
                    config=self.benchmark_config,
                    output_dir=self.experiment_dir,
                    logger_instance=self.logger
                )
                
                # Run training
                training_result = trainer.train()
                all_results[agent_name] = {
                    'status': 'completed',
                    'results': training_result
                }
                
                self.logger.info(f"[OK] Training complete for {agent_name}")
                
            except Exception as e:
                self.logger.error(f"[FAIL] Training failed for {agent_name}: {e}")
                all_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return all_results
    
    def _run_validation(self) -> Dict[str, Any]:
        """Execute the validation phase for all registered agents.

        Loads the best trained model for each agent and evaluates it on
        the validation building set. Training results are forwarded to the
        validator when available. Errors for individual agents are captured
        without aborting the remaining agents.

        Returns:
            A dict mapping agent names to their validation result dicts.
            Failed agents have a ``'status': 'failed'`` entry with an
            ``'error'`` message.
        """
        from hems.core.validation_testing import ValidationTester
        
        all_results = {}
        
        for agent_name, agent in self.agents.items():
            self.logger.info(f"\n[VALIDATION] Validating agent: {agent_name}")
            self.logger.info("-" * 80)
            
            try:
                # Find best model path
                model_path = self._find_best_model(agent_name, 'sequential')
                
                # Create validation tester
                tester = ValidationTester(
                    agent=agent,
                    env_manager=self.env_manager,
                    config=self.benchmark_config,
                    phase="validation",
                    output_dir=self.experiment_dir,
                    logger_instance=self.logger
                )
                
                # Set training results if available
                if hasattr(self, 'training_results') and agent_name in self.training_results:
                    tester.set_training_results(self.training_results[agent_name])
                
                # Run validation
                validation_result = tester.run(model_path)
                all_results[agent_name] = validation_result
                
                self.logger.info(f"[OK] Validation complete for {agent_name}")
                
            except Exception as e:
                self.logger.error(f"[FAIL] Validation failed for {agent_name}: {e}")
                all_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return all_results
    
    def _run_testing(self) -> Dict[str, Any]:
        """Execute the testing phase for all registered agents.

        Loads the best trained model for each agent and evaluates it on
        the testing building set. Training results are forwarded to the
        tester when available. Errors for individual agents are captured
        without aborting the remaining agents.

        Returns:
            A dict mapping agent names to their testing result dicts.
            Failed agents have a ``'status': 'failed'`` entry with an
            ``'error'`` message.
        """
        from hems.core.validation_testing import ValidationTester
        
        all_results = {}
        
        for agent_name, agent in self.agents.items():
            self.logger.info(f"\n[TESTING] Testing agent: {agent_name}")
            self.logger.info("-" * 80)
            
            try:
                # Find best model path
                model_path = self._find_best_model(agent_name, self.benchmark_config.training.mode)
                
                # Create testing tester
                tester = ValidationTester(
                    agent=agent,
                    env_manager=self.env_manager,
                    config=self.benchmark_config,
                    phase="testing",
                    output_dir=self.experiment_dir,
                    logger_instance=self.logger
                )
                
                # Set training results if available
                if hasattr(self, 'training_results') and agent_name in self.training_results:
                    tester.set_training_results(self.training_results[agent_name])
                
                # Run testing
                testing_result = tester.run(model_path)
                all_results[agent_name] = testing_result
                
                self.logger.info(f"[OK] Testing complete for {agent_name}")
                
            except Exception as e:
                self.logger.error(f"[FAIL] Testing failed for {agent_name}: {e}")
                all_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return all_results
    
    def _find_best_model(self, agent_name: str, mode: str) -> Optional[Path]:
        """Find the best saved model file for a given agent and training mode.

        Searches the experiment's ``models/`` directory. For sequential
        mode, it prefers the averaged model (``{agent}_average.pkl``)
        and falls back to any ``*_best.pkl`` file. For parallel mode, it
        looks for ``{agent}_best.pkl``.

        Args:
            agent_name: Name of the agent whose model to locate.
            mode: Training mode used to determine the search subdirectory.
                One of ``'sequential'``, ``'parallel'``, or ``'both'``.

        Returns:
            Path to the best model file, or ``None`` if no model was found.
        """
        models_dir = self.experiment_dir / 'models'
        
        # Check sequential
        if mode in ['sequential', 'both']:
            # Look for average model first
            avg_model = models_dir / 'sequential' / f"{agent_name}_average.pkl"
            if avg_model.exists():
                return avg_model
            
            # Otherwise find any best model
            seq_dir = models_dir / 'sequential'
            if seq_dir.exists():
                best_models = list(seq_dir.glob(f"{agent_name}_*_best.pkl"))
                if best_models:
                    return best_models[0]  # Return first best model
        
        # Check parallel
        if mode in ['parallel', 'both']:
            par_model = models_dir / 'parallel' / f"{agent_name}_best.pkl"
            if par_model.exists():
                return par_model
        
        self.logger.warning(f"No best model found for {agent_name}")
        return None
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """Execute the evaluation phase by aggregating and scoring all results.

        Temporarily adds the ``benchmark_evaluation`` subpackage to
        ``sys.path``, creates a ``BenchmarkEvaluator``, and runs it
        against the collected training, validation, and testing results.

        Returns:
            A dict containing the complete evaluation results including
            aggregated statistics and per-agent scores.
        """
        # Import directly from module files, not from package __init__
        import sys
        from pathlib import Path
        
        # Add the benchmark_evaluation directory to path temporarily
        eval_dir = Path(__file__).parent / 'benchmark_evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        try:
            from benchmark_evaluator import BenchmarkEvaluator
        finally:
            # Remove from path after import
            if str(eval_dir) in sys.path:
                sys.path.remove(str(eval_dir))
        
        self.logger.info("[EVALUATION] Aggregating results and computing statistics")
        
        # Create evaluator
        evaluator = BenchmarkEvaluator(
            experiment_dir=self.experiment_dir,
            logger_instance=self.logger
        )
        
        # Run evaluation
        evaluation_results = evaluator.evaluate(
            benchmark_config=self.benchmark_config,
            training_results=self.training_results,
            validation_results=self.validation_results,
            testing_results=self.testing_results
        )
        
        self.logger.info("[OK] Evaluation complete")
        
        return evaluation_results
    
    def _run_visualization(self, evaluation_results: Dict[str, Any]):
        """Generate benchmark visualizations from evaluation results.

        Temporarily adds the ``benchmark_evaluation`` subpackage to
        ``sys.path``, creates a ``BenchmarkVisualizer``, and generates
        all configured plots. Visualization failures are logged but do
        not propagate as exceptions.

        Args:
            evaluation_results: The evaluation results dict produced by
                ``_run_evaluation``.
        """
        import sys
        from pathlib import Path
        
        # Add the benchmark_evaluation directory to path temporarily
        eval_dir = Path(__file__).parent / 'benchmark_evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        try:
            from benchmark_visualizer import BenchmarkVisualizer
        finally:
            if str(eval_dir) in sys.path:
                sys.path.remove(str(eval_dir))
        
        self.logger.info("[VIZ] Generating plots")
        
        try:
            visualizer = BenchmarkVisualizer(
                experiment_dir=self.experiment_dir,
                logger_instance=self.logger
            )
            
            plot_files = visualizer.generate_all(evaluation_results)
            
            self.logger.info(f"[OK] Generated {len(plot_files)} plots")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Visualization failed: {e}")
            self.logger.exception("Full traceback:")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results and configuration to JSON files.

        Writes two files to the experiment directory:
        - ``benchmark_results.json``: the full results dict.
        - ``config.json``: a summary of the benchmark configuration
          (name, description, seed, training settings, agent list).

        Args:
            results: The complete results dictionary to persist. Values
                that are not JSON-serializable are converted via ``str()``.
        """
        # Save JSON results
        results_file = self.experiment_dir / 'benchmark_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"[SAVE] Results saved to: {results_file}")
        
        # Save configuration
        config_file = self.experiment_dir / 'config.json'
        config_dict = {
            'name': self.benchmark_config.name,
            'description': self.benchmark_config.description,
            'seed': self.benchmark_config.seed,
            'training_mode': self.benchmark_config.training.mode,
            'training_episodes': self.benchmark_config.training.episodes,
            'training_buildings': self.benchmark_config.training.buildings.ids,
            'agents': [a['name'] for a in self.benchmark_config.agents if a.get('enabled', True)]
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"[SAVE] Configuration saved to: {config_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for benchmark execution from the command line.

    Supports two modes of operation:
    - **Full run** (default): loads a YAML config, creates a
      ``BenchmarkRunner``, and executes the complete pipeline.
    - **Plot-only** (``--plot-only``): reloads saved evaluation results
      from a previous experiment directory and regenerates visualizations.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HEMS Benchmark Runner')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to benchmark configuration YAML file'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Skip training/validation/testing and only regenerate plots and metrics from saved results'
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=None,
        help='Experiment directory to load results from (required with --plot-only)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.plot_only:
            # Plot-only mode: load existing results and regenerate visualizations
            if not args.experiment_dir:
                print("[ERROR] --experiment-dir is required when using --plot-only")
                return 1
            
            from pathlib import Path
            experiment_dir = Path(args.experiment_dir)
            
            if not experiment_dir.exists():
                print(f"[ERROR] Experiment directory not found: {experiment_dir}")
                return 1
            
            print(f"\n[PLOT-ONLY MODE] Loading results from: {experiment_dir}")
            
            # Load saved results
            results_file = experiment_dir / 'evaluation' / 'evaluation_results.json'
            if not results_file.exists():
                print(f"[ERROR] Evaluation results not found: {results_file}")
                print("[INFO] Please run a complete benchmark first before using --plot-only")
                return 1
            
            import json
            with open(results_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            
            # Setup logging for plot-only mode
            logger = setup_logging(experiment_dir, "plot_only")
            logger.info("[PLOT-ONLY] Regenerating visualizations from saved results")
            
            # Regenerate visualizations
            import sys
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
            
            plot_files = visualizer.generate_all(evaluation_results)
            
            print(f"\n[SUCCESS] Regenerated {len(plot_files)} plots")
            print(f"Plots saved to: {experiment_dir / 'plots'}")
            
            return 0
        
        else:
            # Normal mode: full benchmark execution
            runner = BenchmarkRunner(args.config)
            results = runner.run()
            
            print("\n[SUCCESS] Benchmark completed successfully!")
            print(f"Results saved to: {runner.experiment_dir}")
            
            return 0
        
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())