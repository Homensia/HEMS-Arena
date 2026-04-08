"""
Benchmark Evaluator - Clean Rebuild
Aggregates training/validation/testing results and computes final statistics.

Key Features:
- Aggregates results from all phases
- Cross-agent comparisons
- Summary statistics
- Generates evaluation_results.json and evaluation_summary.txt
- INTEGRATION: Performance scorer with weighted category scoring
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

try:
    from performance_scorer import PerformanceScorer
except ImportError:
    try:
        from .performance_scorer import PerformanceScorer
    except ImportError:
        PerformanceScorer = None

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Aggregates and evaluates complete benchmark results.
    """
    
    def __init__(
        self,
        experiment_dir: Path,
        logger_instance: logging.Logger
    ):
        """
        Initialize evaluator.
        
        Args:
            experiment_dir: Experiment directory
            logger_instance: Logger
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger_instance
        self.evaluation_dir = self.experiment_dir / 'evaluation'
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        benchmark_config,
        training_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        testing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate complete benchmark results.
        
        Args:
            benchmark_config: Benchmark configuration
            training_results: Training phase results
            validation_results: Validation phase results
            testing_results: Testing phase results
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("[EVAL] Aggregating benchmark results")
        
        # Aggregate per-agent results
        agent_results = self._aggregate_agent_results(
            training_results,
            validation_results,
            testing_results
        )
        
        # Compute summary statistics
        summary = self._compute_summary(agent_results, benchmark_config)
        
        # Compute OPI scores (MAUT-based)
        opi_results = self._compute_opi_scores(
            agent_results,
            testing_results,
            benchmark_config
        )
        
        # Statistical analysis (if multiple agents)
        statistical_tests = None
        if len(agent_results) >= 2:
            import sys
            from pathlib import Path
            
            # Add the benchmark_evaluation directory to path temporarily
            eval_dir = Path(__file__).parent
            if str(eval_dir) not in sys.path:
                sys.path.insert(0, str(eval_dir))
            
            try:
                from statistical_analyzer import StatisticalAnalyzer
            finally:
                if str(eval_dir) in sys.path:
                    sys.path.remove(str(eval_dir))
            
            analyzer = StatisticalAnalyzer(self.logger)
            statistical_tests = analyzer.analyze(agent_results)
        
        # Build evaluation results
        evaluation_results = {
            'benchmark_name': benchmark_config.name,
            'experiment_id': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'seed': benchmark_config.seed,
                'training_mode': benchmark_config.training.mode,
                'training_episodes': benchmark_config.training.episodes,
                'training_buildings': benchmark_config.training.buildings.ids,
                'validation_enabled': benchmark_config.validation.enabled,
                'testing_enabled': benchmark_config.testing.enabled,
            },
            'agent_results': agent_results,
            'summary': summary,
            'opi_results': opi_results,  # MAUT-based OPI scores
            'statistical_tests': statistical_tests,
        }
        
        # Save results
        self._save_results(evaluation_results)
        
        self.logger.info("[OK] Evaluation complete")
        
        return evaluation_results
    
    def _aggregate_agent_results(
        self,
        training_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        testing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results for each agent.
        
        Returns:
            Dict mapping agent_name -> aggregated results
        """
        agent_results = {}
        
        # Get all agent names
        agent_names = set()
        if training_results:
            agent_names.update(training_results.keys())
        if validation_results:
            agent_names.update(validation_results.keys())
        if testing_results:
            agent_names.update(testing_results.keys())
        
        for agent_name in agent_names:
            agent_results[agent_name] = {
                'name': agent_name,
                'training': training_results.get(agent_name, {}),
                'validation': validation_results.get(agent_name, {}),
                'testing': testing_results.get(agent_name, {}),
            }
        
        return agent_results
    
    def _compute_summary(
        self,
        agent_results: Dict[str, Any],
        config
    ) -> Dict[str, Any]:
        """
        Compute summary statistics across all agents.
        
        Returns:
            Summary dictionary with performance scores
        """
        summary = {
            'total_agents': len(agent_results),
            'completed_agents': 0,
            'failed_agents': 0,
            'best_agent': None,
            'best_cost_savings': None,
            'best_reward': None,
            'performance_scores': None,
        }
        
        best_cost_savings = float('-inf')
        best_reward = float('-inf')
        best_agent_cost = None
        best_agent_reward = None
        
        # Collect KPIs for performance scoring
        agent_kpis_dict = {}
        
        for agent_name, results in agent_results.items():
            # Check if completed
            testing = results.get('testing', {})
            if testing.get('status') == 'completed':
                summary['completed_agents'] += 1
                
                # Get savings
                savings = testing.get('savings', {})
                cost_savings = savings.get('cost_savings_percent', 0)
                
                if cost_savings > best_cost_savings:
                    best_cost_savings = cost_savings
                    best_agent_cost = agent_name
                
                # Get reward
                agent_kpis = testing.get('agent_kpis', {})
                avg_reward = self._extract_value(agent_kpis.get('avg_reward', 0))
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_agent_reward = agent_name
                
                # Collect KPIs for scoring
                # Extract flat values from potentially nested dicts
                flat_kpis = {}
                for key, val in agent_kpis.items():
                    flat_kpis[key] = self._extract_value(val)
                agent_kpis_dict[agent_name] = flat_kpis
            else:
                summary['failed_agents'] += 1
        
        summary['best_agent'] = best_agent_cost or best_agent_reward
        summary['best_cost_savings'] = best_cost_savings if best_cost_savings != float('-inf') else None
        summary['best_reward'] = best_reward if best_reward != float('-inf') else None
        
        summary['performance_scores'] = None  
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save complete JSON
        json_file = self.evaluation_dir / 'evaluation_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"[SAVE] Evaluation JSON: {json_file}")
        
        # Save important/filtered JSON (without observations and heavy data)
        important_file = self.evaluation_dir / 'evaluation_important_results.json'
        self._save_important_results(results, important_file)
        self.logger.info(f"[SAVE] Important results JSON: {important_file}")
        
        # Save text summary
        txt_file = self.evaluation_dir / 'evaluation_summary.txt'
        self._write_text_summary(results, txt_file)
        
        self.logger.info(f"[SAVE] Evaluation summary: {txt_file}")
    
    def _save_important_results(self, results: Dict[str, Any], filepath: Path):
        """
        Save filtered results without heavy data (observations, episode_data, large arrays).
        
        Keeps only:
        - Configuration
        - Summary
        - KPIs (agent and baseline)
        - Savings
        - Statistical tests
        - Basic metadata
        
        Excludes:
        - episode_data (observations, actions, rewards arrays)
        - aggregated data (large timestep arrays)
        - training histories
        """
        filtered_results = {
            'benchmark_name': results['benchmark_name'],
            'experiment_id': results['experiment_id'],
            'timestamp': results['timestamp'],
            'configuration': results['configuration'],
            'summary': results['summary'],
            'agent_results': {}
        }
        
        # Filter agent results
        for agent_name, agent_data in results['agent_results'].items():
            filtered_agent = {
                'name': agent_data.get('name', agent_name)
            }
            
            # Training phase (keep only summary stats, no episode data)
            if 'training' in agent_data:
                training = agent_data['training']
                filtered_training = {
                    'status': training.get('status'),
                    'mode': training.get('mode'),
                    'buildings': training.get('buildings'),
                    'total_episodes': training.get('total_episodes')
                }
                
                # Keep results summary but filter out heavy data
                if 'results' in training:
                    train_results = training['results']
                    filtered_train_results = {}
                    
                    # For sequential mode
                    if 'buildings' in train_results and isinstance(train_results['buildings'], dict):
                        filtered_train_results['buildings'] = {}
                        for bldg_id, bldg_data in train_results['buildings'].items():
                            filtered_train_results['buildings'][bldg_id] = {
                                'episodes': bldg_data.get('episodes'),
                                'final_reward': bldg_data.get('final_reward'),
                                'best_reward': bldg_data.get('best_reward'),
                                'avg_last_10': bldg_data.get('avg_last_10')
                                # Exclude: rewards array, losses array
                            }
                    
                    # For parallel mode
                    if 'episodes' in train_results:
                        filtered_train_results['episodes'] = train_results['episodes']
                    if 'final_reward' in train_results:
                        filtered_train_results['final_reward'] = train_results['final_reward']
                    if 'best_reward' in train_results:
                        filtered_train_results['best_reward'] = train_results['best_reward']
                    # Exclude: rewards array
                    
                    filtered_training['results'] = filtered_train_results
                
                filtered_agent['training'] = filtered_training
            
            # Validation phase
            if 'validation' in agent_data:
                validation = agent_data['validation']
                filtered_validation = {
                    'status': validation.get('status'),
                    'phase': validation.get('phase'),
                    'buildings': validation.get('buildings'),
                    'episodes': validation.get('episodes'),
                    'seed': validation.get('seed'),
                    'agent_kpis': validation.get('agent_kpis', {}),
                    'baseline_kpis': validation.get('baseline_kpis', {}),
                    'savings': validation.get('savings', {})
                    # Exclude: agent_data, baseline_data (with episode_data and aggregated)
                }
                filtered_agent['validation'] = filtered_validation
            
            # Testing phase (most important)
            if 'testing' in agent_data:
                testing = agent_data['testing']
                filtered_testing = {
                    'status': testing.get('status'),
                    'phase': testing.get('phase'),
                    'buildings': testing.get('buildings'),
                    'episodes': testing.get('episodes'),
                    'seed': testing.get('seed'),
                    'agent_kpis': testing.get('agent_kpis', {}),
                    'baseline_kpis': testing.get('baseline_kpis', {}),
                    'savings': testing.get('savings', {})
                    # Exclude: agent_data, baseline_data (with episode_data and aggregated)
                }
                
                # Optionally keep basic stats from agent_data/baseline_data
                if 'agent_data' in testing:
                    agent_data_info = testing['agent_data']
                    filtered_testing['agent_stats'] = {
                        'avg_reward': agent_data_info.get('avg_reward'),
                        'std_reward': agent_data_info.get('std_reward'),
                        'episodes': agent_data_info.get('episodes')
                        # Exclude: episode_data, aggregated, rewards array
                    }
                
                if 'baseline_data' in testing:
                    baseline_data_info = testing['baseline_data']
                    filtered_testing['baseline_stats'] = {
                        'avg_reward': baseline_data_info.get('avg_reward'),
                        'std_reward': baseline_data_info.get('std_reward'),
                        'episodes': baseline_data_info.get('episodes')
                        # Exclude: episode_data, aggregated, rewards array
                    }
                
                filtered_agent['testing'] = filtered_testing
            
            filtered_results['agent_results'][agent_name] = filtered_agent
        
        # Add statistical tests if present
        if 'statistical_tests' in results:
            filtered_results['statistical_tests'] = results['statistical_tests']
        
        # Add OPI results if present
        if 'opi_results' in results:
            filtered_results['opi_results'] = results['opi_results']
        
        # Save filtered results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2, default=str)
    
    
    def _write_text_summary(self, results: Dict[str, Any], filepath: Path):
        """Write human-readable summary."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BENCHMARK EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Benchmark: {results['benchmark_name']}\n")
            f.write(f"Experiment ID: {results['experiment_id']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            config = results['configuration']
            f.write(f"Seed: {config['seed']}\n")
            f.write(f"Training Mode: {config['training_mode']}\n")
            f.write(f"Training Episodes: {config['training_episodes']}\n")
            f.write(f"Training Buildings: {', '.join(config['training_buildings'])}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = results['summary']
            f.write(f"Total Agents: {summary['total_agents']}\n")
            f.write(f"Completed: {summary['completed_agents']}\n")
            f.write(f"Failed: {summary['failed_agents']}\n")
            if summary['best_agent']:
                f.write(f"Best Agent: {summary['best_agent']}\n")
            if summary['best_cost_savings'] is not None:
                f.write(f"Best Cost Savings: {summary['best_cost_savings']:.2f}%\n")
            f.write("\n")
            
            # Per-agent results
            f.write("AGENT RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for agent_name, agent_data in results['agent_results'].items():
                f.write(f"Agent: {agent_name}\n")
                f.write("-" * 80 + "\n")
                
                # Testing results
                testing = agent_data.get('testing', {})
                if testing.get('status') == 'completed':
                    agent_kpis = testing.get('agent_kpis', {})
                    baseline_kpis = testing.get('baseline_kpis', {})
                    savings = testing.get('savings', {})
                    
                    f.write(f"Testing Episodes: {testing.get('episodes', 'N/A')}\n")
                    f.write(f"Average Reward: {agent_kpis.get('avg_reward', 0):.2f}\n")
                    f.write(f"Total Cost: €{agent_kpis.get('total_cost', 0):.2f}\n")
                    f.write(f"Baseline Cost: €{baseline_kpis.get('total_cost', 0):.2f}\n")
                    f.write(f"Cost Savings: {savings.get('cost_savings_percent', 0):.2f}%\n")
                    f.write(f"PV Self-Consumption: {agent_kpis.get('pv_self_consumption_rate', 0):.1f}%\n")
                    f.write(f"Battery Cycles: {agent_kpis.get('battery_cycles', 0):.2f}\n")
                    f.write(f"Peak Demand: {agent_kpis.get('peak_demand', 0):.2f} kW\n")
                    f.write(f"Peak Reduction: {savings.get('peak_reduction_percent', 0):.2f}%\n")
                else:
                    f.write(f"Status: {testing.get('status', 'unknown')}\n")
                    if 'error' in testing:
                        f.write(f"Error: {testing['error']}\n")
                
                f.write("\n")
            
            # Statistical tests
            if results.get('statistical_tests'):
                f.write("STATISTICAL ANALYSIS\n")
                f.write("=" * 80 + "\n")
                stats = results['statistical_tests']
                
                if 'comparisons' in stats:
                    f.write("\nPairwise Comparisons:\n")
                    for comp in stats['comparisons']:
                        f.write(f"  {comp['agent_a']} vs {comp['agent_b']}:\n")
                        f.write(f"    p-value: {comp.get('p_value', 'N/A')}\n")
                        f.write(f"    Significant: {comp.get('significant', 'N/A')}\n\n")
            
            # OPI Results (MAUT-based)
            if results.get('opi_results') and results['opi_results'].get('rankings'):
                f.write("\n")
                f.write("OPERATIONAL PERFORMANCE INDEX (OPI)\n")
                f.write("=" * 80 + "\n")
                f.write("MAUT-based scoring with baseline-oracle normalization\n")
                f.write("OPI = fraction of achievable improvement (0=baseline, 1=oracle)\n\n")
                
                f.write("Overall Rankings:\n")
                f.write("-" * 80 + "\n")
                for agent_name, opi_score, rank in results['opi_results']['rankings']:
                    opi_result = results['opi_results']['opi_results'][agent_name]
                    regression_info = f" ({opi_result.regression_count} regressions)" if opi_result.regression_count > 0 else ""
                    f.write(f"  {rank}. {agent_name:30s} | OPI: {opi_score:.4f}{regression_info}\n")
                
                f.write("\n")
                f.write("See opi_report.txt for detailed breakdown\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
    
    def _compute_opi_scores(
        self,
        agent_results: Dict[str, Any],
        testing_results: Dict[str, Any],
        benchmark_config: Any
    ) -> Dict[str, Any]:
        """
        Compute Operational Performance Index (OPI) for all agents.
        
        Uses MAUT-based baseline-oracle normalization.
        
        Args:
            agent_results: Aggregated agent results
            testing_results: Raw testing results
            benchmark_config: Benchmark configuration
        
        Returns:
            Dict with OPI results
        """
        if not PerformanceScorer:
            self.logger.warning("[OPI] PerformanceScorer not available")
            return {}
        
        self.logger.info("[OPI] Computing Operational Performance Index")
        
        # DEBUG: Check what we have
        self.logger.info(f"[OPI DEBUG] Agent results keys: {list(agent_results.keys())}")
        for agent_name, results in agent_results.items():
            testing = results.get('testing', {})
            self.logger.info(f"[OPI DEBUG] {agent_name} - testing keys: {list(testing.keys())}")
            self.logger.info(f"[OPI DEBUG] {agent_name} - status: {testing.get('status')}")
        
        # Initialize scorer
        # Try with sensitivity and stats disabled (MAUT scorer), fall back to basic init
        try:
            scorer = PerformanceScorer(
                logger_instance=self.logger,
                enable_sensitivity_analysis=False,  # Prevent infinite recursion!
                enable_statistical_tests=False      # Faster computation
            )
            self.logger.info("[OPI] Initialized MAUT scorer with analysis features disabled")
        except TypeError:
            # Try just sensitivity disabled
            try:
                scorer = PerformanceScorer(
                    logger_instance=self.logger,
                    enable_sensitivity_analysis=False
                )
                self.logger.info("[OPI] Initialized with sensitivity analysis disabled")
            except TypeError:
                # Simple scorer doesn't accept these parameters
                scorer = PerformanceScorer(logger_instance=self.logger)
                self.logger.info("[OPI] Initialized simple scorer")
        
        # Extract KPIs for each agent
        agent_kpis = {}
        for agent_name, results in agent_results.items():
            testing = results.get('testing', {})
            if testing.get('status') == 'completed':
                kpis = testing.get('agent_kpis', {})
                self.logger.info(f"[OPI DEBUG] {agent_name} - found {len(kpis)} agent KPIs")
                
                # Flatten KPIs
                flat_kpis = {}
                for key, val in kpis.items():
                    flat_kpis[key] = self._extract_value(val)
                agent_kpis[agent_name] = flat_kpis
                
                self.logger.info(f"[OPI DEBUG] {agent_name} - flattened to {len(flat_kpis)} KPIs")
                self.logger.info(f"[OPI DEBUG] {agent_name} - sample KPIs: {list(flat_kpis.keys())[:5]}")
        
        if not agent_kpis:
            self.logger.warning("[OPI] No agent KPIs available - cannot compute OPI")
            self.logger.warning("[OPI] Check that agents completed testing and have agent_kpis in results")
            return {}
        
        # Get baseline KPIs
        baseline_kpis = self._get_baseline_kpis(agent_results)
        
        if not baseline_kpis:
            self.logger.warning("[OPI] No baseline KPIs found - cannot compute OPI")
            self.logger.warning("[OPI] Baseline KPIs should be in testing phase results")
            return {}
        
        self.logger.info(f"[OPI] Found baseline with {len(baseline_kpis)} KPIs")
        self.logger.info(f"[OPI] Sample baseline KPIs: {list(baseline_kpis.keys())[:5]}")
        
        # Get oracle KPIs
        oracle_kpis = self._get_oracle_kpis(agent_results, baseline_kpis)
        
        if not oracle_kpis:
            self.logger.warning("[OPI] No oracle KPIs found - cannot compute OPI")
            self.logger.warning("[OPI] Oracle KPIs are required for MAUT-based scoring")
            return {}
        
        self.logger.info(f"[OPI] Found/estimated oracle with {len(oracle_kpis)} KPIs")
        
        # Calculate OPI (MAUT-based with baseline-oracle normalization)
        try:
            opi_results = scorer.calculate_scores(
                agent_kpis=agent_kpis,
                baseline_kpis=baseline_kpis,
                oracle_kpis=oracle_kpis
            )
            
            # Save OPI report
            opi_report = scorer.format_results(opi_results)
            opi_report_path = self.evaluation_dir / 'opi_report.txt'
            with open(opi_report_path, 'w') as f:
                f.write(opi_report)
            self.logger.info(f"[OPI] Saved report to {opi_report_path}")
            
            # Save OPI JSON
            # Try using scorer's export method if available (MAUT scorer)
            if hasattr(scorer, 'export_to_dict'):
                opi_json = scorer.export_to_dict(opi_results)
            else:
                # Fall back to manual serialization (simple scorer)
                opi_json = self._serialize_opi_results(opi_results)
            
            opi_json_path = self.evaluation_dir / 'opi_results.json'
            with open(opi_json_path, 'w') as f:
                json.dump(opi_json, f, indent=2)
            self.logger.info(f"[OPI] Saved JSON to {opi_json_path}")
            
            return opi_results
            
        except Exception as e:
            self.logger.error(f"[OPI] Failed to compute OPI: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _get_baseline_kpis(self, agent_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract baseline KPIs from agent results.
        
        Checks multiple locations:
        1. testing.baseline_kpis (preferred)
        2. testing.baseline_data.kpis
        3. Any agent with 'baseline' in name
        
        Returns:
            Dict of baseline KPIs
        """
        self.logger.info("[OPI] Searching for baseline KPIs...")
        
        # Try to get baseline from any agent's testing results
        for agent_name, results in agent_results.items():
            testing = results.get('testing', {})
            
            self.logger.info(f"[OPI DEBUG] Checking {agent_name} for baseline...")
            self.logger.info(f"[OPI DEBUG] Testing keys: {list(testing.keys())}")
            
            if testing.get('status') == 'completed':
                # Method 1: Direct baseline_kpis
                baseline_kpis = testing.get('baseline_kpis', {})
                if baseline_kpis:
                    self.logger.info(f"[OPI] Found baseline_kpis in {agent_name}'s testing (Method 1)")
                    flat_kpis = {}
                    for key, val in baseline_kpis.items():
                        flat_kpis[key] = self._extract_value(val)
                    return flat_kpis
                
                # Method 2: baseline_data.kpis
                baseline_data = testing.get('baseline_data', {})
                if baseline_data and 'kpis' in baseline_data:
                    self.logger.info(f"[OPI] Found baseline in baseline_data.kpis (Method 2)")
                    baseline_kpis = baseline_data['kpis']
                    flat_kpis = {}
                    for key, val in baseline_kpis.items():
                        flat_kpis[key] = self._extract_value(val)
                    return flat_kpis
        
        # Method 3: Check for agent with 'baseline' in name
        for agent_name, results in agent_results.items():
            if 'baseline' in agent_name.lower():
                testing = results.get('testing', {})
                if testing.get('status') == 'completed':
                    agent_kpis = testing.get('agent_kpis', {})
                    if agent_kpis:
                        self.logger.info(f"[OPI] Using '{agent_name}' as baseline (Method 3)")
                        flat_kpis = {}
                        for key, val in agent_kpis.items():
                            flat_kpis[key] = self._extract_value(val)
                        return flat_kpis
        
        self.logger.warning("[OPI] No baseline KPIs found in any location")
        self.logger.warning("[OPI] Checked: testing.baseline_kpis, testing.baseline_data.kpis, baseline agents")
        return {}
    
    def _get_oracle_kpis(
        self,
        agent_results: Dict[str, Any],
        baseline_kpis: Dict[str, float]
    ) -> Dict[str, float]:
        """Conservative Relative Improvement Reference Oracle (CRIRO)."""
        if not baseline_kpis:
            return {}
        
        self.logger.warning(
            "[OPI] Using Conservative Relative Improvement Reference Oracle (CRIRO). "
            "For publication, compute true optimum using MILP/MPC."
        )
        
        ALPHA = 0.30
        EPSILON = 1e-6
        
        oracle_kpis = {}
        
        # Import KPI definitions
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from performance_scorer import PerformanceScorer, UtilityType
            kpi_definitions = PerformanceScorer.KPI_DEFINITIONS
            has_definitions = True
            self.logger.info("[ORACLE] Successfully loaded KPI definitions from PerformanceScorer")
        except Exception as e:
            self.logger.error(f"[ORACLE] Failed to import KPI definitions: {e}")
            has_definitions = False
            kpi_definitions = {}
        
        # Debug: sample baseline
        sample_metrics = list(baseline_kpis.items())[:5]
        self.logger.info(f"[ORACLE DEBUG] Sample baseline values: {sample_metrics}")
        
        for metric_name, baseline_val in baseline_kpis.items():
            if not np.isfinite(baseline_val):
                self.logger.warning(f"[ORACLE] Non-finite baseline for {metric_name}: {baseline_val}")
                oracle_kpis[metric_name] = baseline_val
                continue
            
            # Get metric type
            if has_definitions and metric_name in kpi_definitions:
                utility_type = kpi_definitions[metric_name].utility_type
                metric_type_str = utility_type.value
            else:
                is_cost = any(p in metric_name.lower() for p in ['cost', 'volatility', 'curtailment', 'gap', 'peak_demand'])
                metric_type_str = 'cost' if is_cost else 'benefit'
            
            # Apply CRIRO with special cases
            if has_definitions and metric_name in kpi_definitions:
                utility_type_enum = kpi_definitions[metric_name].utility_type
                
                if utility_type_enum.value == 'cost':
                    # COST: oracle should be LOWER than baseline
                    if abs(baseline_val) < EPSILON:
                        # Baseline already at zero (optimal) - oracle = baseline
                        oracle_val = baseline_val
                    else:
                        oracle_val = baseline_val * (1.0 - ALPHA)
                
                elif utility_type_enum.value == 'benefit':
                    # BENEFIT: oracle should be HIGHER than baseline
                    if abs(baseline_val) < EPSILON:
                        # Baseline ~zero: oracle slightly better
                        oracle_val = EPSILON * (1 + ALPHA)
                    elif baseline_val < 0:
                        # Negative values (e.g., negative rewards): reduce magnitude
                        oracle_val = baseline_val * (1.0 - ALPHA)
                    else:
                        # Positive values: check if already at theoretical maximum
                        # Detect percentage scale: if baseline in [90, 100] assume percentage
                        if 90.0 <= baseline_val <= 100.0:
                            # Likely percentage at/near maximum
                            if baseline_val >= 99.5:
                                # Already at max - oracle = baseline
                                oracle_val = baseline_val
                            else:
                                # Close gap to 100 by fraction
                                gap = 100.0 - baseline_val
                                oracle_val = baseline_val + (gap * ALPHA)
                        elif 0.90 <= baseline_val <= 1.0:
                            # Normalized scale at/near maximum
                            if baseline_val >= 0.995:
                                # Already at max - oracle = baseline
                                oracle_val = baseline_val
                            else:
                                # Close gap to 1.0 by fraction
                                gap = 1.0 - baseline_val
                                oracle_val = baseline_val + (gap * ALPHA)
                        else:
                            # Standard case: multiplicative improvement
                            oracle_val = baseline_val * (1.0 + ALPHA)
                
                elif utility_type_enum.value == 'target':
                    # TARGET: use predefined value
                    target = kpi_definitions[metric_name].target_value
                    oracle_val = target if target is not None else baseline_val
                
                else:
                    oracle_val = baseline_val * (1.0 + ALPHA)
            
            else:
                # Fallback for undefined metrics
                if metric_type_str == 'cost':
                    oracle_val = baseline_val * (1.0 - ALPHA) if abs(baseline_val) > EPSILON else baseline_val
                else:
                    oracle_val = baseline_val * (1.0 + ALPHA) if abs(baseline_val) > EPSILON else EPSILON * (1 + ALPHA)
            
            oracle_kpis[metric_name] = oracle_val
            
            # Debug problematic metrics
            if metric_name in ['pv_self_sufficiency_ratio', 'renewable_fraction', 'avg_reward', 'optimal_soc_usage', 
                             'cost_volatility', 'pv_curtailment_rate', 'generalization_gap']:
                self.logger.info(
                    f"[ORACLE DEBUG] {metric_name}: "
                    f"type={metric_type_str}, baseline={baseline_val:.4f}, oracle={oracle_val:.4f}"
                )
        
        n_estimated = len(oracle_kpis)
        self.logger.info(f"[OPI] Estimated oracle with CRIRO (α={ALPHA:.2f}, {n_estimated} KPIs)")
        
        # Check violations
        violations = []
        for metric_name in oracle_kpis:
            if metric_name in baseline_kpis:
                b = baseline_kpis[metric_name]
                o = oracle_kpis[metric_name]
                if has_definitions and metric_name in kpi_definitions:
                    utype = kpi_definitions[metric_name].utility_type.value
                    # Only report real violations (not zero = zero)
                    if utype == 'benefit' and o < b - EPSILON:
                        violations.append(f"{metric_name}: oracle={o:.2f} < baseline={b:.2f}")
                    elif utype == 'cost' and o > b + EPSILON:
                        violations.append(f"{metric_name}: oracle={o:.2f} > baseline={b:.2f}")
        
        if violations:
            self.logger.error(f"[ORACLE] VIOLATIONS DETECTED ({len(violations)}):")
            for v in violations[:10]:
                self.logger.error(f"[ORACLE]   {v}")
        
        return oracle_kpis
    
    def _extract_value(self, metric: Any) -> float:
        """
        Extract float value from metric (handles dict/float/None).
        
        Args:
            metric: Metric value in any format
            
        Returns:
            Float value
        """
        if metric is None:
            return 0.0
        elif isinstance(metric, (int, float)):
            return float(metric)
        elif isinstance(metric, dict):
            return float(metric.get('value', 0))
        elif isinstance(metric, str):
            try:
                return float(metric)
            except (ValueError, TypeError):
                return 0.0
        else:
            return 0.0
    
    def _serialize_opi_results(self, opi_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize OPI results to JSON-compatible format.
        Handles both simple scorer dicts and MAUT scorer dataclasses.
        
        Args:
            opi_results: OPI results from PerformanceScorer
            
        Returns:
            JSON-serializable dict
        """
        serialized = {
            'overall_scores': {},
            'category_scores': {},
            'rankings': [],
            'normalized_kpis': {}
        }
        
        # Handle MAUT scorer's 'opi_results' nested structure
        if 'opi_results' in opi_results:
            # MAUT scorer structure
            maut_results = opi_results['opi_results']
            
            # Extract overall scores from MAUT OPIResult objects
            for agent_name, opi_obj in maut_results.items():
                if hasattr(opi_obj, 'opi_score'):
                    serialized['overall_scores'][agent_name] = float(opi_obj.opi_score)
                    
                    # Extract category scores
                    if hasattr(opi_obj, 'category_scores'):
                        cat_dict = {}
                        for cat_score in opi_obj.category_scores:
                            if hasattr(cat_score, 'category_name') and hasattr(cat_score, 'category_utility'):
                                cat_dict[cat_score.category_name] = float(cat_score.category_utility)
                        serialized['category_scores'][agent_name] = cat_dict
            
            # Extract rankings from MAUT structure
            if 'rankings' in opi_results:
                rankings = opi_results['rankings']
                serialized['rankings'] = [
                    {
                        'agent': agent,
                        'score': float(score),
                        'rank': int(rank)
                    }
                    for agent, score, rank in rankings
                ]
        else:
            # Simple scorer structure
            # Overall scores
            if 'overall_scores' in opi_results:
                serialized['overall_scores'] = {
                    agent: float(score) 
                    for agent, score in opi_results['overall_scores'].items()
                }
            
            # Category scores
            if 'category_scores' in opi_results:
                serialized['category_scores'] = {
                    agent: {
                        cat: float(score) 
                        for cat, score in cat_scores.items()
                    }
                    for agent, cat_scores in opi_results['category_scores'].items()
                }
            
            # Rankings
            if 'rankings' in opi_results:
                serialized['rankings'] = [
                    {
                        'agent': agent,
                        'score': float(score),
                        'rank': int(rank)
                    }
                    for agent, score, rank in opi_results['rankings']
                ]
            
            # Normalized KPIs
            if 'normalized_kpis' in opi_results:
                serialized['normalized_kpis'] = {
                    agent: {
                        kpi: float(val) 
                        for kpi, val in kpis.items()
                    }
                    for agent, kpis in opi_results['normalized_kpis'].items()
                }
        
        return serialized