# =======================================================
# hems/core/benchmark_evaluation/benchmark_visualizer.py
# =======================================================
"""
Benchmark Visualizer - Clean Rebuild
Generates comprehensive visualizations from benchmark results.

Key Features:
- 20+ plots covering all phases
- CRITICAL FIX: _extract_value() helper to unwrap dict metrics
- Handles both dict and float metrics properly
- Training curves, performance comparisons, KPI heatmaps, radar charts
- INTEGRATION: PlotGenerator with all 20 specification plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    from plot_generator import PlotGenerator
except ImportError:
    try:
        from .plot_generator import PlotGenerator
    except ImportError:
        PlotGenerator = None

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BenchmarkVisualizer:
    """
    Generates visualizations from benchmark results.
    """
    
    def __init__(
        self,
        experiment_dir: Path,
        logger_instance: logging.Logger
    ):
        """Initialize visualizer."""
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger_instance
        self.plots_dir = self.experiment_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plot generator if available
        self.plot_generator = PlotGenerator(self.plots_dir, logger_instance) if PlotGenerator else None
    
    def generate_all(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """
        Generate all visualizations.
        
        Args:
            evaluation_results: Complete evaluation results
            
        Returns:
            List of generated plot paths
        """
        self.logger.info("[VIZ] Generating visualizations")
        
        plot_files = []
        
        try:
            # Original plots (backward compatible)
            plot_files.extend(self._plot_training_curves(evaluation_results))
            plot_files.extend(self._plot_performance_comparison(evaluation_results))
            plot_files.extend(self._plot_kpi_heatmap(evaluation_results))
            plot_files.extend(self._plot_radar_chart(evaluation_results))
            plot_files.extend(self._plot_cost_analysis(evaluation_results))
            
            if evaluation_results.get('statistical_tests'):
                plot_files.extend(self._plot_statistical_results(evaluation_results))
            
            #OPI performance scoring plots
            if evaluation_results.get('opi_results'):
                plot_files.extend(self._plot_opi_scores(evaluation_results))
            
            # NEW: Comprehensive plots from specification (20 plots)
            if self.plot_generator:
                try:
                    agent_results = evaluation_results.get('agent_results', {})
                    
                    # Collect all agents' testing data for multi-agent plots
                    all_agents_testing = {}
                    agent_training_data = None
                    
                    for agent_name, agent_data in agent_results.items():
                        testing = agent_data.get('testing', {})
                        training = agent_data.get('training', {})
                        
                        if testing.get('status') == 'completed':
                            # Store full testing data (includes per_building_agent and per_building_baseline)
                            all_agents_testing[agent_name] = testing
                            
                            # Get training data (use first agent with training data)
                            if not agent_training_data and training:
                                agent_training_data = training
                    
                    # Generate plots for first agent (or iterate through all)
                    if all_agents_testing:
                        first_agent_name = list(all_agents_testing.keys())[0]
                        first_agent_testing = all_agents_testing[first_agent_name]
                        
                        # Extract agent and baseline data from testing structure
                        agent_test_data = first_agent_testing.get('agent_data', {})
                        baseline_test_data = first_agent_testing.get('baseline_data', {})
                        
                        self.logger.info(f"[PLOTS] Generating 20 specification plots...")
                        self.logger.info(f"  Agent data available: {len(agent_test_data.get('episode_data', []))} episodes")
                        self.logger.info(f"  Baseline data available: {len(baseline_test_data.get('episode_data', []))} episodes")
                        self.logger.info(f"  Training data available: {agent_training_data is not None}")
                        self.logger.info(f"  Multi-agent comparison: {len(all_agents_testing)} agents")
                        
                        # Generate all plots with proper data
                        spec_plots = self.plot_generator.generate_all_plots(
                            agent_data=agent_test_data,
                            baseline_data=baseline_test_data,
                            training_data=agent_training_data.get('results', {}) if agent_training_data else None,
                            all_agents_data=all_agents_testing
                        )
                        plot_files.extend(spec_plots)
                        
                        self.logger.info(f"  Generated {len(spec_plots)} specification plots")
                except Exception as e:
                    self.logger.warning(f"  Comprehensive plots failed: {e}")
                    self.logger.exception("Full traceback:")
            
            self.logger.info(f"[OK] Generated {len(plot_files)} plots total")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Visualization error: {e}")
            self.logger.exception("Full traceback:")
        
        return plot_files
    
    def _extract_value(self, metric: Any) -> float:
        """
        CRITICAL FIX for Problem #3 and #8: Extract float value from metric.
        
        Handles:
        - Plain float/int: 123.4
        - Dict: {'value': 123.4, 'unit': '€'}
        - String: "123.4"
        - None: 0.0
        
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
    
    def _plot_training_curves(self, results: Dict[str, Any]) -> List[str]:
        """Plot training curves for all agents."""
        plot_files = []
        
        for agent_name, agent_data in results['agent_results'].items():
            training = agent_data.get('training', {})
            if training.get('status') != 'completed':
                continue
            
            training_results = training.get('results', {})
            
            # Sequential mode
            if 'buildings' in training_results and isinstance(training_results['buildings'], dict):
                fig, axes = plt.subplots(1, len(training_results['buildings']), figsize=(15, 5))
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                for idx, (building_id, building_results) in enumerate(training_results['buildings'].items()):
                    rewards = building_results.get('rewards', [])
                    if rewards:
                        axes[idx].plot(rewards, alpha=0.7, label='Episode Reward')
                        # Moving average
                        if len(rewards) > 10:
                            window = min(10, len(rewards))
                            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                            axes[idx].plot(range(window-1, len(rewards)), moving_avg, 
                                         'r-', linewidth=2, label='Moving Average')
                        axes[idx].set_title(f'{building_id}')
                        axes[idx].set_xlabel('Episode')
                        axes[idx].set_ylabel('Reward')
                        axes[idx].legend()
                        axes[idx].grid(True, alpha=0.3)
                
                plt.suptitle(f'Training Curves - {agent_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                plot_file = self.plots_dir / f'{agent_name}_training_curves.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
            
            # Parallel mode
            elif 'rewards' in training_results:
                rewards = training_results['rewards']
                if rewards:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(rewards, alpha=0.7, label='Episode Reward')
                    
                    if len(rewards) > 10:
                        window = min(10, len(rewards))
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(range(window-1, len(rewards)), moving_avg, 
                               'r-', linewidth=2, label='Moving Average')
                    
                    ax.set_title(f'Training Curves - {agent_name}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Reward')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_file = self.plots_dir / f'{agent_name}_training_curves.png'
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(plot_file))
        
        return plot_files
    
    def _plot_performance_comparison(self, results: Dict[str, Any]) -> List[str]:
        """
        Plot performance comparison - REWRITTEN TO MATCH WORKING PLOTS.
        Uses the same data access pattern as plot_generator.
        """
        plot_files = []
        
        if not results or 'agent_results' not in results:
            self.logger.warning("[PLOT] No agent_results")
            return plot_files
        
        # Extract agent_data for each agent (SAME AS plot_generator does)
        agent_names = []
        agent_rewards = []
        baseline_rewards = []
        
        for agent_name, agent_result in results['agent_results'].items():
            testing = agent_result.get('testing', {})
            
            # Only completed
            if testing.get('status') != 'completed':
                continue
            
            # Get agent_data and baseline_data (LIKE plot_generator does at line 95)
            agent_data = testing.get('agent_data', {})
            baseline_data = testing.get('baseline_data', {})
            
            # Get avg_reward directly from agent_data (IT'S A PLAIN FLOAT)
            agent_reward = agent_data.get('avg_reward', 0.0)
            baseline_reward = baseline_data.get('avg_reward', 0.0)
            
            self.logger.info(f"[PLOT] {agent_name}: agent={agent_reward:.2f}, baseline={baseline_reward:.2f}")
            
            agent_names.append(agent_name)
            agent_rewards.append(float(agent_reward))
            baseline_rewards.append(float(baseline_reward))
        
        if not agent_names:
            self.logger.warning("[PLOT] No completed agents")
            return plot_files
        
        n = len(agent_names)
        self.logger.info(f"[PLOT] Plotting {n} agents")
        
        # Create plot (SAME STYLE AS plot_generator plot_11)
        fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 6))
        
        x = np.arange(n)
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, agent_rewards, width, label='Agent',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, baseline_rewards, width, label='Baseline',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Labels and formatting
        ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison: Agent vs Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=0 if n <= 3 else 45, 
                        ha='center' if n <= 3 else 'right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        # Save
        plot_file = self.plots_dir / 'performance_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_files.append(str(plot_file))
        self.logger.info(f"[PLOT] Saved: {plot_file}")
        
        return plot_files
    
    def _plot_kpi_heatmap(self, results: Dict[str, Any]) -> List[str]:
        """Plot KPI heatmap across agents."""
        plot_files = []
        
        # Collect KPIs
        kpi_names = ['total_cost', 'pv_self_consumption_rate', 'battery_cycles', 
                     'peak_demand', 'avg_reward']
        kpi_labels = ['Total Cost (€)', 'PV Self-Consumption (%)', 'Battery Cycles', 
                      'Peak Demand (kW)', 'Avg Reward']
        
        agent_names = []
        kpi_matrix = []
        
        for agent_name, agent_data in results['agent_results'].items():
            testing = agent_data.get('testing', {})
            if testing.get('status') != 'completed':
                continue
            
            agent_kpis = testing.get('agent_kpis', {})
            agent_names.append(agent_name)
            
            row = []
            for kpi in kpi_names:
                value = self._extract_value(agent_kpis.get(kpi, 0))
                row.append(value)
            kpi_matrix.append(row)
        
        if not agent_names:
            return plot_files
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kpi_matrix = np.array(kpi_matrix)
        # Normalize each column
        for col in range(kpi_matrix.shape[1]):
            col_data = kpi_matrix[:, col]
            if col_data.max() != col_data.min():
                kpi_matrix[:, col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
        
        sns.heatmap(kpi_matrix, annot=False, fmt='.2f', cmap='YlOrRd',
                   xticklabels=kpi_labels, yticklabels=agent_names,
                   cbar_kws={'label': 'Normalized Value'}, ax=ax)
        
        ax.set_title('KPI Heatmap (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.plots_dir / 'kpi_heatmap.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def _plot_radar_chart(self, results: Dict[str, Any]) -> List[str]:
        """
        Plot radar chart - CRITICAL FIX for Problem #3.
        Uses _extract_value() to unwrap dict metrics before math operations.
        """
        plot_files = []
        
        # Metrics for radar
        metric_names = [
            'pv_self_consumption_rate',
            'avg_reward',
            'battery_cycles',
        ]
        metric_labels = [
            'PV Self-Consumption',
            'Avg Reward',
            'Battery Cycles',
        ]
        
        # Collect data
        agent_data_list = []
        for agent_name, agent_data in results['agent_results'].items():
            testing = agent_data.get('testing', {})
            if testing.get('status') != 'completed':
                continue
            
            agent_kpis = testing.get('agent_kpis', {})
            
            # CRITICAL FIX: Extract values using helper
            values = []
            for metric in metric_names:
                val = agent_kpis.get(metric, 0)
                val = self._extract_value(val)  # Unwrap dict if needed
                values.append(val)
            
            agent_data_list.append({
                'name': agent_name,
                'values': values
            })
        
        if not agent_data_list:
            return plot_files
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize values
        all_values = np.array([d['values'] for d in agent_data_list])
        normalized_values = []
        
        for i, agent_data_item in enumerate(agent_data_list):
            values = all_values[i]
            
            # Normalize each metric to 0-1
            normalized = []
            for j in range(len(values)):
                col_values = all_values[:, j]
                min_val = col_values.min()
                max_val = col_values.max()
                
                # CRITICAL FIX: Ensure we're working with floats
                if max_val > min_val:
                    norm_val = (values[j] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                
                normalized.append(norm_val)
            
            normalized_values.append(normalized)
        
        # Plot
        angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
        
        for i, agent_data_item in enumerate(agent_data_list):
            values = normalized_values[i] + [normalized_values[i][0]]  # Close the plot
            angles_plot = angles + [angles[0]]
            
            ax.plot(angles_plot, values, 'o-', linewidth=2, label=agent_data_item['name'])
            ax.fill(angles_plot, values, alpha=0.15)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Performance Radar Chart (Normalized)', pad=20, fontsize=14, fontweight='bold')
        ax.grid(True)
        
        plt.tight_layout()
        plot_file = self.plots_dir / 'radar_chart.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def _plot_cost_analysis(self, results: Dict[str, Any]) -> List[str]:
        """Plot cost analysis with numerical results."""
        plot_files = []
        
        agent_names = []
        agent_costs = []
        baseline_costs = []
        import_costs = []
        export_revenues = []
        savings_pct = []
        
        for agent_name, agent_data in results['agent_results'].items():
            testing = agent_data.get('testing', {})
            if testing.get('status') != 'completed':
                continue
            
            agent_kpis = testing.get('agent_kpis', {})
            baseline_kpis = testing.get('baseline_kpis', {})
            savings = testing.get('savings', {})
            
            agent_names.append(agent_name)
            agent_costs.append(self._extract_value(agent_kpis.get('total_cost', 0)))
            baseline_costs.append(self._extract_value(baseline_kpis.get('total_cost', 0)))
            import_costs.append(self._extract_value(agent_kpis.get('import_cost', 0)))
            export_revenues.append(self._extract_value(agent_kpis.get('export_revenue', 0)))
            savings_pct.append(self._extract_value(savings.get('cost_savings_percent', 0)))
        
        if not agent_names:
            return plot_files
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        x = np.arange(len(agent_names))
        width = 0.35
        
        # Left: Agent vs Baseline Total Cost
        bars1 = ax1.bar(x - width/2, agent_costs, width, label='Agent', 
                        color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, baseline_costs, width, label='Baseline', 
                        color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'€{height:,.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Cost (€)', fontsize=12, fontweight='bold')
        ax1.set_title('Total Cost Comparison: Agent vs Baseline', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agent_names, rotation=45, ha='right', fontsize=11)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Right: Cost Savings Percentage
        colors = ['green' if s > 0 else 'red' for s in savings_pct]
        bars3 = ax2.bar(x, savings_pct, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Set y-axis limits to add space at top
        max_savings = max(savings_pct) if savings_pct else 5
        min_savings = min(savings_pct) if savings_pct else 0
        y_range = max_savings - min_savings
        ax2.set_ylim(min_savings - 0.5, max_savings + 0.8)  # Add extra space at top
        
        # Add percentage labels on bars (inside the bar if close to top)
        for i, (bar, val) in enumerate(zip(bars3, savings_pct)):
            height = bar.get_height()
            # Place inside bar if positive and tall, otherwise above
            if height > 0 and height > max_savings * 0.7:
                y_pos = height - 0.3  # Inside bar
                va = 'top'
            else:
                y_pos = height + 0.15 if height > 0 else height - 0.15
                va = 'bottom' if height > 0 else 'top'
            
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.1f}%',
                    ha='center', va=va,
                    fontsize=11, fontweight='bold', color='darkgreen' if val > 0 else 'darkred')
            
            
        
        ax2.set_xlabel('Agent', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cost Savings (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cost Savings vs Baseline', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agent_names, rotation=45, ha='right', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add summary text box
        avg_savings = np.mean(savings_pct)
        total_agent_cost = np.sum(agent_costs)
        total_baseline_cost = np.sum(baseline_costs)
        total_saved = total_baseline_cost - total_agent_cost
        
        summary_text = (f'Summary:\n'
                    f'Avg Savings: {avg_savings:.1f}%\n'
                    f'Total Saved: €{total_saved:,.0f}\n'
                    f'Agent Total: €{total_agent_cost:,.0f}\n'
                    f'Baseline Total: €{total_baseline_cost:,.0f}')
        
        ax2.text(0.98, 0.98, summary_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
                family='monospace')
        
        plt.tight_layout()
        
        plot_file = self.plots_dir / 'cost_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def _plot_statistical_results(self, results: Dict[str, Any]) -> List[str]:
        """Plot statistical test results."""
        plot_files = []
        
        stats = results.get('statistical_tests', {})
        comparisons = stats.get('comparisons', [])
        
        if not comparisons:
            return plot_files
        
        # Comparison matrix
        agent_names = list(set(
            [c['agent_a'] for c in comparisons] + [c['agent_b'] for c in comparisons]
        ))
        n = len(agent_names)
        
        p_value_matrix = np.ones((n, n))
        
        for comp in comparisons:
            i = agent_names.index(comp['agent_a'])
            j = agent_names.index(comp['agent_b'])
            p_val = comp['p_value']
            p_value_matrix[i, j] = p_val
            p_value_matrix[j, i] = p_val
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(-np.log10(p_value_matrix + 1e-10), annot=False, cmap='RdYlGn_r',
                   xticklabels=agent_names, yticklabels=agent_names,
                   cbar_kws={'label': '-log10(p-value)'}, ax=ax)
        
        ax.set_title('Statistical Significance Matrix\n(Darker = More Significant)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.plots_dir / 'statistical_significance.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def _plot_opi_scores(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate OPI performance score visualizations.

        FIXED VERSION - Addresses all 4 plot issues:
        1. Overall OPI scores (horizontal bar) - ✓ Working
        2. Category contributions (stacked bar) - ✓ Fixed to show all categories
        3. Category radar (polar) - ✓ Fixed extraction and scaling
        4. Category heatmap - ✓ Fixed to work with single agent

        Args:
            results: Complete evaluation results with opi_results

        Returns:
            List of generated plot file paths
        """
        plot_files = []

        opi_results = results.get('opi_results', {})
        if not opi_results:
            self.logger.warning("[VIZ] No OPI results available for plotting")
            return plot_files

        self.logger.info("[VIZ-OPI] Starting OPI visualization with enhanced diagnostics")

        # =========================================================================
        # PHASE 1: Extract and validate data structure
        # =========================================================================

        overall_scores = {}
        category_scores = {}
        rankings = []
        metadata = {}

        # Handle the actual PerformanceScorer output format
        if 'opi_results' in opi_results and 'rankings' in opi_results:
            # PerformanceScorer format (current)
            self.logger.info("[VIZ-OPI] Detected PerformanceScorer OPI format")
            agents_data = opi_results.get('opi_results', {})
            rankings_data = opi_results.get('rankings', [])
            metadata = opi_results.get('metadata', {})

            self.logger.info(f"[VIZ-OPI] Found {len(agents_data)} agents in OPI results")

            # Convert OPIResult objects to expected format with detailed logging
            for agent_name, opi_result in agents_data.items():
                self.logger.info(f"[VIZ-OPI] Processing agent: {agent_name}")

                # Get OPI score
                if hasattr(opi_result, 'opi_score'):
                    overall_scores[agent_name] = opi_result.opi_score
                    self.logger.info(f"  - OPI score: {opi_result.opi_score:.4f}")
                else:
                    overall_scores[agent_name] = opi_result.get('opi_score', 0.0)
                    self.logger.info(f"  - OPI score (dict): {opi_result.get('opi_score', 0.0):.4f}")

                # Extract category scores with robust handling
                cat_scores = {}

                if hasattr(opi_result, 'category_scores'):
                    # OPIResult object with CategoryScore objects
                    self.logger.info(f"  - Found {len(opi_result.category_scores)} category scores (object format)")
                    for cat_score in opi_result.category_scores:
                        if hasattr(cat_score, 'category_name'):
                            cat_name = cat_score.category_name
                            cat_utility = cat_score.category_utility
                        else:
                            cat_name = cat_score.get('category_name', '')
                            cat_utility = cat_score.get('category_utility', 0.0)

                        cat_scores[cat_name] = cat_utility
                        self.logger.info(f"    * {cat_name}: {cat_utility:.4f}")

                elif isinstance(opi_result, dict) and 'category_scores' in opi_result:
                    # Dictionary format
                    self.logger.info(f"  - Found category scores (dict format)")
                    for cat_score in opi_result['category_scores']:
                        cat_name = cat_score.get('category_name', '')
                        cat_utility = cat_score.get('category_utility', 0.0)
                        cat_scores[cat_name] = cat_utility
                        self.logger.info(f"    * {cat_name}: {cat_utility:.4f}")
                else:
                    self.logger.warning(f"  - No category scores found for {agent_name}")

                category_scores[agent_name] = cat_scores

            # Convert rankings (already in correct format)
            rankings = rankings_data if isinstance(rankings_data, list) else []
            self.logger.info(f"[VIZ-OPI] Extracted {len(rankings)} rankings")

        # Handle MAUT scorer format (legacy)
        elif 'agents' in opi_results:
            self.logger.info("[VIZ-OPI] Detected legacy MAUT scorer OPI format")
            agents_data = opi_results.get('agents', {})
            rankings_data = opi_results.get('rankings', [])
            metadata = opi_results.get('metadata', {})

            # Convert to expected format
            for agent_name, agent_info in agents_data.items():
                overall_scores[agent_name] = agent_info.get('opi_score', 0.0)

                # Extract category scores
                cat_scores = {}
                agent_cat_scores = agent_info.get('category_scores', {})
                for cat_name, cat_data in agent_cat_scores.items():
                    if isinstance(cat_data, dict):
                        cat_scores[cat_name] = cat_data.get('utility', 0.0)
                    else:
                        cat_scores[cat_name] = cat_data
                category_scores[agent_name] = cat_scores

            # Convert rankings
            for rank_info in rankings_data:
                agent_name = rank_info.get('agent_name', '')
                score = rank_info.get('opi_score', 0.0)
                rank = rank_info.get('rank', 0)
                rankings.append((agent_name, score, rank))

        elif 'overall_scores' in opi_results:
            # Simple scorer format (backward compatible)
            self.logger.info("[VIZ-OPI] Detected simple scorer OPI format")
            overall_scores = opi_results.get('overall_scores', {})
            category_scores = opi_results.get('category_scores', {})
            rankings = opi_results.get('rankings', [])
        else:
            self.logger.warning("[VIZ-OPI] Unknown OPI results format - cannot plot")
            self.logger.warning(f"[VIZ-OPI] Available keys: {list(opi_results.keys())}")
            return plot_files

        if not overall_scores:
            self.logger.warning("[VIZ-OPI] No overall scores in OPI results")
            return plot_files

        # Extract data from rankings
        agent_names = []
        opi_scores = []

        for agent_name, score, rank in rankings:
            agent_names.append(agent_name)
            opi_scores.append(score)

        self.logger.info(f"[VIZ-OPI] Plotting OPI for {len(agent_names)} agents")

        # Get category structure
        if 'category_weights' in metadata:
            category_weights = metadata['category_weights']
            category_names = list(category_weights.keys())
            self.logger.info("[VIZ-OPI] Using category weights from OPI metadata")
        else:
            # Import here to avoid circular dependencies
            try:
                from performance_scorer import PerformanceScorer
                category_names = list(PerformanceScorer.CATEGORY_WEIGHTS.keys())
                category_weights = PerformanceScorer.CATEGORY_WEIGHTS
                self.logger.info("[VIZ-OPI] Using category weights from PerformanceScorer")
            except ImportError:
                self.logger.error("[VIZ-OPI] Cannot import PerformanceScorer - using defaults")
                category_names = ['energy_economics', 'renewable_energy', 'battery_performance', 
                                'ai_learning', 'statistical_research']
                category_weights = {cat: 1.0/len(category_names) for cat in category_names}

        self.logger.info(f"[VIZ-OPI] Category structure: {len(category_names)} categories")
        for cat in category_names:
            self.logger.info(f"  - {cat}: weight={category_weights.get(cat, 0.0):.2f}")

        # =========================================================================
        # PLOT 1: Overall OPI scores (horizontal bar chart)
        # =========================================================================
        self.logger.info("[VIZ-OPI] Generating Plot 1/4: Overall OPI scores")

        fig, ax = plt.subplots(figsize=(12, max(6, len(agent_names) * 0.5)))

        colors = plt.cm.RdYlGn(np.array(opi_scores))
        bars = ax.barh(agent_names, opi_scores, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Operational Performance Index (OPI)', fontsize=12, fontweight='bold')
        ax.set_title('Agent Rankings by OPI Score', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Halfway')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, opi_scores)):
            ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plot_file = self.plots_dir / 'opi_overall_scores.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        self.logger.info(f"[VIZ-OPI] ✓ Saved: {plot_file.name}")

        # =========================================================================
        # PLOT 2: Stacked category contributions
        # =========================================================================
        self.logger.info("[VIZ-OPI] Generating Plot 2/4: Category contributions (stacked)")

        # FIXED: Ensure all categories are included even if score is 0
        category_contributions = {cat: [] for cat in category_names}

        for agent_name in agent_names:
            cat_scores = category_scores.get(agent_name, {})
            self.logger.info(f"  - {agent_name} category breakdown:")

            for cat in category_names:
                cat_utility = cat_scores.get(cat, 0.0)  # Default to 0 if missing
                weight = category_weights.get(cat, 0.0)
                contribution = cat_utility * weight
                category_contributions[cat].append(contribution)

                if contribution > 0.001:  # Log non-zero contributions
                    self.logger.info(f"    * {cat}: utility={cat_utility:.4f}, "
                                   f"weight={weight:.2f}, contrib={contribution:.4f}")

        # Verify all categories have data
        for cat, values in category_contributions.items():
            if len(values) != len(agent_names):
                self.logger.warning(f"  ! Category {cat} has {len(values)} values, expected {len(agent_names)}")
                # Pad with zeros if needed
                while len(values) < len(agent_names):
                    values.append(0.0)
        
        # Log summary of all contributions for debugging
        self.logger.info("  - Category contribution summary:")
        for cat in category_names:
            total = sum(category_contributions[cat])
            self.logger.info(f"    {cat}: {total:.4f}")

        fig, ax = plt.subplots(figsize=(14, max(6, len(agent_names) * 0.5)))

        bottom = np.zeros(len(agent_names))
        colors_breakdown = plt.cm.Set3(np.linspace(0, 1, len(category_names)))

        # Plot all categories in order - no skipping
        for category, color in zip(category_names, colors_breakdown):
            values = np.array(category_contributions[category])
            label = category.replace('_', ' ').title()
            
            # Always plot, even if contribution is very small
            ax.barh(agent_names, values, left=bottom, label=label, 
                   color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
            
            total = np.sum(values)
            if total > 0.001:
                self.logger.info(f"  - Plotted {category}: contrib={total:.4f}")
            else:
                self.logger.info(f"  - Plotted {category}: contrib=~0 (may be too small to see)")
            
            # Always add to bottom for next category
            bottom += values

        ax.set_xlabel('OPI Contribution', fontsize=12, fontweight='bold')
        ax.set_title('Category Contributions to OPI', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_file = self.plots_dir / 'opi_category_breakdown.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        self.logger.info(f"[VIZ-OPI] ✓ Saved: {plot_file.name}")

        # =========================================================================
        # PLOT 3: Radar chart of category scores (for top agents)
        # =========================================================================
        self.logger.info("[VIZ-OPI] Generating Plot 3/4: Category radar (top agents)")

        if len(agent_names) >= 1:
            top_n = min(5, len(agent_names))
            top_agents = agent_names[:top_n]

            # Number of categories
            categories_display = [cat.replace('_', '\n').title() for cat in category_names]
            N = len(categories_display)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            colors_radar = plt.cm.tab10(np.linspace(0, 1, top_n))

            self.logger.info(f"  - Plotting radar for top {top_n} agents:")

            for idx, agent_name in enumerate(top_agents):
                cat_scores = category_scores.get(agent_name, {})

                # FIXED: Ensure all categories are included with default 0
                values = []
                for cat in category_names:
                    val = cat_scores.get(cat, 0.0)
                    values.append(val)

                self.logger.info(f"    * {agent_name}: values={values}")

                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=agent_name, 
                       color=colors_radar[idx], alpha=0.8)
                ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories_display, fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.set_title(f'Category Scores Comparison (Top {top_n} Agents)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.plots_dir / 'opi_category_radar.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
            self.logger.info(f"[VIZ-OPI] ✓ Saved: {plot_file.name}")
        else:
            self.logger.warning("[VIZ-OPI] No agents for radar chart")

        # =========================================================================
        # PLOT 4: Category scores heatmap
        # =========================================================================
        self.logger.info("[VIZ-OPI] Generating Plot 4/4: Category heatmap")

        # FIXED: Generate for single agent as well (removed len(agent_names) >= 2 check)
        if len(agent_names) >= 1:
            # Prepare matrix
            score_matrix = np.zeros((len(agent_names), len(category_names)))

            for i, agent_name in enumerate(agent_names):
                cat_scores = category_scores.get(agent_name, {})
                for j, cat in enumerate(category_names):
                    score_matrix[i, j] = cat_scores.get(cat, 0.0)

            self.logger.info(f"  - Matrix shape: {score_matrix.shape}")
            self.logger.info(f"  - Matrix stats: min={score_matrix.min():.4f}, max={score_matrix.max():.4f}, "
                            f"mean={score_matrix.mean():.4f}")

            fig, ax = plt.subplots(figsize=(12, max(6, len(agent_names) * 0.4)))

            im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Set ticks
            ax.set_xticks(np.arange(len(category_names)))
            ax.set_yticks(np.arange(len(agent_names)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in category_names], 
                              rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(agent_names, fontsize=10)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Category Score', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

            # Add text annotations
            for i in range(len(agent_names)):
                for j in range(len(category_names)):
                    text = ax.text(j, i, f'{score_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

            ax.set_title('Category Scores Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()

            plot_file = self.plots_dir / 'opi_category_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
            self.logger.info(f"[VIZ-OPI] ✓ Saved: {plot_file.name}")
        else:
            self.logger.warning("[VIZ-OPI] No agents for heatmap")

        self.logger.info(f"[VIZ-OPI] ✓ Generated {len(plot_files)} OPI plots successfully")

        return plot_files