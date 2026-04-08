"""
Research Visualizer Module
Generates comprehensive research-grade visualizations and dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


class ResearchVisualizer:
    """
    Research-grade visualization module for comprehensive analysis dashboards.
    """
    
    def __init__(self, experiment_dir: Path, logger):
        """
        Initialize research visualizer.
        
        Args:
            experiment_dir: Experiment directory
            logger: Logger instance
        """
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        # Create visualization directories
        self.plots_dir = experiment_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of plots
        self.training_plots_dir = self.plots_dir / 'training_analytics'
        self.performance_plots_dir = self.plots_dir / 'performance_comparison'
        self.testing_plots_dir = self.plots_dir / 'cross_dataset_analysis'
        self.research_plots_dir = self.plots_dir / 'research_dashboard'
        
        for plot_dir in [self.training_plots_dir, self.performance_plots_dir, 
                        self.testing_plots_dir, self.research_plots_dir]:
            plot_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_research_visualizations(self, training_results: Dict[str, Any], 
                                       testing_results: Dict[str, Any],
                                       evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive research visualizations.
        
        Args:
            training_results: Training phase results
            testing_results: Testing phase results
            evaluation_results: Evaluation phase results
            
        Returns:
            Visualization results with file paths
        """
        self.logger.info("Generating research-grade visualizations...")
        
        visualization_results = {
            'training_analytics': {},
            'performance_comparison': {},
            'cross_dataset_analysis': {},
            'research_dashboard': {},
            'plot_files': []
        }
        
        # Generate training analytics visualizations
        self.logger.info("Creating training analytics visualizations...")
        training_viz = self._generate_training_analytics(training_results)
        visualization_results['training_analytics'] = training_viz
        
        # Generate performance comparison visualizations
        self.logger.info("Creating performance comparison visualizations...")
        performance_viz = self._generate_performance_comparison(evaluation_results)
        visualization_results['performance_comparison'] = performance_viz
        
        # Generate cross-dataset analysis visualizations
        self.logger.info("Creating cross-dataset analysis visualizations...")
        testing_viz = self._generate_cross_dataset_analysis(testing_results)
        visualization_results['cross_dataset_analysis'] = testing_viz
        
        # Generate comprehensive research dashboard
        self.logger.info("Creating research dashboard...")
        dashboard_viz = self._generate_research_dashboard(
            training_results, testing_results, evaluation_results
        )
        visualization_results['research_dashboard'] = dashboard_viz
        
        # Collect all plot files
        visualization_results['plot_files'] = self._collect_plot_files()
        
        self.logger.info("Research visualizations completed")
        return visualization_results
    
    def _generate_training_analytics(self, training_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate training analytics visualizations."""
        plot_files = {}
        
        # Learning curves with confidence bands
        plot_files['learning_curves'] = self._plot_learning_curves(training_results)
        
        # Convergence analysis
        plot_files['convergence_analysis'] = self._plot_convergence_analysis(training_results)
        
        # Training efficiency comparison
        plot_files['training_efficiency'] = self._plot_training_efficiency(training_results)
        
        # Exploration decay analysis
        plot_files['exploration_decay'] = self._plot_exploration_decay(training_results)
        
        return plot_files
    
    def _plot_learning_curves(self, training_results: Dict[str, Any]) -> str:
        """Plot learning curves with confidence bands."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract training data
        agent_data = {}
        for agent_name, results in training_results.get('agent_results', {}).items():
            if results.get('status') == 'completed':
                analytics = results.get('training_analytics', {})
                agent_data[agent_name] = {
                    'rewards': analytics.get('episode_rewards', []),
                    'losses': analytics.get('episode_losses', []),
                    'exploration': analytics.get('exploration_schedule', []),
                    'convergence_episode': analytics.get('convergence_episode', 0)
                }
        
        # Plot 1: Learning curves (rewards)
        ax1 = axes[0, 0]
        for agent_name, data in agent_data.items():
            if data['rewards']:
                episodes = range(len(data['rewards']))
                rewards = data['rewards']
                
                # Plot raw rewards
                ax1.plot(episodes, rewards, alpha=0.3, linewidth=0.5)
                
                # Plot moving average
                window_size = min(20, len(rewards) // 5)
                if window_size > 1:
                    moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                    ax1.plot(episodes, moving_avg, label=agent_name, linewidth=2)
                
                # Mark convergence point
                conv_episode = data['convergence_episode']
                if conv_episode > 0 and conv_episode < len(rewards):
                    ax1.axvline(x=conv_episode, color='red', linestyle='--', alpha=0.7)
        
        ax1.set_title('Learning Curves (Episode Rewards)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss progression
        ax2 = axes[0, 1]
        for agent_name, data in agent_data.items():
            if data['losses']:
                episodes = range(len(data['losses']))
                ax2.plot(episodes, data['losses'], label=agent_name, alpha=0.8)
        
        ax2.set_title('Training Loss Progression')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Exploration schedule
        ax3 = axes[1, 0]
        for agent_name, data in agent_data.items():
            if data['exploration']:
                episodes = range(len(data['exploration']))
                ax3.plot(episodes, data['exploration'], label=agent_name, alpha=0.8)
        
        ax3.set_title('Exploration Rate Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Exploration Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution
        ax4 = axes[1, 1]
        final_performances = []
        agent_names = []
        
        for agent_name, data in agent_data.items():
            if data['rewards']:
                # Use last 10% of episodes for final performance
                final_episodes = max(1, len(data['rewards']) // 10)
                final_perf = np.mean(data['rewards'][-final_episodes:])
                final_performances.append(final_perf)
                agent_names.append(agent_name)
        
        if final_performances:
            bars = ax4.bar(agent_names, final_performances, alpha=0.7)
            ax4.set_title('Final Performance Comparison')
            ax4.set_ylabel('Average Final Reward')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, final_performances):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = self.training_plots_dir / 'learning_curves.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _plot_convergence_analysis(self, training_results: Dict[str, Any]) -> str:
        """Plot convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Extract convergence data
        convergence_data = {}
        for agent_name, results in training_results.get('agent_results', {}).items():
            if results.get('status') == 'completed':
                analytics = results.get('training_analytics', {})
                convergence_metrics = analytics.get('convergence_metrics', {})
                convergence_data[agent_name] = {
                    'episodes': analytics.get('convergence_episode', 0),
                    'stability': convergence_metrics.get('stability_index', 0),
                    'trend_slope': convergence_metrics.get('trend_slope', 0),
                    'final_variance': convergence_metrics.get('reward_variance', 0)
                }
        
        # Plot 1: Convergence episodes
        ax1 = axes[0, 0]
        agents = list(convergence_data.keys())
        conv_episodes = [convergence_data[agent]['episodes'] for agent in agents]
        
        bars = ax1.bar(agents, conv_episodes, alpha=0.7, color='skyblue')
        ax1.set_title('Episodes to Convergence')
        ax1.set_ylabel('Episodes')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, conv_episodes):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value}', ha='center', va='bottom')
        
        # Plot 2: Stability index
        ax2 = axes[0, 1]
        stability_scores = [convergence_data[agent]['stability'] for agent in agents]
        bars = ax2.bar(agents, stability_scores, alpha=0.7, color='lightgreen')
        ax2.set_title('Training Stability Index')
        ax2.set_ylabel('Stability Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Learning trend
        ax3 = axes[1, 0]
        trend_slopes = [convergence_data[agent]['trend_slope'] for agent in agents]
        colors = ['green' if slope > 0 else 'red' for slope in trend_slopes]
        bars = ax3.bar(agents, trend_slopes, alpha=0.7, color=colors)
        ax3.set_title('Learning Trend (Slope)')
        ax3.set_ylabel('Trend Slope')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance variance
        ax4 = axes[1, 1]
        variances = [convergence_data[agent]['final_variance'] for agent in agents]
        bars = ax4.bar(agents, variances, alpha=0.7, color='orange')
        ax4.set_title('Final Performance Variance')
        ax4.set_ylabel('Variance')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = self.training_plots_dir / 'convergence_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _generate_performance_comparison(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate performance comparison visualizations."""
        plot_files = {}
        
        # Comprehensive metrics heatmap
        plot_files['metrics_heatmap'] = self._plot_metrics_heatmap(evaluation_results)
        
        # Radar chart comparison
        plot_files['radar_comparison'] = self._plot_radar_comparison(evaluation_results)
        
        # Performance ranking visualization
        plot_files['performance_ranking'] = self._plot_performance_ranking(evaluation_results)
        
        # Statistical significance matrix
        plot_files['significance_matrix'] = self._plot_significance_matrix(evaluation_results)
        
        return plot_files
    
    def _plot_metrics_heatmap(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot comprehensive metrics heatmap."""
        agent_metrics = evaluation_results.get('agent_metrics', {})
        
        if not agent_metrics:
            return ""
        
        # Prepare data for heatmap
        agents = list(agent_metrics.keys())
        all_metrics = set()
        
        for metrics in agent_metrics.values():
            all_metrics.update(metrics.keys())
        
        all_metrics = sorted(list(all_metrics))
        
        # Create data matrix
        data_matrix = []
        for metric in all_metrics:
            row = []
            for agent in agents:
                if metric in agent_metrics[agent]:
                    value = agent_metrics[agent][metric].value
                    # Normalize to 0-1 scale for better visualization
                    normalized_value = abs(value) / (1 + abs(value))
                    row.append(normalized_value)
                else:
                    row.append(0)
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(len(agents) * 1.2, len(all_metrics) * 0.3))
        
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(agents)))
        ax.set_yticks(range(len(all_metrics)))
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.set_yticklabels(all_metrics, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i, metric in enumerate(all_metrics):
            for j, agent in enumerate(agents):
                if metric in agent_metrics[agent]:
                    value = agent_metrics[agent][metric].value
                    text_value = f'{value:.2f}' if abs(value) < 1000 else f'{value:.1e}'
                    ax.text(j, i, text_value, ha='center', va='center', 
                           fontsize=6, color='white' if data_matrix[i][j] > 0.5 else 'black')
        
        plt.title('Comprehensive Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.performance_plots_dir / 'metrics_heatmap.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _plot_radar_comparison(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot radar chart for multi-dimensional comparison."""
        agent_metrics = evaluation_results.get('agent_metrics', {})
        comparative_analysis = evaluation_results.get('comparative_analysis', {})
        
        if not agent_metrics or not comparative_analysis:
            return ""
        
        # Get category rankings
        category_rankings = comparative_analysis.get('category_rankings', {})
        categories = list(category_rankings.keys())
        agents = list(agent_metrics.keys())
        
        # Calculate category scores for each agent
        category_scores = {}
        for agent in agents:
            category_scores[agent] = {}
            for category in categories:
                if category in category_rankings:
                    # Ranking position (1st = highest score)
                    ranking = category_rankings[category]
                    if agent in ranking:
                        position = ranking.index(agent) + 1
                        # Convert to score (higher is better)
                        score = (len(ranking) - position + 1) / len(ranking)
                        category_scores[agent][category] = score
                    else:
                        category_scores[agent][category] = 0
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
        
        for i, agent in enumerate(agents):
            values = []
            for category in categories:
                values.append(category_scores[agent].get(category, 0))
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Multi-Dimensional Performance Comparison', size=14, fontweight='bold', pad=20)
        
        plot_file = self.performance_plots_dir / 'radar_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _generate_cross_dataset_analysis(self, testing_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate cross-dataset analysis visualizations."""
        plot_files = {}
        
        # Generalization performance plots
        plot_files['generalization_performance'] = self._plot_generalization_performance(testing_results)
        
        # Robustness analysis
        plot_files['robustness_analysis'] = self._plot_robustness_analysis(testing_results)
        
        # Scenario performance breakdown
        plot_files['scenario_breakdown'] = self._plot_scenario_breakdown(testing_results)
        
        return plot_files
    
    def _plot_generalization_performance(self, testing_results: Dict[str, Any]) -> str:
        """Plot generalization performance analysis."""

            # Check if we have valid testing results
        if not testing_results or 'general_testing' not in testing_results or 'specific_testing' not in testing_results:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No testing results available\n(Training phase failed)', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Generalization Performance')
            ax.axis('off')
            
            plot_file = self.testing_plots_dir / 'generalization_performance.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Extract scores safely
        general_results = testing_results.get('general_testing', {})
        specific_results = testing_results.get('specific_testing', {})
        
        # Extract scores with fallbacks
        general_scores = []
        specific_scores = []
        
        for agent_name, results in general_results.items():
            if isinstance(results, dict) and 'score' in results:
                general_scores.append(results['score'])
        
        for agent_name, results in specific_results.items():
            if isinstance(results, dict) and 'score' in results:
                specific_scores.append(results['score'])
        
        # Check if we have any scores
        if not general_scores and not specific_scores:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No valid test scores available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Generalization Performance')
            ax.axis('off')
            
            plot_file = self.testing_plots_dir / 'generalization_performance.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Now we can safely compute min/max
        all_scores = general_scores + specific_scores
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 1

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Generalization Performance Analysis', fontsize=16, fontweight='bold')
        
        agent_results = testing_results.get('agent_results', {})
        
        # Extract data
        agents = []
        general_scores = []
        specific_scores = []
        robustness_scores = []
        
        for agent_name, results in agent_results.items():
            if results.get('status') == 'completed':
                agents.append(agent_name)
                
                general_test = results.get('general_testing', {})
                specific_test = results.get('specific_testing', {})
                robustness = results.get('robustness_metrics', {})
                
                general_scores.append(general_test.get('generalization_score', 0))
                specific_scores.append(specific_test.get('adaptation_score', 0))
                robustness_scores.append(robustness.get('robustness_index', 0))
        
        # Plot 1: General vs Specific Testing
        ax1 = axes[0, 0]
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, general_scores, width, label='General Testing', alpha=0.8)
        bars2 = ax1.bar(x + width/2, specific_scores, width, label='Specific Testing', alpha=0.8)
        
        ax1.set_title('General vs Specific Testing Performance')
        ax1.set_xlabel('Agents')
        ax1.set_ylabel('Performance Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Robustness Index
        ax2 = axes[0, 1]
        bars = ax2.bar(agents, robustness_scores, alpha=0.7, color='orange')
        ax2.set_title('Robustness Index')
        ax2.set_xlabel('Agents')
        ax2.set_ylabel('Robustness Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot - General vs Specific
        ax3 = axes[1, 0]
        scatter = ax3.scatter(general_scores, specific_scores, 
                             c=robustness_scores, cmap='viridis', s=100, alpha=0.7)
        
        for i, agent in enumerate(agents):
            ax3.annotate(agent, (general_scores[i], specific_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('General Testing Score')
        ax3.set_ylabel('Specific Testing Score')
        ax3.set_title('General vs Specific Performance')
        
        # Add diagonal line for reference
        min_score = min(min(general_scores), min(specific_scores))
        max_score = max(max(general_scores), max(specific_scores))
        ax3.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Robustness Index')
        
        # Plot 4: Performance consistency
        ax4 = axes[1, 1]
        consistency_data = []
        
        for agent_name, results in agent_results.items():
            if results.get('status') == 'completed':
                general_test = results.get('general_testing', {})
                general_perfs = general_test.get('scenario_performances', [])
                
                if general_perfs:
                    consistency = 1.0 / (1.0 + np.std(general_perfs))
                    consistency_data.append((agent_name, consistency))
        
        if consistency_data:
            agents_cons, consistency_scores = zip(*consistency_data)
            bars = ax4.bar(agents_cons, consistency_scores, alpha=0.7, color='lightgreen')
            ax4.set_title('Performance Consistency Across Scenarios')
            ax4.set_xlabel('Agents')
            ax4.set_ylabel('Consistency Score')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.testing_plots_dir / 'generalization_performance.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _generate_research_dashboard(self, training_results: Dict[str, Any], 
                                   testing_results: Dict[str, Any],
                                   evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive research dashboard."""
        plot_files = {}
        
        # Executive summary dashboard
        plot_files['executive_summary'] = self._plot_executive_summary(
            training_results, testing_results, evaluation_results
        )
        
        # Performance evolution timeline
        plot_files['performance_timeline'] = self._plot_performance_timeline(training_results)
        
        # Comprehensive comparison matrix
        plot_files['comparison_matrix'] = self._plot_comparison_matrix(evaluation_results)
        
        return plot_files
    
    def _plot_executive_summary(self, training_results: Dict[str, Any], 
                              testing_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> str:
        """Plot executive summary dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a complex grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('HEMS Research Pipeline - Executive Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Get agent rankings
        agent_rankings = evaluation_results.get('agent_rankings', [])
        
        # 1. Overall Performance Ranking (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if agent_rankings:
            y_pos = np.arange(len(agent_rankings))
            ax1.barh(y_pos, range(len(agent_rankings), 0, -1), alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(agent_rankings)
            ax1.set_xlabel('Ranking Score')
            ax1.set_title('Overall Performance Ranking', fontweight='bold')
            ax1.invert_yaxis()
        
        # 2. Training Summary (top-center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        training_summary = training_results.get('training_summary', {})
        convergence_summary = training_summary.get('convergence_summary', {})
        
        if convergence_summary:
            agents = list(convergence_summary.keys())
            conv_episodes = [convergence_summary[agent].get('convergence_episode', 0) 
                           for agent in agents]
            
            bars = ax2.bar(agents, conv_episodes, alpha=0.7, color='skyblue')
            ax2.set_title('Training Convergence', fontweight='bold')
            ax2.set_ylabel('Episodes to Convergence')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Testing Performance (top-center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        testing_summary = testing_results.get('testing_summary', {})
        
        if testing_summary.get('robustness_ranking'):
            robust_agents = testing_summary['robustness_ranking'][:5]  # Top 5
            y_pos = np.arange(len(robust_agents))
            ax3.barh(y_pos, range(len(robust_agents), 0, -1), alpha=0.7, color='lightgreen')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(robust_agents)
            ax3.set_title('Robustness Ranking', fontweight='bold')
            ax3.invert_yaxis()
        
        # 4. Key Metrics Summary (top-right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        # Add text summary
        summary_text = []
        summary_text.append("Key Insights:")
        summary_text.append(f"• Total Agents: {len(agent_rankings)}")
        summary_text.append(f"• Best Agent: {agent_rankings[0] if agent_rankings else 'N/A'}")
        summary_text.append(f"• Successful Training: {training_summary.get('successful_agents', 0)}")
        summary_text.append(f"• Successful Testing: {testing_summary.get('successful_tests', 0)}")
        
        ax4.text(0.1, 0.9, '\n'.join(summary_text), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.5))
        
        # 5. Performance Distribution (middle row, spans 2 columns)
        ax5 = fig.add_subplot(gs[1, :2])
        
        # Extract final performances from all agents
        all_performances = {}
        for agent_name, results in training_results.get('agent_results', {}).items():
            if results.get('status') == 'completed':
                analytics = results.get('training_analytics', {})
                rewards = analytics.get('episode_rewards', [])
                if rewards:
                    final_perf = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                    all_performances[agent_name] = final_perf
        
        if all_performances:
            agents = list(all_performances.keys())
            performances = list(all_performances.values())
            
            ax5.boxplot([performances], labels=['All Agents'])
            ax5.scatter([1] * len(performances), performances, alpha=0.7, s=50)
            
            for i, (agent, perf) in enumerate(all_performances.items()):
                ax5.annotate(agent, (1, perf), xytext=(10, 0), 
                           textcoords='offset points', fontsize=8)
            
            ax5.set_title('Performance Distribution Analysis', fontweight='bold')
            ax5.set_ylabel('Final Performance')
        
        # 6. Category Performance (middle-right, spans 2 columns)
        ax6 = fig.add_subplot(gs[1, 2:])
        
        comparative_analysis = evaluation_results.get('comparative_analysis', {})
        category_rankings = comparative_analysis.get('category_rankings', {})
        
        if category_rankings and agent_rankings:
            categories = list(category_rankings.keys())
            top_agent = agent_rankings[0]
            
            # Show top agent's ranking in each category
            rankings_in_categories = []
            for category in categories:
                if top_agent in category_rankings[category]:
                    rank = category_rankings[category].index(top_agent) + 1
                    rankings_in_categories.append(rank)
                else:
                    rankings_in_categories.append(len(category_rankings[category]) + 1)
            
            # Invert for better visualization (higher is better)
            max_rank = max(rankings_in_categories)
            inverted_ranks = [max_rank - rank + 1 for rank in rankings_in_categories]
            
            bars = ax6.bar(categories, inverted_ranks, alpha=0.7, color='gold')
            ax6.set_title(f'Category Performance - {top_agent}', fontweight='bold')
            ax6.set_ylabel('Performance Score')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Timeline (bottom row, spans all columns)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Create a timeline of the experiment
        timeline_data = []
        for agent_name, results in training_results.get('agent_results', {}).items():
            if results.get('status') == 'completed':
                analytics = results.get('training_analytics', {})
                training_time = analytics.get('training_time', 0)
                timeline_data.append((agent_name, training_time))
        
        if timeline_data:
            agents, times = zip(*timeline_data)
            cumulative_times = np.cumsum([0] + list(times))
            
            for i, (agent, time) in enumerate(timeline_data):
                ax7.barh(i, time, left=cumulative_times[i], alpha=0.7, 
                        label=agent if i < 5 else "")  # Limit legend entries
            
            ax7.set_title('Training Timeline', fontweight='bold')
            ax7.set_xlabel('Cumulative Time (seconds)')
            ax7.set_yticks(range(len(agents)))
            ax7.set_yticklabels(agents)
            
            if len(agents) <= 5:
                ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_file = self.research_plots_dir / 'executive_summary.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _collect_plot_files(self) -> List[str]:
        """Collect all generated plot files."""
        plot_files = []
        
        for plot_dir in [self.training_plots_dir, self.performance_plots_dir, 
                        self.testing_plots_dir, self.research_plots_dir]:
            if plot_dir.exists():
                for plot_file in plot_dir.glob('*.png'):
                    plot_files.append(str(plot_file))
        
        return plot_files
    
    # Additional helper methods for specific plots
    def _plot_training_efficiency(self, training_results: Dict[str, Any]) -> str:
        """Plot training efficiency analysis."""
        # Implementation for training efficiency plots
        # This would include time per episode, sample efficiency, etc.
        pass
    
    def _plot_exploration_decay(self, training_results: Dict[str, Any]) -> str:
        """Plot exploration decay analysis."""
        # Implementation for exploration decay visualization
        pass
    
    def _plot_performance_ranking(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot detailed performance ranking visualization."""
        # Implementation for detailed ranking visualization
        pass
    
    def _plot_significance_matrix(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot statistical significance matrix."""
        # Implementation for significance testing visualization
        pass


    def _plot_robustness_analysis(self, testing_results: Dict[str, Any]) -> str:
        """Plot robustness analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robustness Analysis', fontsize=16, fontweight='bold')
        
        # Check if we have testing results
        if not testing_results or 'agent_results' not in testing_results:
            # Create empty plot with message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No testing results available\n(Training phase failed)', 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
            
            plot_file = self.testing_plots_dir / 'robustness_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        agent_results = testing_results.get('agent_results', {})
        
        # Extract robustness data
        agents = []
        robustness_scores = []
        variance_scores = []
        adaptation_scores = []
        
        for agent_name, results in agent_results.items():
            if results.get('status') == 'completed':
                agents.append(agent_name)
                
                robustness = results.get('robustness_metrics', {})
                robustness_scores.append(robustness.get('robustness_index', 0))
                variance_scores.append(robustness.get('performance_variance', 0))
                adaptation_scores.append(robustness.get('adaptation_rate', 0))
        
        # Plot 1: Robustness scores
        if agents:
            axes[0, 0].bar(agents, robustness_scores, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Robustness Index')
            axes[0, 0].set_ylabel('Robustness Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Performance variance
            axes[0, 1].bar(agents, variance_scores, alpha=0.7, color='orange')
            axes[0, 1].set_title('Performance Variance')
            axes[0, 1].set_ylabel('Variance Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Adaptation scores
            axes[1, 0].bar(agents, adaptation_scores, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Adaptation Rate')
            axes[1, 0].set_ylabel('Adaptation Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Robustness vs Adaptation scatter
            axes[1, 1].scatter(robustness_scores, adaptation_scores, s=100, alpha=0.7)
            for i, agent in enumerate(agents):
                axes[1, 1].annotate(agent, (robustness_scores[i], adaptation_scores[i]), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_xlabel('Robustness Index')
            axes[1, 1].set_ylabel('Adaptation Rate')
            axes[1, 1].set_title('Robustness vs Adaptation')
        else:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No completed agent results', 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.testing_plots_dir / 'robustness_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)

    def _plot_scenario_breakdown(self, testing_results: Dict[str, Any]) -> str:
        """Plot scenario performance breakdown."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check if we have testing results
        if not testing_results or 'agent_results' not in testing_results:
            ax.text(0.5, 0.5, 'No testing results available\n(Training phase failed)', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Scenario Performance Breakdown')
            
            plot_file = self.testing_plots_dir / 'scenario_breakdown.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        agent_results = testing_results.get('agent_results', {})
        
        # Create a simple breakdown visualization
        agents = list(agent_results.keys())
        scenarios = ['General Testing', 'Specific Testing']
        
        if agents:
            x = np.arange(len(agents))
            width = 0.35
            
            general_scores = []
            specific_scores = []
            
            for agent_name in agents:
                results = agent_results.get(agent_name, {})
                general_test = results.get('general_testing', {})
                specific_test = results.get('specific_testing', {})
                
                general_scores.append(general_test.get('generalization_score', 0))
                specific_scores.append(specific_test.get('adaptation_score', 0))
            
            bars1 = ax.bar(x - width/2, general_scores, width, label='General Testing', alpha=0.8)
            bars2 = ax.bar(x + width/2, specific_scores, width, label='Specific Testing', alpha=0.8)
            
            ax.set_xlabel('Agents')
            ax.set_ylabel('Performance Score')
            ax.set_title('Scenario Performance Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(agents, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No agent results to display', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.testing_plots_dir / 'scenario_breakdown.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)

    def _plot_training_efficiency(self, training_results: Dict[str, Any]) -> str:
        """Plot training efficiency analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Check if we have training results
        if not training_results or 'agent_results' not in training_results:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No training results available', 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
            
            plot_file = self.training_plots_dir / 'training_efficiency.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Create placeholder plots
        for i, ax in enumerate(axes.flat):
            ax.text(0.5, 0.5, f'Training Efficiency Plot {i+1}\n(To be implemented)', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'Efficiency Metric {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.training_plots_dir / 'training_efficiency.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)

    def _plot_exploration_decay(self, training_results: Dict[str, Any]) -> str:
        """Plot exploration decay analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if we have training results
        if not training_results or 'agent_results' not in training_results:
            ax.text(0.5, 0.5, 'No training results available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Exploration Decay Analysis')
            
            plot_file = self.training_plots_dir / 'exploration_decay.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Create placeholder plot
        ax.text(0.5, 0.5, 'Exploration Decay Analysis\n(To be implemented)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Exploration Decay Analysis')
        ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.training_plots_dir / 'exploration_decay.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)

    def _plot_performance_ranking(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot detailed performance ranking visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if we have evaluation results
        if not evaluation_results or 'agent_metrics' not in evaluation_results:
            ax.text(0.5, 0.5, 'No evaluation results available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Performance Ranking')
            
            plot_file = self.performance_plots_dir / 'performance_ranking.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Create placeholder plot
        ax.text(0.5, 0.5, 'Performance Ranking\n(To be implemented)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Performance Ranking')
        ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.performance_plots_dir / 'performance_ranking.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)

    def _plot_significance_matrix(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot statistical significance matrix."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Check if we have evaluation results
        if not evaluation_results or 'agent_metrics' not in evaluation_results:
            ax.text(0.5, 0.5, 'No evaluation results available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Statistical Significance Matrix')
            
            plot_file = self.performance_plots_dir / 'significance_matrix.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Create placeholder plot
        ax.text(0.5, 0.5, 'Statistical Significance Matrix\n(To be implemented)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Statistical Significance Matrix')
        ax.axis('off')
        
        plt.tight_layout()
        plot_file = self.performance_plots_dir / 'significance_matrix.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)    


    def _plot_performance_timeline(self, training_results: Dict[str, Any]) -> str:
            """Plot performance timeline visualization."""
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Check if we have training results
            if not training_results or 'agent_results' not in training_results:
                ax.text(0.5, 0.5, 'No training results available', 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                ax.set_title('Performance Timeline')
                
                plot_file = self.research_plots_dir / 'performance_timeline.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                return str(plot_file)
            
            agent_results = training_results.get('agent_results', {})
            
            # Plot learning curves for each agent
            for agent_name, results in agent_results.items():
                if results.get('status') == 'completed':
                    analytics = results.get('training_analytics', {})
                    episode_rewards = analytics.get('episode_rewards', [])
                    
                    if episode_rewards:
                        episodes = range(len(episode_rewards))
                        ax.plot(episodes, episode_rewards, label=agent_name, alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Reward')
            ax.set_title('Training Performance Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add moving average lines if data exists
            for agent_name, results in agent_results.items():
                if results.get('status') == 'completed':
                    analytics = results.get('training_analytics', {})
                    episode_rewards = analytics.get('episode_rewards', [])
                    
                    if len(episode_rewards) > 10:
                        # Calculate moving average
                        window = min(10, len(episode_rewards) // 4)
                        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                        episodes_avg = range(window-1, len(episode_rewards))
                        ax.plot(episodes_avg, moving_avg, '--', alpha=0.8, linewidth=1)
            
            plt.tight_layout()
            plot_file = self.research_plots_dir / 'performance_timeline.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)    


    def _plot_comparison_matrix(self, evaluation_results: Dict[str, Any]) -> str:
        """Plot comparison matrix visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Check if we have evaluation results
        if not evaluation_results or 'agent_metrics' not in evaluation_results:
            ax.text(0.5, 0.5, 'No evaluation results available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Agent Comparison Matrix')
            
            plot_file = self.research_plots_dir / 'comparison_matrix.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        agent_metrics = evaluation_results.get('agent_metrics', {})
        
        if not agent_metrics:
            ax.text(0.5, 0.5, 'No agent metrics available', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            ax.set_title('Agent Comparison Matrix')
            
            plot_file = self.research_plots_dir / 'comparison_matrix.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        # Create a simple comparison matrix
        agents = list(agent_metrics.keys())
        n_agents = len(agents)
        
        # Create matrix with some example metrics
        comparison_matrix = np.zeros((n_agents, n_agents))
        
        # Fill matrix with comparison values (example: performance differences)
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i == j:
                    comparison_matrix[i, j] = 1.0  # Same agent
                else:
                    # Example comparison metric
                    metrics_i = agent_metrics[agent_i]
                    metrics_j = agent_metrics[agent_j]
                    
                    # Try to get a performance metric for comparison
                    perf_i = 0
                    perf_j = 0
                    
                    for metric_name, metric_data in metrics_i.items():
                        if hasattr(metric_data, 'value') and 'cost' in metric_name.lower():
                            perf_i = metric_data.value
                            break
                    
                    for metric_name, metric_data in metrics_j.items():
                        if hasattr(metric_data, 'value') and 'cost' in metric_name.lower():
                            perf_j = metric_data.value
                            break
                    
                    # Calculate relative performance (smaller is better for cost)
                    if perf_j != 0:
                        comparison_matrix[i, j] = perf_i / perf_j
                    else:
                        comparison_matrix[i, j] = 1.0
        
        # Create heatmap
        im = ax.imshow(comparison_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n_agents))
        ax.set_yticks(range(n_agents))
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.set_yticklabels(agents)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Performance', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(n_agents):
            for j in range(n_agents):
                value = comparison_matrix[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=text_color, fontsize=10)
        
        ax.set_title('Agent Performance Comparison Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Compared Agent')
        ax.set_ylabel('Reference Agent')
        
        plt.tight_layout()
        plot_file = self.research_plots_dir / 'comparison_matrix.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)        