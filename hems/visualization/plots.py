"""
HEMS Visualization Module
Comprehensive visualization tools for HEMS simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path


class HEMSVisualizer:
    """Comprehensive visualization tools for HEMS simulation results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("Set2")
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
        })
    
    def plot_training_comparison(self, training_stats: Dict[str, Dict]):
        """Plot training curves comparison for all agents."""
        if not training_stats:
            return
        
        # Determine which agents have training data
        trainable_agents = {name: stats for name, stats in training_stats.items() 
                          if 'episode_rewards' in stats}
        
        if not trainable_agents:
            return
        
        n_agents = len(trainable_agents)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Episode rewards
        ax = axes[0]
        for agent_name, stats in trainable_agents.items():
            if 'episode_rewards' in stats:
                rewards = stats['episode_rewards']
                # Smooth with moving average
                window = max(1, len(rewards) // 20)
                if len(rewards) > window:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                    ax.plot(smoothed, label=f'{agent_name} (smoothed)', alpha=0.8)
                ax.plot(rewards, alpha=0.3, linewidth=0.5)
        
        ax.set_title('Training Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training time comparison
        ax = axes[1]
        agents = list(trainable_agents.keys())
        times = [trainable_agents[agent].get('total_training_time', 0) for agent in agents]
        bars = ax.bar(agents, times, color=sns.color_palette("Set2", len(agents)))
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        # Episodes completed
        ax = axes[2]
        episodes = [trainable_agents[agent].get('episodes_completed', 0) for agent in agents]
        bars = ax.bar(agents, episodes, color=sns.color_palette("Set2", len(agents)))
        ax.set_title('Episodes Completed')
        ax.set_ylabel('Episodes')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, ep_val in zip(bars, episodes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(episodes)*0.01,
                   f'{ep_val}', ha='center', va='bottom')
        
        # Learning curves (loss for DQN)
        ax = axes[3]
        for agent_name, stats in trainable_agents.items():
            if 'training_losses' in stats and stats['training_losses']:
                losses = stats['training_losses']
                # Smooth losses
                window = max(1, len(losses) // 50)
                if len(losses) > window:
                    smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
                    ax.plot(smoothed, label=f'{agent_name} loss', alpha=0.8)
        
        if any('training_losses' in stats for stats in trainable_agents.values()):
            ax.set_title('Training Loss (DQN)')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Training Loss')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_comparison.png', bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, results: Dict[str, Dict]):
        """Plot comprehensive performance comparison."""
        # Filter successful results
        successful_results = {name: result for name, result in results.items() 
                            if 'error' not in result}
        
        if len(successful_results) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract agents and metrics
        agents = list(successful_results.keys())
        
        # Total electricity cost comparison
        ax = axes[0, 0]
        costs = [successful_results[agent].get('total_electricity_cost', 0) for agent in agents]
        bars = ax.bar(agents, costs, color=sns.color_palette("Set1", len(agents)))
        ax.set_title('Total Electricity Cost')
        ax.set_ylabel('Cost (€)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                   f'€{cost:.2f}', ha='center', va='bottom')
        
        # Peak demand comparison
        ax = axes[0, 1]
        peaks = [successful_results[agent].get('peak_demand', 0) for agent in agents]
        bars = ax.bar(agents, peaks, color=sns.color_palette("Set2", len(agents)))
        ax.set_title('Peak Demand')
        ax.set_ylabel('Peak Demand (kWh)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, peak in zip(bars, peaks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(peaks)*0.01,
                   f'{peak:.2f}', ha='center', va='bottom')
        
        # Total reward comparison
        ax = axes[0, 2]
        rewards = [successful_results[agent].get('total_reward', 0) for agent in agents]
        bars = ax.bar(agents, rewards, color=sns.color_palette("Set3", len(agents)))
        ax.set_title('Total Reward')
        ax.set_ylabel('Reward')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, reward in zip(bars, rewards):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + (max(rewards) - min(rewards))*0.01
            else:
                va = 'top'
                y_pos = height - (max(rewards) - min(rewards))*0.01
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{reward:.1f}', ha='center', va=va)
        
        # Battery utilization
        ax = axes[1, 0]
        battery_means = [successful_results[agent].get('battery_soc_mean', 0.5) for agent in agents]
        bars = ax.bar(agents, battery_means, color=sns.color_palette("viridis", len(agents)))
        ax.set_title('Average Battery SoC')
        ax.set_ylabel('SoC')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, soc in zip(bars, battery_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{soc:.2f}', ha='center', va='bottom')
        
        # PV self-consumption
        ax = axes[1, 1]
        pv_rates = [successful_results[agent].get('pv_self_consumption_rate', 0) for agent in agents]
        bars = ax.bar(agents, pv_rates, color=sns.color_palette("plasma", len(agents)))
        ax.set_title('PV Self-Consumption Rate')
        ax.set_ylabel('Self-Consumption Rate')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars, pv_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # Cost savings (relative to baseline/highest cost)
        ax = axes[1, 2]
        if costs:
            baseline_cost = max(costs)
            savings_pct = [(baseline_cost - cost) / baseline_cost * 100 for cost in costs]
            colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in savings_pct]
            bars = ax.bar(agents, savings_pct, color=colors, alpha=0.7)
            ax.set_title('Cost Savings vs Highest Cost')
            ax.set_ylabel('Savings (%)')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, saving in zip(bars, savings_pct):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                y_offset = 1 if height >= 0 else -1
                ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                       f'{saving:.1f}%', ha='center', va=va)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', bbox_inches='tight')
        plt.close()
    
    def plot_load_profiles_comparison(self, environments: Dict[str, Any]):
        """Plot load profiles comparison across agents."""
        if not environments:
            return
        
        # Collect load data for each agent
        load_data = {}
        for agent_name, env in environments.items():
            try:
                consumption = np.array(env.unwrapped.net_electricity_consumption, dtype=float)
                if len(consumption) > 0:
                    load_data[agent_name] = consumption
            except Exception:
                continue
        
        if not load_data:
            return
        
        # Determine time period to plot (first week if available)
        max_length = max(len(data) for data in load_data.values())
        plot_length = min(168, max_length)  # One week max
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Time series comparison
        ax = axes[0]
        for agent_name, consumption in load_data.items():
            time_steps = range(min(plot_length, len(consumption)))
            ax.plot(time_steps, consumption[:plot_length], label=agent_name, linewidth=1.5)
        
        ax.set_title(f'Load Profiles Comparison (First {plot_length} hours)')
        ax.set_xlabel('Time Step (hours)')
        ax.set_ylabel('Net Consumption (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        
        # Daily average patterns
        ax = axes[1]
        for agent_name, consumption in load_data.items():
            if len(consumption) >= 24:
                # Calculate daily average pattern
                daily_pattern = self._calculate_daily_pattern(consumption)
                ax.plot(range(24), daily_pattern, label=agent_name, marker='o', linewidth=2)
        
        ax.set_title('Average Daily Load Patterns')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Consumption (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))
        
        # Load distribution comparison
        ax = axes[2]
        load_distributions = []
        labels = []
        for agent_name, consumption in load_data.items():
            load_distributions.append(consumption)
            labels.append(agent_name)
        
        ax.boxplot(load_distributions, labels=labels)
        ax.set_title('Load Distribution Comparison')
        ax.set_ylabel('Net Consumption (kWh)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'load_profiles_comparison.png', bbox_inches='tight')
        plt.close()
    
    def plot_battery_comparison(self, environments: Dict[str, Any]):
        """Plot battery operation comparison across agents."""
        if not environments:
            return
        
        # Collect battery data
        battery_data = {}
        for agent_name, env in environments.items():
            try:
                if env.unwrapped.buildings:
                    building = env.unwrapped.buildings[0]  # Use first building
                    soc = np.array(building.electrical_storage.soc, dtype=float)
                    if len(soc) > 0:
                        battery_data[agent_name] = soc
            except Exception:
                continue
        
        if not battery_data:
            return
        
        # Determine time period
        max_length = max(len(data) for data in battery_data.values())
        plot_length = min(168, max_length)  # One week
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SoC time series
        ax = axes[0, 0]
        for agent_name, soc in battery_data.items():
            time_steps = range(min(plot_length, len(soc)))
            ax.plot(time_steps, soc[:plot_length], label=agent_name, linewidth=1.5)
        
        ax.set_title(f'Battery SoC Comparison (First {plot_length} hours)')
        ax.set_xlabel('Time Step (hours)')
        ax.set_ylabel('State of Charge')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        
        # SoC distribution
        ax = axes[0, 1]
        soc_distributions = []
        labels = []
        for agent_name, soc in battery_data.items():
            soc_distributions.append(soc)
            labels.append(agent_name)
        
        ax.boxplot(soc_distributions, labels=labels)
        ax.set_title('SoC Distribution Comparison')
        ax.set_ylabel('State of Charge')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Daily SoC patterns
        ax = axes[1, 0]
        for agent_name, soc in battery_data.items():
            if len(soc) >= 24:
                daily_soc = self._calculate_daily_pattern(soc)
                ax.plot(range(24), daily_soc, label=agent_name, marker='o', linewidth=2)
        
        ax.set_title('Average Daily SoC Patterns')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average SoC')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))
        
        # SoC statistics
        ax = axes[1, 1]
        stats_data = []
        for agent_name, soc in battery_data.items():
            stats = {
                'Agent': agent_name,
                'Mean SoC': np.mean(soc),
                'SoC Std': np.std(soc),
                'Min SoC': np.min(soc),
                'Max SoC': np.max(soc),
                'Range': np.max(soc) - np.min(soc)
            }
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create a simple table
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=stats_df.round(3).values,
                        colLabels=stats_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Battery Statistics Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'battery_comparison.png', bbox_inches='tight')
        plt.close()
    
    def plot_cost_breakdown(self, results: Dict[str, Dict]):
        """Plot detailed cost breakdown comparison."""
        successful_results = {name: result for name, result in results.items() 
                            if 'error' not in result}
        
        if len(successful_results) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        agents = list(successful_results.keys())
        
        # Total cost vs components
        ax = axes[0, 0]
        import_costs = [successful_results[agent].get('import_cost', 0) for agent in agents]
        export_revenues = [successful_results[agent].get('export_revenue', 0) for agent in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, import_costs, width, label='Import Cost', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, export_revenues, width, label='Export Revenue', color='green', alpha=0.7)
        
        ax.set_title('Import Cost vs Export Revenue')
        ax.set_ylabel('Cost/Revenue (€)')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Net cost comparison
        ax = axes[0, 1]
        net_costs = [successful_results[agent].get('net_electricity_cost', 0) for agent in agents]
        colors = sns.color_palette("RdYlGn_r", len(agents))
        bars = ax.bar(agents, net_costs, color=colors)
        ax.set_title('Net Electricity Cost')
        ax.set_ylabel('Net Cost (€)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars, net_costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(net_costs)*0.01,
                   f'€{cost:.2f}', ha='center', va='bottom')
        
        # Cost per kWh
        ax = axes[1, 0]
        total_costs = [successful_results[agent].get('total_electricity_cost', 0) for agent in agents]
        total_steps = [successful_results[agent].get('total_steps', 1) for agent in agents]
        cost_per_step = [cost/steps if steps > 0 else 0 for cost, steps in zip(total_costs, total_steps)]
        
        bars = ax.bar(agents, cost_per_step, color=sns.color_palette("viridis", len(agents)))
        ax.set_title('Average Cost per Time Step')
        ax.set_ylabel('Cost per Step (€/hour)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars, cost_per_step):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(cost_per_step)*0.01,
                   f'€{cost:.4f}', ha='center', va='bottom')
        
        # Cost savings summary
        ax = axes[1, 1]
        if total_costs:
            baseline_cost = max(total_costs)  # Use highest cost as baseline
            absolute_savings = [baseline_cost - cost for cost in total_costs]
            relative_savings = [saving / baseline_cost * 100 for saving in absolute_savings]
            
            # Create combined bar chart
            x = np.arange(len(agents))
            
            # Absolute savings (left y-axis)
            color = 'tab:blue'
            ax.set_xlabel('Agent')
            ax.set_ylabel('Absolute Savings (€)', color=color)
            bars1 = ax.bar(x - 0.2, absolute_savings, 0.4, color=color, alpha=0.7, label='Absolute')
            ax.tick_params(axis='y', labelcolor=color)
            ax.tick_params(axis='x', rotation=45)
            
            # Relative savings (right y-axis)
            ax2 = ax.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Relative Savings (%)', color=color)
            bars2 = ax2.bar(x + 0.2, relative_savings, 0.4, color=color, alpha=0.7, label='Relative')
            ax2.tick_params(axis='y', labelcolor=color)
            
            ax.set_title('Cost Savings vs Highest Cost Agent')
            ax.set_xticks(x)
            ax.set_xticklabels(agents)
            
            # Add value labels
            for bar, saving in zip(bars1, absolute_savings):
                height = bar.get_height()
                if height >= 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(absolute_savings)*0.01,
                           f'€{saving:.2f}', ha='center', va='bottom', color='blue')
            
            for bar, saving in zip(bars2, relative_savings):
                height = bar.get_height()
                if height >= 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(relative_savings)*0.01,
                           f'{saving:.1f}%', ha='center', va='bottom', color='orange')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_breakdown.png', bbox_inches='tight')
        plt.close()
    
    def plot_action_analysis(self, agent_name: str, actions: np.ndarray, 
                           environment: Any, observations: List = None):
        """Plot detailed action analysis for a specific agent."""
        if len(actions) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Action time series
        ax = axes[0, 0]
        if actions.ndim == 2:
            for i in range(actions.shape[1]):
                ax.plot(actions[:, i], label=f'Building {i+1}', linewidth=1.5)
            ax.legend()
        else:
            ax.plot(actions, linewidth=1.5)
        
        ax.set_title(f'{agent_name} - Action Time Series')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action (charge-/discharge+)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Action distribution
        ax = axes[0, 1]
        if actions.ndim == 2:
            for i in range(actions.shape[1]):
                ax.hist(actions[:, i], bins=30, alpha=0.7, label=f'Building {i+1}')
            ax.legend()
        else:
            ax.hist(actions, bins=30, alpha=0.7)
        
        ax.set_title(f'{agent_name} - Action Distribution')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Action vs time of day
        ax = axes[1, 0]
        if len(actions) >= 24:
            hours = np.arange(len(actions)) % 24
            if actions.ndim == 2:
                action_mean = np.mean(actions, axis=1)
            else:
                action_mean = actions
            
            # Calculate hourly average actions
            hourly_actions = [np.mean(action_mean[hours == h]) for h in range(24)]
            
            ax.plot(range(24), hourly_actions, marker='o', linewidth=2)
            ax.set_title(f'{agent_name} - Average Action by Hour')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Average Action')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xticks(range(0, 24, 4))
        
        # Action statistics
        ax = axes[1, 1]
        if actions.ndim == 2:
            stats_data = []
            for i in range(actions.shape[1]):
                building_actions = actions[:, i]
                stats = {
                    'Building': f'Building {i+1}',
                    'Mean': np.mean(building_actions),
                    'Std': np.std(building_actions),
                    'Min': np.min(building_actions),
                    'Max': np.max(building_actions),
                    'Charge %': np.mean(building_actions < -0.1) * 100,
                    'Discharge %': np.mean(building_actions > 0.1) * 100
                }
                stats_data.append(stats)
        else:
            stats_data = [{
                'Building': 'Single',
                'Mean': np.mean(actions),
                'Std': np.std(actions),
                'Min': np.min(actions),
                'Max': np.max(actions),
                'Charge %': np.mean(actions < -0.1) * 100,
                'Discharge %': np.mean(actions > 0.1) * 100
            }]
        
        stats_df = pd.DataFrame(stats_data)
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=stats_df.round(3).values,
                        colLabels=stats_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title(f'{agent_name} - Action Statistics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{agent_name}_action_analysis.png', bbox_inches='tight')
        plt.close()
    
    def _calculate_daily_pattern(self, data: np.ndarray) -> np.ndarray:
        """Calculate average daily pattern from time series data."""
        if len(data) < 24:
            return data
        
        # Reshape data into days and hours, then average across days
        n_complete_days = len(data) // 24
        daily_data = data[:n_complete_days * 24].reshape(-1, 24)
        return np.mean(daily_data, axis=0)
    
    def create_summary_dashboard(self, results: Dict[str, Dict], 
                               environments: Dict[str, Any] = None):
        """Create a comprehensive summary dashboard."""
        if not results:
            return
        
        successful_results = {name: result for name, result in results.items() 
                            if 'error' not in result}
        
        if not successful_results:
            return
        
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        agents = list(successful_results.keys())
        
        # Title
        fig.suptitle('HEMS Simulation Results Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Cost comparison
        ax = fig.add_subplot(gs[0, :2])
        costs = [successful_results[agent].get('total_electricity_cost', 0) for agent in agents]
        bars = ax.bar(agents, costs, color=sns.color_palette("Set1", len(agents)))
        ax.set_title('Total Electricity Cost Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cost (€)')
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                   f'€{cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance metrics
        ax = fig.add_subplot(gs[0, 2:])
        metrics = ['peak_demand', 'average_demand', 'battery_soc_mean']
        metric_labels = ['Peak Demand (kWh)', 'Avg Demand (kWh)', 'Avg Battery SoC']
        
        x = np.arange(len(agents))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [successful_results[agent].get(metric, 0) for agent in agents]
            if any(v != 0 for v in values):  # Only plot if we have data
                ax.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents, rotation=45)
        ax.legend()
        
        # Add more plots if environments are available
        if environments:
            # 3. Load profiles (first 48 hours)
            ax = fig.add_subplot(gs[1, :])
            for agent_name, env in environments.items():
                try:
                    consumption = np.array(env.unwrapped.net_electricity_consumption, dtype=float)
                    plot_hours = min(48, len(consumption))
                    ax.plot(range(plot_hours), consumption[:plot_hours], 
                           label=agent_name, linewidth=2)
                except Exception:
                    continue
            
            ax.set_title('Load Profiles Comparison (First 48 Hours)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Net Consumption (kWh)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Results summary table
        ax = fig.add_subplot(gs[2:, :])
        
        # Create comprehensive results table
        table_data = []
        for agent in agents:
            result = successful_results[agent]
            row = {
                'Agent': agent,
                'Total Cost (€)': f"{result.get('total_electricity_cost', 0):.3f}",
                'Net Cost (€)': f"{result.get('net_electricity_cost', 0):.3f}",
                'Peak Demand (kWh)': f"{result.get('peak_demand', 0):.3f}",
                'Avg Demand (kWh)': f"{result.get('average_demand', 0):.3f}",
                'Total Reward': f"{result.get('total_reward', 0):.2f}",
                'Avg Battery SoC': f"{result.get('battery_soc_mean', 0):.3f}",
                'PV Self-Consumption': f"{result.get('pv_self_consumption_rate', 0):.1%}",
            }
            table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_df.values,
                        colLabels=table_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(table_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Comprehensive Results Summary', fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(self.output_dir / 'summary_dashboard.png', bbox_inches='tight')
        plt.close()