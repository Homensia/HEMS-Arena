# =======================================================
# hems/core/benchmark_evaluation/statistical_analyzer.py
# =======================================================

"""
Statistical Analyzer
Performs statistical tests to compare agents.

Key Features:
- Pairwise comparisons (t-tests, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Confidence intervals
- Significance testing
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on agent results.
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize analyzer."""
        self.logger = logger_instance or logger
        self.alpha = 0.05  # Significance level
    
    def analyze(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on agent results.
        
        Args:
            agent_results: Results for all agents
            
        Returns:
            Statistical analysis results
        """
        self.logger.info("[STATS] Running statistical analysis")
        
        # Extract reward distributions
        agent_rewards = self._extract_rewards(agent_results)
        
        if len(agent_rewards) < 2:
            self.logger.warning("[STATS] Need at least 2 agents for comparison")
            return {}
        
        # Pairwise comparisons
        comparisons = self._pairwise_comparisons(agent_rewards)
        
        # Rankings
        rankings = self._compute_rankings(agent_rewards)
        
        # Overall statistics
        overall = self._compute_overall_stats(agent_rewards)
        
        results = {
            'num_agents': len(agent_rewards),
            'comparisons': comparisons,
            'rankings': rankings,
            'overall': overall,
        }
        
        self.logger.info(f"[STATS] Completed {len(comparisons)} pairwise comparisons")
        
        return results
    
    def _extract_rewards(self, agent_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract reward distributions for each agent.
        
        Returns:
            Dict mapping agent_name -> reward array
        """
        agent_rewards = {}
        
        for agent_name, results in agent_results.items():
            testing = results.get('testing', {})
            if testing.get('status') != 'completed':
                continue
            
            # Get episode rewards
            agent_data = testing.get('agent_data', {})
            rewards = agent_data.get('rewards', [])
            
            if rewards and len(rewards) > 0:
                agent_rewards[agent_name] = np.array(rewards)
        
        return agent_rewards
    
    def _pairwise_comparisons(
        self,
        agent_rewards: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Perform pairwise statistical comparisons.
        
        Returns:
            List of comparison results
        """
        comparisons = []
        agent_names = list(agent_rewards.keys())
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent_a = agent_names[i]
                agent_b = agent_names[j]
                
                rewards_a = agent_rewards[agent_a]
                rewards_b = agent_rewards[agent_b]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)
                
                # Mann-Whitney U test (non-parametric alternative)
                u_stat, u_p_value = stats.mannwhitneyu(
                    rewards_a, rewards_b, alternative='two-sided'
                )
                
                # Effect size (Cohen's d)
                effect_size = self._cohens_d(rewards_a, rewards_b)
                
                # Confidence intervals
                ci_a = self._confidence_interval(rewards_a)
                ci_b = self._confidence_interval(rewards_b)
                
                comparison = {
                    'agent_a': agent_a,
                    'agent_b': agent_b,
                    'mean_a': float(np.mean(rewards_a)),
                    'mean_b': float(np.mean(rewards_b)),
                    'std_a': float(np.std(rewards_a)),
                    'std_b': float(np.std(rewards_b)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'u_statistic': float(u_stat),
                    'u_p_value': float(u_p_value),
                    'effect_size': float(effect_size),
                    'effect_size_interpretation': self._interpret_effect_size(effect_size),
                    'significant': p_value < self.alpha,
                    'ci_a': ci_a,
                    'ci_b': ci_b,
                }
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            a: First sample
            b: Second sample
            
        Returns:
            Cohen's d value
        """
        n_a = len(a)
        n_b = len(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(a) - np.mean(b)) / pooled_std
    
    def _interpret_effect_size(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval.
        
        Args:
            data: Data array
            confidence: Confidence level (default 0.95)
            
        Returns:
            (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return (float(mean - margin), float(mean + margin))
    
    def _compute_rankings(
        self,
        agent_rewards: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Rank agents by performance.
        
        Returns:
            List of rankings with confidence intervals
        """
        rankings = []
        
        for agent_name, rewards in agent_rewards.items():
            mean_reward = np.mean(rewards)
            ci = self._confidence_interval(rewards)
            
            rankings.append({
                'agent': agent_name,
                'mean_reward': float(mean_reward),
                'std_reward': float(np.std(rewards)),
                'confidence_interval': ci,
            })
        
        # Sort by mean reward (descending)
        rankings.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        # Add rank
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _compute_overall_stats(
        self,
        agent_rewards: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute overall statistics across all agents.
        
        Returns:
            Overall statistics
        """
        all_rewards = np.concatenate(list(agent_rewards.values()))
        
        return {
            'total_episodes': len(all_rewards),
            'overall_mean': float(np.mean(all_rewards)),
            'overall_std': float(np.std(all_rewards)),
            'overall_min': float(np.min(all_rewards)),
            'overall_max': float(np.max(all_rewards)),
        }