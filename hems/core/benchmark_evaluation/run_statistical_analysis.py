#!/usr/bin/env python3
"""
Standalone Statistical Analyzer (t-tests, effect sizes, CI)
Reads evaluation_results.json and produces statistical comparisons.

Usage:
    python run_statistical_analysis.py <evaluation_results.json>
    
Output:
    - statistical_results.json (detailed results)
    - statistical_report.txt (human-readable)
"""

import sys
import json
from pathlib import Path
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_results(filepath: str) -> dict:
    """Load evaluation results JSON file."""
    logger.info(f"Loading evaluation results from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data for {len(data.get('agent_results', {}))} agents")
    return data


def extract_agent_rewards(evaluation_data: dict) -> dict:
    """
    Extract episode rewards from evaluation results.
    
    Returns:
        {agent_name: np.array of rewards}
    """
    agent_rewards = {}
    
    agent_results = evaluation_data.get('agent_results', {})
    
    for agent_name, agent_data in agent_results.items():
        # Get testing phase data
        testing = agent_data.get('testing', {})
        
        if testing.get('status') != 'completed':
            logger.warning(f"Skipping {agent_name}: testing not completed")
            continue
        
        # Try to get rewards from testing data
        agent_test_data = testing.get('agent_data', {})
        rewards = agent_test_data.get('rewards', [])
        
        if not rewards:
            # Try episode data
            episode_data = agent_test_data.get('episode_data', [])
            if episode_data:
                rewards = [ep.get('total_reward', 0) for ep in episode_data]
        
        if not rewards:
            logger.warning(f"Skipping {agent_name}: no rewards found")
            continue
        
        agent_rewards[agent_name] = np.array(rewards)
        logger.info(f"Extracted {len(rewards)} episode rewards for {agent_name}")
    
    return agent_rewards


def run_statistical_analysis(agent_rewards: dict) -> dict:
    """Run statistical analysis (t-tests, effect sizes, CI)."""
    from statistical_analyzer import StatisticalAnalyzer
    
    logger.info(f"Running statistical analysis on {len(agent_rewards)} agents")
    
    if len(agent_rewards) < 2:
        logger.error("Need at least 2 agents for statistical comparison!")
        return {
            'error': 'Insufficient agents',
            'message': 'Need at least 2 agents with completed testing for statistical analysis'
        }
    
    # Build agent_results structure expected by StatisticalAnalyzer
    agent_results = {}
    for agent_name, rewards in agent_rewards.items():
        agent_results[agent_name] = {
            'testing': {
                'status': 'completed',
                'agent_data': {
                    'rewards': rewards.tolist()
                }
            }
        }
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(logger_instance=logger)
    
    # Run analysis
    results = analyzer.analyze(agent_results)
    
    logger.info("Statistical analysis completed successfully")
    return results


def format_statistical_report(results: dict) -> str:
    """Format statistical results as human-readable report."""
    if 'error' in results:
        return f"Error: {results['message']}"
    
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overview
    n_agents = results.get('num_agents', 0)
    comparisons = results.get('comparisons', [])
    report.append(f"Number of agents: {n_agents}")
    report.append(f"Pairwise comparisons: {len(comparisons)}")
    report.append("")
    
    # Rankings
    report.append("-" * 80)
    report.append("RANKINGS (by mean reward)")
    report.append("-" * 80)
    rankings = results.get('rankings', [])
    for ranking in rankings:
        agent = ranking['agent']
        mean = ranking['mean_reward']
        std = ranking['std_reward']
        ci = ranking['confidence_interval']
        rank = ranking['rank']
        
        report.append(f"\n#{rank}: {agent}")
        report.append(f"  Mean reward: {mean:.2f} (±{std:.2f})")
        report.append(f"  95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    # Comparisons
    report.append("")
    report.append("-" * 80)
    report.append("PAIRWISE COMPARISONS")
    report.append("-" * 80)
    
    for comp in comparisons:
        agent_a = comp['agent_a']
        agent_b = comp['agent_b']
        
        report.append(f"\n{agent_a} vs {agent_b}:")
        report.append(f"  Mean {agent_a}: {comp['mean_a']:.2f}")
        report.append(f"  Mean {agent_b}: {comp['mean_b']:.2f}")
        report.append(f"  Difference: {comp['difference']:.2f}")
        report.append(f"  p-value: {comp['p_value']:.4f} {'***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*' if comp['p_value'] < 0.05 else ''}")
        report.append(f"  Cohen's d: {comp['cohens_d']:.3f} ({'Large' if abs(comp['cohens_d']) >= 0.8 else 'Medium' if abs(comp['cohens_d']) >= 0.5 else 'Small' if abs(comp['cohens_d']) >= 0.2 else 'Negligible'})")
        report.append(f"  95% CI: [{comp['ci_95_lower']:.2f}, {comp['ci_95_upper']:.2f}]")
        report.append(f"  Significant: {'YES' if comp['significant'] else 'NO'}")
        report.append(f"  Winner: {comp['winner']}")
    
    # Overall stats
    report.append("")
    report.append("-" * 80)
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    overall = results.get('overall', {})
    if overall:
        report.append(f"\nBest mean reward: {overall.get('best_mean', 0):.2f}")
        report.append(f"Worst mean reward: {overall.get('worst_mean', 0):.2f}")
        report.append(f"Range: {overall.get('range', 0):.2f}")
        report.append(f"Average std: {overall.get('avg_std', 0):.2f}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def save_results(results: dict, output_dir: Path):
    """Save statistical results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_file = output_dir / 'statistical_results.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed results: {json_file}")
    
    # Save human-readable report
    if 'error' not in results:
        report = format_statistical_report(results)
        
        txt_file = output_dir / 'statistical_report.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved readable report: {txt_file}")


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_statistical_analysis.py <evaluation_results.json>")
        print("\nExample:")
        print("  python run_statistical_analysis.py experiments/exp_001/evaluation/evaluation_results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)
    
    # Determine output directory (same as input)
    output_dir = Path(input_file).parent / 'statistical_analysis'
    
    print("=" * 80)
    print("STANDALONE STATISTICAL ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    evaluation_data = load_evaluation_results(input_file)
    
    # Extract rewards
    agent_rewards = extract_agent_rewards(evaluation_data)
    
    if len(agent_rewards) < 2:
        logger.error("Cannot proceed: Need at least 2 agents with reward data")
        logger.info(f"Found {len(agent_rewards)} agent(s) with rewards")
        sys.exit(1)
    
    # Run analysis
    results = run_statistical_analysis(agent_rewards)
    
    # Save results
    save_results(results, output_dir)
    
    print()
    print("=" * 80)
    print(f"✅ STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - statistical_results.json (detailed)")
    print(f"  - statistical_report.txt (readable)")
    print()
    
    # Show quick summary
    if 'error' not in results:
        rankings = results.get('rankings', [])
        comparisons = results.get('comparisons', [])
        
        print("Agent Rankings:")
        for rank_info in rankings[:5]:
            print(f"  #{rank_info['rank']}: {rank_info['agent']} (mean: {rank_info['mean_reward']:.2f})")
        
        print(f"\nSignificant differences found: {sum(1 for c in comparisons if c['significant'])}/{len(comparisons)}")
        print()


if __name__ == '__main__':
    main()