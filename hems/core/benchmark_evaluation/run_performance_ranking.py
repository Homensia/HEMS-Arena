#!/usr/bin/env python3
"""
Standalone Performance Scorer (TOPSIS + DEA + Sensitivity)
Reads evaluation_results.json and produces scientific rankings.

Usage:
    python run_performance_ranking.py <evaluation_results.json>
    
Output:
    - ranking_results.json (detailed results)
    - ranking_report.txt (human-readable)
"""

import sys
import json
from pathlib import Path
import logging

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


def extract_agent_kpis(evaluation_data: dict) -> dict:
    """
    Extract agent KPIs from evaluation results.
    
    Returns:
        {agent_name: {kpi_name: value}}
    """
    agent_kpis = {}
    
    agent_results = evaluation_data.get('agent_results', {})
    
    for agent_name, agent_data in agent_results.items():
        # Get testing phase KPIs
        testing = agent_data.get('testing', {})
        
        if testing.get('status') != 'completed':
            logger.warning(f"Skipping {agent_name}: testing not completed")
            continue
        
        kpis = testing.get('agent_kpis', {})
        
        if not kpis:
            logger.warning(f"Skipping {agent_name}: no KPIs found")
            continue
        
        agent_kpis[agent_name] = kpis
        logger.info(f"Extracted {len(kpis)} KPIs for {agent_name}")
    
    return agent_kpis


def run_ranking(agent_kpis: dict) -> dict:
    """Run scientific ranking (TOPSIS + DEA + Sensitivity)."""
    from scientific_ranking import ScientificRankingSystem
    
    logger.info(f"Running scientific ranking on {len(agent_kpis)} agents")
    
    if len(agent_kpis) < 2:
        logger.error("Need at least 2 agents for ranking!")
        return {
            'error': 'Insufficient agents',
            'message': 'Need at least 2 agents with completed testing for ranking'
        }
    
    # Initialize ranker
    ranker = ScientificRankingSystem(logger_instance=logger)
    
    # Run complete evaluation
    results = ranker.evaluate_all(agent_kpis)
    
    logger.info("Ranking completed successfully")
    return results


def save_results(results: dict, output_dir: Path):
    """Save ranking results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_file = output_dir / 'ranking_results.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed results: {json_file}")
    
    # Save human-readable report
    if 'error' not in results:
        from scientific_ranking import ScientificRankingSystem
        ranker = ScientificRankingSystem()
        report = ranker.format_results(results)
        
        txt_file = output_dir / 'ranking_report.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved readable report: {txt_file}")


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_performance_ranking.py <evaluation_results.json>")
        print("\nExample:")
        print("  python run_performance_ranking.py experiments/exp_001/evaluation/evaluation_results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)
    
    # Determine output directory (same as input)
    output_dir = Path(input_file).parent / 'ranking_analysis'
    
    print("=" * 80)
    print("STANDALONE PERFORMANCE RANKING ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    evaluation_data = load_evaluation_results(input_file)
    
    # Extract KPIs
    agent_kpis = extract_agent_kpis(evaluation_data)
    
    if len(agent_kpis) < 2:
        logger.error("Cannot proceed: Need at least 2 agents with completed testing")
        logger.info(f"Found {len(agent_kpis)} agent(s) with KPIs")
        sys.exit(1)
    
    # Run ranking
    results = run_ranking(agent_kpis)
    
    # Save results
    save_results(results, output_dir)
    
    print()
    print("=" * 80)
    print(f"✅ RANKING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - ranking_results.json (detailed)")
    print(f"  - ranking_report.txt (readable)")
    print()
    
    # Show quick summary
    if 'error' not in results:
        primary = results.get('primary_ranking', {})
        rankings = primary.get('rankings', [])
        
        print("Top 3 Agents (TOPSIS):")
        for item in rankings[:3]:
            print(f"  #{item['rank']}: {item['agent']} (score: {item['topsis_score']:.4f})")
        print()


if __name__ == '__main__':
    main()