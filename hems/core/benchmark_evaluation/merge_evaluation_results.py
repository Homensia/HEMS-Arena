#!/usr/bin/env python3
"""
Merge Evaluation Results
Combines evaluation_results.json from multiple experiments into one file.

Usage:
    python merge_evaluation_results.py <file1.json> <file2.json> ... -o <output.json>
    
Example:
    python merge_evaluation_results.py exp_001/evaluation_results.json \
                                       exp_002/evaluation_results.json \
                                       exp_003/evaluation_results.json \
                                       -o merged_results.json
"""

import sys
import json
from pathlib import Path
import argparse
from datetime import datetime


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_evaluation_results(files: list) -> dict:
    """
    Merge multiple evaluation_results.json files.
    
    Takes agent_results from all files and combines them.
    Uses metadata from first file as base.
    """
    if not files:
        raise ValueError("No files provided")
    
    print(f"Merging {len(files)} evaluation result files...")
    
    # Load first file as base
    base = load_json(files[0])
    print(f"✓ Loaded base: {files[0]}")
    
    merged = {
        'benchmark_name': base.get('benchmark_name', 'merged_benchmark'),
        'experiment_id': f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'configuration': base.get('configuration', {}),
        'agent_results': {},
        'summary': {
            'num_agents': 0,
            'agents': [],
            'source_files': files
        }
    }
    
    # Collect all agent_results
    all_agents = set()
    
    for filepath in files:
        data = load_json(filepath)
        agent_results = data.get('agent_results', {})
        
        for agent_name, agent_data in agent_results.items():
            if agent_name in merged['agent_results']:
                print(f"⚠ Warning: Agent '{agent_name}' appears in multiple files - using first occurrence")
                continue
            
            merged['agent_results'][agent_name] = agent_data
            all_agents.add(agent_name)
            print(f"  ✓ Added agent: {agent_name}")
    
    # Update summary
    merged['summary']['num_agents'] = len(all_agents)
    merged['summary']['agents'] = sorted(list(all_agents))
    
    print(f"\n✓ Merged {len(all_agents)} unique agents")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge evaluation results from multiple experiments'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Evaluation result JSON files to merge'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MERGE EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    # Check all files exist
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"✗ Error: File not found: {filepath}")
            sys.exit(1)
    
    # Merge
    try:
        merged = merge_evaluation_results(args.files)
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        sys.exit(1)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, default=str)
    
    print()
    print("=" * 80)
    print("✓ MERGE COMPLETE")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print(f"Total agents: {merged['summary']['num_agents']}")
    print(f"Agents: {', '.join(merged['summary']['agents'])}")
    print()
    print("You can now run:")
    print(f"  python run_performance_ranking.py {output_path}")
    print(f"  python run_statistical_analysis.py {output_path}")
    print()


if __name__ == '__main__':
    main()