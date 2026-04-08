# ========================================================================
# hems/agents/ambitious_engineers_algorithm.py - run_complete_training.py
# ========================================================================


"""
run_complete_training.py - Complete Training Pipeline (FIXED)
==============================================================
Master script to run the entire Team ambitiousengineers training pipeline.

FIXED: Correct ordering - Phase 1 (Step 6) now runs BEFORE Phase 2/3 (Step 5)

Total Expected Runtime: ~4 days with 50 parallel workers

Pipeline Stages:
1. Initialize Data (5 seconds)
2. Train Demand Model (20 minutes)
3. Train Solar Model (5 minutes)
4. Compute DP Baseline (15 minutes)
5. Train Phase 1 Policy (2.5 days) ← CORRECTED
6. Train Multi-Agent Policy (1.5 days)
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta


def run_script(script_path: str, description: str, expected_time: str) -> bool:
    """Run a training script and handle errors."""
    print("\n" + "="*80)
    print(f"  {description}")
    print("="*80)
    print(f"Expected runtime: {expected_time}")
    print(f"Script: {script_path}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {format_time(elapsed)}")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {format_time(elapsed)}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n✗ {description} interrupted by user")
        return False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print("="*80)
    print("  Checking Prerequisites")
    print("="*80 + "\n")
    
    # Check for data directory
    data_path = Path("datasets/citylearn_datasets/citylearn_challenge_2022_phase_1")
    if not data_path.exists():
        print("✗ CityLearn Challenge 2022 Phase 1 data not found!")
        print(f"  Expected location: {data_path}")
        print("\nPlease download the dataset and place it in the correct location.")
        return False
    print(f"✓ Data directory found: {data_path}")
    
    # Check for required Python packages
    required_packages = [
        'numpy', 'pandas', 'torch', 'cma', 'sklearn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ Package {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ Package {package} not found")
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    # Check for GPU (optional but recommended)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠  No GPU detected - training will be slower")
    except:
        pass
    
    print("\n✓ All prerequisites met\n")
    return True


def main():
    print("="*80)
    print("  TEAM AMBITIOUSENGINEERS - COMPLETE TRAINING PIPELINE")
    print("="*80)
    print("\nThis script will train the complete Team ambitiousengineers model")
    print("from the CityLearn Challenge 2022 (2nd Place Solution).\n")
    
    print("Training Pipeline (CORRECTED ORDER):")
    print("  1. Initialize Data                 (5 seconds)")
    print("  2. Train Demand Forecaster         (20 minutes)")
    print("  3. Train Solar Forecaster          (5 minutes)")
    print("  4. Compute DP Baseline             (15 minutes)")
    print("  5. Train Phase 1 Policy (CMA-ES)   (2.5 days) ← Step 6 script")
    print("  6. Train Multi-Agent Policy         (1.5 days) ← Step 5 script")
    print(f"  {'-'*50}")
    print("  TOTAL EXPECTED TIME: ~4 days with 50 parallel workers\n")
    
    print("⚠  WARNING: This training will take approximately 4 DAYS!")
    print("   Make sure you have:")
    print("   - A stable system that won't be interrupted")
    print("   - Sufficient compute resources (50+ CPU cores recommended)")
    print("   - ~100GB free disk space for checkpoints")
    print("   - GPU for forecasting model training (recommended)\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nTraining cancelled.")
        return 0
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Get algorithm directory
    algo_dir = Path(__file__).parent
    
    # Training pipeline stages - CORRECTED ORDER
    stages = [
        {
            'script': algo_dir / '1_initialize.py',
            'description': 'STAGE 1: Initialize Data',
            'expected_time': '5 seconds'
        },
        {
            'script': algo_dir / '2_create_demand_model.py',
            'description': 'STAGE 2: Train Demand Forecaster',
            'expected_time': '20 minutes'
        },
        {
            'script': algo_dir / '3_create_solar_model.py',
            'description': 'STAGE 3: Train Solar Forecaster',
            'expected_time': '5 minutes'
        },
        {
            'script': algo_dir / '4_create_training_single_agents.py',
            'description': 'STAGE 4: Compute DP Baseline',
            'expected_time': '15 minutes'
        },
        {
            'script': algo_dir / '6_policy_optimization.py',  # ← CORRECTED: Now uses Step 6
            'description': 'STAGE 5: Train Phase 1 Policy (CMA-ES)',
            'expected_time': '2.5 days'
        },
        {
            'script': algo_dir / '5_create_training_multi_agents.py',  # ← CORRECTED: Now runs last
            'description': 'STAGE 6: Train Multi-Agent Policy',
            'expected_time': '1.5 days'
        }
    ]
    
    # Start training
    pipeline_start = time.time()
    completed_stages = 0
    
    for i, stage in enumerate(stages, 1):
        success = run_script(
            str(stage['script']),
            stage['description'],
            stage['expected_time']
        )
        
        if not success:
            print("\n" + "="*80)
            print("  TRAINING PIPELINE FAILED")
            print("="*80)
            print(f"\nFailed at stage {i}/{len(stages)}: {stage['description']}")
            print(f"Completed stages: {completed_stages}/{len(stages)}")
            
            elapsed = time.time() - pipeline_start
            print(f"Time elapsed: {format_time(elapsed)}")
            
            print("\nPartial results may be available in:")
            print("  - data/external/")
            print("  - data/models/")
            
            return 1
        
        completed_stages += 1
        
        # Progress update
        elapsed = time.time() - pipeline_start
        print(f"\nProgress: {completed_stages}/{len(stages)} stages completed")
        print(f"Time elapsed: {format_time(elapsed)}")
        
        if completed_stages < len(stages):
            print(f"Next stage: {stages[completed_stages]['description']}")
    
    # All stages completed
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*80)
    print("  TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal training time: {format_time(total_time)}")
    print(f"All {len(stages)} stages completed successfully.\n")
    
    print("Generated Models:")
    print("  ✓ data/models/demand_forecaster.pth - Demand forecasting MLP")
    print("  ✓ data/models/solar_forecaster.pth - Solar forecasting MLP")
    print("  ✓ data/models/phase1_best_policy.npy - Phase 1 policy (184 params)")
    print("  ✓ data/models/multi_agent_policy.npy - Multi-agent policy (465 params)")
    
    print("\nGenerated Data:")
    print("  ✓ data/external/single_agent_dp.npy - DP baseline actions")
    print("  ✓ data/external/observations.csv - Building observations")
    print("  ✓ data/external/carbon_intensity.npy - Carbon intensity data")
    print("  ✓ data/external/pricing.npy - Electricity pricing data")
    
    print("\n" + "="*80)
    print("  NEXT STEPS")
    print("="*80)
    print("\nYou can now:")
    print("  1. Evaluate the trained models on test data")
    print("  2. Compare performance with other algorithms")
    print("  3. Use the models in your HEMS simulator")
    print("  4. Analyze the training statistics and checkpoints")
    
    print("\nTo use in HEMS framework:")
    print("  python3 -m hems.main --buildings 5 --days 30 \\")
    print("      --agents ambitious_engineers \\")
    print("      --ae-mode phase23 \\")
    print("      --ae-weights data/models/multi_agent_policy.npy")
    
    print("\n" + "="*80)
    print("  CONGRATULATIONS!")
    print("="*80)
    print("\nYou have successfully trained the complete Team ambitiousengineers")
    print("model from the CityLearn Challenge 2022 (2nd Place Solution).")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be available in data/ directories.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)