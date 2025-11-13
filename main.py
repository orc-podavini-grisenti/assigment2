#!/usr/bin/env python3
"""
Main entry point for robot path tracking experiments.
Run with: python main.py [--experiment Q1|Q2|Q3|Q4|all] [--no-viz]
"""
import argparse
import sys
from pathlib import Path

from experiments import run_q1, run_q2, run_q3, run_q4
from utility.plotting import plot_infinity


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='UR5 Robot Path Tracking Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiment Q1          # Run experiment Q1 only
  python main.py --experiment all         # Run all experiments
  python main.py --experiment Q3 --no-viz # Run Q3 without visualization
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        choices=['Q1', 'Q2', 'Q3', 'Q4', 'all'],
        default='all',
        help='Select which experiment to run (default: all)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable robot visualization'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable result plotting'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Create results directory structure
    Path('./results/Q1').mkdir(parents=True, exist_ok=True)
    Path('./results/Q2').mkdir(parents=True, exist_ok=True)
    Path('./results/Q3').mkdir(parents=True, exist_ok=True)
    Path('./results/Q4').mkdir(parents=True, exist_ok=True)
    
    # Plot reference infinity curve
    print("\n" + "="*70)
    print("Plotting reference infinity curve...")
    print("="*70)
    plot_infinity(0, 1)
    
    # Experiment mapping
    experiments = {
        'Q1': ('Q1: Path Tracking without Terminal Cost', run_q1),
        'Q2': ('Q2: Path Tracking with Cyclic Terminal Cost', run_q2),
        'Q3': ('Q3: Cyclic Trajectory Tracking (Hard & Soft Constraints)', run_q3),
        'Q4': ('Q4: Minimum Time Path Tracking', run_q4)
    }
    
    # Determine which experiments to run
    if args.experiment == 'all':
        experiments_to_run = ['Q1', 'Q2', 'Q3', 'Q4']
    else:
        experiments_to_run = [args.experiment]
    
    # Run selected experiments
    for exp_id in experiments_to_run:
        exp_name, exp_func = experiments[exp_id]
        
        print("\n" + "="*70)
        print(f"Running {exp_name}")
        print("="*70)
        
        try:
            exp_func(
                visualize=not args.no_viz,
                plot_results=not args.no_plot
            )
            print(f"\n✓ {exp_id} completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error in {exp_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        if exp_id != experiments_to_run[-1]:
            input("\nPress ENTER to continue to next experiment...")
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    main()