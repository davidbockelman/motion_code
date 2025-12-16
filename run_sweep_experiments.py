"""
Script to run parameter sweeps on Lightning7 dataset.

Sweep 1: Fix m=5, sweep R=[1,2,3,5,8]
Sweep 2: Fix R=2, sweep m=[3,5,8,10,15]

Each combination runs once.
Outputs to sweepLightning7.json
"""
import subprocess
import sys
import os

def run_experiment(dataset, m, R, json_file, run_forecasting=False):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        'test_mixture_experts.py',
        '--dataset', dataset,
        '--m', str(m),
        '--R', str(R),
        '--json-file', json_file
    ]
    
    if run_forecasting:
        cmd.append('--run-forecasting')
    
    print(f"\n{'='*80}")
    print(f"Running: dataset={dataset}, m={m}, R={R}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Completed: m={m}, R={R}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running m={m}, R={R}: {e}\n")
        return False

def main():
    dataset = 'Lightning7'
    json_file = 'sweepLightning7.json'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(json_file) if os.path.dirname(json_file) else '.'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Remove existing output file to start fresh
    if os.path.exists(json_file):
        print(f"Removing existing output file: {json_file}")
        os.remove(json_file)
    
    total_runs = 0
    successful_runs = 0
    
    # Sweep 1: Fix m=5, sweep R=[1,2,3,5,8]
    # print("\n" + "="*80)
    # print("SWEEP 1: Fix m=5, sweep R=[1,2,3,5,8]")
    # print("="*80)
    
    # m_fixed = 5
    # R_values = [1, 2, 3, 5, 8]
    
    # for R in R_values:
    #     total_runs += 1
    #     if run_experiment(dataset, m_fixed, R, json_file):
    #         successful_runs += 1
    
    # Sweep 2: Fix R=2, sweep m=[3,5,8,10,15]
    print("\n" + "="*80)
    print("SWEEP 2: Fix R=2, sweep m=[3,5,8,10,15]")
    print("="*80)
    
    R_fixed = 2
    m_values = [3, 5, 8, 10, 15]
    
    for m in m_values:
        total_runs += 1
        if run_experiment(dataset, m, R_fixed, json_file):
            successful_runs += 1
    
    # Summary
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {total_runs - successful_runs}")
    print(f"Results saved to: {json_file}")
    print("="*80)

if __name__ == '__main__':
    main()

