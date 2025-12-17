#!/usr/bin/env python3
"""
Script to compare Motion Code vs MOE (Mixture of Experts) performance
for Synthetic3Class dataset.
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from collections import defaultdict


def load_moe_results(json_file='results/Synthetic3Class.json'):
    """
    Load MOE results from JSON file and average across runs for each m value.
    
    Returns
    -------
    dict
        Dictionary mapping m to averaged metrics
    """
    with open(json_file, 'r') as f:
        moe_data = json.load(f)
    
    # Group by m value
    moe_by_m = defaultdict(lambda: {
        'accuracies': [],
        'rmses': [],
        'train_times': []
    })
    
    for entry in moe_data:
        m = entry['m']
        moe_by_m[m]['accuracies'].append(entry['classification_accuracy'])
        moe_by_m[m]['rmses'].append(entry['forecasting_metrics']['average_rmse'])
        moe_by_m[m]['train_times'].append(entry['train_time'])
    
    # Average across runs
    moe_averaged = {}
    for m, metrics in moe_by_m.items():
        moe_averaged[m] = {
            'accuracy': np.mean(metrics['accuracies']) * 100,  # Convert to percentage
            'accuracy_std': np.std(metrics['accuracies']) * 100,
            'rmse': np.mean(metrics['rmses']),
            'rmse_std': np.std(metrics['rmses']),
            'train_time': np.mean(metrics['train_times']),
            'num_runs': len(metrics['accuracies'])
        }
    
    return moe_averaged


def load_motion_code_results(csv_file='Synthetic3Class_motioncode_results.csv'):
    """
    Load Motion Code results from CSV file.
    
    Returns
    -------
    dict
        Dictionary mapping m to metrics
    """
    motion_code_results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row['m'])
            motion_code_results[m] = {
                'accuracy': float(row['accuracy']),
                'forecast_error': float(row['forecast_error']),
                'train_time': float(row['train_time'])
            }
    return motion_code_results


def create_comparison_graphs(moe_results, motion_code_results, output_prefix='synthetic3class'):
    """Create comparison graphs for Motion Code vs MOE."""
    
    # Get common m values
    moe_m_values = sorted(moe_results.keys())
    motion_code_m_values = sorted(motion_code_results.keys())
    common_m_values = sorted(set(moe_m_values) & set(motion_code_m_values))
    
    if not common_m_values:
        print("Error: No common m values found between MOE and Motion Code results")
        return
    
    # Extract data
    motion_code_acc = [motion_code_results[m]['accuracy'] for m in common_m_values]
    motion_code_rmse = [motion_code_results[m]['forecast_error'] for m in common_m_values]
    
    moe_acc = [moe_results[m]['accuracy'] for m in common_m_values]
    moe_acc_std = [moe_results[m]['accuracy_std'] for m in common_m_values]
    moe_rmse = [moe_results[m]['rmse'] for m in common_m_values]
    moe_rmse_std = [moe_results[m]['rmse_std'] for m in common_m_values]
    
    # Create the graphs - 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classification Accuracy
    ax1.plot(common_m_values, motion_code_acc, 'o-', label='Motion Code', 
             linewidth=2, markersize=10, color='#2E86AB')
    ax1.errorbar(common_m_values, moe_acc, yerr=moe_acc_std, fmt='s-', 
                 label='MOE', linewidth=2, markersize=10, color='#A23B72', 
                 capsize=5, capthick=2)
    ax1.set_xlabel('m', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Synthetic3Class: Classification Accuracy (R=3)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Forecasting RMSE
    ax2.plot(common_m_values, motion_code_rmse, 'o-', label='Motion Code', 
             linewidth=2, markersize=10, color='#2E86AB')
    ax2.errorbar(common_m_values, moe_rmse, yerr=moe_rmse_std, fmt='s-', 
                 label='MOE', linewidth=2, markersize=10, color='#A23B72',
                 capsize=5, capthick=2)
    ax2.set_xlabel('m', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Forecasting RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Synthetic3Class: Forecasting RMSE (R=3)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f'{output_prefix}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plots to {output_file}")
    plt.close()
    
    # Create combined graph with dual y-axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(common_m_values, motion_code_acc, 'o-', label='Motion Code (Accuracy)', 
            linewidth=2, markersize=10, color='#2E86AB')
    ax.errorbar(common_m_values, moe_acc, yerr=moe_acc_std, fmt='s-', 
                label='MOE (Accuracy)', linewidth=2, markersize=10, color='#A23B72',
                capsize=5, capthick=2)
    ax_twin = ax.twinx()
    ax_twin.plot(common_m_values, motion_code_rmse, 'o--', label='Motion Code (RMSE)', 
                 linewidth=2, markersize=10, color='#06A77D')
    ax_twin.errorbar(common_m_values, moe_rmse, yerr=moe_rmse_std, fmt='s--', 
                     label='MOE (RMSE)', linewidth=2, markersize=10, color='#F18F01',
                     capsize=5, capthick=2)
    
    ax.set_xlabel('m', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, color='#2E86AB', fontweight='bold')
    ax_twin.set_ylabel('Forecasting RMSE', fontsize=12, color='#06A77D', fontweight='bold')
    ax.set_title('Synthetic3Class: Motion Code vs MOE Performance (R=3)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax_twin.tick_params(axis='y', labelcolor='#06A77D')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    plt.tight_layout()
    output_file = f'{output_prefix}_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot to {output_file}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'m':<5} {'Motion Code Acc':<18} {'MOE Acc':<18} {'Motion Code RMSE':<18} {'MOE RMSE':<18}")
    print("-"*80)
    for m in common_m_values:
        mc_acc = motion_code_results[m]['accuracy']
        mc_rmse = motion_code_results[m]['forecast_error']
        moe_acc = moe_results[m]['accuracy']
        moe_rmse = moe_results[m]['rmse']
        moe_acc_std = moe_results[m]['accuracy_std']
        moe_rmse_std = moe_results[m]['rmse_std']
        print(f"{m:<5} {mc_acc:<18.2f} {moe_acc:.2f}±{moe_acc_std:.2f}{'':<8} {mc_rmse:<18.4f} {moe_rmse:.4f}±{moe_rmse_std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Motion Code vs MOE performance for Synthetic3Class dataset'
    )
    parser.add_argument('--moe-json', default='results/Synthetic3Class.json',
                       help='Path to MOE results JSON file')
    parser.add_argument('--motion-code-csv', default='Synthetic3Class_motioncode_results.csv',
                       help='Path to Motion Code results CSV file')
    parser.add_argument('--output-prefix', default='synthetic3class',
                       help='Prefix for output image files')
    
    args = parser.parse_args()
    
    print("Loading data...")
    moe_results = load_moe_results(args.moe_json)
    motion_code_results = load_motion_code_results(args.motion_code_csv)
    
    print(f"Loaded MOE results for m values: {sorted(moe_results.keys())}")
    print(f"Loaded Motion Code results for m values: {sorted(motion_code_results.keys())}")
    
    print("\nCreating comparison graphs...")
    create_comparison_graphs(moe_results, motion_code_results, args.output_prefix)
    print("\nDone!")


if __name__ == '__main__':
    main()

