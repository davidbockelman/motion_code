#!/usr/bin/env python3
"""
Script to compare Motion Code vs MOE (Mixture of Experts) performance
for Lightning7 dataset across m sweep.
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_moe_m_sweep(m_sweep_file='sweepLightning7.json'):
    """Load MOE m sweep data from JSON file."""
    with open(m_sweep_file, 'r') as f:
        moe_m_sweep = json.load(f)
    return moe_m_sweep


def load_motion_code_m_sweep(csv_file='lightning7_sweep_m_values.csv'):
    """Load normal Motion Code m sweep data from CSV."""
    motion_code_m_sweep = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row['m'])
            motion_code_m_sweep[m] = {
                'accuracy': float(row['accuracy']),
                'forecast_error': float(row['forecast_error']),
                'train_time': float(row['train_time'])
            }
    return motion_code_m_sweep


def create_m_sweep_comparison_graphs(moe_m_sweep, motion_code_m_sweep, output_prefix='lightning7'):
    """Create comparison graphs for m sweep."""
    
    # Prepare data for m sweep graph
    m_sweep_m_values = sorted(motion_code_m_sweep.keys())
    motion_code_m_acc = [motion_code_m_sweep[m]['accuracy'] for m in m_sweep_m_values]
    motion_code_m_rmse = [motion_code_m_sweep[m]['forecast_error'] for m in m_sweep_m_values]

    moe_m_m_values = sorted([entry['m'] for entry in moe_m_sweep])
    moe_m_acc = [entry['classification_accuracy'] * 100 for entry in sorted(moe_m_sweep, key=lambda x: x['m'])]
    moe_m_rmse = [entry['forecasting_metrics']['average_rmse'] for entry in sorted(moe_m_sweep, key=lambda x: x['m'])]

    # Create the graphs - 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # M Sweep - Classification Accuracy
    ax1.plot(m_sweep_m_values, motion_code_m_acc, 'o-', label='Motion Code', 
             linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(moe_m_m_values, moe_m_acc, 's-', label='MOE', 
             linewidth=2, markersize=8, color='#A23B72')
    ax1.set_xlabel('m', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('m Sweep: Classification Accuracy (R=2)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # M Sweep - Forecasting RMSE
    ax2.plot(m_sweep_m_values, motion_code_m_rmse, 'o-', label='Motion Code', 
             linewidth=2, markersize=8, color='#2E86AB')
    ax2.plot(moe_m_m_values, moe_m_rmse, 's-', label='MOE', 
             linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('m', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Forecasting RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('m Sweep: Forecasting RMSE (R=2)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f'{output_prefix}_m_sweep_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved m sweep comparison to {output_file}")
    plt.close()

    # Create combined graph with dual y-axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(m_sweep_m_values, motion_code_m_acc, 'o-', label='Motion Code (Accuracy)', 
            linewidth=2, markersize=8, color='#2E86AB')
    ax_twin = ax.twinx()
    ax_twin.plot(m_sweep_m_values, motion_code_m_rmse, 'o--', label='Motion Code (RMSE)', 
                 linewidth=2, markersize=8, color='#06A77D')
    ax.plot(moe_m_m_values, moe_m_acc, 's-', label='MOE (Accuracy)', 
            linewidth=2, markersize=8, color='#A23B72')
    ax_twin.plot(moe_m_m_values, moe_m_rmse, 's--', label='MOE (RMSE)', 
                 linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('m', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, color='#2E86AB', fontweight='bold')
    ax_twin.set_ylabel('Forecasting RMSE', fontsize=12, color='#06A77D', fontweight='bold')
    ax.set_title('m Sweep Comparison: Motion Code vs MOE (R=2)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax_twin.tick_params(axis='y', labelcolor='#06A77D')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    plt.tight_layout()
    output_file = f'{output_prefix}_m_sweep_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined m sweep graph to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare Motion Code vs MOE performance for Lightning7 dataset (m sweep only)'
    )
    parser.add_argument('--m-sweep-json', default='sweepLightning7.json',
                       help='Path to m sweep MOE JSON file')
    parser.add_argument('--m-sweep-csv', default='lightning7_sweep_m_values.csv',
                       help='Path to m sweep Motion Code CSV file')
    parser.add_argument('--output-prefix', default='lightning7',
                       help='Prefix for output image files')
    
    args = parser.parse_args()
    
    print("Loading data...")
    moe_m_sweep = load_moe_m_sweep(args.m_sweep_json)
    motion_code_m_sweep = load_motion_code_m_sweep(args.m_sweep_csv)
    
    print(f"Loaded {len(moe_m_sweep)} m sweep MOE entries")
    print(f"Loaded {len(motion_code_m_sweep)} m sweep Motion Code entries")
    
    print("\nCreating graphs...")
    create_m_sweep_comparison_graphs(moe_m_sweep, motion_code_m_sweep, args.output_prefix)
    print("\nDone!")


if __name__ == '__main__':
    main()

