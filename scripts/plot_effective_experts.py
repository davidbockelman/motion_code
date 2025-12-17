#!/usr/bin/env python3
"""
Script to plot MOE results with effective number of experts * m on the x-axis.
Effective number of experts = exp(-sum over experts {weight_i * log(weight_i)})
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_model(model_path):
    """Load a saved Motion Code model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try with .npy extension if not provided
    if not model_path.endswith('.npy'):
        model_path = model_path + '.npy'
    
    params = np.load(model_path, allow_pickle=True).item()
    return params


def compute_effective_experts(expert_weights):
    """
    Compute effective number of experts using entropy formula.
    
    Parameters
    ----------
    expert_weights : numpy.ndarray
        Shape (num_motion, R) - weights for each class-expert combination
    
    Returns
    -------
    float
        Effective number of experts (averaged across classes)
    """
    num_motion, R = expert_weights.shape
    effective_experts_per_class = []
    
    for k in range(num_motion):
        weights_k = expert_weights[k]
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        weights_k = np.clip(weights_k, epsilon, 1.0)
        
        # Compute entropy: -sum(weight_i * log(weight_i))
        entropy = -np.sum(weights_k * np.log(weights_k))
        
        # Effective number of experts: exp(entropy)
        effective = np.exp(entropy)
        effective_experts_per_class.append(effective)
    
    # Average across classes
    return np.mean(effective_experts_per_class)


def load_moe_data_from_models(json_file, model_dir='saved_models'):
    """
    Load MOE data from JSON file and extract expert weights from corresponding model files.
    
    Parameters
    ----------
    json_file : str
        Path to JSON file with MOE results
    model_dir : str
        Directory containing saved model files
    
    Returns
    -------
    list
        List of dictionaries with m, R, metrics, and effective_experts * m
    """
    with open(json_file, 'r') as f:
        moe_data = json.load(f)
    
    results = []
    
    for entry in moe_data:
        m = entry['m']
        R = entry['R']
        dataset_name = entry['dataset_name']
        
        # Construct model file path
        model_file = os.path.join(model_dir, f'test_{dataset_name}_m{m}_R{R}_classify.npy')
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file not found: {model_file}")
            continue
        
        try:
            # Load model to get expert weights
            params = load_model(model_file)
            expert_weights = params.get('expert_weights')
            
            if expert_weights is None:
                print(f"Warning: expert_weights not found in {model_file}")
                # Use uniform weights as fallback
                num_motion = params.get('Sigma', np.array([])).shape[0] if 'Sigma' in params else 1
                expert_weights = np.ones((num_motion, R)) / R
            
            # Compute effective number of experts
            effective_experts = compute_effective_experts(expert_weights)
            
            # Compute effective_experts * m
            effective_m = effective_experts * m
            
            # Store results
            result = {
                'm': m,
                'R': R,
                'effective_experts': effective_experts,
                'effective_m': effective_m,
                'classification_accuracy': entry['classification_accuracy'],
                'average_rmse': entry['forecasting_metrics']['average_rmse'],
                'model_file': model_file
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue
    
    return results


def create_plot(results, output_file='lightning7_effective_experts.png'):
    """
    Create plots showing MOE results vs effective_experts * m.
    
    Parameters
    ----------
    results : list
        List of result dictionaries
    output_file : str
        Output file path for the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract data
    effective_m = [r['effective_m'] for r in results]
    accuracies = [r['classification_accuracy'] * 100 for r in results]  # Convert to percentage
    rmses = [r['average_rmse'] for r in results]
    
    # Sort by effective_m for better visualization
    sorted_indices = np.argsort(effective_m)
    effective_m = [effective_m[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    rmses = [rmses[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Classification Accuracy
    ax1.plot(effective_m, accuracies, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax1.set_xlabel('Effective Number of Experts × m', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('MOE: Classification Accuracy vs Effective Experts × m', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations with m and R values
    for i, r in enumerate([results[j] for j in sorted_indices]):
        ax1.annotate(f"m={r['m']}, R={r['R']}", 
                     (effective_m[i], accuracies[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)
    
    # Plot 2: Forecasting RMSE
    ax2.plot(effective_m, rmses, 's-', linewidth=2, markersize=10, color='#A23B72')
    ax2.set_xlabel('Effective Number of Experts × m', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Forecasting RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('MOE: Forecasting RMSE vs Effective Experts × m', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations with m and R values
    for i, r in enumerate([results[j] for j in sorted_indices]):
        ax2.annotate(f"m={r['m']}, R={r['R']}", 
                     (effective_m[i], rmses[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()
    
    # Also create a combined plot with dual y-axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(effective_m, accuracies, 'o-', label='Classification Accuracy', 
            linewidth=2, markersize=10, color='#2E86AB')
    ax_twin = ax.twinx()
    ax_twin.plot(effective_m, rmses, 's-', label='Forecasting RMSE', 
                 linewidth=2, markersize=10, color='#A23B72')
    
    ax.set_xlabel('Effective Number of Experts × m', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, color='#2E86AB', fontweight='bold')
    ax_twin.set_ylabel('Forecasting RMSE', fontsize=12, color='#A23B72', fontweight='bold')
    ax.set_title('MOE: Performance vs Effective Experts × m', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax_twin.tick_params(axis='y', labelcolor='#A23B72')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
    # Add annotations
    for i, r in enumerate([results[j] for j in sorted_indices]):
        ax.annotate(f"m={r['m']}, R={r['R']}", 
                   (effective_m[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    combined_output = output_file.replace('.png', '_combined.png')
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_output}")
    plt.close()
    
    # Print summary
    print("\nSummary:")
    print(f"{'m':<5} {'R':<5} {'Eff. Experts':<15} {'Eff. × m':<12} {'Accuracy':<12} {'RMSE':<12}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['effective_m']):
        print(f"{r['m']:<5} {r['R']:<5} {r['effective_experts']:<15.4f} {r['effective_m']:<12.2f} "
              f"{r['classification_accuracy']*100:<12.2f} {r['average_rmse']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot MOE results vs effective number of experts × m'
    )
    parser.add_argument('json_file', type=str,
                       help='Path to JSON file with MOE results (e.g., sweepLightning7.json)')
    parser.add_argument('--model-dir', type=str, default='saved_models',
                       help='Directory containing saved model files')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for plot (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_effective_experts.png"
    
    print("Loading MOE data and computing effective experts...")
    results = load_moe_data_from_models(args.json_file, args.model_dir)
    
    if not results:
        print("Error: No results found. Check that model files exist.")
        return 1
    
    print(f"Loaded {len(results)} results")
    print("\nCreating plots...")
    create_plot(results, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

