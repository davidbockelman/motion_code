#!/usr/bin/env python3
"""
Script to plot MOE R sweep results with effective number of experts on the x-axis.
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


def load_moe_r_sweep_data(json_file, model_dir='saved_models'):
    """
    Load MOE R sweep data from JSON file and extract expert weights from corresponding model files.
    
    Parameters
    ----------
    json_file : str
        Path to JSON file with MOE R sweep results
    model_dir : str
        Directory containing saved model files
    
    Returns
    -------
    list
        List of dictionaries with m, R, and effective_experts
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
            
            # Store results
            result = {
                'm': m,
                'R': R,
                'effective_experts': effective_experts,
                'model_file': model_file
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue
    
    return results


def create_plot(results, output_file='lightning7_r_sweep_effective_experts.png'):
    """
    Create plots showing effective number of experts vs R (number of experts).
    
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
    effective_experts = [r['effective_experts'] for r in results]
    R_values = [r['R'] for r in results]
    
    # Sort by R for better visualization
    sorted_indices = np.argsort(R_values)
    effective_experts = [effective_experts[i] for i in sorted_indices]
    R_values = [R_values[i] for i in sorted_indices]
    
    # Create figure with effective experts vs R
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot effective number of experts vs R
    ax.plot(R_values, effective_experts, 'o-', linewidth=2, markersize=12, color='#2E86AB', label='Effective Number of Experts')
    
    # Add diagonal line showing theoretical maximum (if all experts used equally)
    max_R = max(R_values)
    ax.plot([1, max_R], [1, max_R], '--', linewidth=1.5, color='gray', alpha=0.5, label='Theoretical Maximum (R)')
    
    ax.set_xlabel('R (Number of Experts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Number of Experts', fontsize=12, fontweight='bold')
    ax.set_title('R Sweep: Effective Number of Experts vs R', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotations with effective expert values
    for i, r in enumerate([results[j] for j in sorted_indices]):
        ax.annotate(f"{effective_experts[i]:.2f}", 
                   (R_values[i], effective_experts[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()
    
    # Print summary
    print("\nSummary:")
    print(f"{'R':<5} {'Eff. Experts':<15}")
    print("-" * 25)
    for r in sorted(results, key=lambda x: x['R']):
        print(f"{r['R']:<5} {r['effective_experts']:<15.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot MOE R sweep results vs effective number of experts'
    )
    parser.add_argument('json_file', type=str,
                       help='Path to JSON file with MOE R sweep results (e.g., rSweepLightning7.json)')
    parser.add_argument('--model-dir', type=str, default='saved_models',
                       help='Directory containing saved model files')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for plot (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_effective_experts.png"
    
    print("Loading MOE R sweep data and computing effective experts...")
    results = load_moe_r_sweep_data(args.json_file, args.model_dir)
    
    if not results:
        print("Error: No results found. Check that model files exist.")
        return 1
    
    print(f"Loaded {len(results)} results")
    print("\nCreating plots...")
    create_plot(results, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

