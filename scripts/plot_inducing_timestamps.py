#!/usr/bin/env python3
"""
Script to plot a random sample from a dataset with inducing timestamps overlaid.
Dot sizes are proportional to expert weights for the sample's class.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from motion_code import MotionCode
from data_processing import load_data, process_data_for_motion_codes


def sigmoid_np(x):
    """Sigmoid function using numpy."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_inducing_timestamps(X_m, Z):
    """
    Compute inducing timestamps for each expert.
    
    Parameters
    ----------
    X_m : numpy.ndarray
        Inducing points in latent space, shape (m, latent_dim)
    Z : numpy.ndarray
        Global motion codes (experts), shape (R, latent_dim)
    
    Returns
    -------
    dict
        Dictionary mapping expert index to inducing timestamps array
    """
    R = Z.shape[0]
    inducing_timestamps = {}
    
    for r in range(R):
        # Inducing timestamps for expert r: sigmoid(X_m @ Z[r])
        X_m_r = sigmoid_np(X_m @ Z[r])
        inducing_timestamps[r] = X_m_r
    
    return inducing_timestamps


def plot_sample_with_inducing_timestamps(model_path, dataset_name, split='test', 
                                         sample_idx=None, output_file=None, 
                                         min_dot_size=20, max_dot_size=300):
    """
    Plot a random sample from a dataset with inducing timestamps overlaid.
    
    Parameters
    ----------
    model_path : str
        Path to saved model file
    dataset_name : str
        Name of the dataset
    split : str
        'train' or 'test'
    sample_idx : int, optional
        Index of sample to plot. If None, picks random sample.
    output_file : str, optional
        Output file path. If None, auto-generates.
    min_dot_size : float
        Minimum dot size for smallest weight
    max_dot_size : float
        Maximum dot size for largest weight
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = MotionCode()
    model.load(model_path)
    
    # Load dataset
    print(f"Loading {split} data for {dataset_name}...")
    try:
        Y, labels = load_data(name=dataset_name, split=split, add_noise=False)
        X, Y, labels = process_data_for_motion_codes(Y, labels)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Handle different data formats
    # If X, Y are 2D arrays, convert to lists of 1D arrays
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = [X[i] for i in range(len(X))]
        elif X.ndim == 1:
            X = [X]  # Single sample
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")
    
    if isinstance(Y, np.ndarray):
        if Y.ndim == 2:
            Y = [Y[i] for i in range(len(Y))]
        elif Y.ndim == 1:
            Y = [Y]  # Single sample
        else:
            raise ValueError(f"Unexpected Y shape: {Y.shape}")
    
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = [labels]  # Single label
    
    # Ensure X and Y have the same length
    if len(X) != len(Y):
        raise ValueError(f"X and Y have different lengths: {len(X)} vs {len(Y)}")
    
    print(f"Loaded {len(X)} samples")
    
    # Select sample
    num_samples = len(X)
    if sample_idx is None:
        sample_idx = random.randint(0, num_samples - 1)
    
    if sample_idx >= num_samples:
        raise ValueError(f"Sample index {sample_idx} out of range [0, {num_samples-1}]")
    
    X_sample = X[sample_idx]
    Y_sample = Y[sample_idx]
    label = labels[sample_idx]
    
    print(f"Selected sample {sample_idx} from class {label}")
    
    # Get expert weights for this class
    if hasattr(model, 'expert_weights') and model.expert_weights is not None:
        class_weights = model.expert_weights[label]  # Shape (R,)
    else:
        # Fallback to uniform weights
        class_weights = np.ones(model.R) / model.R
        print("Warning: expert_weights not found, using uniform weights")
    
    print(f"Expert weights for class {label}: {class_weights}")
    
    # Compute inducing timestamps for each expert
    inducing_timestamps = compute_inducing_timestamps(model.X_m, model.Z)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot the time series
    ax.plot(X_sample, Y_sample, 'k-', linewidth=2, label='Time Series', alpha=0.7)
    
    # Plot inducing timestamps for each expert with sizes proportional to weights
    colors = plt.cm.Set3(np.linspace(0, 1, model.R))
    
    for r in range(model.R):
        timestamps = inducing_timestamps[r]
        weight = class_weights[r]
        
            # Map timestamps to Y values by interpolation
        # Ensure timestamps are within the range of X_sample
        timestamps_clipped = np.clip(timestamps, X_sample.min(), X_sample.max())
        Y_at_timestamps = np.interp(timestamps_clipped, X_sample, Y_sample)
        
        # Calculate dot size based on weight (normalize to [min_dot_size, max_dot_size])
        # Normalize weights to [0, 1] range
        weight_normalized = (weight - class_weights.min()) / (class_weights.max() - class_weights.min() + 1e-10)
        dot_size = min_dot_size + weight_normalized * (max_dot_size - min_dot_size)
        
        # Plot inducing timestamps
        ax.scatter(timestamps, Y_at_timestamps, s=dot_size, alpha=0.6, 
                  color=colors[r], edgecolors='black', linewidths=1.5,
                  label=f'Expert {r} (weight={weight:.3f})', zorder=5)
    
    ax.set_xlabel('Time (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    # Get class name if available, otherwise use label number
    class_name = f'Class {label}'
    ax.set_title(f'{dataset_name} - Sample {sample_idx} ({class_name})\nInducing Timestamps by Expert Weight', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_file = f'{base_name}_{dataset_name}_inducing.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Split: {split}")
    print(f"  Sample index: {sample_idx}")
    print(f"  Class: {label}")
    print(f"  Number of experts: {model.R}")
    print(f"  Number of inducing points per expert: {len(inducing_timestamps[0])}")
    print(f"\nExpert weights:")
    for r in range(model.R):
        print(f"    Expert {r}: {class_weights[r]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot random sample with inducing timestamps overlaid, sized by expert weights'
    )
    parser.add_argument('model_path', type=str,
                       help='Path to saved model file (with or without .npy extension)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Synthetic3Class, ItalyPowerDemand, ECGFiveDays, Lightning7, etc.)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                       help='Data split to use (default: test)')
    parser.add_argument('--sample-idx', type=int, default=None,
                       help='Index of sample to plot (default: random)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--min-dot-size', type=float, default=20,
                       help='Minimum dot size for smallest weight (default: 20)')
    parser.add_argument('--max-dot-size', type=float, default=300,
                       help='Maximum dot size for largest weight (default: 300)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    try:
        plot_sample_with_inducing_timestamps(
            model_path=args.model_path,
            dataset_name=args.dataset,
            split=args.split,
            sample_idx=args.sample_idx,
            output_file=args.output,
            min_dot_size=args.min_dot_size,
            max_dot_size=args.max_dot_size
        )
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

