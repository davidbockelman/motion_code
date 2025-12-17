#!/usr/bin/env python3
"""
Script to plot a random sample from a dataset with its forecasting.
Shows the overall forecasting and the contributions per expert.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from motion_code import MotionCode
from data_processing import load_data, process_data_for_motion_codes, split_train_test_forecasting
from sparse_gp import sigmoid, q


def forecast_predict_with_experts(model, test_time_horizon, label):
    """
    Get forecasting predictions including individual expert contributions.
    
    Returns:
        weighted_mean: Overall weighted prediction
        weighted_covar: Overall weighted covariance
        expert_means: List of mean predictions for each expert
        expert_weights: Weights for each expert
    """
    k = label
    expert_predictions = []
    expert_means = []
    
    for r in range(model.R):
        X_m_r = sigmoid(model.X_m @ model.Z[r])
        if isinstance(model.mu_ms[k], list):
            mean, covar = q(test_time_horizon, X_m_r, 
                           model.kernel_params[k], model.mu_ms[k][r], 
                           model.A_ms[k][r], model.K_mm_invs[k][r])
        else:
            # Backward compatibility: single expert
            mean, covar = q(test_time_horizon, X_m_r, 
                           model.kernel_params[k], model.mu_ms[k], 
                           model.A_ms[k], model.K_mm_invs[k])
        expert_predictions.append((mean, covar))
        expert_means.append(mean)
    
    # Get expert weights for this class
    if hasattr(model, 'expert_weights') and model.expert_weights is not None:
        weights = model.expert_weights[k]  # Shape (R,)
    else:
        # Fallback to uniform weights
        weights = np.ones(model.R) / model.R
    
    # Weight and combine predictions
    weighted_mean = np.zeros_like(expert_predictions[0][0])
    weighted_covar = np.zeros_like(expert_predictions[0][1])
    
    for r, (mean, covar) in enumerate(expert_predictions):
        weighted_mean += weights[r] * mean
        weighted_covar += weights[r] * covar
    
    return weighted_mean, weighted_covar, expert_means, weights


def plot_forecasting_with_experts(model_path, dataset_name, split='test', 
                                  sample_idx=None, output_file=None, 
                                  forecast_percentage=0.2, seed=None):
    """
    Plot a random sample with its forecasting, showing overall and per-expert contributions.
    
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
    forecast_percentage : float
        Percentage of sequence length to forecast (default: 0.2 = 20%)
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
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
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = [X[i] for i in range(len(X))]
        elif X.ndim == 1:
            X = [X]
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")
    
    if isinstance(Y, np.ndarray):
        if Y.ndim == 2:
            Y = [Y[i] for i in range(len(Y))]
        elif Y.ndim == 1:
            Y = [Y]
        else:
            raise ValueError(f"Unexpected Y shape: {Y.shape}")
    
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = [labels]
    
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
    
    # Split into train and forecast portions
    seq_length = len(X_sample)
    train_length = int(seq_length * (1 - forecast_percentage))
    test_length = seq_length - train_length
    
    X_train_sample = X_sample[:train_length]
    Y_train_sample = Y_sample[:train_length]
    X_forecast = X_sample[train_length:]
    Y_forecast = Y_sample[train_length:]
    
    # Create test time horizon (normalized to [0, 1] range)
    # The model expects time in [0, 1], so we need to map the forecast horizon
    time_max = X_sample.max()
    time_min = X_sample.min()
    test_time_horizon = (X_forecast - time_min) / (time_max - time_min + 1e-10)
    
    # Get forecasting predictions
    print("Computing forecasting predictions...")
    weighted_mean, weighted_covar, expert_means, expert_weights = forecast_predict_with_experts(
        model, test_time_horizon, label
    )
    std = np.sqrt(np.diag(weighted_covar)).reshape(-1)
    
    print(f"Expert weights for class {label}: {expert_weights}")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Plot the training portion of the time series
    ax.plot(X_train_sample, Y_train_sample, 'k-', linewidth=2.5, 
            label='Observed Time Series', alpha=0.8, zorder=1)
    
    # Plot the actual forecast portion (ground truth)
    ax.plot(X_forecast, Y_forecast, 'g--', linewidth=2, 
            label='Ground Truth (Forecast)', alpha=0.7, zorder=2)
    
    # Plot individual expert contributions
    colors = plt.cm.Set3(np.linspace(0, 1, model.R))
    for r in range(model.R):
        expert_mean = expert_means[r]
        weight = expert_weights[r]
        ax.plot(X_forecast, expert_mean, '--', linewidth=1.5, 
                color=colors[r], alpha=0.6, 
                label=f'Expert {r} (weight={weight:.3f})', zorder=3)
    
    # Plot overall weighted prediction
    ax.plot(X_forecast, weighted_mean, 'r-', linewidth=3, 
            label='Overall Prediction (Weighted)', alpha=0.9, zorder=4)
    
    # Plot uncertainty region (2 standard deviations)
    ax.fill_between(X_forecast, weighted_mean + 2*std, weighted_mean - 2*std, 
                    color='red', alpha=0.15, zorder=0, label='Uncertainty (±2σ)')
    
    # Add vertical line to separate train and forecast
    ax.axvline(x=X_train_sample[-1], color='gray', linestyle=':', 
               linewidth=2, alpha=0.5, label='Train/Forecast Boundary')
    
    ax.set_xlabel('Time (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name} - Sample {sample_idx} (Class {label})\nForecasting with Expert Contributions', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_file = f'{base_name}_{dataset_name}_forecast_experts.png'
    
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
    print(f"  Forecast percentage: {forecast_percentage*100:.1f}%")
    print(f"\nExpert weights:")
    for r in range(model.R):
        print(f"    Expert {r}: {expert_weights[r]:.4f}")
    
    # Calculate RMSE for overall prediction
    from utils import RMSE
    rmse = RMSE(weighted_mean, Y_forecast)
    print(f"\nForecasting RMSE: {rmse:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot random sample with forecasting, showing overall and per-expert contributions'
    )
    parser.add_argument('model_path', type=str,
                       help='Path to saved model file (with or without .npy extension)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Synthetic3Class, ItalyPowerDemand, ECGFiveDays, etc.)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                       help='Data split to use (default: test)')
    parser.add_argument('--sample-idx', type=int, default=None,
                       help='Index of sample to plot (default: random)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--forecast-percentage', type=float, default=0.2,
                       help='Percentage of sequence to forecast (default: 0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        plot_forecasting_with_experts(
            model_path=args.model_path,
            dataset_name=args.dataset,
            split=args.split,
            sample_idx=args.sample_idx,
            output_file=args.output,
            forecast_percentage=args.forecast_percentage,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

