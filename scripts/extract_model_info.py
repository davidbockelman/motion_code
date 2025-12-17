#!/usr/bin/env python3
"""
Script to extract and display inducing timestamps and per-class weights from a saved Motion Code model.
"""

import numpy as np
import argparse
import os


def sigmoid(x):
    """Sigmoid function: 1/(1+exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow


def load_model(model_path):
    """Load a saved Motion Code model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try with .npy extension if not provided
    if not model_path.endswith('.npy'):
        model_path = model_path + '.npy'
    
    params = np.load(model_path, allow_pickle=True).item()
    return params


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
        X_m_r = sigmoid(X_m @ Z[r])
        inducing_timestamps[r] = X_m_r
    
    return inducing_timestamps


def extract_model_info(model_path, json_output_file=None, text_output_file=None, verbose=True):
    """
    Extract and display inducing timestamps and per-class weights from a saved model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file
    json_output_file : str, optional
        Path for JSON output file. If not provided, auto-generates from model_path.
    text_output_file : str, optional
        Path for text output file (optional).
    verbose : bool
        If True, print output to console
    """
    # Load model
    params = load_model(model_path)
    
    X_m = params.get('X_m')
    Z = params.get('Z')
    expert_weights = params.get('expert_weights')
    R = params.get('R', Z.shape[0] if Z is not None else None)
    num_motion = expert_weights.shape[0] if expert_weights is not None else None
    
    if X_m is None or Z is None:
        raise ValueError("Model missing required parameters: X_m or Z")
    
    if expert_weights is None:
        print("Warning: expert_weights not found in model. Using uniform weights.")
        if num_motion is None:
            num_motion = params.get('Sigma', np.array([])).shape[0] if 'Sigma' in params else 1
        expert_weights = np.ones((num_motion, R)) / R
    
    # Compute inducing timestamps for each expert
    inducing_timestamps = compute_inducing_timestamps(X_m, Z)
    
    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("MOTION CODE MODEL INFORMATION")
    output_lines.append("=" * 80)
    output_lines.append(f"Model path: {model_path}")
    output_lines.append(f"Number of experts (R): {R}")
    output_lines.append(f"Number of classes (num_motion): {num_motion}")
    output_lines.append(f"Number of inducing points (m): {X_m.shape[0]}")
    output_lines.append(f"Latent dimension: {X_m.shape[1]}")
    output_lines.append("")
    
    # Display inducing timestamps for each expert
    output_lines.append("=" * 80)
    output_lines.append("INDUCING TIMESTAMPS FOR EACH EXPERT")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for r in range(R):
        timestamps = inducing_timestamps[r]
        output_lines.append(f"Expert {r}:")
        output_lines.append(f"  Shape: {timestamps.shape}")
        output_lines.append(f"  Timestamps: {np.array2string(timestamps, precision=6, separator=', ')}")
        output_lines.append(f"  Min: {np.min(timestamps):.6f}, Max: {np.max(timestamps):.6f}, Mean: {np.mean(timestamps):.6f}")
        output_lines.append("")
    
    # Display per-class weights
    output_lines.append("=" * 80)
    output_lines.append("PER-CLASS EXPERT WEIGHTS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Each row represents a class, each column represents an expert.")
    output_lines.append("Values in each row sum to 1.0 (softmax normalization).")
    output_lines.append("")
    
    # Create formatted table
    header = "Class" + "".join([f"  Expert {r:3d}" for r in range(R)])
    output_lines.append(header)
    output_lines.append("-" * len(header))
    
    for k in range(num_motion):
        row = f"{k:5d}" + "".join([f"  {expert_weights[k, r]:8.6f}" for r in range(R)])
        output_lines.append(row)
    
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("SUMMARY STATISTICS")
    output_lines.append("=" * 80)
    
    # Statistics for each class
    for k in range(num_motion):
        weights_k = expert_weights[k]
        max_expert = np.argmax(weights_k)
        output_lines.append(f"Class {k}:")
        output_lines.append(f"  Max weight: {weights_k[max_expert]:.6f} (Expert {max_expert})")
        output_lines.append(f"  Min weight: {np.min(weights_k):.6f} (Expert {np.argmin(weights_k)})")
        output_lines.append(f"  Weight std: {np.std(weights_k):.6f}")
        output_lines.append("")
    
    # Prepare JSON output (always created)
    import json
    
    json_output = {
        'model_path': model_path,
        'R': int(R),
        'num_motion': int(num_motion),
        'm': int(X_m.shape[0]),
        'latent_dim': int(X_m.shape[1]),
        'inducing_timestamps': {
            f'expert_{r}': {
                'values': inducing_timestamps[r].tolist(),
                'min': float(np.min(inducing_timestamps[r])),
                'max': float(np.max(inducing_timestamps[r])),
                'mean': float(np.mean(inducing_timestamps[r])),
                'std': float(np.std(inducing_timestamps[r]))
            }
            for r in range(R)
        },
        'expert_weights': {
            f'class_{k}': {
                'weights': expert_weights[k].tolist(),
                'max_weight': float(np.max(expert_weights[k])),
                'max_expert': int(np.argmax(expert_weights[k])),
                'min_weight': float(np.min(expert_weights[k])),
                'min_expert': int(np.argmin(expert_weights[k])),
                'std': float(np.std(expert_weights[k]))
            }
            for k in range(num_motion)
        }
    }
    
    # Determine JSON output file path
    if json_output_file is None:
        # Auto-generate from model path
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        if base_name.endswith('.npy'):
            base_name = base_name[:-4]
        json_output_file = f"{base_name}_model_info.json"
    
    # Save JSON file
    with open(json_output_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    if verbose:
        print(f"JSON output saved to: {json_output_file}")
    
    # Print to console if verbose
    if verbose:
        output_text = "\n".join(output_lines)
        print(output_text)
    
    # Save text file if specified
    if text_output_file:
        output_text = "\n".join(output_lines)
        with open(text_output_file, 'w') as f:
            f.write(output_text)
        if verbose:
            print(f"Text output saved to: {text_output_file}")
    
    return {
        'inducing_timestamps': inducing_timestamps,
        'expert_weights': expert_weights,
        'R': R,
        'num_motion': num_motion
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract inducing timestamps and per-class weights from a saved Motion Code model'
    )
    parser.add_argument('model_path', type=str,
                       help='Path to the saved model file (with or without .npy extension)')
    parser.add_argument('--json', '-j', type=str, default=None,
                       help='JSON output file path (default: auto-generated from model name)')
    parser.add_argument('--text', '-t', type=str, default=None,
                       help='Text output file path (optional)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    try:
        extract_model_info(
            args.model_path,
            json_output_file=args.json,
            text_output_file=args.text,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

