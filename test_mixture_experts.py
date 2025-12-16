"""
Test script for mixture of experts motion code on sktime datasets.
"""
import numpy as np
import os
import json
import argparse
from data_processing import get_train_test_data_forecast, process_data_for_motion_codes
from motion_code import MotionCode

def test_classification(dataset_name='ItalyPowerDemand', m=5, R=3):
    """Test classification with mixture of experts."""
    print(f"\n{'='*60}")
    print(f"Testing Classification on {dataset_name} with m={m}, R={R}")
    print(f"{'='*60}\n")
    
    # Load data from saved .npy file
    print("Loading data from saved file...")
    data_path = f'data/{dataset_name}.npy'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Please run add_noise_and_save_data.py first.")
    
    data = np.load(data_path, allow_pickle=True).item()
    Y_train_bm = data['Y_train']
    labels_train_bm = data['labels_train']
    Y_test_bm = data['Y_test']
    labels_test_bm = data['labels_test']
    
    # Process data for motion codes
    X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train_bm, labels_train_bm)
    X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test_bm, labels_test_bm)
    
    # Convert to lists if needed
    if not isinstance(X_train, list):
        X_train = [X_train[i] for i in range(len(X_train))]
        Y_train = [Y_train[i] for i in range(len(Y_train))]
    if not isinstance(X_test, list):
        X_test = [X_test[i] for i in range(len(X_test))]
        Y_test = [Y_test[i] for i in range(len(Y_test))]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(labels_train))}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = MotionCode(m=m, Q=1, latent_dim=2, sigma_y=0.1, R=R, lambda_reg=1.0, lambda_weight_reg=0.0)
    
    print("Training model...")
    model_path = f'saved_models/test_{dataset_name}_m{m}_R{R}_classify'
    model.fit(X_train, Y_train, labels_train, model_path)
    print(f"Training completed in {model.train_time:.2f} seconds")
    
    # Load model
    print("\nLoading model...")
    model.load(model_path)
    print(f"Model loaded. R={model.R}, num_motion={model.num_motion}")
    
    # Print learned static weights for each class
    print("\n" + "="*60)
    print("Learned Static Weights for Each Class:")
    print("="*60)
    for k in range(model.num_motion):
        print(f"Class {k}: {model.expert_weights[k]}")
        print(f"  (sums to {np.sum(model.expert_weights[k]):.6f})")
    
    # Print inducing timestamps for each expert
    print("\n" + "="*60)
    print("Inducing Timestamps (Normalized 0-1) for Each Expert:")
    print("="*60)
    from sparse_gp import sigmoid
    for r in range(model.R):
        X_m_r = sigmoid(model.X_m @ model.Z[r])
        print(f"Expert {r}: {X_m_r}")
        print(f"  (min={np.min(X_m_r):.6f}, max={np.max(X_m_r):.6f}, mean={np.mean(X_m_r):.6f})")
    
    # Test classification
    print("\n" + "="*60)
    print("Testing classification...")
    print("="*60)
    accuracy = model.classify_predict_on_batches(X_test, Y_test, labels_test)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    return accuracy, model.train_time

def test_forecasting(dataset_name='ItalyPowerDemand', m=10, R=3):
    """Test forecasting with mixture of experts."""
    print(f"\n{'='*60}")
    print(f"Testing Forecasting on {dataset_name} with m={m}, R={R}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    benchmark_data, motion_code_data = get_train_test_data_forecast(dataset_name)
    X_train, Y_train, labels, test_time_horizon, Y_test = motion_code_data
    
    # Convert to lists if needed
    if not isinstance(X_train, list):
        X_train = [X_train[i] for i in range(len(X_train))]
        Y_train = [Y_train[i] for i in range(len(Y_train))]
    if not isinstance(Y_test, list):
        Y_test = [Y_test[i] for i in range(len(Y_test))]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(Y_test)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = MotionCode(m=m, Q=1, latent_dim=2, sigma_y=0.1, R=R)
    
    print("Training model...")
    model_path = f'saved_models/test_{dataset_name}_m{m}_R{R}_forecast'
    model.fit(X_train, Y_train, labels, model_path)
    print(f"Training completed in {model.train_time:.2f} seconds")
    
    # Load model
    print("\nLoading model...")
    model.load(model_path)
    print(f"Model loaded. R={model.R}, num_motion={model.num_motion}")
    
    # Print learned static weights for each class
    print("\n" + "="*60)
    print("Learned Static Weights for Each Class:")
    print("="*60)
    for k in range(model.num_motion):
        print(f"Class {k}: {model.expert_weights[k]}")
        print(f"  (sums to {np.sum(model.expert_weights[k]):.6f})")
    
    # Print inducing timestamps for each expert
    print("\n" + "="*60)
    print("Inducing Timestamps (Normalized 0-1) for Each Expert:")
    print("="*60)
    from sparse_gp import sigmoid
    for r in range(model.R):
        X_m_r = sigmoid(model.X_m @ model.Z[r])
        print(f"Expert {r}: {X_m_r}")
        print(f"  (min={np.min(X_m_r):.6f}, max={np.max(X_m_r):.6f}, mean={np.mean(X_m_r):.6f})")
    
    # Test forecasting
    print("\n" + "="*60)
    print("Testing forecasting...")
    print("="*60)
    errors = model.forecast_predict_on_batches(test_time_horizon, Y_test, labels)
    print(f"Forecasting RMSE per class:")
    for k, err in enumerate(errors):
        print(f"  Class {k}: {err:.4f}")
    print(f"Average RMSE: {np.mean(errors):.4f}")
    
    return errors, model.train_time

def save_results_to_json(results, json_file):
    """Append results as a JSON object to a JSON file (one JSON object per line)."""
    os.makedirs(os.path.dirname(json_file) if os.path.dirname(json_file) else '.', exist_ok=True)
    
    # Append to file (one JSON object per line)
    with open(json_file, 'a') as f:
        json.dump(results, f)
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test mixture of experts motion code')
    parser.add_argument('--dataset', type=str, default='Lightning7', help='Dataset name')
    parser.add_argument('--m', type=int, default=5, help='Number of inducing points')
    parser.add_argument('--R', type=int, default=3, help='Number of experts')
    parser.add_argument('--json-file', type=str, default='results.json', help='JSON file to append results to')
    parser.add_argument('--run-forecasting', action='store_true', help='Also run forecasting tests')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Mixture of Experts Motion Code Test")
    print("="*60)
    print(f"Dataset: {args.dataset}, m={args.m}, R={args.R}")
    print("="*60)
    
    results = {
        'dataset_name': args.dataset,
        'm': args.m,
        'R': args.R,
        'classification_accuracy': None,
        'forecasting_metrics': None,
        'train_time': None
    }
    
    try:
        # Run classification
        print(f"\n\nTesting classification with m={args.m}, R={args.R}")
        accuracy, train_time = test_classification(args.dataset, m=args.m, R=args.R)
        results['classification_accuracy'] = float(accuracy)
        results['train_time'] = float(train_time)
        
        # Run forecasting if requested
        print(f"\n\nTesting forecasting with m={args.m}, R={args.R}")
        errors, forecast_train_time = test_forecasting(args.dataset, m=args.m, R=args.R)
        results['forecasting_metrics'] = {
            'rmse_per_class': [float(err) for err in errors],
            'average_rmse': float(np.mean(errors)),
            'train_time': float(forecast_train_time)
        }
        
        # Save results to JSON file
        save_results_to_json(results, args.json_file)
        print(f"\nResults saved to {args.json_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        save_results_to_json(results, args.json_file)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

