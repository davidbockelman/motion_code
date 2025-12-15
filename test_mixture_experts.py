"""
Test script for mixture of experts motion code on sktime datasets.
"""
import numpy as np
from data_processing import get_train_test_data_classify, get_train_test_data_forecast
from motion_code import MotionCode

def test_classification(dataset_name='ItalyPowerDemand', R=3):
    """Test classification with mixture of experts."""
    print(f"\n{'='*60}")
    print(f"Testing Classification on {dataset_name} with R={R}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    benchmark_data, motion_code_data = get_train_test_data_classify(
        dataset_name, load_existing_data=False, add_noise=True
    )
    X_train, Y_train, labels_train, X_test, Y_test, labels_test = motion_code_data
    
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
    model = MotionCode(m=10, Q=1, latent_dim=2, sigma_y=0.1, R=R)
    
    print("Training model...")
    model_path = f'saved_models/test_{dataset_name}_R{R}_classify'
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
    
    return accuracy

def test_forecasting(dataset_name='ItalyPowerDemand', R=3):
    """Test forecasting with mixture of experts."""
    print(f"\n{'='*60}")
    print(f"Testing Forecasting on {dataset_name} with R={R}")
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
    model = MotionCode(m=10, Q=1, latent_dim=2, sigma_y=0.1, R=R)
    
    print("Training model...")
    model_path = f'saved_models/test_{dataset_name}_R{R}_forecast'
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
    
    return errors

if __name__ == '__main__':
    # Test with different R values
    R_values = [2]
    dataset = 'Lightning7'
    
    print("="*60)
    print("Mixture of Experts Motion Code Test")
    print("="*60)
    
    for R in R_values:
        try:
            print(f"\n\nTesting with R={R}")
            acc = test_classification(dataset, R=R)
            # errors = test_forecasting(dataset, R=R)
        except Exception as e:
            print(f"Error with R={R}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

