"""
Create a synthetic dataset with 3 classes based on combinations of periodic, trend, and step bases.

Each class is a fixed combination of 2 bases with specific weights.
Time series have 150 timestamps each.
Noise with sigma=0.1 is added to each series.
"""
import numpy as np
import os

def generate_periodic(t, a, b):
    """Generate periodic signal: sin(2π*a*t + b)"""
    return np.sin(2 * np.pi * a * t + b)

def generate_trend(t, a, b):
    """Generate trend signal: a*t + b"""
    return a * t + b

def generate_step(t, a):
    """Generate step signal: 1[t > a]"""
    return (t > a).astype(float)

def generate_synthetic_dataset(
    num_train=100, 
    num_test=50, 
    num_timestamps=150,
    noise_sigma=0.1,
    random_seed=42
):
    """
    Generate synthetic dataset with 3 classes.
    
    Class 0: 0.8 * periodic + 0.2 * step
    Class 1: 0.7 * periodic + 0.3 * trend
    Class 2: 0.6 * trend + 0.4 * step
    """
    np.random.seed(random_seed)
    
    # Time points (normalized to [0, 1])
    t = np.linspace(0, 1, num_timestamps)
    
    # Parameter ranges for sampling
    periodic_a_range = (0.5, 3.0)  # Frequency
    periodic_b_range = (0, 2 * np.pi)  # Phase
    trend_a_range = (-2.0, 2.0)  # Slope
    trend_b_range = (-1.0, 1.0)  # Intercept
    step_a_range = (0.3, 0.7)  # Step location
    
    def generate_class_0_series():
        """Class 0: 0.8 * periodic + 0.2 * step"""
        a_per = np.random.uniform(*periodic_a_range)
        b_per = np.random.uniform(*periodic_b_range)
        a_step = np.random.uniform(*step_a_range)
        
        periodic = generate_periodic(t, a_per, b_per)
        step = generate_step(t, a_step)
        signal = 0.8 * periodic + 0.2 * step
        return signal
    
    def generate_class_1_series():
        """Class 1: 0.7 * periodic + 0.3 * trend"""
        a_per = np.random.uniform(*periodic_a_range)
        b_per = np.random.uniform(*periodic_b_range)
        a_trend = np.random.uniform(*trend_a_range)
        b_trend = np.random.uniform(*trend_b_range)
        
        periodic = generate_periodic(t, a_per, b_per)
        trend = generate_trend(t, a_trend, b_trend)
        signal = 0.7 * periodic + 0.3 * trend
        return signal
    
    def generate_class_2_series():
        """Class 2: 0.6 * trend + 0.4 * step"""
        a_trend = np.random.uniform(*trend_a_range)
        b_trend = np.random.uniform(*trend_b_range)
        a_step = np.random.uniform(*step_a_range)
        
        trend = generate_trend(t, a_trend, b_trend)
        step = generate_step(t, a_step)
        signal = 0.6 * trend + 0.4 * step
        return signal
    
    # Generate training data
    print("Generating training data...")
    Y_train_list = []
    labels_train_list = []
    
    samples_per_class_train = num_train // 3
    for class_idx in range(3):
        for _ in range(samples_per_class_train):
            if class_idx == 0:
                signal = generate_class_0_series()
            elif class_idx == 1:
                signal = generate_class_1_series()
            else:  # class_idx == 2
                signal = generate_class_2_series()
            
            # Add noise
            noise = np.random.normal(0, noise_sigma, size=signal.shape)
            noisy_signal = signal + noise
            
            # Reshape to (1, num_timestamps) for univariate time series
            Y_train_list.append(noisy_signal.reshape(1, -1))
            labels_train_list.append(class_idx)
    
    # Generate test data
    print("Generating test data...")
    Y_test_list = []
    labels_test_list = []
    
    samples_per_class_test = num_test // 3
    for class_idx in range(3):
        for _ in range(samples_per_class_test):
            if class_idx == 0:
                signal = generate_class_0_series()
            elif class_idx == 1:
                signal = generate_class_1_series()
            else:  # class_idx == 2
                signal = generate_class_2_series()
            
            # Add noise
            noise = np.random.normal(0, noise_sigma, size=signal.shape)
            noisy_signal = signal + noise
            
            # Reshape to (1, num_timestamps) for univariate time series
            Y_test_list.append(noisy_signal.reshape(1, -1))
            labels_test_list.append(class_idx)
    
    # Convert to numpy arrays in numpy3d format: (samples, channels, timesteps)
    Y_train = np.array(Y_train_list)  # Shape: (num_train, 1, num_timestamps)
    Y_test = np.array(Y_test_list)    # Shape: (num_test, 1, num_timestamps)
    labels_train = np.array(labels_train_list, dtype=int)
    labels_test = np.array(labels_test_list, dtype=int)
    
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print(f"labels_train shape: {labels_train.shape}, unique: {np.unique(labels_train)}")
    print(f"labels_test shape: {labels_test.shape}, unique: {np.unique(labels_test)}")
    
    return Y_train, labels_train, Y_test, labels_test

def main():
    dataset_name = 'Synthetic3Class'
    
    # Generate dataset
    Y_train, labels_train, Y_test, labels_test = generate_synthetic_dataset(
        num_train=90,   # 20 per class
        num_test=60,    # 20 per class
        num_timestamps=150,
        noise_sigma=0.1,
        random_seed=42
    )
    
    # Save to .npy file in the same format as add_noise_and_save_data.py
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}.npy')
    
    data_dict = {
        'Y_train': Y_train,
        'labels_train': labels_train,
        'Y_test': Y_test,
        'labels_test': labels_test
    }
    
    np.save(output_path, data_dict, allow_pickle=True)
    print(f"\nSaved synthetic dataset to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"  Classes: 3")
    print(f"  Train samples: {len(Y_train)} ({len(Y_train)//3} per class)")
    print(f"  Test samples: {len(Y_test)} ({len(Y_test)//3} per class)")
    print(f"  Timestamps per series: {Y_train.shape[2]}")
    print(f"  Noise sigma: 0.1")
    print("\nClass definitions:")
    print("  Class 0: 0.8 * periodic + 0.2 * step")
    print("  Class 1: 0.7 * periodic + 0.3 * trend")
    print("  Class 2: 0.6 * trend + 0.4 * step")

if __name__ == '__main__':
    main()

