"""
Script to add noise to datasets and save them to data/dataset_name.npy

Usage:
    python add_noise_and_save_data.py <dataset_name>
    python add_noise_and_save_data.py --all  # Process all basic datasets
"""
import numpy as np
import os
import sys
import argparse
from data_processing import load_data

# List of basic datasets
BASIC_DATASETS = [
    'Chinatown', 'ECGFiveDays', 'FreezerSmallTrain', 'HouseTwenty',
    'InsectEPGRegularTrain', 'ItalyPowerDemand', 'Lightning7',
    'MoteStrain', 'PowerCons', 'SonyAIBORobotSurface2', 
    'UWaveGestureLibraryAll', 'GunPointOldVersusYoung'
]

def add_noise_to_data(Y, noise_level=0.3):
    """
    Add noise to data.
    
    Parameters
    ----------
    Y: numpy.ndarray
        Time series data
    noise_level: float
        Noise level as fraction of max absolute value (default: 0.3 = 30%)
    
    Returns
    -------
    Y_noisy: numpy.ndarray
        Data with noise added
    """
    noise = np.random.normal(size=Y.shape) * noise_level * np.max(np.abs(Y))
    return Y + noise

def save_dataset_with_noise(dataset_name, output_dir='data', noise_level=0.3):
    """
    Load dataset, add noise, and save to npy file.
    
    Parameters
    ----------
    dataset_name: str
        Name of the dataset
    output_dir: str
        Output directory (default: 'data')
    noise_level: float
        Noise level as fraction of max absolute value (default: 0.3 = 30%)
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load train and test data
        print("Loading train data...")
        Y_train, labels_train = load_data(dataset_name, split='train', add_noise=False)
        print(f"Train shape: {Y_train.shape}, Labels shape: {labels_train.shape}")
        
        print("Loading test data...")
        Y_test, labels_test = load_data(dataset_name, split='test', add_noise=False)
        print(f"Test shape: {Y_test.shape}, Labels shape: {labels_test.shape}")
        
        # Add noise to train and test data
        print(f"Adding noise (level={noise_level*100}% of max absolute value)...")
        Y_train_noisy = add_noise_to_data(Y_train, noise_level)
        Y_test_noisy = add_noise_to_data(Y_test, noise_level)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to npy file as dictionary
        output_path = os.path.join(output_dir, f"{dataset_name}.npy")
        data_dict = {
            'Y_train': Y_train_noisy,
            'labels_train': labels_train,
            'Y_test': Y_test_noisy,
            'labels_test': labels_test
        }
        
        np.save(output_path, data_dict, allow_pickle=True)
        print(f"Saved noisy data to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Add noise to datasets and save to .npy files')
    parser.add_argument('dataset_name', nargs='?', help='Name of the dataset to process')
    parser.add_argument('--all', action='store_true', help='Process all basic datasets')
    parser.add_argument('--output-dir', default='data', help='Output directory (default: data)')
    parser.add_argument('--noise-level', type=float, default=0.3, 
                       help='Noise level as fraction of max absolute value (default: 0.3)')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all basic datasets
        print(f"Processing {len(BASIC_DATASETS)} datasets...")
        success_count = 0
        for dataset_name in BASIC_DATASETS:
            if save_dataset_with_noise(dataset_name, args.output_dir, args.noise_level):
                success_count += 1
        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{len(BASIC_DATASETS)} datasets processed successfully")
        print(f"{'='*60}")
    elif args.dataset_name:
        # Process single dataset
        save_dataset_with_noise(args.dataset_name, args.output_dir, args.noise_level)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()

