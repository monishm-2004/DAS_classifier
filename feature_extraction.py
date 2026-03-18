import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
from scipy import signal, stats

# Path setup
base_path_train = Path('/home/monish-m/Downloads/das/70-30 Data/train')
base_path_test = Path('/home/monish-m/Downloads/das/70-30 Data/test')
label_file_train = base_path_train / 'label.txt'
label_file_test = base_path_test / 'label.txt'

def extract_features(data):
    """
    Extract specific vibration analysis features from a single data sample (10000 x 12 array).
    
    Features extracted per channel:
    - dif_max, dif_min, dif_pk: Amplitude-based features
    - dif_mean, dif_var, dif_std, dif_rms: Statistical features
    - dif_energy: Average squared magnitude of frequency spectrum
    - dif_arv: Average rectified value
    - dif_boxing: Waveform factor (RMS/ARV)
    - dif_maichong: Pulse factor (Max/ARV)
    - dif_fengzhi: Crest factor (Max/RMS)
    - dif_yudu: Margin factor (Max / sqrt(mean(sqrt(|x|))))
    - dif_kurt: Kurtosis
    - dif_qiaodu: Skewness
    - dif_entropy: Information entropy (frequency domain)
    
    Parameters:
    data: numpy array of shape (samples, channels)
    
    Returns:
    dict with aggregated features across all channels
    """
    features = {}
    
    # Process each channel
    channel_stats = []
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        
        # Basic amplitude features
        dif_max = np.max(channel_data)
        dif_min = np.min(channel_data)
        dif_pk = dif_max - dif_min  # Peak-to-peak
        
        # Statistical features
        dif_mean = np.mean(channel_data)
        dif_var = np.var(channel_data)
        dif_std = np.std(channel_data)
        
        # Energy and RMS
        dif_rms = np.sqrt(np.mean(channel_data ** 2))
        
        # Energy: Average squared magnitude of frequency spectrum
        fft = np.abs(np.fft.fft(channel_data))
        dif_energy = np.mean(fft ** 2)
        
        # Average Rectified Value (ARV)
        dif_arv = np.mean(np.abs(channel_data))
        
        # Waveform Factor (RMS / ARV) - describes wave shape
        dif_boxing = dif_rms / dif_arv if dif_arv != 0 else 0
        
        # Pulse Factor (Max / ARV) - detects sudden impacts
        dif_maichong = dif_max / dif_arv if dif_arv != 0 else 0
        
        # Crest Factor (Max / RMS) - detects sharp peaks
        dif_fengzhi = dif_max / dif_rms if dif_rms != 0 else 0
        
        # Margin Factor (Max / sqrt(mean(sqrt(|x|))))
        margin_divisor = np.sqrt(np.mean(np.sqrt(np.abs(channel_data))))
        dif_yudu = dif_max / margin_divisor if margin_divisor != 0 else 0
        
        # Kurtosis - measures "peakiness" or impulses
        dif_kurt = stats.kurtosis(channel_data)
        
        # Skewness/Clearance Factor - asymmetry/sharpness
        dif_qiaodu = stats.skew(channel_data)
        
        # Information Entropy - complexity/randomness in frequency domain
        fft_normalized = fft / np.sum(fft)
        dif_entropy = -np.sum(fft_normalized[fft_normalized > 0] * np.log2(fft_normalized[fft_normalized > 0]))
        
        channel_stats.append({
            'dif_max': dif_max,
            'dif_min': dif_min,
            'dif_pk': dif_pk,
            'dif_mean': dif_mean,
            'dif_var': dif_var,
            'dif_std': dif_std,
            'dif_rms': dif_rms,
            'dif_energy': dif_energy,
            'dif_arv': dif_arv,
            'dif_boxing': dif_boxing,
            'dif_maichong': dif_maichong,
            'dif_fengzhi': dif_fengzhi,
            'dif_yudu': dif_yudu,
            'dif_kurt': dif_kurt,
            'dif_qiaodu': dif_qiaodu,
            'dif_entropy': dif_entropy,
        })
    
    # Aggregate features across channels (mean, std, min, max)
    channel_stats_array = np.array([list(d.values()) for d in channel_stats])
    
    for i, stat_name in enumerate(channel_stats[0].keys()):
        # Mean across channels
        features[f'{stat_name}_mean'] = np.mean(channel_stats_array[:, i])
        # Std across channels
        features[f'{stat_name}_std'] = np.std(channel_stats_array[:, i])
        # Min across channels
        features[f'{stat_name}_min'] = np.min(channel_stats_array[:, i])
        # Max across channels
        features[f'{stat_name}_max'] = np.max(channel_stats_array[:, i])
    
    return features

def load_and_extract_features(file_path, base_path):
    """
    Load a .mat file and extract features.
    
    Parameters:
    file_path: path to the .mat file (relative to base_path)
    base_path: base directory path
    
    Returns:
    dict with features or None if error occurs
    """
    try:
        full_path = base_path / file_path.lstrip('/')
        mat_data = scipy.io.loadmat(full_path)
        data = mat_data['data']
        features = extract_features(data)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(label_file, base_path, dataset_name):
    """
    Process a dataset (train or test) and extract features.
    
    Parameters:
    label_file: path to the label file
    base_path: base directory path
    dataset_name: name of the dataset ('train' or 'test')
    
    Returns:
    DataFrame with extracted features
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    print("Reading label file...")
    
    # Read label file
    file_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_path = parts[0]
            label = int(parts[1])
            file_labels.append((file_path, label))
    
    print(f"Found {len(file_labels)} labeled files")
    
    # Extract features for all files
    print("Extracting features...")
    all_features = []
    all_labels = []
    
    for file_path, label in tqdm.tqdm(file_labels, desc=f"Processing {dataset_name} files"):
        features = load_and_extract_features(file_path, base_path)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
    
    print(f"Successfully processed {len(all_features)} files")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    
    # Create label mapping
    label_mapping = {
        0: 'background',
        1: 'dig',
        2: 'knock',
        3: 'water',
        4: 'shake',
        5: 'walk'
    }
    df['activity'] = df['label'].map(label_mapping)
    
    # Display info
    print("\n" + "-"*60)
    print(f"{dataset_name.upper()} DATASET SUMMARY")
    print("-"*60)
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 2}")  # Exclude label and activity
    print("\nActivity distribution:")
    print(df['activity'].value_counts().sort_index())
    print("\nDataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Save the DataFrame
    output_csv = base_path / f'features_dataset_{dataset_name}.csv'
    output_pkl = base_path / f'features_dataset_{dataset_name}.pkl'
    
    df.to_csv(output_csv, index=False)
    df.to_pickle(output_pkl)
    
    print(f"\n✓ CSV saved to: {output_csv}")
    print(f"✓ Pickle saved to: {output_pkl}")
    
    return df

def main():
    """
    Main function to process train and test datasets separately.
    """
    # Process train dataset
    df_train = process_dataset(label_file_train, base_path_train, 'train')
    
    # Process test dataset
    df_test = process_dataset(label_file_test, base_path_test, 'test')
    
    return df_train, df_test

if __name__ == '__main__':
    df_train, df_test = main()
