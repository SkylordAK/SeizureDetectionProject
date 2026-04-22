import os
import pandas as pd
import numpy as np
import mne
from sklearn.model_selection import train_test_split

# Configurations
FS = 256  # Sampling frequency
Wn = 2    # 2 second window for seizure data
overlap = 0.5 * Wn # 50% overlap
CHANNELS = 4

# Assuming data is in a Data directory within the project folder for now
# Users can point to their specific data folder by changing this
DATA_DIR = r"Data"
OUTPUT_DIR = r"ProcessedData"

def load_and_preprocess_data():
    files = {
        'Normal': os.path.join(DATA_DIR, "Normal.csv"),
        'Seizure': os.path.join(DATA_DIR, "Seizure.csv"),
        'Epilepsy': os.path.join(DATA_DIR, "Epilepsy.csv")
    }
    
    classes = {'Normal': 0, 'Seizure': 1, 'Epilepsy': 2}
    
    X = []
    y = []
    
    for class_name, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {class_name}...")
            continue
            
        print(f"Loading {class_name} data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Keep only the EEG columns (the raw sensor data)
        eeg_cols = [col for col in df.columns if 'RAW' in col]
        if len(eeg_cols) != CHANNELS:
            print(f"Expected {CHANNELS} EEG channels, found {len(eeg_cols)} in {file_path}.")
            continue
            
        eeg_data = df[eeg_cols].values # Shape: (Total Samples, Channels)
        
        # Creating MNE raw array
        # Muse channels: TP9, AF7, AF8, TP10
        ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
        info = mne.create_info(ch_names=ch_names, ch_types=['eeg'] * CHANNELS, sfreq=FS)
        info.set_montage('standard_1020')
        raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
        raw.set_eeg_reference(verbose=False)
        
        # Create fixed length epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=Wn, overlap=overlap, verbose=False)
        epoch_data = epochs.get_data(copy=True) # Shape: (Epochs, Channels, Samples)
        
        # For our 1D CNN / LSTM, Keras expects (Epochs, Samples, Channels)
        epoch_data = np.transpose(epoch_data, (0, 2, 1))
        
        X.append(epoch_data)
        y.extend([classes[class_name]] * len(epoch_data))
        
    if not X:
        print("No valid data loaded. Please check your DATA_DIR and CSV files.")
        return None, None
        
    X = np.concatenate(X, axis=0) # Shape: (Total Epochs, Samples, Channels)
    y = np.array(y)
    
    print(f"\nTotal extracted Epochs: {X.shape[0]}")
    print(f"Data shape: {X.shape}") 
    
    return X, y

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Step 1: Loading and Preprocessing Data...")
    X, y = load_and_preprocess_data()
    
    if X is None:
        return
        
    print(f"\nStep 2: Saving Preprocessed Data to '{OUTPUT_DIR}'...")
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    print("Data saved successfully.")

if __name__ == "__main__":
    main()
