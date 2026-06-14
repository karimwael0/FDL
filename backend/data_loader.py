import pandas as pd
import numpy as np
import torch

def load_and_preprocess_chunk(file_path, scaler, sequence_length=10):
    """
    Loads a chunk of the CICIoT2023 dataset, normalizes features using a PRE-FITTED scaler,
    applies a sliding window, and returns PyTorch tensors.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Drop any missing values
    df = df.dropna()
    
    # Ensure there's enough data for at least one sequence
    if df.empty or len(df) < sequence_length:
        return torch.tensor([]), torch.tensor([])

    # Separate label
    y = df['label'].values
    X = df.drop(columns=['label'])
    
    # Convert labels to binary format based on CICIoT2023 specs
    if y.dtype == object or y.dtype.name == 'category':
        y = np.where(y == 'BenignTraffic', 0, 1)
    else:
        y = np.array(y)

    # Normalize the 47 numerical features using the GLOBAL pre-fitted scaler
    # DO NOT use fit_transform here.
    X_scaled = scaler.transform(X)
    
    # --- Sliding Window Algorithm ---
    num_samples = len(X_scaled) - sequence_length + 1
    
    # Process features into 3D array: (num_samples, sequence_length, 47)
    X_seqs = np.array([X_scaled[i : i + sequence_length] for i in range(num_samples)], dtype=np.float32)
    
    # Assign the label of the last packet in each sequence window
    y_seqs = y[sequence_length - 1:].astype(np.float32).reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_seqs, dtype=torch.float32)
    y_train = torch.tensor(y_seqs, dtype=torch.float32)

    return X_train, y_train