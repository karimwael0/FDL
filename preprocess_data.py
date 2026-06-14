import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_benign_data(raw_dir='data/raw', out_dir='data/federated'):
    """
    Loads, cleans, labels, and partitions raw benign data into federated split nodes.
    """
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if not csv_files:
        print(f"[-] No CSV files found in {raw_dir}")
        print("Please ensure the CIC IoT 2023 benign CSVs are placed there.")
        return
    
    print(f"[+] Found {len(csv_files)} dataset files in {raw_dir}. Loading...")
    
    dfs = []
    for file in csv_files:
        print(f"    -> Loading {os.path.basename(file)}")
        dfs.append(pd.read_csv(file))
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"[+] Combined Raw Dataset Shape: {df.shape}")
    
    # 1. Clean: Drop Identifier Columns
    identifiers = ['Source IP', 'Destination IP', 'Timestamp']
    # Safely drop only those that exist
    drop_targets = [col for col in identifiers if col in df.columns]
    if drop_targets:
        df.drop(columns=drop_targets, inplace=True)
        print(f"[+] Dropped identifier columns: {drop_targets}")
        
    # Handle NaN and Infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_rows:
        print(f"[+] Dropped {initial_rows - len(df)} rows containing NaN or Infinity.")
        
    # 2. Labeling (0 = Benign)
    df['label'] = 0
    print("[+] Added 'label' column (0 = Benign)")
    
    # 3. Scaling
    print("[+] Normalizing numeric features using StandardScaler...")
    # Get all features except the 'label'
    features = [col for col in df.columns if col != 'label']
    numeric_features = df[features].select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # 4. FDL Distribution (Split equally into 3 nodes)
    # Shuffle the dataset thoroughly to prevent time-series grouping bias across nodes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    partitions = np.array_split(df, 3)
    
    for i, partition in enumerate(partitions, 1):
        node_folder = os.path.join(out_dir, f'node_{i}')
        os.makedirs(node_folder, exist_ok=True)
        
        save_path = os.path.join(node_folder, 'processed_data.csv')
        partition.to_csv(save_path, index=False)
        print(f"[+] Node {i} provisioned: Saved {len(partition)} rows to {save_path}")
        
    print("\n[SUCCESS] Benign data Federated pre-processing complete!")


def add_attack_data(node_path, attack_csv_path):
    """
    PLACEHOLDER FUNCTION
    Use this later to inject malicious attack vectors (Mirai, DDoS) 
    into a specific node's processed dataset for the Federated evaluation rounds.
    """
    print(f"Warning: add_attack_data not fully implemented yet.")
    print(f"Would mix attack data from '{attack_csv_path}' into '{node_path}/processed_data.csv'")
    # Example logical flow for later:
    # 1. Load node's processed_data.csv
    # 2. Load attack_csv_path
    # 3. Drop identifiers, handle inf/nan, Label Attack as 1
    # 4. Scale Attack numeric features (Ideally using the same scaler fitted on Benign)
    # 5. pd.concat both DataFrames, Shuffle, and save back to node_path
    pass


if __name__ == "__main__":
    # Assumes execution from project root
    preprocess_benign_data(raw_dir='data/raw', out_dir='data/federated')
