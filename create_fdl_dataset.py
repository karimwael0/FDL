import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_sample_data(directory, target_size, label_value):
    print(f"Reading from {directory}/ ...")
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    if not csv_files:
        print(f"[-] No CSV files found in {directory}")
        return pd.DataFrame()
        
    dfs = []
    for file in csv_files:
        # Read low_memory=False to avoid DtypeWarning from mixed type columns
        dfs.append(pd.read_csv(file, low_memory=False))
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Sample randomly if we have more than target_size
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=42)
    else:
        print(f"[!] Warning: Only found {len(df)} rows in {directory}, which is less than requested {target_size}.")
        
    df['label'] = label_value
    return df

def main():
    # 1. Load Data and Sample
    df_benign = load_and_sample_data('data/raw/benign', 20000, 0)
    df_mirai = load_and_sample_data('data/raw/mirai', 20000, 1)

    if df_benign.empty and df_mirai.empty:
        print("Failed to load any data.")
        return

    # 2. Combine Datasets
    df_combined = pd.concat([df_benign, df_mirai], ignore_index=True)
    print(f"\n[+] Combined dataset size: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns.")

    # 3. Clean Features: Drop non-numeric (drops IP, Timestamps mapped as strings, etc)
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    df_combined = df_combined[numeric_cols]
    
    # Clean NaN and Inf values just in case
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)
    print(f"[+] After cleaning non-numeric, NaNs, and Infs: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns.")

    # 4. Standard Scaler
    print("[+] Applying StandardScaler...")
    scaler = StandardScaler()
    feature_cols = [c for c in df_combined.columns if c != 'label']
    df_combined[feature_cols] = scaler.fit_transform(df_combined[feature_cols])

    # 5. Distribute to Nodes
    # Shuffle entire combined dataset prior to partition
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into 3 nodes by index to preserve DataFrame
    idx_partitions = np.array_split(df_combined.index, 3)
    partitions = [df_combined.loc[idx] for idx in idx_partitions]
    
    node_configs = [
        {'id': '1', 'name': 'node_1', 'desc': 'Smart Lock'},
        {'id': '2', 'name': 'node_2', 'desc': 'Camera'},
        {'id': '3', 'name': 'node_3', 'desc': 'Fridge'}
    ]

    print("\n--- Federated Node Distribution Summary ---")
    for partition, config in zip(partitions, node_configs):
        node_dir = os.path.join('data', 'federated', config['name'])
        os.makedirs(node_dir, exist_ok=True)
        
        # 80/20 train/test split per node
        train_df, test_df = train_test_split(partition, test_size=0.2, random_state=42)
        
        train_df.to_csv(os.path.join(node_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(node_dir, 'test.csv'), index=False)
        
        safe_count = len(partition[partition['label'] == 0])
        attack_count = len(partition[partition['label'] == 1])
        
        print(f"\n{config['name'].upper()} ({config['desc']})")
        print(f"  -> Total points: {len(partition)}")
        print(f"  -> 'Safe': {safe_count}")
        print(f"  -> 'Attack' (Mirai): {attack_count}")
        print(f"  -> Train points: {len(train_df)}")
        print(f"  -> Test points: {len(test_df)}")

    print("\n[SUCCESS] Dataset created and distributed successfully!")

if __name__ == "__main__":
    main()
