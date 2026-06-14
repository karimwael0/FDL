import os
import glob
import numpy as np
import tensorflow as tf

from model_arch import create_hybrid_model

def main():
    # 1. Find the latest saved weights from the Flower Server NPZ dump
    weight_files = glob.glob("round-*-weights.npz")
    if not weight_files:
        print("[-] Could not find any round-*-weights.npz files in the current directory.")
        print("[-] Please ensure the Flower Server generated the NPZ parameters successfully.")
        return

    # Sort to find the latest round
    def get_round_number(filename):
        return int(filename.split('-')[1])
        
    latest_weights_file = sorted(weight_files, key=get_round_number)[-1]
    print(f"[+] Discovered aggregated global weights: {latest_weights_file}")

    # 2. Load the npz file
    npz_data = np.load(latest_weights_file)
    weights = [npz_data[key] for key in npz_data.files]

    # 3. Instantiate the Model Architecture
    input_dim = weights[0].shape[1]
    print(f"[+] Reconstructing 1D-CNN-LSTM model with input_dim = {input_dim}")
    
    model = create_hybrid_model(input_dim)
    
    # 4. Inject the weights into the model
    try:
        model.set_weights(weights)
        print(f"[+] Successfully injected global weights into the neural network graph.")
    except ValueError as e:
        print(f"[-] ValueError injecting weights: {e}")
        return

    # 5. Save the final .h5 model
    keras_target_path = "iot_defense_model.h5"
    model.save(keras_target_path)
    print(f"[SUCCESS] Deep Learning intelligence formally saved to {keras_target_path}!")

if __name__ == "__main__":
    main()
