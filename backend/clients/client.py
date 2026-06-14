import argparse
import sys
import os
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from sklearn.preprocessing import StandardScaler

# Add backend to path to import properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import IoTHybridDefense
from data_loader import load_and_preprocess_chunk

# ==========================================
# Crucial PyTorch State Dict Helper Functions
# ==========================================

def get_parameters_helper(model: nn.Module):
    """
    Extracts the PyTorch state dict into a list of NumPy arrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters_helper(model: nn.Module, parameters):
    """
    Loads a list of NumPy arrays back into the PyTorch state dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class IoTHybridClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, criterion, optimizer):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer

    def get_parameters(self, config):
        return get_parameters_helper(self.model)

    def set_parameters(self, parameters):
        set_parameters_helper(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.train()
        for epoch in range(1): # Local epochs
            for sequences, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
        return get_parameters_helper(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in self.testloader:
                outputs = self.model(sequences)
                loss += self.criterion(outputs, labels).item()
                
                preds = torch.sigmoid(outputs) >= 0.5
                total += labels.size(0)
                correct += (preds.float() == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser(description="Flower IoT Client")
    parser.add_argument("--data", type=str, default="../../data/chunk_1.csv", help="Dataset chunk path")
    args = parser.parse_args()

    # Create a pre-fitted scaler mock (in production, this would be loaded via pickle/joblib)
    # This aligns correctly with your data_loader requirement.
    scaler = StandardScaler()
    scaler.fit(np.zeros((1, 47))) # Mock fitting for feature size mapping

    try:
        X_data, y_data = load_and_preprocess_chunk(args.data, scaler=scaler, sequence_length=10)
        
        # Split into train/test (80/20)
        split_idx = int(len(X_data) * 0.8)
        X_train, y_train = X_data[:split_idx], y_data[:split_idx]
        X_test, y_test = X_data[split_idx:], y_data[split_idx:]
        
    except (FileNotFoundError, OSError):
        print("[WARNING] Data not found. Using dummy PyTorch Tensors for scaffolding.")
        X_train = torch.randn(100, 10, 47)
        y_train = torch.randint(0, 2, (100, 1)).float()
        X_test = torch.randn(20, 10, 47)
        y_test = torch.randint(0, 2, (20, 1)).float()

    # Load heavily typed PyTorch DataLoaders natively
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Initialize Model, Loss Function, and Optimizer
    model = IoTHybridDefense()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    client = IoTHybridClient(model, trainloader, testloader, criterion, optimizer)
    
    # Start the client loop referencing 8080
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
