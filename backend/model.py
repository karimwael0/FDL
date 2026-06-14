import torch
import torch.nn as nn
import torch.nn.functional as F

class IoTHybridDefense(nn.Module):
    def __init__(self, num_features=47, cnn_out_channels=64, lstm_hidden_dim=128, lstm_layers=1, num_classes=1):
        """
        Hybrid CNN-LSTM Architecture for IoT Intrusion Detection.
        Expects input sequences of shape: (batch_size, sequence_length, num_features=47)
        """
        super(IoTHybridDefense, self).__init__()
        
        # --- CNN Block ---
        # The input will be permuted to (batch, features, seq_len) 
        # so in_channels matches num_features (47)
        self.conv1 = nn.Conv1d(
            in_channels=num_features, 
            out_channels=cnn_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # --- LSTM Block ---
        # Output of CNN permuted back to (batch, seq_len_out, cnn_out_channels)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=lstm_hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # --- Classifier ---
        self.dropout = nn.Dropout(p=0.5)
        # Using a single output node for binary classification with BCEWithLogitsLoss
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x):
        # Initial shape of x: (batch_size, sequence_length, num_features)
        
        # 1. Permute dimensions for Conv1d -> (batch_size, num_features, sequence_length)
        x = x.permute(0, 2, 1)
        
        # 2. CNN Block (Conv, ReLU, MaxPool)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 3. Permute dimensions back for LSTM -> (batch_size, sequence_length_out, cnn_out_channels)
        x = x.permute(0, 2, 1)
        
        # 4. LSTM Block
        # lstm_out contains all states: (batch_size, sequence_length, hidden_dim)
        # hn shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 5. Extract the final hidden state from the LSTM
        # We index the last time step of lstm_out
        final_hidden_state = lstm_out[:, -1, :]
        
        # 6. Classifier
        x = self.dropout(final_hidden_state)
        logits = self.fc(x)
        
        return logits
